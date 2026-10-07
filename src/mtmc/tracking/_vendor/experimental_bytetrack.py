# Adapted from supervision 0.30.7 tracker/byte_tracker/core.py.
# Copyright (c) 2022 Roboflow. MIT license: LICENSE.supervision.
# Original SHA256: 018911df3dd600c7a0abbd059138b116185a458f2b4877b4c1fd32f5c341c90d
# Modifications: candidate provenance, bounded appearance history hooks, optional
# pre-assignment appearance mask, direct output control, and fail-stop validation.
import numpy as np
import numpy.typing as npt

from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import box_iou_batch
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from supervision.tracker.byte_tracker.single_object_track import TrackState
from mtmc.tracking.appearance_track import CandidateTrack as STrack
from supervision.tracker.byte_tracker.utils import IdCounter


def _valid_tracking_tensors(
    tensors: npt.NDArray[np.float32],
) -> npt.NDArray[np.bool_]:
    """Identify finite tensors with positive-width and positive-height boxes."""
    bboxes = tensors[:, :4]
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    return np.isfinite(tensors).all(axis=1) & (widths > 0) & (heights > 0)


class ExperimentalByteTrack:
    """Pinned ByteTrack lifecycle with candidate provenance and optional appearance gating."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: float = 30,
        minimum_consecutive_frames: int = 1,
        *, direct_output: bool = False, appearance_threshold: float | None = None,
    ) -> None:
        if appearance_threshold is not None and (not np.isfinite(appearance_threshold) or not -1 <= appearance_threshold <= 1):
            raise ValueError("Invalid appearance threshold")
        if appearance_threshold is not None and not direct_output:
            raise ValueError("Appearance mode requires direct candidate output")
        self.direct_output = direct_output
        self.appearance_threshold = appearance_threshold
        self._features = ()
        self.statistics = {}
        self.selected_indices = ()
        self.failed = False
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold

        self.frame_id = 0
        self.det_thresh = self.track_activation_threshold + 0.1
        if self.det_thresh > 1.0:
            self.det_thresh = self.track_activation_threshold
        self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.kalman_filter = KalmanFilter()
        self.shared_kalman = KalmanFilter()

        self.tracked_tracks: list[STrack] = []
        self.lost_tracks: list[STrack] = []
        self.removed_tracks: list[STrack] = []

        # Warning, possible bug: If you also set internal_id to start at 1,
        # all traces will be connected across objects.
        self.internal_id_counter = IdCounter()
        self.external_id_counter = IdCounter(start_id=1)

    def update_candidates(self, boxes, scores, features):
        if self.failed:
            raise RuntimeError("Tracker is failed; construct a fresh instance")
        if (not isinstance(boxes, np.ndarray) or boxes.dtype != np.float32 or boxes.ndim != 2
                or boxes.shape[1] != 4 or not np.isfinite(boxes).all()
                or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError("Expected positive finite float32 detection boxes")
        if (not isinstance(scores, np.ndarray) or scores.dtype != np.float32
                or scores.shape != (len(boxes),) or not np.isfinite(scores).all()
                or np.any((scores < 0) | (scores > 1)) or len(features) != len(boxes)):
            raise ValueError("Invalid confidence or feature mapping")
        copied = []
        for feature in features:
            if feature is None:
                copied.append(None)
                continue
            if (not isinstance(feature, np.ndarray) or feature.dtype != np.float32
                    or feature.shape != (512,) or not np.isfinite(feature).all()
                    or abs(float(np.linalg.norm(feature)) - 1) > 1e-5):
                raise ValueError("Invalid candidate descriptor")
            copied.append(feature.copy())
        self._features = tuple(copied)
        self.statistics = {"high_compared": 0, "high_blocked": 0, "high_unavailable": 0,
                           "unconfirmed_compared": 0, "unconfirmed_blocked": 0, "unconfirmed_unavailable": 0}
        try:
            detections = Detections(xyxy=boxes.copy(), confidence=scores.copy(),
                                    class_id=np.zeros(len(boxes), dtype=int))
            return self.update_with_detections(detections)
        except Exception:
            self.failed = True
            raise

    def _appearance_gate(self, costs, tracks, detections, cost_limit, phase):
        if self.appearance_threshold is None:
            return costs
        from mtmc.tracking.appearance_track import gate_costs
        output, counters = gate_costs(costs, [t.descriptor(self.frame_id) for t in tracks],
            [d.current_feature for d in detections], cost_limit, self.appearance_threshold)
        for name, count in counters.items():
            self.statistics[phase + "_" + name] += count
        return output

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the tracker with the provided detections and returns the updated
        detection results.

        Args:
            detections: The detections to pass through the tracker.

        Example:
            ```python
            import supervision as sv
            from rfdetr import RFDETRMedium

            model = RFDETRMedium()
            tracker = sv.ByteTrack()

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            def callback(frame: np.ndarray, index: int) -> np.ndarray:
                detections = model.predict(frame[:, :, ::-1])
                detections = tracker.update_with_detections(detections)

                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels)
                return annotated_frame

            sv.process_video(
                source_path="<SOURCE_VIDEO_PATH>",
                target_path="<TARGET_VIDEO_PATH>",
                callback=callback
            )
            ```
        """
        if detections.confidence is None:
            raise ValueError("Detections confidence must be provided for tracking.")

        tensors = np.hstack(
            (
                detections.xyxy,
                detections.confidence[:, np.newaxis],
            )
        )
        tracks = self.update_with_tensors(tensors=tensors)

        if self.direct_output:
            selected = sorted((track.candidate_index, track.external_track_id) for track in tracks)
            if any(track.candidate_frame != self.frame_id for track in tracks):
                raise RuntimeError("Returned track has no current candidate")
            indices = np.asarray([i for i, _ in selected], dtype=int)
            if len(set(indices.tolist())) != len(indices):
                raise RuntimeError("A detection was assigned twice")
            result = detections.select(indices)
            result.tracker_id = np.asarray([gid for _, gid in selected], dtype=int)
            self.selected_indices = tuple(int(i) for i in indices)
            return result

        if len(tracks) > 0:
            detection_bounding_boxes = np.asarray([det[:4] for det in tensors])
            track_bounding_boxes = np.asarray([track.tlbr for track in tracks])

            ious = box_iou_batch(detection_bounding_boxes, track_bounding_boxes)

            iou_costs: npt.NDArray[np.float32] = 1 - ious

            matches, _, _ = matching.linear_assignment(iou_costs, 0.5)
            tracked_detections = detections.select(slice(None))
            tracked_detections.tracker_id = np.full(len(detections), -1, dtype=int)
            for i_detection, i_track in matches:
                tracked_detections.tracker_id[i_detection] = int(
                    tracks[i_track].external_track_id
                )

            return tracked_detections.select(tracked_detections.tracker_id != -1)

        else:
            detections = Detections.empty()
            detections.tracker_id = np.array([], dtype=int)

            return detections

    def reset(self) -> None:
        """
        Resets the internal state of the ByteTrack tracker.

        This method clears the tracking data, including tracked, lost,
        and removed tracks, as well as resetting the frame counter. It's
        particularly useful when processing multiple videos sequentially,
        ensuring the tracker starts with a clean state for each new video.
        """
        self.frame_id = 0
        self.internal_id_counter.reset()
        self.external_id_counter.reset()
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []

    def update_with_tensors(self, tensors: npt.NDArray[np.float32]) -> list[STrack]:
        """
        Updates the tracker with the provided tensors and returns the updated tracks.

        Args:
            tensors: The new tensors to update with.

        Returns:
            Updated tracks.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        tensors = tensors[_valid_tracking_tensors(tensors)]
        scores = tensors[:, 4]
        bboxes = tensors[:, :4]

        remain_inds = scores >= self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_keep,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                    candidate_index=int(index), feature=self._features[index], learn_feature=True,
                )
                for (tlbr, score_keep, index) in zip(dets, scores_keep, np.flatnonzero(remain_inds))
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks: list[STrack] = []

        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.shared_kalman)
        dists = matching.iou_distance(strack_pool, detections)

        dists = matching.fuse_score(dists, detections)
        dists = self._appearance_gate(dists, strack_pool, detections, self.minimum_matching_threshold, "high")
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_second,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                    candidate_index=int(index), feature=self._features[index], learn_feature=False,
                )
                for (tlbr, score_second, index) in zip(dets_second, scores_second, np.flatnonzero(inds_second))
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        dists = matching.fuse_score(dists, detections)
        dists = self._appearance_gate(dists, unconfirmed, detections, 0.7, "unconfirmed")
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]

        return output_stracks


def joint_tracks(
    track_list_a: list[STrack], track_list_b: list[STrack]
) -> list[STrack]:
    """
    Joins two lists of tracks, ensuring that the resulting list does not
    contain tracks with duplicate internal_track_id values.

    Args:
        track_list_a: First list of tracks.
        track_list_b: Second list of tracks.

    Returns:
        Combined list of tracks from track_list_a and track_list_b
            without duplicate internal_track_id values.
    """
    seen_track_ids = set()
    result = []

    for track in track_list_a + track_list_b:
        if track.internal_track_id not in seen_track_ids:
            seen_track_ids.add(track.internal_track_id)
            result.append(track)

    return result


def sub_tracks(track_list_a: list[STrack], track_list_b: list[STrack]) -> list[STrack]:
    """
    Returns a list of tracks from track_list_a after removing any tracks
    that share the same internal_track_id with tracks in track_list_b.

    Args:
        track_list_a: List of tracks.
        track_list_b: List of tracks to be subtracted from track_list_a.
    Returns:
        List of remaining tracks from track_list_a after subtraction.
    """
    tracks = {track.internal_track_id: track for track in track_list_a}
    track_ids_b = {track.internal_track_id for track in track_list_b}

    for track_id in track_ids_b:
        tracks.pop(track_id, None)

    return list(tracks.values())


def remove_duplicate_tracks(
    tracks_a: list[STrack], tracks_b: list[STrack]
) -> tuple[list[STrack], list[STrack]]:
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
    matching_pairs = np.where(pairwise_distance < 0.05)

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
        time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    result_a = [
        track for index, track in enumerate(tracks_a) if index not in duplicates_a
    ]
    result_b = [
        track for index, track in enumerate(tracks_b) if index not in duplicates_b
    ]

    return result_a, result_b
