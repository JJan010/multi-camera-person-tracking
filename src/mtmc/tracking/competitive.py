# Tracking state machine adapted from supervision 0.30.7 / experimental_bytetrack.py.
# Copyright (c) 2022 Roboflow. MIT license: _vendor/LICENSE.supervision.
# Changes: joint low-score challenges to incumbent high matches, before any updates.
import numpy as np
import numpy.typing as npt

from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.single_object_track import TrackState
from mtmc.association.pairwise import solve_partial_assignment
from mtmc.tracking.appearance_track import CandidateTrack as STrack
from mtmc.tracking._vendor.experimental_bytetrack import (
    ExperimentalByteTrack, _valid_tracking_tensors, joint_tracks, sub_tracks, remove_duplicate_tracks,
)


def challenge_assignment(high_iou, low_iou, high_cos, low_cos, *, threshold, mode):
    """Keep incumbents at zero gain; jointly allocate only strictly better weak alternatives.

    Both ablations share geometry and appearance gates. Only the ranking differs.
    Missing appearance disables a challenge, never the incumbent/fallback tracker.
    """
    if mode not in ("iou", "appearance") or not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Invalid challenge configuration")
    high_iou, low_iou, high_cos, low_cos = [np.asarray(x, np.float64) for x in
                                            (high_iou, low_iou, high_cos, low_cos)]
    if (low_iou.ndim != 2 or high_iou.shape != (low_iou.shape[0],)
            or high_cos.shape != high_iou.shape or low_cos.shape != low_iou.shape
            or not np.isfinite(high_iou).all() or not np.isfinite(low_iou).all()
            or np.any((high_iou < 0) | (high_iou > 1)) or np.any((low_iou < 0) | (low_iou > 1))
            or np.isinf(high_cos).any() or np.isinf(low_cos).any()
            or np.any(np.abs(high_cos[np.isfinite(high_cos)]) > 1)
            or np.any(np.abs(low_cos[np.isfinite(low_cos)]) > 1)):
        raise ValueError("Invalid challenge matrices")
    eligible = ((low_iou >= .5) & np.isfinite(high_cos)[:, None]
                & np.isfinite(low_cos) & (low_cos >= threshold))
    gain = (low_cos - high_cos[:, None]) if mode == "appearance" else (low_iou - high_iou[:, None])
    # Division by two preserves the maximizer and fits the solver's [-1, 1] API.
    gains = np.where(eligible & (gain > 0), gain / 2, 0.)
    return solve_partial_assignment(gains, min_similarity=0.), eligible


def _cosine(a, b):
    if a is None or b is None:
        return np.nan
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))


class CompetitiveByteTrack(ExperimentalByteTrack):
    """Experimental pre-update refinement; not a general all-stage optimal tracker."""
    def __init__(self, *args, refinement="appearance", **kwargs):
        if refinement not in ("disabled", "iou", "appearance"):
            raise ValueError("Unknown refinement")
        super().__init__(*args, **kwargs)
        if not self.direct_output or self.appearance_threshold is None:
            raise ValueError("Competitive tracker requires direct output and the existing appearance gate")
        self.refinement = refinement
        self.refinement_events = []
        self.refinement_counts = {}

    def update_candidates(self, boxes, scores, features):
        self.refinement_events = []
        self.refinement_counts = {"incumbents": 0, "weak_candidates": 0,
                                  "weak_reserved_for_unmatched": 0,
                                  "eligible_pairs": 0, "replacements": 0}
        return super().update_candidates(boxes, scores, features)

    def _refine(self, matches, pool, high, tensors, weak_mask):
        original_high_count = len(high)
        # Lost tracks retain their original high-only recovery policy.
        positions = sorted((p for p, (i, j) in enumerate(matches)
                            if pool[i].state == TrackState.Tracked and pool[i].is_activated),
                           key=lambda p: pool[matches[p][0]].internal_track_id)
        weak_indices = np.flatnonzero(weak_mask).tolist()
        self.refinement_counts.update(incumbents=len(positions), weak_candidates=len(weak_indices))
        # Preserve every motion-feasible weak option for previously unmatched
        # active tracks. Do not steal their second-stage recovery candidates.
        matched_rows = set(matches[:, 0])
        unmatched_active = [t for i, t in enumerate(pool)
                            if i not in matched_rows and t.state == TrackState.Tracked]
        if unmatched_active and weak_indices:
            costs = matching.iou_distance([np.asarray(t.tlbr, np.float32) for t in unmatched_active],
                                         [np.asarray(tensors[i, :4], np.float32) for i in weak_indices])
            reserved = set(j for j in range(len(weak_indices)) if np.any(costs[:, j] <= .5))
            self.refinement_counts['weak_reserved_for_unmatched'] = len(reserved)
            weak_indices = [index for j, index in enumerate(weak_indices) if j not in reserved]
        if not positions or not weak_indices:
            return matches, tuple(i for i in range(original_high_count) if i not in set(matches[:, 1])), set()
        tracks = [pool[matches[p][0]] for p in positions]
        incumbents = [high[matches[p][1]] for p in positions]
        weak_boxes = [np.asarray(tensors[i, :4], np.float32) for i in weak_indices]
        high_iou = np.array([1. - float(matching.iou_distance([t], [d])[0, 0]) for t, d in zip(tracks, incumbents)])
        low_iou = np.clip(1. - matching.iou_distance([np.asarray(t.tlbr, np.float32) for t in tracks], weak_boxes), 0., 1.)
        descriptors = [t.descriptor(self.frame_id) for t in tracks]
        high_cos = np.array([_cosine(v, d.current_feature) for v, d in zip(descriptors, incumbents)])
        low_cos = np.array([[_cosine(v, self._features[i]) for i in weak_indices] for v in descriptors])
        choices, eligible = challenge_assignment(high_iou, low_iou, high_cos, low_cos,
                                                 threshold=self.appearance_threshold, mode=self.refinement)
        self.refinement_counts['eligible_pairs'] = int(eligible.sum())
        result = matches.copy()
        used = set()
        for i, j in choices:
            index = weak_indices[j]
            weak = STrack(STrack.tlbr_to_tlwh(tensors[index, :4]), tensors[index, 4],
                          self.minimum_consecutive_frames, self.shared_kalman,
                          self.internal_id_counter, self.external_id_counter,
                          candidate_index=index, feature=self._features[index], learn_feature=False)
            high.append(weak)
            result[positions[i], 1] = len(high) - 1
            used.add(index)
            self.refinement_events.append({
                'local_id': int(tracks[i].external_track_id),
                'high_detection_index': int(incumbents[i].candidate_index), 'low_detection_index': int(index),
                'high_score': float(incumbents[i].score), 'low_score': float(tensors[index, 4]),
                'high_cosine': float(high_cos[i]), 'low_cosine': float(low_cos[i, j]),
                'high_predicted_iou': float(high_iou[i]), 'low_predicted_iou': float(low_iou[i, j]),
                'prior_sample_frames': [int(f - 1) for f, _ in tracks[i].appearance_history
                                        if 0 <= self.frame_id - f <= 30]})
        self.refinement_counts['replacements'] = len(choices)
        # Released high candidates remain available for unconfirmed tracks / births.
        remaining_high = tuple(i for i in range(original_high_count) if i not in set(result[:, 1]))
        return result, remaining_high, used

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

        # The incumbent high assignment is frozen before the optional refinement.
        used_weak = set()
        if self.refinement != "disabled":
            matches, u_detection, used_weak = self._refine(
                matches, strack_pool, detections, tensors, inds_second)

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
                if int(index) not in used_weak
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
