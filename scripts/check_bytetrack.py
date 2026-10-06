"""Check local identity continuity, low scores and independent camera state."""

import numpy as np
import supervision as sv


def make_detections(frame_index, score, x_offset=0):
    if score is None:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
        )

    x = 100 + x_offset + 2 * frame_index
    return sv.Detections(
        xyxy=np.array([[x, 100, x + 50, 220]], dtype=np.float32),
        confidence=np.array([score], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )


trackers = {
    camera: sv.ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=1,
    )
    for camera in (4, 5, 8)
}

# Camera 4: motion, low confidence, one missing observation, return.
scores_4 = [0.9, 0.9, 0.2, None, 0.9]
initial_ids = {}

for frame_index, score_4 in enumerate(scores_4):
    inputs = {
        4: make_detections(frame_index, score_4),
        # Low-confidence observations alone must not start a track.
        5: make_detections(frame_index, 0.2),
        # Another camera maintains its own independent track.
        8: make_detections(frame_index, 0.9, x_offset=400),
    }
    outputs = {
        camera: trackers[camera].update_with_detections(detections)
        for camera, detections in inputs.items()
    }
    ids = {
        camera: result.tracker_id.tolist()
        for camera, result in outputs.items()
    }
    print(f"Frame {frame_index}: {ids}")

    if frame_index == 0:
        for camera in (4, 8):
            if len(ids[camera]) != 1:
                raise RuntimeError(f"Camera {camera}: track was not created")
            initial_ids[camera] = ids[camera][0]

    if ids[5]:
        raise RuntimeError("Low-confidence detections started a new track")

    if ids[8] != [initial_ids[8]]:
        raise RuntimeError("Camera 8 lost its independent identity")

    expected_4 = [] if score_4 is None else [initial_ids[4]]
    if ids[4] != expected_4:
        raise RuntimeError(
            f"Camera 4: expected {expected_4}, received {ids[4]}"
        )

print("Low-confidence observation maintains an active ID: OK")
print("Missing observation produces no visible track: OK")
print("Return after a gap preserves the ID: OK")
print("Low-confidence observations alone create no track: OK")
print("Independent camera trackers: OK")
print("ByteTrack smoke test: PASSED")
