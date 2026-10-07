"""Causal candidate provenance and appearance history for an experimental local tracker."""
from collections import deque
import numpy as np
from supervision.tracker.byte_tracker.single_object_track import STrack


def gate_costs(costs, track_features, detection_features, cost_limit, threshold):
    """Mask motion-eligible pairs before assignment; preserve every allowed cost.

    Missing features explicitly fall back to the existing motion/score cost.
    Cosine threshold is inclusive and is not a calibrated person probability.
    """
    costs = np.asarray(costs)
    if costs.shape != (len(track_features), len(detection_features)) or not np.isfinite(costs).all():
        raise ValueError('Invalid appearance cost matrix')
    if not np.isfinite([cost_limit, threshold]).all() or not -1 <= threshold <= 1:
        raise ValueError('Invalid appearance gate settings')
    for vector in [*track_features, *detection_features]:
        if vector is not None and (np.asarray(vector).shape != (512,) or not np.isfinite(vector).all()
                                   or abs(np.linalg.norm(vector) - 1) > 1e-4):
            raise ValueError('Invalid normalized appearance vector')
    out = costs.copy()
    counters = {'compared': 0, 'blocked': 0, 'unavailable': 0}
    for i, j in zip(*np.nonzero(costs <= cost_limit)):
        a, b = track_features[i], detection_features[j]
        if a is None or b is None:
            counters['unavailable'] += 1
            continue
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        similarity = float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
        counters['compared'] += 1
        if similarity < threshold:
            out[i, j] = np.inf
            counters['blocked'] += 1
    return out, counters


class CandidateTrack(STrack):
    """Retain the exact accepted detection index and only past strong appearance samples."""
    def __init__(self, *args, candidate_index, feature, learn_feature):
        super().__init__(*args)
        self.candidate_index = candidate_index
        self.current_feature = feature
        self.learn_feature = learn_feature
        self.candidate_frame = None
        self.appearance_history = deque(maxlen=8)

    def _observe(self, candidate, frame):
        self.candidate_index = candidate.candidate_index
        self.candidate_frame = frame
        self.current_feature = candidate.current_feature
        while self.appearance_history and frame - self.appearance_history[0][0] > 30:
            self.appearance_history.popleft()
        if candidate.learn_feature and candidate.current_feature is not None:
            self.appearance_history.append((frame, candidate.current_feature.copy()))

    def descriptor(self, frame):
        # Called before accepting the current candidate; never includes it.
        values = [v for f, v in self.appearance_history if 0 <= frame - f <= 30]
        if not values:
            return None
        mean = np.mean(np.asarray(values, dtype=np.float64), axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-12 else values[-1].copy()

    def activate(self, kalman_filter, frame_id):
        super().activate(kalman_filter, frame_id)
        self._observe(self, frame_id)

    def update(self, new_track, frame_id):
        super().update(new_track, frame_id)
        self._observe(new_track, frame_id)

    def re_activate(self, new_track, frame_id):
        super().re_activate(new_track, frame_id)
        self._observe(new_track, frame_id)
