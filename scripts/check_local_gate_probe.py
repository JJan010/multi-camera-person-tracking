"""Verify that gate observation preserves outputs and distinguishes high/low stages."""
import numpy as np

from probe_local_appearance_gate import observer_class, observed_step
from mtmc.tracking._vendor.experimental_bytetrack import ExperimentalByteTrack


def main():
    a, b = np.eye(2, 512, dtype=np.float32)
    settings = dict(track_activation_threshold=.5, lost_track_buffer=30, minimum_matching_threshold=.8,
                    frame_rate=30, minimum_consecutive_frames=1, direct_output=True, appearance_threshold=.6)
    for case in ('high_gate_then_low_recovery', 'low_dissimilar', 'missing_descriptor'):
        reference = ExperimentalByteTrack(**settings)
        observer = observer_class()(target_local_id=1, **settings)
        steps = [([[10, 10, 30, 60]], [.9], [None if case == 'missing_descriptor' else a])]
        if case == 'high_gate_then_low_recovery':
            steps.append(([[10, 10, 30, 60], [11, 10, 31, 60]], [.9, .2], [b, a]))
        elif case == 'low_dissimilar':
            steps.append(([[10, 10, 30, 60]], [.2], [b]))
        else:
            steps.append(([[10, 10, 30, 60]], [.9], [a]))
        for frame, (boxes, scores, features) in enumerate(steps):
            boxes = np.asarray(boxes, np.float32); scores = np.asarray(scores, np.float32)
            expected = reference.update_candidates(boxes, scores, features)
            actual, row = observed_step(observer, boxes, scores, features, 1, frame, True)
            for name in ('tracker_id', 'xyxy', 'confidence'):
                assert np.array_equal(getattr(actual, name), getattr(expected, name))
            assert observer.selected_indices == reference.selected_indices
        if case == 'high_gate_then_low_recovery':
            assert row['selected_phase'] == 'low_score_iou_without_appearance_gate' and row['cosine_prior'] == 1
            assert row['gate_audit'][0]['candidates'][0]['appearance_blocked']
        elif case == 'low_dissimilar':
            assert row['selected_phase'] == 'low_score_iou_without_appearance_gate' and row['cosine_prior'] == 0
        else:
            assert row['selected_phase'] == 'high' and row['cosine_prior'] is None
            assert row['gate_audit'][0]['candidates'][0]['appearance_unavailable']
        print(case + ': correct audit and unchanged output/provenance: OK')
    print('Local gate observer checks: PASSED')


if __name__ == '__main__':
    main()
