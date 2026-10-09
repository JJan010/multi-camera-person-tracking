"""One fixed feature space per identity stage; legacy runtime remains unchanged."""
from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from time import perf_counter_ns
import numpy as np
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch, validate_batch
from mtmc.reid.osnet import ObservationKey
from mtmc.association.geometry import CameraGroundPositions, GroundObservation, project_box_foot
from mtmc.association.feature_geometry import CameraFeatures, associate_feature_pair, group_feature_pairs
from mtmc.association.grouping import key_order
from .core import IdentityStage, require


class FeatureIdentityStage(IdentityStage):
    """Same identity policy, explicit encoder space; positive-area crops required.

    Model/shape/key preflight completes before the existing manager is called.
    Inputs to the manager are scalar links/groups, not model descriptors.
    """
    def __init__(self, run_id, matrices, coordinate_space, *, space, **kwargs):
        require(isinstance(space,FeatureSpace),'Expected explicit feature space')
        super().__init__(run_id,matrices,coordinate_space,**kwargs)
        self.space=space

    def update(self, frame_index, timestamp, descriptors, records):
        require(type(frame_index) is int and frame_index >= 0
                and isinstance(timestamp, Fraction) and timestamp >= 0, "Invalid round context")
        validate_batch(descriptors,run_id=self.run_id,space=self.space,
                       frame_index=frame_index,timestamp=timestamp)
        keys, vectors = descriptors.keys, descriptors.embeddings
        rows = {}
        for record in records:
            key = record.key
            require(isinstance(key, ObservationKey) and all(type(v) is int and v >= 0 for v in key_order(key))
                    and key.frame_index == frame_index and key.camera_id in self.cameras
                    and key not in rows, "Invalid or duplicate track key")
            box = np.asarray(record.source_xyxy, dtype=np.float64)
            require(box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] > box[:2]), "Invalid tracked box")
            rows[key] = record
        expected = {key for key, record in rows.items() if record.crop_xyxy_int is not None}
        require(set(keys) == expected and all(t == timestamp and isinstance(t, Fraction) for t in descriptors.timestamps),
                "Current embedding/track coverage or timestamp mismatch")
        appearances, grounds = {}, {}
        t0 = perf_counter_ns()
        for camera in self.cameras:
            indices = sorted((i for i, key in enumerate(keys) if key.camera_id == camera), key=lambda i: key_order(keys[i]))
            selected = tuple(keys[i] for i in indices)
            features = FeatureBatch(self.run_id, self.space, selected, (timestamp,) * len(selected), vectors[indices].copy())
            appearances[camera] = CameraFeatures(self.run_id, camera, frame_index, timestamp, self.variant, features)
            positions = tuple(GroundObservation(key, project_box_foot(self.matrices[camera], rows[key].source_xyxy))
                              for key in selected)
            grounds[camera] = CameraGroundPositions(self.run_id, camera, frame_index, timestamp,
                                                    self.coordinate_space, positions)
        t1 = perf_counter_ns()
        pairs = tuple(associate_feature_pair(appearances[a], appearances[b], grounds[a], grounds[b],
                      min_similarity=self.threshold, **self.geometry_configuration) for a, b in combinations(self.cameras, 2))
        t2 = perf_counter_ns()
        grouped = group_feature_pairs(pairs)
        missing = tuple(sorted(set(rows) - expected, key=key_order))
        # Preserve every local track, including fully outside, unencoded boxes.
        groups = (*grouped.groups, *((key,) for key in missing))
        grouped = replace(grouped, groups=tuple(sorted(groups, key=lambda g: tuple(key_order(k) for k in g))))
        t3 = perf_counter_ns()
        result = self.manager.update(grouped)
        t4 = perf_counter_ns()
        require({a.key for a in result.assignments} == set(rows), "Global assignment coverage differs")
        timing = {name: (end-start)/1e6 for name, start, end in (
            ("project_ms", t0, t1), ("associate_ms", t1, t2), ("group_ms", t2, t3), ("identity_ms", t3, t4))}
        return result, grouped, pairs, grounds, missing, timing
