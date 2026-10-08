"""Experimental identity adapter with explicit descriptor availability.

Keeps original crop metadata and all local observations. Baseline core.py is
unchanged. Pairing, grouping and identity management use the existing policies.
"""
from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from time import perf_counter_ns
import numpy as np

from mtmc.association.geometry import (CameraGroundPositions, GroundObservation,
    associate_camera_pair_with_geometry, group_geometry_associations, project_box_foot)
from mtmc.association.grouping import key_order
from mtmc.association.pairwise import CameraAppearance
from mtmc.pipeline.core import IdentityStage, require
from mtmc.reid.osnet import ObservationKey, ReIDBatch


class AvailableIdentityStage(IdentityStage):
    """Unavailable descriptors cannot make new appearance/geometry links.

    Their observations enter grouping as singletons. Existing local-to-global
    bindings can preserve continuity. Missing appearance does not erase a crop
    or make a positive-area crop appear fully outside the image.
    """
    def update(self, frame_index, timestamp, descriptors, records, *, unavailable_keys):
        require(type(frame_index) is int and frame_index >= 0
                and isinstance(timestamp, Fraction) and timestamp >= 0, 'Invalid round scope')
        require(isinstance(descriptors, ReIDBatch), 'Expected ReIDBatch')
        keys, vectors = descriptors.keys, descriptors.embeddings
        require(len(set(keys)) == len(keys) and len(descriptors.timestamps) == len(keys), 'Invalid descriptor keys')
        require(isinstance(vectors, np.ndarray) and vectors.dtype == np.float32
                and vectors.shape == (len(keys),512) and np.isfinite(vectors).all()
                and np.allclose(np.linalg.norm(vectors,axis=1),1,rtol=0,atol=1e-5), 'Invalid descriptors')
        rows = {}
        for record in records:
            key = record.key
            require(isinstance(key, ObservationKey) and all(type(v) is int and v >= 0 for v in key_order(key))
                    and key.frame_index == frame_index and key.camera_id in self.cameras and key not in rows,
                    'Invalid tracked observation key')
            box = np.asarray(record.source_xyxy, np.float64)
            require(box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] > box[:2]), 'Invalid tracked box')
            rows[key] = record
        unavailable = tuple(unavailable_keys)
        require(all(isinstance(k, ObservationKey) for k in unavailable) and len(set(unavailable)) == len(unavailable)
                and set(unavailable) == set(rows)-set(keys) and set(keys) <= set(rows),
                'Available/unavailable keys must exactly partition tracked observations')
        require(all(rows[k].crop_xyxy_int is not None for k in keys), 'A fully outside crop has a descriptor')
        require(all(isinstance(t,Fraction) and t == timestamp for t in descriptors.timestamps), 'Descriptor timestamp differs')
        appearances, grounds = {}, {}
        t0 = perf_counter_ns()
        for camera in self.cameras:
            indices = sorted((i for i,k in enumerate(keys) if k.camera_id == camera), key=lambda i:key_order(keys[i]))
            selected = tuple(keys[i] for i in indices)
            features = ReIDBatch(selected, (timestamp,)*len(selected), vectors[indices].copy())
            appearances[camera] = CameraAppearance(self.run_id,camera,frame_index,timestamp,self.variant,features)
            positions = tuple(GroundObservation(key,project_box_foot(self.matrices[camera],rows[key].source_xyxy))
                              for key in selected)
            grounds[camera] = CameraGroundPositions(self.run_id,camera,frame_index,timestamp,self.coordinate_space,positions)
        t1 = perf_counter_ns()
        pairs = tuple(associate_camera_pair_with_geometry(appearances[a],appearances[b],grounds[a],grounds[b],
                      min_similarity=self.threshold,**self.geometry_configuration) for a,b in combinations(self.cameras,2))
        t2 = perf_counter_ns()
        grouped = group_geometry_associations(pairs)
        missing = tuple(sorted(unavailable,key=key_order))
        groups = (*grouped.groups,*((key,) for key in missing))
        grouped = replace(grouped,groups=tuple(sorted(groups,key=lambda group:tuple(key_order(k) for k in group))))
        t3 = perf_counter_ns()
        result = self.manager.update(grouped)
        t4 = perf_counter_ns()
        require({a.key for a in result.assignments} == set(rows), 'Global observation coverage differs')
        timings = {name:(end-start)/1e6 for name,start,end in
                   (('project_ms',t0,t1),('associate_ms',t1,t2),('group_ms',t2,t3),('identity_ms',t3,t4))}
        return result, grouped, pairs, grounds, missing, timings
