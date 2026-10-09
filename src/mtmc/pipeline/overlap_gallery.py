"""Experimental gallery-only admission by same-camera visible-box overlap."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import math

from mtmc.reid.sample_quality import assess_sample
from .core import require
from .confirmed_return import ConfirmedReturnRegistry, ConfirmedReturnStage, _Bridge


def overlap_scores(records, sizes):
    """Max intersection / own clipped continuous area; all visible tracks compete."""
    boxes = {}
    for record in records:
        key = record.key
        require(key not in boxes and key.camera_id in sizes, 'Duplicate or unknown overlap key')
        width, height = sizes[key.camera_id]
        assess_sample(record, width, height, min_confidence=.5, border_fraction=.01)
        x1, y1, x2, y2 = record.source_xyxy
        boxes[key] = (max(0., min(width, x1)), max(0., min(height, y1)),
                      max(0., min(width, x2)), max(0., min(height, y2)))
    scores = {}
    for key, (x1, y1, x2, y2) in boxes.items():
        area = max(0., x2-x1) * max(0., y2-y1)
        maximum = 0.
        if area:
            for other, (a1, b1, a2, b2) in boxes.items():
                if other == key or other.camera_id != key.camera_id:
                    continue
                overlap = max(0., min(x2, a2)-max(x1, a1)) * max(0., min(y2, b2)-max(y1, b1))
                maximum = max(maximum, overlap / area)
        scores[key] = maximum
    return scores


class OverlapGalleryRegistry(ConfirmedReturnRegistry):
    def __init__(self, *args, overlap_enabled, max_overlap, **kwargs):
        super().__init__(*args, **kwargs)
        require(self.enabled and type(overlap_enabled) is bool, 'Recovery must be enabled in both variants')
        require(type(max_overlap) in (int, float) and math.isfinite(max_overlap) and 0 <= max_overlap <= 1,
                'Invalid overlap threshold')
        self.overlap_enabled = overlap_enabled
        self.max_overlap = float(max_overlap)
        self.admission_scores = None
        self.last_overlap = None

    def _advance(self, grouped, evidence):
        require(self.admission_scores is not None and set(self.admission_scores) == set(evidence),
                'Missing overlap evidence')
        before = deepcopy(self.galleries)
        output = super()._advance(grouped, evidence)
        # The original transaction evaluates queries/retirements with prior galleries.
        # Rebuild its final gallery, rather than deleting rejected rows after truncation:
        # rejected updates must not evict older accepted samples.
        for event in self.last_audit.reactivations:
            before[event['global_id']] = before.pop(event['provisional_id'], [])
        gc = self.configuration['gallery']
        fresh = defaultdict(list)
        counts = Counter()
        decisions = []
        for assignment in output.assignments:
            sample = evidence[assignment.key]
            eligible = (sample is not None and sample.confidence >= gc['min_confidence']
                        and sample.normalized_clearance >= gc['border_fraction'])
            blocked = eligible and self.overlap_enabled and self.admission_scores[assignment.key] > self.max_overlap
            outcome = 'overlap_rejected' if blocked else 'accepted' if eligible else 'base_quality_rejected'
            counts[outcome] += 1
            decisions.append(dict(key=assignment.key, global_id=assignment.global_id,
                                  maximum_overlap=self.admission_scores[assignment.key], outcome=outcome))
            if eligible and not blocked:
                fresh[assignment.global_id].append(sample)
        galleries = {}
        for state in output.identities:
            gid = state.global_id
            per_camera = defaultdict(list)
            for sample in (*before.get(gid, ()), *fresh[gid]):
                if grouped.timestamp - sample.timestamp <= Fraction(gc['max_sample_age_seconds']):
                    per_camera[sample.key.camera_id].append(sample)
            galleries[gid] = [s for camera in sorted(per_camera) for s in sorted(
                per_camera[camera], key=lambda s: (s.timestamp, s.key.frame_index, s.key.local_id))[-gc['max_per_camera']:]]
        if not self.overlap_enabled:
            # A true control also checks the gallery reconstruction against the original code.
            import pickle
            require(set(galleries) == set(self.galleries) and all(
                pickle.dumps(galleries[g]) == pickle.dumps(self.galleries[g]) for g in galleries),
                'Unfiltered gallery reconstruction differs')
        self.galleries = galleries
        self.last_audit = replace(self.last_audit, lifecycle={**self.last_audit.lifecycle,
            'gallery_vectors': sum(map(len, galleries.values()))})
        counts['retained_id_rounds_without_gallery'] = sum(not v for v in galleries.values())
        self.last_overlap = dict(counts=dict(counts), decisions=tuple(decisions))
        return output


class OverlapGalleryStage(ConfirmedReturnStage):
    def __init__(self, *args, overlap_enabled, max_overlap, **kwargs):
        super().__init__(*args, **kwargs)
        previous = self.registry
        self.registry = OverlapGalleryRegistry(previous.base, space=previous.space,
            coordinate_space=previous.coordinates, camera_ids=previous.cameras, enabled=previous.enabled,
            configuration=previous.configuration, overlap_enabled=overlap_enabled, max_overlap=max_overlap)
        self.manager = _Bridge(self.registry)

    def update_with_raw(self, frame, timestamp, means, raw, records):
        records = tuple(records)
        require(not self.failed, 'Failed stage cannot be retried')
        scores = overlap_scores(records, self.sizes)
        require(self.registry.admission_scores is None, 'Overlapping stage calls')
        self.registry.admission_scores = scores
        try:
            return super().update_with_raw(frame, timestamp, means, raw, records)
        finally:
            # Also restore this transient input on validation/transaction failure.
            self.registry.admission_scores = None
