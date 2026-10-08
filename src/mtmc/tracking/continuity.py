"""Experimental appearance-discontinuity policy over fixed local observations.

Current descriptors are compared BEFORE reference updates. A pending candidate
freezes its older reference. This is not a calibrated person-switch detector.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from mtmc.reid.osnet import ObservationKey
from .segments import LocalTrackSegments, SegmentBreak, SegmentedRound, require


@dataclass(frozen=True)
class ContinuitySettings:
    max_reference_samples: int
    max_reference_age: Fraction
    reference_sample_interval: Fraction
    min_reference_samples: int
    min_confidence: float
    break_similarity: float
    candidate_similarity: float
    min_support_rounds: int
    min_support_seconds: Fraction
    max_support_gap: Fraction


@dataclass(frozen=True)
class ContinuityDecision:
    key: ObservationKey
    outcome: str
    reference_similarity: float | None
    candidate_similarity: float | None
    reference_frames: tuple[int, ...]
    reference_times: tuple[Fraction, ...]
    reference_fallback: bool
    support_rounds: int
    support_span: Fraction


@dataclass(frozen=True)
class ContinuityReset:
    camera_id: int
    local_id: int
    reason: str


@dataclass(frozen=True)
class ContinuityFrame:
    segmented: SegmentedRound
    decisions: tuple[ContinuityDecision, ...]
    resets: tuple[ContinuityReset, ...]


@dataclass(frozen=True)
class _Sample:
    frame: int
    time: Fraction
    vector: np.ndarray


@dataclass(frozen=True)
class _Pending:
    reference: np.ndarray
    reference_frames: tuple[int, ...]
    reference_times: tuple[Fraction, ...]
    reference_fallback: bool
    seed: np.ndarray
    first_time: Fraction
    last_time: Fraction
    support_rounds: int


def cosine(a, b):
    # Inputs are already validated unit float32 features or normalized means.
    return float(np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)), -1., 1.))


def reference(samples):
    mean = np.mean(np.stack([s.vector for s in samples]), axis=0, dtype=np.float64)
    length = float(np.linalg.norm(mean))
    fallback = length <= 1e-12
    effective = samples[-1:] if fallback else samples
    vector = samples[-1].vector.copy() if fallback else (mean / length).astype(np.float32)
    return vector, tuple(s.frame for s in effective), tuple(s.time for s in effective), fallback


class AppearanceContinuity:
    """Atomic owner of the causal decision policy and local segment mapping.

    One immutable setting set per run. No GT/files/images/global-ID information
    enters this policy. Quality is not inferred from detector confidence alone.
    Low-confidence/missing-feature/absent observations cannot confirm a cut.
    This guard may miss gradual identity drift and may cut after pose/occlusion
    changes. A future paired experiment must measure both effects.
    """
    def __init__(self, run_id, camera_ids, *, enabled, settings):
        require(isinstance(settings, ContinuitySettings), 'Expected ContinuitySettings')
        cfg = settings
        require(type(cfg.max_reference_samples) is int and type(cfg.min_reference_samples) is int
                and 2 <= cfg.min_reference_samples <= cfg.max_reference_samples, 'Invalid reference sample limits')
        require(type(cfg.min_support_rounds) is int and cfg.min_support_rounds >= 2, 'At least two support rounds required')
        require(all(isinstance(x, Fraction) and x > 0 for x in
                    (cfg.max_reference_age, cfg.reference_sample_interval,
                     cfg.min_support_seconds, cfg.max_support_gap))
                and cfg.min_support_seconds < cfg.max_reference_age
                and (cfg.min_reference_samples - 1) * cfg.reference_sample_interval <= cfg.max_reference_age,
                'Invalid exact time settings')
        require(type(cfg.min_confidence) in (int, float) and np.isfinite(cfg.min_confidence)
                and 0 <= cfg.min_confidence <= 1, 'Invalid confidence gate')
        require(all(type(x) in (int, float) and np.isfinite(x) and -1 <= x <= 1
                    for x in (cfg.break_similarity, cfg.candidate_similarity)), 'Invalid cosine settings')
        self.settings = cfg
        self.segmenter = LocalTrackSegments(run_id, camera_ids, enabled=enabled)
        self._samples = {}; self._pending = {}

    @property
    def stored_vectors(self):
        return sum(len(v) for v in self._samples.values()) + 2 * len(self._pending)

    @property
    def pending_tracks(self):
        return len(self._pending)

    def update(self, batch):
        # Full all-camera preflight before cloning/planning any new policy state.
        self.segmenter._validate(batch)
        require(not batch.breaks, 'Automatic continuity owner does not accept external break requests')
        if not self.segmenter.enabled:
            return ContinuityFrame(self.segmenter.update(batch), (), ())
        working = deepcopy(self)
        result = working._advance(batch)
        self.__dict__.update(working.__dict__)
        return result

    def _advance(self, batch):
        cfg = self.settings; now = batch.timestamp; frame = batch.frame_index
        features = dict(zip(batch.features.keys, batch.features.embeddings))
        visible = {(r.key.camera_id, r.key.local_id) for r in batch.records}
        resets = []; decisions = []; breaks = []

        def clear(local, reason):
            if local in self._pending:
                del self._pending[local]
                resets.append(ContinuityReset(*local, reason))

        # Empty rounds age references and explicitly interrupt pending evidence.
        self._samples = {local: tuple(s for s in samples if now - s.time <= cfg.max_reference_age)
                         for local, samples in self._samples.items()}
        self._samples = {k: v for k, v in self._samples.items() if v}
        for local, candidate in sorted(tuple(self._pending.items())):
            if local not in visible:
                clear(local, 'absent_observation')
            elif any(now - t > cfg.max_reference_age for t in candidate.reference_times):
                clear(local, 'frozen_reference_expired')
            elif now - candidate.last_time > cfg.max_support_gap:
                clear(local, 'support_time_gap')

        for record in sorted(batch.records, key=lambda r: (r.key.camera_id, r.key.local_id)):
            key = record.key; local = key.camera_id, key.local_id
            vector = features.get(key); samples = self._samples.get(local, ())
            candidate = self._pending.get(local)
            sim = seed_sim = None; frames = (); times = (); fallback = False
            count = 0; span = Fraction(0)
            if vector is None or record.confidence < cfg.min_confidence:
                outcome = 'missing_feature' if vector is None else 'weak_observation'
                clear(local, outcome)
            elif candidate is None and len(samples) < cfg.min_reference_samples:
                if not samples or now - samples[-1].time >= cfg.reference_sample_interval:
                    outcome = 'reference_bootstrap_sampled'
                    self._samples[local] = (*samples, _Sample(frame, now, vector.copy()))[-cfg.max_reference_samples:]
                else:
                    outcome = 'reference_bootstrap_wait'
            else:
                if candidate is None:
                    anchor, frames, times, fallback = reference(samples)
                else:
                    anchor = candidate.reference
                    frames, times, fallback = candidate.reference_frames, candidate.reference_times, candidate.reference_fallback
                # The current vector is never included in its own prior reference.
                require(all(t < now for t in times) and all(f < frame for f in frames), 'Noncausal reference')
                sim = cosine(vector, anchor)
                if sim >= cfg.break_similarity:
                    clear(local, 'reference_agreement')
                    if not samples or now - samples[-1].time >= cfg.reference_sample_interval:
                        self._samples[local] = (*samples, _Sample(frame, now, vector.copy()))[-cfg.max_reference_samples:]
                        outcome = 'reference_updated'
                    else:
                        outcome = 'reference_agreement_unsampled'
                else:
                    if candidate is not None:
                        seed_sim = cosine(vector, candidate.seed)
                        if seed_sim < cfg.candidate_similarity:
                            clear(local, 'candidate_incoherent')
                            candidate = None
                    if candidate is None:
                        candidate = _Pending(anchor.copy(), frames, times, fallback, vector.copy(), now, now, 1)
                        outcome = 'contradiction_started'
                    else:
                        candidate = replace(candidate, last_time=now, support_rounds=candidate.support_rounds + 1)
                        outcome = 'contradiction_pending'
                    count = candidate.support_rounds; span = now - candidate.first_time
                    if count >= cfg.min_support_rounds and span >= cfg.min_support_seconds:
                        old = self.segmenter._state.get(local)
                        require(old is not None, 'Established reference without segment state')
                        breaks.append(SegmentBreak(key, old.generation, 'confirmed_appearance_discontinuity'))
                        # A cut starts a new reference; rejected pre-cut samples are not seeded into it.
                        self._samples[local] = (_Sample(frame, now, vector.copy()),)
                        clear(local, 'split_confirmed')
                        outcome = 'split_confirmed'
                    else:
                        self._pending[local] = candidate
            decisions.append(ContinuityDecision(key, outcome, sim, seed_sim, frames, times, fallback, count, span))
        segmented = self.segmenter.update(replace(batch, breaks=tuple(breaks)))
        require(len(segmented.events) == sum(d.outcome == 'split_confirmed' for d in decisions), 'Cut count differs')
        return ContinuityFrame(segmented, tuple(decisions), tuple(resets))
