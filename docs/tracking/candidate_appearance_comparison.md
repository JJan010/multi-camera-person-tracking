# Camera 8 / local 44: high-stage assignment and low-score alternatives

Source preview: `20261007T182120464197Z`.
Source gate probe: `20261007T181400956969Z`.
Variant: `direct_appearance`, local appearance threshold 0.6.

Visual review shows two overlapping people inside accepted crops. The gallery
already contains mixed crops, notably frames 1435, 1439 and 1440. The diagnostic
GT label is 1 at frame 1438, ambiguous at 1439/1440, 13 at 1441 and 1 again at
1446. This is not sufficient evidence for a permanent identity takeover.
Mutually unique IoU evidence does not imply a visually pure crop.

The frozen candidate inspection found:

| Frame | Accepted score | Accepted IoU with GT 1 | Alternative score | Alternative IoU with GT 1 |
|---|---:|---:|---:|---:|
| 1439 | 0.7489 | 0.6510 | 0.2168 | 0.8765 |
| 1440 | 0.7213 | 0.6202 | 0.3547 | 0.7937 |
| 1441 | 0.6299 | 0.4742 | 0.4437 | 0.7751 |

All alternatives above belong to the low-score partition. An accepted high-stage
match prevents that track from being considered in the subsequent low-stage
recovery. However, better GT overlap does not establish that the low candidate
is appearance-compatible, motion-eligible, or available given other tracks.

## Frozen appearance comparison

Run `scripts/compare_transition_candidates.py --preview-report <report.json>`.
Only Python/NumPy and frozen artifacts are used; no model, video, tracker replay,
assignment modification, or threshold selection takes place.

The script follows checksummed preview/probe/local/cache references, joins actual
accepted candidates by index and embedding row, and reconstructs each frame's
exact prior strong gallery. All samples must precede the current frame, be at
most 30 frames old, and have high detector confidence. The normalized float64
mean and cancellation fallback reproduce the pinned CandidateTrack descriptor.
The accepted candidate's cosine must match the observed gate within 1e-10.

Every candidate in the camera is compared against that same prior descriptor.
The report saves all similarities. GT IoU filters console rows for readability
only; it is never used to construct the descriptor. Positive delta_vs_accepted
means higher appearance similarity than the frozen accepted candidate.

This diagnostic does not inspect the predicted Kalman box, low-stage eligibility,
or assignment competition. It must not be presented as a counterfactual tracking
improvement. If appearance does not distinguish alternatives, changing stage
order alone has no demonstrated appearance-based justification for this case.
Any policy change requires causal multi-track replay and full local/global
evaluation, including fragmentation, false positives and missed observations.

The two-minute fragment is development data after inspection. A later untouched
scene/segment is needed for validation; no deployment thresholds are calibrated.

References:
- Zhang et al., ByteTrack (ECCV 2022): https://arxiv.org/abs/2110.06864
- Wojke et al., Deep SORT (2017): https://arxiv.org/abs/1703.07402

ByteTrack motivates retaining low-score detections for recovering occluded
objects; Deep SORT motivates appearance-informed association. This comparison
does not implement or claim to reproduce Deep SORT.
