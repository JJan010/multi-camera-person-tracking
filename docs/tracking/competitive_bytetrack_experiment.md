# Constrained high/low candidate competition

## Motivation and hypothesis

Camera 8 / local 44 has mixed-person crops near scene frame 1441. Frozen
candidate inspection found low-score boxes with better overlap with diagnostic
GT 1. Comparison against the *actual prior strong gallery* also preferred them:

| Frame | Accepted cosine | Alternative cosine | Alternative score |
|---|---:|---:|---:|
| 1439 | 0.9027 | 0.9512 | 0.2168 |
| 1440 | 0.9036 | 0.9466 | 0.3547 |
| 1441 | 0.7862 | 0.9302 | 0.4437 |

Reference probe: `artifacts/candidate_appearance_probe/20261007T183937481045Z/report.json`.
The gallery and accepted-candidate cosine were verified against the frozen gate
probe. The case motivates a hypothesis, not an estimated global improvement.

Hypothesis: a weak candidate can sometimes preserve identity better than an
otherwise admissible high-score incumbent. Binary rejection alone cannot express
this preference when both candidates exceed the existing appearance threshold.

## Three variants and exact controls

- `staged`: the pinned `direct_appearance` policy, including its original costs,
  low-stage motion-only recovery and strong-only appearance updates.
- `competitive_iou`: constrained refinement, ranking improvement in predicted
  track/detection IoU.
- `competitive_appearance`: identical refinement eligibility, ranking improvement
  in cosine similarity to the track's prior strong gallery.

The experiment also replays a disabled refinement instance. Both disabled and
staged outputs must reproduce every frozen baseline camera update exactly,
including accepted candidate indices and embedding rows. Staged local metrics
must reproduce the input report. No source detector or model inference is run.

The IoU control still uses the same appearance eligibility gate as the appearance
variant. It is not an appearance-free tracker. Comparing the two new variants
isolates the ranking criterion at equal eligibility rules; their later states
can diverge causally.

## Refinement contract

1. Run the original high-score assignment against Kalman-predicted track boxes.
2. Before updating any track, consider only high-matched tracks that were already
   activated and in `Tracked` state. Their high detections are the incumbents.
3. Weak candidates have `0.1 < score < track_activation_threshold` (0.5 here).
   Reserve a weak candidate for original second-stage recovery if its predicted
   IoU is at least 0.5 to **any** unmatched active track. It cannot be stolen by
   the refinement.
4. Remaining weak candidates require predicted IoU >= 0.5, both appearance
   descriptors available, and cosine >= the unchanged local threshold (0.6).
   These are runtime features, not GT geometry or labels.
5. For each eligible incumbent track / weak candidate pair, calculate improvement
   over that track's incumbent (IoU or cosine, depending on variant). Require
   strictly positive improvement. No tuned margin is introduced.
6. Solve one partial assignment maximizing total positive improvement. Each weak
   candidate is used at most once. Keeping an incumbent has zero gain. The solver
   is the existing dummy-column partial-assignment implementation.
7. Update tracks once using the selected detections. A weak selection does not
   enter or refresh the strong appearance gallery. Remove used weak candidates
   from the subsequent low-score recovery pool.
8. Released high detections remain available for unconfirmed tracks and original
   high-score births. Lost-track high recovery, unmatched-track low recovery,
   removal and duplicate handling retain the pinned implementation's rules.

This is an optimal refinement only for the constrained candidate matrix at that
round, not a globally optimal tracker. Incumbent high detections are not
redistributed between tracks. Exact-gain ties follow the pinned solver and
canonical track/candidate ordering; no arbitrary input-order invariance is claimed.

## Evaluation and risks

Replay all source frames from zero without GT input and without a temporal reset.
Save all three traces and refinement decisions before loading GT. Use the existing
per-camera motmetrics protocol on frames 2..end, with pooled local IDF1, precision,
recall, IDSW, FP and FN. Inspect cameras 5 and 8 only as supplementary examples.
Every refinement records old/new detection indices, scores, prior cosine,
predicted IoU and actual prior gallery frame indices.

A released high box can create a duplicate track. Small positive gains can be
noise, weak crops may contain occlusion, and the historical descriptor may be
contaminated. Measure these effects rather than assuming an improvement. Keep
the full runtime baseline unchanged until global evaluation is also completed.
The inspected two-minute training-scene fragment is development data, not an
independent validation set. Do not select deployment thresholds from it.

This stage evaluates local tracking; global IDF1, independent validation and
end-to-end latency remain separate requirements. Existing global replay code
expects the earlier experiment protocol and must not be pointed at this new
report without a dedicated adapter.

## Running in WSL

From `/home/jakjan/projects/multi-camera-person-tracking`, activate `.venv`, then:

```bash
PYTHONPATH="$PWD/src" python scripts/check_competitive_bytetrack.py
```

After the checks pass:

```bash
PYTHONPATH="$PWD/src" python scripts/experiment_competitive_bytetrack.py \
  --local-report artifacts/appearance_bytetrack/20261007T171955616342Z/report.json
```

Outputs are saved under `artifacts/competitive_bytetrack/<UTC-run>/`.

## Scientific context

Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*,
ECCV 2022, https://arxiv.org/abs/2110.06864.
Section 3 and Algorithm 1 describe high-first association followed by weak-box
recovery for unmatched tracks. The authors also motivate motion-only matching
for weak boxes because their appearance can be unreliable under occlusion.
This experiment deliberately tests a constrained exception to that staging,
not a claim that the paper recommends our modification.

Wojke et al., *Simple Online and Realtime Tracking with a Deep Association Metric*,
2017, https://arxiv.org/abs/1703.07402, provides the broader appearance-association
motivation. This experiment is not a reproduction of Deep SORT or BoT-SORT.

## Implementation and technical verification

`competitive.py` reuses the pinned candidate/history implementation and copies
its update state machine with a pre-update refinement and consumed-weak filtering.
The inherited Supervision code remains under `_vendor/LICENSE.supervision`.
Existing source files and installed Supervision are not edited.

Synthetic checks cover positive-gain competition, exhaustive small assignment
solutions, missing features, motion gates, weak-only births, lost-track handling,
unmatched-track reservation, no weak history refresh and disabled-control parity.
A 300-round technical CLI fixture uses cached early boxes and synthetic constant
features; it checks baseline reproduction, complete evaluation and output hashes.
That fixture is an implementation test, not an OSNet quality result. Actual pinned
environment checks and the two-minute experiment are run in the user's WSL.
