# Geometry and controlled identity: paired results

Experiment: `20261007T124001420051Z`.
Baseline: `20261007T104647560154Z`.
Scene 001, cameras 4/5/8, frames 2..299 at 30 FPS.

The baseline appearance groups and complete controlled identity records were
reproduced before comparison. Both variants use the same tracked boxes,
appearance descriptors and observation rows. New outputs were frozen before
offline evaluation; runtime association and identity management received no GT.

## Configuration

- Mean appearance descriptors, cosine threshold 0.70.
- Ground-plane distance <= 2 native dataset units before pair assignment.
- Explicit appearance-only fallback for unavailable projected positions.
- Complete-support greedy grouping.
- Existing controlled merge policy, with unchanged idle and confirmation
  settings: idle 1 second; at least 3 rounds AND 1/5 second support;
  maximum evidence gap 1/10 second.

The distance is an in-sample integration candidate selected after inspecting
this training fragment, not an independently calibrated deployment threshold.

## Shared global identity metrics

| Metric | Appearance + controlled merge | Geometry + controlled merge |
| --- | ---: | ---: |
| IDF1 | 85.03% | 86.02% |
| IDP | 86.34% | 87.35% |
| IDR | 83.75% | 84.73% |
| IDTP | 13666 | 13826 |
| IDFP | 2162 | 2002 |
| IDFN | 2651 | 2491 |

The reported difference is +1.00 percentage point when computed before
rounding. There are 160 additional identity-correct observations. GT and
prediction observation totals remain 16317 and 15828 respectively.

## Link diagnostics

| Population | All links | Correct | Different known GT | Unknown GT |
| --- | ---: | ---: | ---: | ---: |
| Baseline pairs | 10968 | 10507 | 363 | 98 |
| Geometry pairs | 10762 | 10620 | 52 | 90 |
| Baseline group links | 10431 | 10140 | 210 | 81 |
| Geometry group links | 10472 | 10362 | 37 | 73 |
| Removed pair links | 319 | 0 | 311 | 8 |
| Newly selected pair links | 113 | 113 | 0 | 0 |

The gate removes 311 known-wrong pair links without removing diagnostically
correct ones in this comparison. Assignment also selects 113 new correct
links. That second effect was not measured by the earlier fixed-pair filter.
All counts represent repeated framewise links, not independent people or
independent trials. Labels come from the stated offline IoU matching protocol.

## Identity lifecycle and merge evidence

- Allocated IDs: 49; ever emitted: 49.
- Absorbed IDs: 18; expired IDs: 3; retained at end: 28.
- Lifecycle accounting: 49 = 18 + 3 + 28.
- Accepted merge events: 18.
- Acceptance-time visible-member GT diagnostics: 18 all-same-GT, 0 different
  known GT, 0 unresolved.
- Unavailable projected observations: 0.

The acceptance-time diagnostic is not a guarantee that every global identity
has a correct full history. Initial groups, local-track identity changes and
later attachments can still affect identity quality. The presence of 37
known-wrong grouped links and IDF1 below 100% makes the remaining limitation
explicit. The fallback policy was not exercised by this real-data run.

## Development direction

Preserve this run as a quality reference while integrating the complete
video-to-global-ID pipeline. Establish its stage timings, memory behavior,
restart behavior and quality using the current PyTorch CUDA backends.

Before further threshold tuning, create versioned scene/time manifests for
development validation and a separate final test. Assess a longer sequence
and another scene with the current configuration. Adjacent frames from this
same short fragment are not an independent validation set. If inspected
validation results inform changes, do not relabel them as final test results.

Subsequent quality changes should address measured failure modes, such as
remaining wrong associations, fragmented identities, local identity switches
or unreliable ground-contact proxies. Change one mechanism at a time and
compare on fixed observations where appropriate. Improvement on this short
integration clip alone is insufficient evidence of generalization.

TensorRT and video-GPU work retain the same quality reference. Evaluate not
only tensor differences, but detection/retrieval decisions and local/global
identity metrics. Record quality and latency jointly. Optimizing execution
does not require first eliminating every identity error.

The full report is copied to `identity_experiment_metrics.json`; raw traces
remain in the ignored timestamped artifact directory. A small CSV comparison
is copied to `identity_experiment_comparison.csv`.
