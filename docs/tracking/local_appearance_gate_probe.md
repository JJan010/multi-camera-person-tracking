# Observe local appearance decisions before changing the policy

Continuity diagnostic: `20261007T180250829032Z`, direct_appearance.
Full global IDF1 is 53.03% while the fixed-box framewise spatial F1 ceiling is
97.28%. The ceiling is an offline diagnostic, not detector AP or an achievable
online tracking guarantee. Evidence includes 58 mixed local tracks and 45 mixed
global identities; 95 evidence-label transitions are not the CLEAR IDSW count.

Select camera 8 / local ID 44, which has evidence of 3 GT identities and 909
observations outside its dominant GT label. This probe selects its earliest
recorded evidence-label transition deterministically, including gaps in evidence.
It does not assume that the transition is instantaneous or that it is the cause
of all downstream errors.

## Mechanism being inspected

The pinned experimental tracker checks appearance before high-score matching
and unconfirmed-track matching. The low-score recovery stage still uses the
original IoU matching. This is intentional in the existing experimental policy:
weak detections can recover partially occluded people, but their embeddings can
also be unreliable. ByteTrack's paper motivates retaining low-score detections
instead of discarding them: https://arxiv.org/abs/2110.06864.

Possible explanations to distinguish include low-stage acceptance without an
appearance gate, a high-stage match passing the gate, and missing prior features
causing the configured motion/score fallback. These are hypotheses until the
target replay is inspected. No threshold or policy changes are made here.

## Probe behavior

The probe replays the selected camera from frame zero using its frozen candidate
boxes, scores and embeddings. It subclasses the existing tracker solely to
observe `_appearance_gate`: the original method is called and its result is
returned unchanged. Source hashes are checked against the frozen local experiment.
Every replayed camera update must reproduce all output IDs, boxes, confidences,
candidate indices and embedding-row mappings exactly.

The report saves a bounded context around the previous/current evidence frames:

- Accepted detection index and score.
- Accepted matching stage (or initial identity/preconfirmation case).
- Cosine similarity to the actual causal strong-sample gallery before update.
- Scene-frame indices of the prior gallery samples.
- Original high/unconfirmed costs, motion eligibility, feature availability and
  appearance rejection flags for target-track candidates.

The low stage is identified using the pinned disjoint score partition; its IoU
cost matrix is not instrumented. Scores below the activation threshold and above
0.1 belong to that stage. High-score acceptance for an established external ID
must agree with the observed gate. GT evidence selects the case offline but is
never supplied to the tracker or used to modify its decisions.

## WSL commands

```bash
PYTHONPATH="$PWD/src" python scripts/check_local_gate_probe.py

PYTHONPATH="$PWD/src" python scripts/probe_local_appearance_gate.py \
  --diagnostic-report artifacts/appearance_continuity/20261007T180250829032Z/report.json \
  --camera 8 \
  --local-id 44
```

Outputs: `artifacts/local_gate_probe/<run_id>/report.json` and `timeline.json`.
No models, video decoder or GPU are needed. Existing tracker sources and frozen
results are not modified.

Synthetic checks cover high-score appearance rejection followed by low-score
recovery, low-score acceptance of a dissimilar feature, missing-feature fallback,
and equality with an unobserved tracker. A complete synthetic CLI run additionally
checks source joins, final input checksums, output writing and timeline hashes.

Use the observation to choose a targeted paired experiment. Do not assume that
applying appearance gating to every weak detection is an improvement: it can
also break recovery and increase fragmentation. Any modified policy needs its
own comparison on fixed inputs and subsequent untouched validation material.
