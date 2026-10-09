# Confirmed return context diagnostic

Run `scripts/diagnose_confirmed_return_context.py --self-check`, then pass the
`--recovery-report` from a completed `confirmed_clip_return_paired_v1` experiment.
The default context is 30 frames before each previous endpoint and 30 frames
after each confirmation. It audits every return, including previously resolved
returns. Both scenes use the same interval; no thresholds are changed.

This is offline GT analysis. No models, video decoding, tracking or recovery
execute. The diagnostic verifies trace, event, scene and GT checksums and repeats
the checks after reading. Original events and endpoint labels must reproduce.

Spatial matching keeps all predictions in each camera before selecting a target;
competing predictions and ambiguous GT remain unknown. Context labels only
include the endpoint's original local track, effective segment and global ID.
Other segments or IDs are recorded as excluded rather than assimilated. ID zero
is valid. No nearest-label imputation replaces an unknown endpoint.

Context categories summarize only available mutually unique labels:
- same_context_gt: each side has one known label, equal across the sides;
- different_context_gt: each side has one known label, different across sides;
- mixed_context: a side contains several known labels;
- insufficient_context: one side has no known label.

These are evidence descriptions, not correctness certificates. Same-context
labels do not establish gallery purity, correctness across the full identity's
lifetime, or absence of switches outside the interval. Future context must not
be used as an online input.

Outputs under `artifacts/confirmed_return_context/<run>/`:
- report.json: all event summaries, endpoints, nearest evidence and checksums;
- observations.json: individual requested observations, including exclusions;
- confirmation_decisions.json: saved online decisions for the provisional IDs
  that eventually returned, without recomputing or reranking candidates.

Dependencies: existing `scripts/diagnose_recovery_context.py` (spatial evidence
helper), `mtmc.data.scene`, and `mtmc.data.ground_truth`. No existing source is
replaced. Local checks include serialized context with ID zero, unknown GT and
segment boundaries, and actual serialized confirmed-return registry output.
