# Frozen CLIP/OSNet identity context

This diagnostic investigates the existing fixed-track experiment. It changes
no assignments, features, thresholds or runtime state and executes no models.
GT is used offline to explain frozen outputs.

The development target is GT 15: it had the largest separate-minus-shared IDTP
contribution in both models. This retrospective choice is for diagnosis, not
an input to the tracking policy. Numeric GIDs are scoped to each model/run;
similar GID numbers across variants do not identify equivalent identities.

```bash
PYTHONPATH="$PWD/src" python scripts/check_clipreid_global_diagnostic.py

PYTHONPATH="$PWD/src" python scripts/diagnose_clipreid_global.py \
  --experiment-report artifacts/clipreid_global_experiment/20261009T191501187432Z/report.json \
  --gt-id 15
```

The reader reproduces both models' full/window identity metrics, lifecycle
event counts and accepted-merge categories. It uses the original scene's
clipping and IoU rules and retains all prediction candidates when assigning
mutually unique GT evidence. An unknown match is never inferred from nearby
frames. Source observation keys must map exactly to internal segment keys and
their assigned GIDs.

For the selected GT, it records a per-frame multi-camera evidence timeline
and per-camera GID transitions. Each transition includes the evidence gap,
original local ID, internal segment ID, assignment reason, related events,
and the previous GID's state before and after the current round. States are
visible, lost, absorbed or expired where available; they explain lifecycle,
not whether an identity assignment is correct. These are evidence transitions,
not a replacement for CLEAR IDSW counts.

For every merge it records current visible-member GT labels and boxes, support
duration provenance, retained membership before merging, and per-GID evidence
counts from the preceding one second. The current frame is excluded from that
prior context. Absorbed IDs are not retroactively aliased in those counts.
Only merges with different known current labels print in full to the terminal;
all merge contexts remain available in the JSON artifact. Neither a successful
current match nor a short pure-looking context proves lifetime identity purity.

Outputs under `artifacts/clipreid_global_diagnostic/<run>/`:

- `report.json`: reproduced metrics, accounting, checksums and target summaries;
- `target_timeline.json`: all selected-GT evidence, without inventing missing frames;
- `target_transitions.json`: all per-camera evidence transitions;
- `merge_context.json`: all merges and their prior context.

Synthetic checks cover ID zero, a known incorrect merge, past-only evidence,
absorbed/lost identity states, local/segment changes, empty GT slots and rejection
of inconsistent internal-to-original assignment mappings.
