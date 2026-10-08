# Frozen continuity transfer: mapping and lifecycle context

Purpose: explain the difference between independent minute evaluations and one
shared identity assignment across the entire scene_041 sequence. Input is the
completed appearance-continuity transfer report. This is offline diagnosis with
GT; it never changes observations, assignments, settings or runtime state.

Run from the WSL project root with its virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/diagnose_continuity_transfer.py --self-check

PYTHONPATH="$PWD/src" python scripts/diagnose_continuity_transfer.py \
  --validation-report artifacts/appearance_continuity_validation/20261008T111045386342Z/report.json
```

The script verifies recorded trace/audit/config/GT checksums, preserves original
scene dimensions and empty GT slots, and recomputes all six full/window identity
metric records. They must exactly match the completed transfer experiment. All
spatially admissible IoU candidates contribute to the global identity assignment.
No early framewise matching is used to manufacture identity labels for metrics.

For each variant it records:

- selected optimal GT/global-ID assignments for full, first and second windows;
- per-GT contributions to separate-window IDTP minus shared-full IDTP;
- overlap counts for each selected global ID in both windows and the full run;
- global-ID changes between consecutive mutually unique GT observations within
  the same camera, including original local ID and internal segment generation;
- the previous global ID's recorded status before and after the current round;
- an allocation, expiration, absorption/merge and segment-cut event ledger.

The expected total assignment gaps for this exact input are 8489 (disabled) and
10499 (enabled). These are identity-correct observation differences, not counts
of people, switches or guaranteed recoverable errors. A per-GT contribution can
be negative because identity assignments compete one-to-one. Optimal assignment
ties can alter which GT receives a contribution while preserving total metrics.
Selected matching IDs are not necessarily the dominant unique-evidence IDs.

Output: artifacts/continuity_transfer_diagnostic/<run-id>/

- report.json: verified metrics, mapping contributions, evidence/event counts;
- evidence_transitions.json: same-camera GT evidence changes for both variants;
- lifecycle_events.json: complete event ledger for the inspected runtime;
- top_mapping_context.json: eight largest mapping-gap contributors per variant,
  their evidence transitions and all lifecycle events involving selected IDs.

Status visible means the previous ID has a current member; lost means it is
still retained without a current observation. Expired and absorbed are distinct.
An ID can have been absorbed into another ID without representing a person
change. A split can happen before an observed global-ID transition, or may not
change the global ID at all. The full event ledger is therefore retained in
addition to same-round event context.

Evidence transitions use mutually unique IoU matches and may span unknown or
absent observations. Their frame gap is explicit. They are not CLEAR IDSW and
must not be treated as proof that a particular event caused an error. Even
same-camera GT identity evidence is an offline diagnostic, never a runtime input.
Lifecycle counts must reproduce the source allocated IDs, expired IDs, merges
and split counts. Source checksums are verified again after diagnosis.

Before delivery: known-answer checks cover stable identity, fragmentation,
identity mixing, empty observations, ID zero, retained/lost, expired and absorbed
states. Integration fixtures cover 120 rounds with arbitrary camera IDs/sizes,
empty cameras/rounds/GT slots, outside predictions, constant appearance and real
synthetic segment cuts. All full/window metrics reproduce the prior evaluator;
allocation/expiration/merge/split counts reproduce the source runtime. These are
technical fixtures, not measurements of model quality or GPU performance.
