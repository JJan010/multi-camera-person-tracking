# Frozen global identity replay

This CPU-only step replays the saved groups from the scene_001 grouping diagnostic
through `local_anchor_no_merge_v1`. It records persistent assignments and lifecycle
events. It does not run the detector, tracker, OSNet or pairwise assignment again.
It does not yet compute identity quality metrics or select a deployment threshold.

## Run

From the project root with its virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_global_identity_replay.py
PYTHONPATH="$PWD/src" python scripts/replay_global_identity.py \
  --grouping-report artifacts/multicamera_grouping/20261007T084841755095Z/report.json \
  --max-idle-seconds 1
```

The positive idle duration is mandatory. Here one second is an explicit integration
setting, not a calibrated value and not evidence that it is optimal. Exact rational
values such as `1/2` are accepted. Expiry uses scene timestamps, not elapsed program
time. No new dependencies are required beyond the current evaluation environment.

## State isolation and initialization

One independent manager is created for each descriptor variant and threshold in
the source report. All managers start empty at frame 2 and process frames 2..299.
There is no warmup, retrospective relabeling, future-frame access or extra empty
round after the clip. A full run from frame zero can produce different IDs.

A global integer is meaningful only with `identity_scope`. This field includes
the replay run, descriptor variant, threshold and idle duration. In particular,
`global_id=1` from `latest/0.7` and `global_id=1` from `mean/0.7` are not the same
identity. The original video/trace `run_id` is preserved separately.

## Provenance and input boundary

The grouping report and compressed trace are hashed. The replay checks source
protocol, order, timeline and experiment scope; every setting must contain the
same observations, embedding rows and inherited labels at a given frame. Rows
cannot be reused across frames. Each partition has unique observations and at most
one observation per camera inside a group.

The grouping decision audit must support exactly all retained within-group edges.
Aggregate grouping counters and correct/wrong/unresolved retained edge counts are
reconstructed and compared with the source report. Input hashes are checked again
at completion. This is consistency verification, not a rerun of upstream matching.

Runtime `FrameGroups` passed to the identity manager contain observation keys and
scene context only. Saved GT labels and source embedding rows are kept in a separate
lookup and copied into the output trace after assignment. The smoke test confirms
that changing diagnostic GT leaves the runtime input and manager output unchanged.
Raw images, embeddings, boxes and GT files are not reread or independently reverified.
Their earlier provenance is inherited through the hashed source reports.

## Outputs

Each invocation creates `artifacts/global_identity/<UTC replay ID>/`:

- `report.json`: protocol, source hashes, code hashes, summary and completed checks;
- `summary.csv`: one row per variant/threshold;
- `by_frame.csv`: counters and current registry sizes per frame and setting;
- `assignments.jsonl.gz`: one record per frame and setting, containing assignments,
  diagnostic metadata, group decisions, visibility state and expiration events.

Each assignment retains `(camera_id, local_id, frame_index)`, `embedding_row`,
`diagnostic_gt_id`, assigned global ID and assignment reason. Exact timestamps are
stored as rational strings. Registry state includes visible/lost identities and
retained local bindings. Thus later evaluation can join predictions back to the
frozen source trace without rerunning inference.

## Console columns

| Column | Meaning |
| --- | --- |
| `GlobalIDs` | Distinct global IDs allocated over the clip; not unique true people. |
| `Attached` | New local-track observations attached to an existing global ID. |
| `MultiIDConf` | Group decisions rejected because they contained multiple existing global IDs. |
| `ReservedCam` | Group decisions rejected because a retained binding reserved a requested camera slot. |
| `Competing` | Group decisions rejected because multiple eligible groups requested the same identity/camera slot. |
| `ExpiredIDs` | Global IDs removed after all their local bindings expired. |

Conflict counts are summed over groups and frames. The same persistent conflict
may be counted many times; these are not counts of people or unique failures.
`Attached` counts assignment events, not the total number of multi-camera identities.

The report additionally counts observations assigned through local continuity or
new-ID creation, distinct local tracks, expired bindings, retained identities at
clip end, and peaks of visible/lost IDs and retained local bindings.

Checks enforce observation coverage and current-camera uniqueness. Lifecycle
accounting requires allocated IDs = expired IDs + retained IDs at clip end.
Global IDs are not reused. These checks establish bookkeeping integrity; they do
not establish that an ID corresponds to the correct person.

## Interpretation boundary

Do not choose the setting with the fewest IDs or conflicts as the best tracker.
Too few IDs can mean different people were merged; too many can mean one person
was fragmented. No-merge can preserve fragmentation, and local continuity can
propagate a wrong initial association or a local-track ID switch.

This step makes the actual causal identity output available for evaluation.
The subsequent quality analysis must measure identity correctness across cameras
and time, with explicit matching and denominators. This replay is neither that
quality evaluation nor a performance benchmark. The short reused training clip
is still an integration diagnostic.
