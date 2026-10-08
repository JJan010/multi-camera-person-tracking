# Dormant return context diagnostic

The first fixed-radius experiment recovered five IDs in scene_001. Full-sequence
IDTP/IDFP/IDFN stayed exactly 96696/83765/87511. The second-minute identity
assignment gained 170 IDTP (60.64% to 60.84% IDF1); this is a separate optimal
identity matching and does not imply a full-sequence gain.

All five return endpoints used the same camera before and after the gap:

| Global ID | Camera | Last frame | Return frame | Gap in frames |
|---|---:|---:|---:|---:|
| 154 | 5 | 2016 | 2054 | 38 |
| 186 | 4 | 2434 | 2528 | 94 |
| 175 | 5 | 2570 | 2616 | 46 |
| 202 | 4 | 2643 | 2674 | 31 |
| 213 | 8 | 3393 | 3431 | 38 |

GIDs 154 and 213 have matching last-visible and returning unique GT labels
(9 and 14, respectively). The other three events remain unresolved at their
endpoints. These checks do not establish the purity of historical descriptors
or the correctness of an identity's entire lifetime.

`scripts/diagnose_recovery_context.py` reads the frozen enabled trace and checks
15 preceding frames for each previous local member and 15 subsequent frames
for each returning local member, including both boundary observations. At
30 FPS these are half-second context intervals. Absent observations are explicit.

It uses the existing clipping and inclusive IoU gate, retaining all predictions
in each camera/frame. It distinguishes no admissible GT, multiple admissible GT,
and one admissible GT with competing predictions. Best IoU is diagnostic only;
it never substitutes a nearest GT label for mutually unique evidence. ID zero
is valid. All original event members and boundary labels must reproduce exactly.

This is offline evidence, including future context after a return. Context
labels must not enter runtime decisions or automatically replace unresolved
endpoint labels. No predictions, quality metrics, archive settings or old source
files are changed. The default runtime policy remains unchanged.

Run from the project root with the project's venv active:

```bash
PYTHONPATH="$PWD/src" python scripts/diagnose_recovery_context.py --self-check
PYTHONPATH="$PWD/src" python scripts/diagnose_recovery_context.py \
  --recovery-report artifacts/dormant_recovery/20261008T094135187275Z/report.json
```

Outputs are a new timestamped `artifacts/dormant_recovery_context/` directory,
with `report.json` and detailed `observations.json`. The report pins input hashes,
the diagnostic code hash and the observation artifact hash. Models and video
decoding are not run; the existing scene loader verifies video files by hash.
