# Visual review of a local identity transition

The 60-second diagnostic identified stable framewise spatial matching
(97.62% F1 ceiling) but declining independent-camera pooled local IDF1
(64.45%) and global IDF1 (71.75%). The shared-identity assignment gap was
25115 observations, compared with fixed-box minimum FN=3166 and FP=1447.
These quantities prioritize identity investigation but do not identify the
causal contribution of an individual module.

The highest-ranked local mixing case was camera 5 / local ID 11:
two distinct diagnostic GT identities and 811 mutually unique observations
outside the dominant GT label. Select this case for visual review before
changing tracker settings, appearance gating or global identity bindings.

## Run

From the WSL project root with the project environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_identity_switch_preview.py

PYTHONPATH="$PWD/src" python scripts/preview_identity_switch.py \
  --diagnostic-report artifacts/mtmc_sequence_diagnostic/20261007T135553276618Z/report.json \
  --camera 5 --local-id 11
```

The command uses existing NumPy, Pillow and PyAV packages. No models execute.
Only the selected camera is decoded on CPU. Frames are decoded sequentially
with exact checks that PTS * time_base equals frame_index / 30.

## Deterministic selection

Read the checksum-pinned `local_label_transitions.csv` from the diagnostic.
Select the earliest change between two different mutually unique GT labels
for the requested camera/local ID. Its previous and current evidence frames
can be separated by a gap; the report explicitly records this gap.

Select context frames at previous-15, previous-5, previous, current, current+5
and current+15, clipped to valid runtime boundaries and deduplicated. At 30 FPS,
15 frames provide half a second of context. This is not a claim that the true
physical transition occurred exactly at the first newly labeled frame.

Recompute unique IoU evidence at both endpoint frames and require it to match
the diagnostic GT and global IDs. Validate all frozen local/global observation
keys and embedding rows while reading the source trace. Verify the original
video manifest, video hash and pipeline input video checksum before decoding.

If camera/local ID arguments are omitted, select the top local-mixing case in
the diagnostic ranking. That ranking orders evidence outside the dominant GT;
it is not an official identity-switch metric or a ranking of causal damage.

## Images and interpretation

Output: `artifacts/identity_switch_preview/<UTC-run-id>/`.

- `camera_0005_local_0011_contact.jpg`: chronological context, left-to-right
  then top-to-bottom, at most six panels;
- `*_annotated.png`: original-resolution frames with overlays;
- `*_raw_context.png`: unannotated context crops for visual inspection;
- `report.json`: selection, exact times, source coordinates, current global IDs,
  GT candidates, unique GT labels, hashes and image filenames.

Colors are stable across the contact sheet:

- **cyan, solid box:** the selected local track, labeled with its local/global ID;
- **green, dashed box:** the earlier diagnostic GT person;
- **pink, dashed box:** the later diagnostic GT person;
- **gray:** neighboring local tracks with their local/global IDs.

Every panel uses the same context rectangle, computed from the selected track
and the two GT persons across all selected frames, with padding. This avoids
independent recentering that could conceal movement. Presentation cropping and
resizing never change the original boxes used in evaluation.

Inspect whether the cyan track follows the same visible person, disappears
behind another person, or begins following the other GT person. Compare the
global ID before and after. A unique IoU label alone is not visual proof of a
track switch; occlusion and box overlap can make labeling misleading.

The current global manager uses retained camera/local bindings as continuity
anchors. If a local track changes person while keeping its ID, the binding can
preserve the old global identity. This is a hypothesis to investigate in the
selected images, not a conclusion established by the histogram alone.

Review the annotated frames together with the raw context if labels obscure
important details. If the event is not clear, inspect additional neighboring
frames rather than choosing a new threshold based only on the contact sheet.
This preview does not alter history, local IDs, global IDs or merge decisions.
