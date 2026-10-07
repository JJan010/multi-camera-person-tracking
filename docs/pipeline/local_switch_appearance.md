# Local identity takeover: frozen appearance diagnostic

## Observed case

Run `20261007T133915524799Z`, camera 5, local track 11. The visual preview
`20261007T141100069044Z` supports a local identity takeover during occlusion:

- Frames 798, 808 and 813: L11/G15 follows GT 0, the plaid-shirt person.
- Frame 815: the first selected unique overlap evidence associates L11/G15 with
  GT 23, the gray-jacket person. Frame 814 was not shown in the contact sheet.
- Frame 830: the two people are visually separate. L11/G15 follows GT 23,
  while the plaid-shirt person is represented by L40/G78.

GT 0 is a valid person ID. Missing GT annotation at an individual frame does
not establish that the person physically disappeared. This is one inspected
failure, not a causal explanation of every identity error in the sequence.

The local-ID anchor can propagate a local identity takeover into a global ID.
Appearance history keyed by the local track can also accumulate the new
person's descriptors. These observations motivate inspecting appearance
before changing local association or global identity policies.

## Diagnostic

From the project root in the existing virtual environment:

```bash
PYTHONPATH="$PWD/src" python scripts/diagnose_switch_appearance.py \
  --preview-report artifacts/identity_switch_preview/20261007T141100069044Z/report.json
```

The script verifies hashes and observation-to-embedding row mapping, then
reads frozen latest and mean embeddings on CPU. It never runs a model or
changes a prediction. Default context extends 30 frames before the previous
evidence frame and 30 frames after the new evidence frame (783..845 here).

The fixed anchor is the first encoded target observation in this time window,
selected without GT. It is a diagnostic reference, not a calibrated runtime
gallery or a guaranteed clean crop.

The target timeline reports normalized cosine similarity between:

- Current and previous encoded observation, with the frame gap recorded.
- Current observation and the previous encoded observation's mean descriptor.
  This mean precedes the current observation.
- Current observation and the fixed anchor.
- Current mean, which includes the current observation, and the fixed anchor.

The camera-track table provides the same anchor comparison for other local
tracks, including any later track assigned to the original person by unique
GT overlap. GT only labels this offline diagnostic.

Detector evidence lists saved candidate indices, confidence and clipped-box
IoU against the two selected GT identities. The inclusive IoU >= 0.5 gate is
diagnostic. A single detection can overlap both GT boxes; these candidate
lists are not the tracker's actual assignments. Saved detections already
passed the configured detector threshold. Tracked-box embeddings do not
provide embeddings for every competing detection.

Outputs under `artifacts/identity_switch_appearance/<timestamp>/`:

- `target_timeline.csv`: every frame in the context, including target absence.
- `camera_tracks.csv`: all visible local tracks in the selected camera.
- `detector_evidence.json`: detector candidates for both diagnostic GT IDs.
- `report.json`: input/output checksums, selected results and limitations.

## Interpretation and next decision

A sustained change relative to the old reference could support an experiment
with appearance-aware local association or a track-continuity safeguard.
Strong adjacent-frame similarity alone cannot exclude gradual appearance
drift. Neither a cosine decrease nor a single inspected failure defines a
deployment threshold. A safeguard also needs tests for legitimate pose,
viewpoint and occlusion changes and evaluation on a separate sequence.

Deep SORT is relevant background: Wojke, Bewley and Paulus (2017),
*Simple Online and Realtime Tracking with a Deep Association Metric*,
https://arxiv.org/abs/1703.07402. It combines appearance and motion for local
association. This diagnostic does not implement or claim to reproduce Deep SORT.

## Verification

The script was exercised end to end on an existing technical fixture with
synthetic descriptors. Known-answer checks covered cosine values, missing and
invalid vectors, the inclusive IoU boundary, shared detection candidates and
empty detections. Actual OSNet values for this minute-long run must be computed
in the user's environment from the pinned arrays.
