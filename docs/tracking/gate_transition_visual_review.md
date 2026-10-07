# Visual review of a high-stage appearance match

Probe: `20261007T181400956969Z`, camera 8 / local ID 44.
Previous unique GT evidence: frame 1438, GT 1, global ID 112.
Current unique GT evidence: frame 1441, GT 13, global ID 112.

The current observation was accepted in the high-score stage with detector
score 0.6299 and cosine similarity 0.7862 to the actual prior strong-sample
history. It passes the configured appearance threshold 0.6. The neighboring
frames 1439 and 1440 also used the high stage, with cosine approximately
0.9027 and 0.9036. This does not implicate low-score bypass at the observed
transition. The three-frame evidence gap is not a three-frame tracker outage.

Do not raise a threshold to reject one inspected match before understanding
the images and comparing the trade-off across fixed observations. Candidate
confidence is a detector score, not identity confidence. A strong-sample history
does not guarantee unoccluded, single-person crop contents.

## Preview

`preview_gate_transition.py` follows the pinned probe/diagnostic/evaluation
references to the frozen direct_appearance boxes and the original video. It
renders context frames around the transition, including the evidence gap when
short, and the exact prior strong-sample frame indices recorded by the observer.

For this event the context frames are 1428, 1438, 1439, 1440, 1441 and 1446.
The prior gallery is read from the actual audit, not guessed from the last eight
video frames. The current crop is displayed separately and is not counted as
a prior sample.

The accepted boxes and original integer crop bounds determine the displayed
raw crops. Their resizing in the contact sheet is only presentation; raw PNGs
are also saved. The script fails if a required prior sample cannot be located
in emitted local records instead of silently substituting a different crop.

Outputs in `artifacts/gate_transition_preview/<run_id>/`:

- `transition_annotated.jpg`: shared context ROI with target/local/global IDs
  and selected GT boxes.
- `transition_raw.jpg`: the same context without box overlays.
- `prior_gallery_and_current.jpg`: prior strong crops plus the current crop.
- Individual context/crop PNGs and `report.json` with source hashes and evidence.

Colors: cyan = target local track; green = previous diagnostic GT person;
pink = current diagnostic GT person. `Unique GT: None` means no mutually unique
IoU label for that observation, not absence of a person.

## Run

```bash
PYTHONPATH="$PWD/src" python scripts/preview_gate_transition.py \
  --probe-report artifacts/local_gate_probe/20261007T181400956969Z/report.json
```

Only the selected camera is decoded, with exact PTS checks. Models, association
and tracking are not run. No configuration or prediction is modified.

Inspect whether the target box changes person, contains both people, or shifts
its overlap between GT boxes while crop contents remain ambiguous. Inspect the
prior gallery for earlier mixed crops or appearance changes. These possibilities
are hypotheses until the images are reviewed.

A complete synthetic CLI fixture checked reference resolution, frame selection,
gallery-to-box mapping, PTS validation, rendered image/report output and all
artifact checksums. No real-scene visual conclusion was drawn from that fixture.
