# Gallery-only overlap admission experiment

This is one development hypothesis after visual review of contaminated return galleries.
It does not replace the runtime baseline or select a deployment setting.

## Fixed comparison

Both variants enable the existing confirmed CLIP return registry. `disabled` means
**overlap gate off**, and `enabled` means **overlap gate on**. This differs from the
meaning of those labels in the source recovery experiment.

The control reproduces the source recovered variant's complete global runtime records,
archive decisions, reactivation events and lifecycle at every frame. Offline evaluation
also reproduces its full/window identity metrics, merge labels and endpoint return labels.
Source hashes are checked before replay and after evaluation. Predictions are frozen
before GT is loaded. No inference, decoding, detector or local tracker is run.

Local observations, confidence, CLIP raw features and means, segment cuts, scene geometry,
ordinary association/merge settings, query admission and confirmation settings stay fixed.
Future global decisions can differ as a consequence of different stored gallery evidence.
Both scenes have previously been inspected; this is development evidence, not an untouched test.

## Single new rule

For each current observation, clip its continuous xyxy box to the image. Compute the
intersection with every other current local-track box in the same camera, divided by
**the target box's own clipped continuous area**. Take the maximum individual overlap.
This is asymmetric, is not IoU, and is not the union of all intersecting boxes.

A sample already passing the existing confidence/border requirements may be appended to
the return gallery only when this maximum is **at most 0.3**. Equality is accepted.
This threshold is one declared hypothesis, not fitted here. All current track boxes,
including weak observations, are competitors. Fully outside boxes have no intersection;
other cameras do not compete. Keys exclude the observation itself.

Tracking observations, current return queries and association/history descriptors are
not removed or gated. The gallery is rebuilt from the pre-round stored samples and
accepted current updates, following the original ID remapping, age and per-camera cap.
Removing rejected samples only after the original truncation would incorrectly evict
older good samples, so the implementation deliberately avoids that shortcut.

Retained samples keep their actual timestamps and expire normally. No rejected update
refreshes a descriptor. Empty/unavailable galleries remain unavailable. Disabled
reconstruction is checked against the original gallery update implementation.

## Outputs

`artifacts/overlap_gallery/<run>/report.json` contains full/first/second global IDF1,
IDTP/IDFP/IDFN verified with motmetrics, lifecycle, admission counters, no-gallery retained
identity-rounds, archive events (including skipped retirement reasons), return decision
counts and endpoint GT diagnostics for both variants. No-gallery counts are identity-rounds,
not unique identities. Gallery admission counters include all runtime frames; quality
metrics use the configured evaluation interval (2..3599).

`global_tracks.jsonl.gz` retains all observations, decisions and per-observation overlap
admission with original and effective segment keys. `reactivation_diagnostics.json`
contains both variants' endpoint labels. Endpoint labels are not a proof of whole-gallery
purity and must not be interpreted as such.

## Checks and execution

```bash
PYTHONPATH="$PWD/src" python scripts/check_overlap_gallery.py

PYTHONPATH="$PWD/src" python scripts/experiment_overlap_gallery.py \
  --recovery-report artifacts/confirmed_clip_return/20261009T194904982876Z/report.json

PYTHONPATH="$PWD/src" python scripts/experiment_overlap_gallery.py \
  --recovery-report artifacts/confirmed_clip_return/20261009T195106432886Z/report.json
```

The synthetic checks cover exact control/clean-enabled parity, a known delayed return,
clipped and asymmetric overlap, weak competitors, inclusive overlap and age boundaries,
old-sample preservation, loss of recovery after rejecting all source evidence, ordering,
ID zero, preflight correction and transactional failure. A separate local serialized
150-round fixture checked the experiment join, recovered control, full/window motmetrics,
lifecycle and return labels. These checks do not measure real tracking quality.

## Limitations

Overlapping boxes are a proxy for ambiguous crops, not measured visibility. Duplicate
predictions or close people may reject useful evidence. Missing person boxes, furniture,
door frames and other static occluders are not detected by this rule. It cannot guarantee
that the scene_041 doorway case is resolved. Existing mixed identities, local switches,
poor queries and association errors remain possible.

Judge the paired result by full-run IDF1 and return errors/coverage, not only the previously
inspected examples. Do not promote the gate just because fewer returns are made.
