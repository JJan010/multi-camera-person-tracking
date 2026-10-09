# Dimension-explicit appearance history

## Starting evidence

Full-rate CLIP-ReID caches completed for both 3600-frame scenes:

| Scene | Encoded observations | Checked temporal crops | Max absolute feature error |
|---|---:|---:|---:|
| train/scene_001 | 180547 | 6021 | 0.0 |
| val/scene_041 | 60638 | 2019 | 0.0 |

Source cache runs: 20261008T144440650354Z and 20261008T145233226492Z.
Both retain all original local observations, with zero fully outside crops.
Exact feature equality is established for the 120-frame reference grid in each
scene. Full mapping and vector normalization checks cover the complete caches.
These facts establish cache validity, not global tracking quality.

## Contract

`mtmc.reid.feature_history` is separate from the frozen OSNet history. A
FeatureSpace includes the pinned model/preprocessing configuration SHA-256,
feature dimension and normalization. FeatureBatch also carries run scope,
observation keys and exact scene timestamps. A 1280-dimensional vector cannot
be passed to the 512-dimensional history, and two different model configurations
cannot share a history even if they have equal dimensions.

The numerical policy is unchanged: include current sample, retain at most eight
samples no older than one scene second (inclusive), average in float64, normalize
jointly, return float32. If a mean cancels to near zero, use the latest vector.
Empty rounds advance eviction without emitting absent tracks. Input and output
arrays do not expose retained state. Invalid batches are rejected before commit.

## Controlled CPU replay

`scripts/replay_feature_history.py --cache-report <CLIP full-rate report>` reads
only frozen caches, source traces and configuration. It executes no neural
model, decoder, tracker, GT evaluator or identity manager.

For every observation it joins the original camera/local/frame key to the
source trace's enabled `segment_bindings`. Both encoders therefore use the SAME
previously recorded continuity cuts. It does not rerun an OSNet-calibrated
continuity detector with CLIP vectors. This is a conditional comparison on
OSNet-derived local tracks and segments, not an end-to-end CLIP tracking system.

Three histories run in parallel in the same logical scene sequence:

1. Existing AppearanceHistory on frozen 512-dimensional OSNet candidate vectors.
2. New FeatureHistory on those identical OSNet vectors and segment keys.
3. New FeatureHistory on the corresponding frozen 1280-dimensional CLIP vectors.

For 1 versus 2, compare latest/mean arrays exactly, observation keys, timestamps,
source sample frames/times, cancellation flags, normalization lengths and state
counts. This proves implementation compatibility for history; it does not yet
prove exact association/global-manager compatibility. CLIP and OSNet histories
must admit the same samples, but their vector values are intentionally different.

## Outputs

`artifacts/feature_history/<UTC run>/` contains a report, run status,
`mean_embeddings.npy` and `history_rows.jsonl.gz`. CLIP mean rows have the SAME
row indices as the source CLIP full-rate cache. The mapping preserves original
observation keys and records effective segment keys and causal sample provenance.
Unavailable observations retain null rows. The means are written using a memory
map and checked for finite normalized values after writing.

Mean-vector storage adds approximately another 1.15 GiB across the two scenes.
Raw vectors remain in the source caches; no input, baseline or historical output
is overwritten. Partial output has an incomplete status; rerun into a new run.

## Scope and next step

No thresholds or deployment policies are selected. Both scenes are already part
of development, so neither is an untouched final test set. Next implement the
model-scoped association interface and verify the OSNet path before measuring
CLIP global-ID quality with declared, model-appropriate thresholds. Improvements
in retrieval ranking do not by themselves establish improvements in global IDF1.
