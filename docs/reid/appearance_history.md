# Causal appearance history: latest versus normalized mean

This step constructs descriptors from the frozen observations of run
`20261006T204237314810Z`. It does not rerun decoding, detection, tracking or
OSNet. Both descriptor variants therefore share exactly the same input crops
and local track decisions, including their known errors.

## First experiment

For each local track, compare:

- **Latest:** the current observation's normalized embedding, copied exactly.
- **Mean:** the L2-normalized arithmetic mean of the current and recent
  embeddings from that same camera/local ID.

The initial configuration retains at most 8 observations whose age is at most
1 second in scene time. Current observations are included. These are explicit
starting parameters, not values tuned on validation data or claimed optimal.
At 30 FPS, eight consecutive observations span 7/30 seconds (about 0.233 s),
not eight seconds. Gaps can increase this span up to the configured age limit.

For retained normalized vectors e_1 ... e_n, first compute
`m = (e_1 + ... + e_n) / n`, then return `m / ||m||_2`.
Accumulation uses float64; the stored/output descriptors are float32.
Normalization keeps cosine comparisons consistent with the single-frame
baseline. If the mean norm is <= 1e-12, the output uses the latest vector and
explicitly records this fallback. That threshold is a numerical guard against
cancellation, not an identity-matching threshold.

## Runtime contract

`AppearanceHistory` is instantiated once for one run/session/generation.
The internal key is `(camera_id, local_id)`, isolated by this owning instance.
A restart requires a new instance and run ID. Global identity is not inferred.

`update(frame_index, timestamp, observations)` accepts the existing `ReIDBatch`.
Frame indices and exact `Fraction` scene timestamps must increase. All keys and
timestamps must belong to the current round. Unit-length, finite float32
vectors and unique camera/local-ID pairs are required. Invalid inputs are
rejected before state changes. Stored vectors and output arrays have independent
ownership, so modifying caller arrays does not corrupt history.

Call update for every round, including empty batches. Samples are removed when
their age is strictly greater than the configured maximum; the exact boundary
is retained. An empty track history is removed. Absent tracks may remain in
memory temporarily but are not emitted as visible observations. Slow processing
does not age samples: only elapsed scene time does.

The result contains latest and mean `ReIDBatch` objects, source frame/time lists,
mean norms before normalization and fallback indicators. Row order follows the
input observations. No GT, quality selection, ID merging or switch correction
is used by this module. It retains only feature vectors and small metadata,
never full images or person crops.

## Commands

From the project root with the project's `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_appearance_history.py
PYTHONPATH="$PWD/src" python scripts/preview_appearance_history.py \
  --run-report artifacts/local_reid/20261006T204237314810Z/report.json
```

The known-answer check covers averaging, memory ownership, bounded sample count,
camera isolation, scene-time expiry, empty rounds, run restarts, invalid update
atomicity, cancellation and preservation of already-produced outputs.

The replay script verifies input hashes, trace chronology and coverage of every
embedding row. It feeds only current observations to history in chronological
order. It checks that latest descriptors remain bitwise equal to the source,
mean descriptors are normalized, and source samples respect count/age/causality
limits. The 300-round source should produce 15,914 descriptors for each variant.

## Artifacts

A new `artifacts/appearance_history/<UTC>/` directory contains:

- `mean_embeddings.npy`: one mean descriptor per source embedding row. Row i
  still describes the current observation associated with source row i.
- `observations.jsonl`: current key, source embedding rows/frames, sample count,
  time span, fallback status and latest-to-mean cosine for each observation.
- `report.json`: source checksums, configuration, checks, history size counts,
  vector cache peak, CPU update timings and selected diagnostic examples.

The latest variant is the original source `embeddings.npy`; it is referenced
by path/hash rather than duplicated. The module's in-memory state is limited
per track and by age. The offline preview also holds the trace and row lookup
for provenance and writes a full descriptor matrix; this is separate from the
runtime history's memory requirements. Vector payload size excludes Python
object overhead and temporary arrays. CPU timing here covers history.update,
not the full detection/tracking pipeline, and is only an initial diagnostic.

## Interpretation and next evaluation

A high cosine between the latest vector and the mean indicates little descriptor
change. It does not establish identity correctness or retrieval improvement.
Average descriptors can mix different people after a local ID switch. The
existing camera-8 ID 10/12 swap is therefore retained for analysis, not removed
to improve a score. Selected examples around that event are recorded without
using GT to build descriptors or tune the policy.

Next, evaluate latest and mean descriptors on the same eligible query/gallery
observations, using GT only for evaluation. Report failures as well as pooled
metrics, especially around local identity switches. Neither variant is assumed
better before measurement. No global ID or cosine acceptance threshold is
chosen in this step.

## Scientific context

[Wojke, Bewley and Paulus, Simple Online and Realtime Tracking with a Deep
Association Metric, 2017](https://arxiv.org/abs/1703.07402) uses learned
appearance descriptors for association. Its
[official nearest-neighbor implementation](https://github.com/nwojke/deep_sort/blob/master/deep_sort/nn_matching.py)
keeps samples per target, optionally limits the sample budget and compares
against the nearest sample. That supports considering a bounded appearance
history; it does not establish that an eight-observation arithmetic mean or a
one-second age limit is optimal. Our mean and expiry policy are separate,
explicit project experiments. A gallery-based variant remains a later comparison.
