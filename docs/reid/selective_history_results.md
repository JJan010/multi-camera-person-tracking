# Selective global appearance history: development result

Source run: `20261008T082757689939Z`.
Source report: `artifacts/selective_history/20261008T082757689939Z/report.json`.
Scene: scene_001, cameras 4/5/8, runtime frames 0..3599.
Evaluation: frames 2..3599; staged local tracks frozen for both variants.

This note transcribes the reported WSL console result. The full report should
be copied alongside it as `selective_history_metrics.json`; this note does not
substitute for the report's checksums, configuration or observation-level traces.

## Result

| Interval | All updates IDF1 | Strong updates IDF1 | Change in IDTP |
|---|---:|---:|---:|
| Full: 2..3599 | 53.03% | 53.03% | +1 |
| First: 2..1799 | 73.57% | 73.57% | +2 |
| Second: 1800..3599 | 60.64% | 60.64% | +1 |

Full-run counts:

| Variant | IDTP | IDFP | IDFN |
|---|---:|---:|---:|
| all_updates | 96696 | 83765 | 87511 |
| strong_updates | 96697 | 83764 | 87510 |

Both variants have 184207 GT observations and 180461 prediction observations
in the evaluation interval. IDF1 is `2 * IDTP / (GT + predictions)`.
The full-run difference is approximately **+0.000548 percentage points**.
This is a negligible measured change, not evidence of a useful quality gain.
Window identity assignments are optimized separately; their IDTP changes need
not sum to the full-run change. Runtime state was continuous across the windows.

## History and lifecycle

Across all runtime frames, both variants preserve 180547 local observations.
The strong-update rule accepted 175438 samples and rejected 5109 (2.830%).
Of rejected observations, 4618 reused unexpired accepted history and 491 had no
descriptor (0.272% of runtime observations). Those observations were retained
in the global output. All-updates provided descriptors for all 180547 observations.

Lifecycle totals are equal: 267 allocated/ever-emitted IDs, 155 absorbed IDs,
84 expired IDs, 151 merge events and 28 retained IDs at the end. Equal totals
do not imply identical assignment traces or identical merge events.

Visible-member merge diagnostics changed from 148 same-GT / 3 unresolved to
149 same-GT / 2 unresolved. This aggregate change does not establish that a
specific wrong merge was corrected. Neither branch reports a known mixed-GT
accepted merge in this diagnostic; the diagnostic does not certify the entire
history of each merged identity.

## Verification and decision

The run reports exact reproduction of all-updates global records throughout
3600 rounds and verifies full/window identity metrics against motmetrics.
Local tracks and the evaluation denominators remain fixed. GT is evaluated
after the predictions are frozen.

**Decision: keep all_updates as the reference global-history policy.** Retain
the selective implementation as a tested experiment, without promoting it to
the runtime baseline or sweeping the threshold on this sequence. The result
does not show that every crop-selection method is ineffective; it tests only
the score >= 0.5 update rule, eight accepted samples and one-second sample age.

Every embedding was already computed. Rejecting a history update in this replay
does not demonstrate fewer OSNet calls, lower GPU load or faster end-to-end
processing. Such a scheduling optimization needs a separate runtime measurement.

## Next bounded question: identity continuity across track loss

The current identity manager retains local-to-global bindings for the configured
idle interval. After all bindings expire it removes the identity from active
state; it has no separate appearance archive for reactivating a departed person.
In addition, a visible local ID continues its existing global binding even if
the local tracker has changed the person occupying that ID. Selective global
history alone repairs neither mechanism.

The next architectural experiment should separate a local track's lifetime from
the lifetime of the person's identity. A bounded archive of inactive identities
would allow a new local track to request a previous global ID using appearance
and scene-time/spatial constraints, with an explicit unmatched outcome and
one-to-one conflict resolution. It must not blindly extend same-camera slot
reservation or rewrite previously emitted outputs. It also cannot be described
as a fix for active local-ID switches.

This is a proposed causal policy to test, not an implemented improvement or a
conclusion that expiry explains all current errors. Compare against the unchanged
staged/all-updates reference on frozen observations, then assess transfer under
a frozen policy. Scene_041 has already been inspected; an untouched final test
must be kept separate from future development choices.
