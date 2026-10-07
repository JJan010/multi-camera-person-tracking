# Global identity failure analysis results

Source diagnostic: 20261007T100559052947Z.
Source identity replay: 20261007T093510175809Z.
Setting: mean descriptors, threshold 0.70, idle duration 1 second.
Scene_001, cameras 4/5/8, frames 2..299.
This is an explicitly selected integration case, not deployment calibration.

## Verified error accounting

Global IDF1 was reproduced at 72.7329%, with IDTP=11690, IDFP=4138 and IDFN=4627.
Independent maximum-cardinality spatial matching supports 15666 observations,
yielding a fixed-box framewise F1 ceiling of 97.4708%.

The exact decomposition is:

- IDFN 4627 = fixed-box minimum 651 + shared-identity assignment gap 3976.
- IDFP 4138 = fixed-box minimum 162 + shared-identity assignment gap 3976.

The 3976 term appears in both equations because each additional credited match
reduces both counters by one. It must not be presented as 7952 independently
recoverable observations. The ceiling is 24.7379 percentage points above the
current IDF1, but is not a promised achievable online identity score.

A lack of any spatial candidate accounts for 606 GT observations and 136
predictions. These are smaller than the respective spatial matching minima:
matching competition also matters. The identity assignment gap can contain
local-track errors and duplicate-box competition as well as global association
errors. Aggregate decomposition alone does not allocate blame among modules.

## Fragmentation and mixing evidence

Mutually unique overlaps provided 15493 supported pairs. All 25 GT identities
and all 50 predicted global IDs had some such evidence.

- 18 GT identities had support under multiple global IDs.
- 7 global IDs had support for multiple GT identities.
- 8 camera/local-track keys had support for multiple GT identities over time.

GT 21 is the largest displayed fragmentation example: IDs 22, 25, 29 and 34
have 63, 250, 273 and 184 supporting observations respectively. The total is 770;
497 observations lie outside the largest bucket. Multiple global IDs support
this GT identity simultaneously in 260 frames.

GT 14 has supporting global IDs 14, 30 and 48. Its conflict example at frame 30
is especially useful: camera 4/local 12 has global ID 30, while camera 5/local 10
and camera 8/local 11 have global ID 14. All three have unique overlap evidence
for GT 14. The same pattern is recorded at frames 31..34. The no-merge policy
explicitly prevents these two established IDs from unifying.

The displayed global-mixing examples include:

- Global 20: GT 19 has 378 supporting observations and GT 10 has 87.
- Global 23: GT 10 has 200 supporting observations and GT 19 has 87.
- Global 21: GT 18 has 206 supporting observations and GT 9 has 82.

All five displayed top mixing IDs have zero frames with simultaneously supported
different GT identities. This is consistent with identity contamination over time,
but does not prove a specific ByteTrack switch or rule out global association
errors. The detailed local-track support and timeline are needed for attribution.

## Existing-ID conflicts

Of 2333 rejected multiple-existing-ID group decisions:

| Diagnostic category | Decisions | Share of all decisions |
| --- | ---: | ---: |
| All members support the same GT | 2077 | 89.03% |
| Different known GT identities | 171 | 7.33% |
| Unresolved | 85 | 3.64% |

These are repeated group/frame events, not independent examples, unique people,
or an estimated probability that an arbitrary merge is correct.

The report demonstrates both unnecessary rejection of true links and useful
rejection of false links. At frame 26, global IDs 10 and 25 are proposed together,
but the camera 5/local 6 observation supports GT 1 and camera 8/local 16 supports
GT 21. Blindly merging all existing-ID conflicts would accept this wrong link.

Temporal repetition alone is also insufficient: at frames 43, 44, 45 and 46,
global IDs 10 and 29 are repeatedly proposed together while supporting GT 1 and
GT 21 respectively. A rule accepting any three consecutive proposals would not
reject this example on persistence grounds alone.

## Next implementation boundary

Keep local_anchor_no_merge_v1 and its reports intact as the frozen baseline.
The next candidate policy should be a separate causal variant with explicit,
synthetic-testable rules. Proposed safeguards to specify and evaluate:

1. Accumulate evidence in scene time, counting at most one observation round
   per candidate rather than counting each camera edge as independent evidence.
2. Check compatible retained camera/local-track bindings across entire identities,
   not only the members of the triggering pair or group.
3. Check support from other currently visible members; missing support and
   contradictory evidence must have explicit effects on pending candidates.
4. Use deterministic decisions for competing candidates and revalidate state
   before each accepted merge to prevent transitive camera conflicts.
5. Record merge provenance and canonical-ID changes only from acceptance time
   onward; never rewrite earlier emitted assignments to improve measured IDF1.
6. Re-evaluate both incorrect merges and fragmentation with identical GT/box
   denominators. Use separate data later for parameter calibration.

These are design requirements for the next experiment, not validated guarantees.
They do not repair local-track identity switches by themselves. GT labels remain
strictly evaluation-only and must not enter merge decisions.
