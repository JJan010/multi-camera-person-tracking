# Multi-camera grouping results

Source run: 20261007T084841755095Z.
Scene: scene_001; cameras: 4, 5, 8; frames: 2 through 299.
Policy: complete_support_greedy_v1.

The frozen source counts were reproduced. All 15,828 observations
were preserved for each variant/threshold setting. Retained links
were a subset of the original pairwise links.

Grouping increased labeled pair precision for most tested settings,
while removing both correct and incorrect links. Recall never increased.

For mean descriptors at threshold 0.70:
- Precision: 96.66% -> 97.97%.
- Conditional pair recall: 87.33% -> 84.28%.
- Removed links: 367 correct, 153 wrong-known, 17 unresolved.

For mean descriptors at threshold 0.80:
- Precision: 98.76% -> 99.51%.
- Conditional pair recall: 57.39% -> 47.71%.
- Removed links: 1164 correct, 59 wrong-known, 7 unresolved.
- Retained links: 5740 correct, 28 wrong-known, 15 unresolved.

Complete-support grouping can split a true three-camera person
when one pairwise link is missing. Camera uniqueness is a structural
invariant, not proof of correct identity.

Singleton counts are summed over frames, not unique people.
Recall is conditional on available GT-assigned observation pairs.
These are frame-local association diagnostics, not global IDF1.

No deployment threshold was selected. This short reused training-scene
clip is an integration diagnostic, not independent validation.
The next step is persistent global identity management across time.
