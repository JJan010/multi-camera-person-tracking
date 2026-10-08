# Scene 041 camera 364: visual review of missed observations

Source diagnostic: `20261007T220413363583Z`, staged, frames 2..3599.
All six supplied JPEGs were reviewed and their source artifact hashes checked.
The frozen diagnostic report remains unchanged, including its pending-review flag.

## Reproduced counts

Local GT observations 32361, predictions 30459, FP 1086, FN 2988,
IDSW 64, IDF1 61.01%, precision 96.43%, recall 90.77%.

| MISS description | Count | Share |
|---|---:|---:|
| No admissible cached detection | 1595 | 53.38% |
| Admissible detection, none emitted | 1316 | 44.04% |
| Admissible prediction, GT unmatched | 77 | 2.58% |

For the 1316 cases with available but un-emitted detections, the maximum
score among admissible candidates was below 0.5 in 1150 cases, between
0.5 inclusive and 0.6 exclusive in 109 cases, and at least 0.6 in 57 cases.
Thus 87.39% of that category had only weak-stage candidates.
This does not by itself prove the reason for the missed tracking update.

Median clipped GT width/height is 27/76 pixels for the no-candidate category
and 39/65 pixels for available-but-un-emitted detections. These are dimensions
in the original 1920x1080 image, not network input or Re-ID resize dimensions.
Only 23/2988 missed boxes cross the image boundary. Being fully inside the
image does not establish that a person is unoccluded.

## Visual observations

- Frame 2303, GT13 (27x56): a tiny GT-marked fragment behind shelving;
  the crop alone is insufficient to recognize a full person reliably.
- Frame 345, GT18 (24x142): a narrow fragment mostly hidden by shelving.
- Frame 268, GT24 (39x65): a visible head/upper-body fragment behind shelving.
  Candidate IoU 0.8397, score 0.1858, no admissible emitted track.
- Frame 74, GT21 (40x36): head/shoulders above shelving. Candidate IoU 0.8727,
  score 0.4632, no admissible emitted track.
- Frame 3038, GT22 and frame 190, GT17: overlapping people/GT regions.
  An emitted observation is spatially admissible, but the target is unmatched
  in the temporal evaluator. The specific competing assignment is not
  established from the image alone.

The examples intentionally emphasize frequent failure cases; they do not
measure the prevalence of occlusion among all 2988 misses.
In no-candidate examples with best IoU zero, the displayed yellow box is
an arbitrary tied maximum elsewhere in the frame. Its high confidence is
not confidence in the missed person.

## Combined validation conclusion

Camera 362 exposed repeated background detections of pallet/shelving, plus
one clearly visible person absent from GT in a reviewed frame. Camera 364
exposes substantial occlusion and small visible fragments, often with
low-confidence candidates. A universal confidence increase could trade
background FP for additional person FN. A decrease could create more false
tracks. Neither direction is validated by this review.

Close this bounded visual audit, retain staged and its original full-scene
metrics, and use these findings to motivate hypotheses tested on training
scenes. Do not change validation annotations or mask image regions after
observing the metrics. This validation scene is now inspected development
feedback; future final performance claims need additional untouched data.

The next quality experiment should target persistent identity continuity,
with one isolated change and both local/global metrics. Crop-quality-aware
appearance updates are a candidate hypothesis, not an approved runtime
replacement. Any detector fine-tuning is a separate training-data experiment.
Faster inference through TensorRT is a separate performance task and does
not directly resolve these identity or annotation issues.

Reference: Zhang et al., ByteTrack, ECCV 2022,
https://arxiv.org/abs/2110.06864. Weak detections can help recover occluded
objects; the above category counts and visual conclusions are project findings.
