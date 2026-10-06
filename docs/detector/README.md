# Initial RF-DETR integration

## Configuration

- RF-DETR Small, rfdetr 1.11.2.
- Pretrained weights; SHA-256 recorded in first_preview.json.
- PyTorch CUDA inference, FP32, TF32 disabled.
- Default model input resolution: 512 x 512.
- One batch containing three RGB images, ordered by camera ID.
- Confidence threshold: 0.5.
- No inference optimization, compilation or TensorRT.
- Model weights move from CPU to CUDA on the first predict() call.
- No OpenCV package installed; supervision reports its NumPy fallback.

## Diagnostic result

Scene: MTMC_Tracking_2024/train/scene_001.
Zero-based frame: 300. Cameras: 4, 5, 8.

Both GT and predicted boxes were clipped to the image boundaries.
Matching prioritized the number of valid pairs at IoU >= 0.5,
then their total IoU. Each box could participate in at most one pair.

| Camera | TP | FP | FN | Unmatched GT IDs |
|--------|----|----|----|------------------|
| 4      | 18 | 0  | 0  | none             |
| 5      | 21 | 0  | 1  | 12               |
| 8      | 19 | 0  | 2  | 0, 13            |

Aggregate precision: 100%.
Aggregate recall: 58/61, approximately 95.1%.

This is a one-frame integration diagnostic, not a full-scene,
held-out, COCO AP or official MTMC evaluation.
GT is used only by the diagnostic comparison script.

## Reproduction

Run from the repository root with the virtual environment active:

    PYTHONPATH="$PWD/src" python scripts/preview_detector.py
    python scripts/check_detector_gt.py

Generated images, predictions and model weights are stored in artifacts/.
The preview is not a latency benchmark.
