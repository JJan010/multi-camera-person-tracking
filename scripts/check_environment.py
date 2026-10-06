"""Check the Python environment and basic CUDA operations."""

import sys

import torch
import torchvision
from torchvision.ops import nms


def main():
    print("Python:", sys.version.split()[0])
    print("Interpreter:", sys.executable)
    print("Virtual environment:", sys.prefix != sys.base_prefix)
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(device))
    print("Compute capability:", torch.cuda.get_device_capability(device))

    with torch.inference_mode():
        matrix = torch.ones((512, 512), device=device)
        result = matrix @ matrix
        expected = torch.full_like(result, 512.0)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        print("Matrix multiplication: OK; device:", result.device)

        boxes = torch.tensor(
            [[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
            dtype=torch.float32,
            device=device,
        )
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)
        kept = nms(boxes, scores, iou_threshold=0.5)

        if kept.tolist() != [0, 2]:
            raise RuntimeError(f"Unexpected NMS result: {kept.tolist()}")

        torch.cuda.synchronize()
        print("torchvision CUDA operator: OK")

    print("Environment smoke test: PASSED")


if __name__ == "__main__":
    main()
