"""Strict checkpoint loading and pinned official inference parity, not Re-ID quality."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from torch import nn
from mtmc.reid.clipreid_model import load_clipreid_visual, preprocess_clipreid, require, sha256

ROOT = Path(__file__).resolve().parents[1]


def reference_forward(model, configuration):
    """Execute the exact verified upstream forward with its trained BN branches.

    The original constructor also initializes text training machinery and downloads
    base CLIP. Neither is used by this image-only evaluation branch.
    """
    source = ROOT / configuration['assets']['inference_reference']['path']
    require(sha256(source) == configuration['assets']['inference_reference']['sha256'], 'Changed forward reference')
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'build_transformer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    ref = nn.Module()
    ref.model_name = 'ViT-B-16'
    ref.neck_feat = 'before'
    ref.image_encoder = model.visual
    ref.bottleneck = nn.BatchNorm1d(768)
    ref.bottleneck_proj = nn.BatchNorm1d(512)
    state = torch.load(ROOT / configuration['assets']['weights']['path'], map_location='cpu', weights_only=True)
    for name in ('bottleneck', 'bottleneck_proj'):
        getattr(ref, name).load_state_dict({k[len(name)+1:]: v for k, v in state.items() if k.startswith(name+'.')}, strict=True)
    ref.to(device=next(model.parameters()).device, dtype=torch.float32).eval()
    ref.forward = MethodType(namespace['forward'], ref)
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='Default CUDA; CPU is for technical portability checks only')
    args = parser.parse_args()
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda'), 'Expected CPU or CUDA')
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    configuration_path = ROOT / 'configs/models/clipreid_vit_b16_msmt17.json'
    config = json.loads(configuration_path.read_text())
    print('Verifying official assets and loading visual weights with strict=True...', flush=True)
    model, info = load_clipreid_visual(configuration_path, project_root=ROOT, device=str(device))
    reference = reference_forward(model, config)
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'Not all model parameters FP32')
    rng = np.random.default_rng(20261008)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in ((91, 37), (256, 128), (143, 61))]
    x = preprocess_clipreid(images)
    contiguous_check = preprocess_clipreid([images[0][:, ::-1]])
    require(torch.equal(contiguous_check, preprocess_clipreid([images[0][:, ::-1].copy()])), 'Non-contiguous RGB preprocessing differs')
    constant = np.zeros((31, 19, 3), dtype=np.uint8)
    constant[:] = [255, 0, 0]
    transformed = preprocess_clipreid([constant])
    require(torch.equal(transformed[:, 0], torch.ones_like(transformed[:, 0]))
        and torch.equal(transformed[:, 1:], -torch.ones_like(transformed[:, 1:])), 'RGB/normalization contract differs')
    x = x.to(device)
    with torch.inference_mode():
        raw = model.raw_features(x)
        expected_raw = reference(x)
        result = model(x)
        expected = torch.nn.functional.normalize(expected_raw, p=2, dim=1)
        permutation = model(x[[2, 0, 1]])
        single = model(x[:1])
    require(raw.shape == result.shape == (3, 1280), 'Unexpected feature dimension')
    require(torch.isfinite(raw).all().item() and (raw.norm(dim=1) > 0).all().item(), 'Invalid raw features')
    torch.testing.assert_close(raw, expected_raw, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.norm(dim=1), torch.ones(3, device=device), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(permutation, result[[2, 0, 1]], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(single, result[:1], atol=1e-4, rtol=1e-4)
    for image in (images[0].astype(np.float32), np.zeros((0, 20, 3), dtype=np.uint8)):
        try:
            preprocess_clipreid([image])
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed RGB image accepted')
    parameter = next(model.parameters())
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'artifacts/clipreid_model_checks' / run
    out.mkdir(parents=True, exist_ok=False)
    report = dict(completed=True, protocol='clipreid_visual_smoke_v1', run_id=run,
        torch_version=torch.__version__, device=str(parameter.device), dtype=str(parameter.dtype),
        gpu_name=torch.cuda.get_device_name(parameter.device) if device.type == 'cuda' else None,
        input_shape=list(x.shape), output_shape=list(result.shape),
        raw_reference_max_abs=float((raw-expected_raw).abs().max()),
        normalized_reference_max_abs=float((result-expected).abs().max()),
        batch_single_max_abs=float((single-result[:1]).abs().max()),
        normalized_norms=result.norm(dim=1).cpu().tolist(), model=info,
        code_checksums={p: sha256(ROOT/p) for p in ('scripts/check_clipreid_model.py','src/mtmc/reid/clipreid_model.py')},
        checks={'strict_visual_load': True, 'official_forward_parity': True, 'rgb_preprocessing': True,
            'single_and_permuted_batches': True, 'finite_normalized_features': True},
        limitations=['Synthetic pixels only: no person retrieval or global identity quality measured.',
            'No comparison with the authors original software environment or published benchmark metrics.',
            'No ONNX/TensorRT or end-to-end integration tested.'])
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Checkpoint tensors:', info['checkpoint_tensor_count'])
    print('Visual tensors loaded with strict=True:', info['loaded_visual_tensors'])
    print('Visual parameters:', info['visual_parameters'])
    print('Model device:', parameter.device)
    print('Model dtype:', parameter.dtype)
    print('Input shape:', tuple(x.shape))
    print('Output shape:', tuple(result.shape))
    print('Official forward parity: max_abs=', report['raw_reference_max_abs'])
    print('L2 norms:', report['normalized_norms'])
    print('RGB preprocessing, single batch and input permutation: PASSED')
    print('Report:', out/'report.json')
    print('CLIP-ReID visual smoke test: PASSED; retrieval quality not yet evaluated')


if __name__ == '__main__':
    main()
