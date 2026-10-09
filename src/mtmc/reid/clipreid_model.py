"""Pinned CLIP-ReID visual inference, separate from the 512-D OSNet contracts."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ClipReIDVisual(nn.Module):
    """Official pre-BN image CLS(768) + projected CLS(512), then joint L2 norm."""
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def raw_features(self, x):
        require(x.ndim == 4 and tuple(x.shape[1:]) == (3, 256, 128), 'Expected NCHW RGB 256x128')
        require(x.shape[0] > 0 and x.dtype == torch.float32, 'Expected a nonempty FP32 batch')
        require(x.device == next(self.visual.parameters()).device, 'Input/model device mismatch')
        _, last, projected = self.visual(x)
        # Same TEST.NECK_FEAT='before' expression as upstream build_transformer.forward.
        return torch.cat((last[:, 0], projected[:, 0]), dim=1)

    def forward(self, x):
        return torch.nn.functional.normalize(self.raw_features(x), p=2, dim=1)


def load_clipreid_visual(config_path, *, project_root, device='cuda:0'):
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    require(config['architecture'] == 'ViT-B-16' and config['feature_dim'] == 1280
            and config['input_size_hw'] == [256, 128]
            and config['neck_feat'] == 'before' and config['stride_size'] == [16, 16]
            and config['sie_camera'] is False and config['sie_view'] is False,
            'Unsupported model contract')
    require(config['normalization'] == {'mean': [.5, .5, .5], 'std': [.5, .5, .5]}, 'Unsupported normalization')
    assets = {}
    for name, item in config['assets'].items():
        path = (root / item['path']).resolve()
        require(root in path.parents, 'Asset outside project')
        require(path.is_file() and path.stat().st_size == item['bytes']
                and sha256(path) == item['sha256'], f'Missing/changed asset: {name}')
        assets[name] = path
    require(torch.get_default_dtype() == torch.float32, 'Default dtype must be FP32')
    # Import only the verified, unmodified architecture file, not training/tokenizer code.
    spec = importlib.util.spec_from_file_location('_mtmc_pinned_clipreid_visual', assets['visual_source'])
    require(spec is not None and spec.loader is not None, 'Cannot load visual architecture')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    state = torch.load(assets['weights'], map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and len(state) == 315, 'Unexpected checkpoint structure')
    require(all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()), 'Expected tensor state dict')
    prefixes = dict(Counter(k.split('.')[0] for k in state))
    require(prefixes == {'classifier': 1, 'classifier_proj': 1, 'bottleneck': 5,
        'bottleneck_proj': 5, 'image_encoder': 152, 'prompt_learner': 3, 'text_encoder': 148},
        'Unexpected checkpoint branches (including possible camera embeddings)')
    for key, shape in {
        'classifier.weight': (1041, 768), 'classifier_proj.weight': (1041, 512),
        'image_encoder.positional_embedding': (129, 768),
        'image_encoder.conv1.weight': (768, 3, 16, 16), 'image_encoder.proj': (768, 512)
    }.items():
        require(tuple(state[key].shape) == shape, f'Unexpected tensor shape: {key}')
    visual = upstream.VisionTransformer(h_resolution=16, w_resolution=8,
        patch_size=16, stride_size=16, width=768, layers=12, heads=12, output_dim=512)
    weights = {k[len('image_encoder.'):]: v for k, v in state.items() if k.startswith('image_encoder.')}
    require(len(weights) == 152 and all(torch.isfinite(v).all().item() for v in weights.values()), 'Invalid visual tensors')
    visual.load_state_dict(weights, strict=True)
    model = ClipReIDVisual(visual).eval().to(device=device, dtype=torch.float32)
    info = {'checkpoint_tensor_count': len(state), 'loaded_visual_tensors': len(weights),
        'unused_inference_branches': {k: n for k, n in prefixes.items() if k != 'image_encoder'},
        'visual_parameters': sum(p.numel() for p in model.parameters()),
        'config_sha256': sha256(config_path), 'assets': config['assets'],
        'upstream_revision': config['upstream_revision'], 'strict_visual_load': True}
    return model, info


def preprocess_clipreid(rgb_images):
    """Upstream validation transform: PIL RGB bilinear resize, tensor, (x-.5)/.5."""
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    images = tuple(rgb_images)
    require(images, 'Expected at least one image')
    for image in images:
        require(isinstance(image, np.ndarray) and image.dtype == np.uint8
                and image.ndim == 3 and image.shape[2] == 3 and min(image.shape[:2]) > 0,
                'Expected nonempty HWC uint8 RGB images')
    transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
    return torch.stack([transform(Image.fromarray(np.ascontiguousarray(image))) for image in images])
