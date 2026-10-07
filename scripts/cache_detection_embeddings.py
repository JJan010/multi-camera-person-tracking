"""Cache OSNet features of frozen detector candidates before local association."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mtmc.reid.crops import crop_geometry
from mtmc.reid.osnet import ObservationKey, PersonCrop, sha256
from run_mtmc import EmbeddingArchive

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare_candidates(local, batch, *, source_run, row_offset):
    """The encoder's local_id field temporarily carries a per-frame detection index.

    Persisted records call this field detection_index. It is never a track ID.
    Keep all saved candidates, including duplicate boxes and fully outside boxes.
    """
    frame = batch.frame_index
    require(local['run_id'] == source_run and local['frame_index'] == frame
            and Fraction(local['timestamp']) == batch.timestamp == Fraction(frame, 30),
            'Frozen detection/video scope or time differs')
    images = {p.camera_id: p for p in batch.frames}
    cameras = {c['camera']: c for c in local['cameras']}
    require(len(images) == len(batch.frames) == len(cameras) == len(local['cameras']) == 3
            and set(images) == set(cameras) == {4, 5, 8}, 'Camera coverage differs')
    crops, records = [], []
    for camera in sorted(cameras):
        packet, saved = images[camera], cameras[camera]
        require(packet.frame_index == frame and packet.timestamp == batch.timestamp
                and packet.rgb.dtype == np.uint8 and packet.rgb.shape == (1080, 1920, 3),
                'Invalid frame packet')
        boxes = np.asarray(saved['detector_xyxy'], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(saved['detector_confidence'], dtype=np.float64)
        require(scores.shape == (len(boxes),) and np.isfinite(boxes).all()
                and np.all(boxes[:, 2:] > boxes[:, :2]) and np.isfinite(scores).all()
                and np.all((scores >= 0) & (scores <= 1)), 'Invalid saved detections')
        for index, (box, score) in enumerate(zip(boxes, scores)):
            bounds, fraction = crop_geometry(box, 1920, 1080)
            row = row_offset + len(crops) if bounds is not None else None
            records.append({'camera_id': camera, 'detection_index': index, 'frame_index': frame,
                            'xyxy': box.tolist(), 'confidence': float(score),
                            'crop_xyxy_int': bounds, 'inside_image_fraction': fraction,
                            'embedding_row': row})
            if bounds is not None:
                x1, y1, x2, y2 = bounds
                rgb = packet.rgb[y1:y2, x1:x2].view()
                rgb.setflags(write=False)
                crops.append(PersonCrop(ObservationKey(camera, index, frame), batch.timestamp, rgb))
    return tuple(crops), records


def encode_chunks(encoder, crops, batch_size):
    require(type(batch_size) is int and batch_size > 0, 'Invalid batch size')
    chunks = []
    for start in range(0, len(crops), batch_size):
        selected = crops[start:start + batch_size]
        result = encoder.encode(selected)
        require(result.keys == tuple(c.key for c in selected)
                and result.timestamps == tuple(c.timestamp for c in selected), 'Encoder mapping changed')
        a = result.embeddings
        require(a.shape == (len(selected), 512) and a.dtype == np.float32 and np.isfinite(a).all()
                and np.allclose(np.linalg.norm(a, axis=1), 1, rtol=0, atol=1e-5),
                'Invalid normalized embeddings')
        chunks.append(a)
    return np.concatenate(chunks) if chunks else np.empty((0, 512), dtype=np.float32)


def parity_sample(local, records, features, row_offset, reference):
    """Only compare identical RGB crop bounds; do not infer detection-to-track assignments."""
    lookup = {}
    for item in records:
        if item['embedding_row'] is not None:
            lookup.setdefault((item['camera_id'], tuple(item['crop_xyxy_int'])),
                              item['embedding_row'] - row_offset)
    results = []
    for item in local['reid_observations']:
        if item['embedding_row'] is None:
            continue
        key = (item['camera'], tuple(item['crop_xyxy_int']))
        if key not in lookup:
            continue
        index = item['embedding_row']
        require(type(index) is int and 0 <= index < len(reference), 'Invalid reference row')
        a = features[lookup[key]].astype(np.float64)
        b = reference[index].astype(np.float64)
        require(np.isfinite(b).all() and abs(np.linalg.norm(b) - 1) < 1e-4, 'Invalid reference vector')
        similarity = float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
        error = float(np.max(np.abs(a - b)))
        require(similarity >= .9999, 'Identical crop failed reference cosine parity >= 0.9999')
        results.append({'camera_id': item['camera'], 'frame_index': local['frame_index'],
                        'reference_local_id': item['local_id'], 'reference_embedding_row': index,
                        'cache_embedding_row': row_offset + lookup[key],
                        'cosine': similarity, 'max_absolute_error': error})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-report', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--torch-threads', type=int, default=1)
    args = parser.parse_args()
    require(args.batch_size > 0 and args.torch_threads > 0, 'Batch size and threads must be positive')
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve()
        actual = sha256(path)
        require(expected is None or actual == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': actual}
        return path

    print('Verifying frozen detections, videos and OSNet configuration...', flush=True)
    source_path = checked('pipeline_report', args.run_report)
    source = json.loads(source_path.read_text())
    cfg, rounds = source['configuration'], source['summary']['rounds']
    require(source.get('completed') is True and source['protocol'] == 'scene_001_mtmc_sequential_fp32_v1'
            and cfg['fps'] == 30 and cfg['cameras'] == [4, 5, 8]
            and type(rounds) is int and 0 < rounds <= 23994, 'Unsupported source run')
    for name in ('tracks', 'embeddings'):
        spec = source['artifacts'][name]
        checked(name, source_path.parent / spec['path'], spec['sha256'])
    reference = np.load(inputs['embeddings']['path'], mmap_mode='r', allow_pickle=False)
    require(reference.shape == (source['summary']['total_embeddings'], 512)
            and reference.dtype == np.float32, 'Reference feature shape/dtype differs')
    for name in ('video_manifest', 'osnet_configuration'):
        spec = source['inputs'][name]
        checked(name, spec['path'], spec['sha256'])
    manifest = json.loads(Path(inputs['video_manifest']['path']).read_text())
    require(manifest['dataset'] == 'nvidia/PhysicalAI-SmartSpaces'
            and manifest['revision'] == '2cbe9563cbe9f47f846e5c871ee994572bbbc60e'
            and manifest['scene'] == 'MTMC_Tracking_2024/train/scene_001'
            and sorted(x['camera'] for x in manifest['files']) == [4, 5, 8], 'Different video source')
    sources = {}
    for spec in manifest['files']:
        camera = spec['camera']
        require(spec['sha256'] == source['inputs'][f'video_{camera}']['sha256'], 'Source video differs')
        path = checked(f'video_{camera}', ROOT / spec['local_path'], spec['sha256'])
        require(path.stat().st_size == spec['size_bytes'], 'Video size differs')
        sources[camera] = path
    osnet_config = Path(inputs['osnet_configuration']['path'])
    for name, spec in json.loads(osnet_config.read_text())['assets'].items():
        checked(f'osnet_asset_{name}', ROOT / spec['path'], spec['sha256'])

    import torch
    from mtmc.reid.osnet import OSNetEncoder
    from mtmc.video.replay import synchronized_replay
    torch.set_num_threads(args.torch_threads)
    print('Loading OSNet CUDA FP32; RF-DETR and trackers will not run...', flush=True)
    encoder = OSNetEncoder(osnet_config, project_root=ROOT)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts/detection_embeddings' / run
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run_status.json').write_text(json.dumps({'completed': False, 'run_id': run}) + '\n')
    archive = EmbeddingArchive(output / 'embeddings.npy')
    counts, samples = Counter(), []
    sample_frames = {0, rounds // 4, rounds // 2, rounds - 1}
    started = perf_counter()
    try:
        with synchronized_replay(sources, fps=30, threads=1) as batches, \
                Path(inputs['tracks']['path']).open() as trace, (output / 'detections.jsonl').open('w') as saved:
            for frame in range(rounds):
                line = trace.readline()
                require(bool(line), 'Truncated source trace')
                local, batch = json.loads(line), next(batches)
                require(batch.frame_index == frame, 'Replay frame order differs')
                offset = archive.rows
                crops, records = prepare_candidates(local, batch, source_run=source['run_id'], row_offset=offset)
                features = encode_chunks(encoder, crops, args.batch_size)
                if frame in sample_frames:
                    samples.extend(parity_sample(local, records, features, offset, reference))
                archive.append(features)
                saved.write(json.dumps({'cache_run_id': run, 'source_run_id': source['run_id'],
                            'frame_index': frame, 'timestamp': str(batch.timestamp),
                            'detections': records}, allow_nan=False) + '\n')
                counts['detections'] += len(records)
                counts['encoded'] += len(crops)
                counts['fully_outside'] += len(records) - len(crops)
                for item in records:
                    counts[f"camera_{item['camera_id']}"] += 1
                if (frame + 1) % 60 == 0 or frame == rounds - 1:
                    print(f'Processed {frame + 1}/{rounds}; candidate embeddings={archive.rows}', flush=True)
                del batch, crops, features, records, local
            require(trace.readline() == '', 'Trailing source trace rows')
        archive.finalize()
    finally:
        archive.close()
    elapsed = perf_counter() - started
    require(counts['encoded'] == archive.rows
            and counts['detections'] == counts['encoded'] + counts['fully_outside'], 'Count mismatch')
    require(bool(samples), 'No identical-crop reference samples available')
    for item in inputs.values():
        require(sha256(Path(item['path'])) == item['sha256'], 'Input changed while caching')
    parity = {'samples': len(samples), 'minimum_cosine': min(x['cosine'] for x in samples),
              'max_absolute_error': max(x['max_absolute_error'] for x in samples)}
    (output / 'reference_parity.json').write_text(json.dumps(samples, indent=2, allow_nan=False) + '\n')
    code = [ROOT / 'scripts/cache_detection_embeddings.py', ROOT / 'src/mtmc/reid/osnet.py', ROOT / 'src/mtmc/reid/crops.py',
            ROOT / 'src/mtmc/video/replay.py', ROOT / 'src/mtmc/video/reader.py', ROOT / 'scripts/run_mtmc.py']
    report = {'completed': True, 'protocol': 'frozen_detector_candidate_embeddings_v1',
              'run_id': run, 'source_run_id': source['run_id'], 'inputs': inputs,
              'configuration': {'frames': [0, rounds - 1], 'rounds': rounds, 'cameras': [4, 5, 8],
                                'fps': 30, 'batch_size': args.batch_size, 'torch_threads': args.torch_threads,
                                'dtype': 'float32', 'feature_dim': 512, 'ground_truth_used': False,
                                'candidate_key': ['source_run_id', 'frame_index', 'camera_id', 'detection_index'],
                                'sampling': 'All saved detector candidates; original within-camera order'},
              'summary': dict(counts), 'reference_parity': parity, 'elapsed_s': elapsed,
              'timing_note': 'Cache construction includes decode, inference, parity and writes; not a tracker benchmark',
              'versions': {name: version(name) for name in ('numpy', 'torch', 'torchvision', 'av', 'pillow')},
              'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code},
              'artifacts': {name: {'path': name, 'sha256': sha256(output / name)}
                            for name in ('detections.jsonl', 'embeddings.npy', 'reference_parity.json')},
              'limits': ['Detection index is frame-local, not a persistent local track ID.',
                         'No candidate selection by GT or future observations.',
                         'Available candidates are limited by the original saved detector threshold.',
                         'No crop-quality gate; fully outside boxes retain a null embedding row.',
                         'Encoder batches differ from the baseline; sampled parity allows small numerical changes.',
                         'No local/global tracking outcomes changed or evaluated by this step.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'run_status.json').write_text(json.dumps({'completed': True, 'run_id': run}) + '\n')
    print('Candidate counts:', json.dumps(dict(counts)))
    print('Identical-crop reference parity:', json.dumps(parity))
    print(f'Report: {output / "report.json"}')
    print('Frozen detection embedding cache: COMPLETED; tracker experiments can reuse it')


if __name__ == '__main__':
    main()
