"""Compare the complete shorter frozen run against the longer run's prefix."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value, scope):
    """Normalize only declared run metadata, never IDs, rows, scores or times."""
    if isinstance(value, list):
        return [normalize(v, scope) for v in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in scope:
                require(item == scope[key], f'Unexpected {key} in frozen record')
                result[key] = '<run-scope>'
            else:
                result[key] = normalize(item, scope)
        return result
    return value


def compare_jsonl(short, long, rounds, short_scope, long_scope):
    mismatches, first = 0, []
    with Path(short).open() as left, Path(long).open() as right:
        for frame in range(rounds):
            a, b = left.readline(), right.readline()
            require(bool(a) and bool(b), 'Truncated trace prefix')
            a, b = json.loads(a), json.loads(b)
            for record in (a, b):
                require(record['frame_index'] == frame and Fraction(record['timestamp']) == Fraction(frame, 30),
                        'Invalid frame/time order')
            if normalize(a, short_scope) != normalize(b, long_scope):
                mismatches += 1
                if len(first) < 5:
                    first.append(frame)
        require(left.readline() == '', 'Shorter trace has trailing frames')
    return {'compared_frames': rounds, 'mismatching_frames': mismatches, 'first_mismatches': first,
            'exact': mismatches == 0}


def compare_vectors(short, long, short_rows, long_rows):
    left = np.load(short, mmap_mode='r', allow_pickle=False)
    right = np.load(long, mmap_mode='r', allow_pickle=False)
    require(left.dtype == right.dtype == np.float32 and left.shape == (short_rows, 512)
            and right.shape == (long_rows, 512) and long_rows >= short_rows, 'Invalid feature arrays')
    changed, maximum = 0, 0.0
    for start in range(0, short_rows, 2048):
        a = left[start:start + 2048]
        b = right[start:start + len(a)]
        require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite feature')
        changed += int(np.count_nonzero(np.any(a != b, axis=1)))
        maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    return {'compared_rows': short_rows, 'changed_rows': changed, 'max_absolute_error': maximum,
            'exact': changed == 0}


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        records = [{'run_id': 'short', 'frame_index': i, 'timestamp': str(Fraction(i, 30)),
                    'nested': {'identity_scope': 'short', 'global_id': i + 1}, 'embedding_row': i}
                   for i in range(2)]
        extended = json.loads(json.dumps(records).replace('short', 'long'))
        extended.append({'run_id': 'long', 'frame_index': 2, 'timestamp': '1/15'})
        def save(name, rows):
            path = root / name
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            return path
        a, b = save('a.jsonl', records), save('b.jsonl', extended)
        scopes = ({'run_id': 'short', 'identity_scope': 'short'}, {'run_id': 'long', 'identity_scope': 'long'})
        assert compare_jsonl(a, b, 2, *scopes)['exact']
        extended[1]['nested']['global_id'] = 9
        save('b.jsonl', extended)
        assert compare_jsonl(a, b, 2, *scopes)['first_mismatches'] == [1]
        vectors = np.eye(2, 512, dtype=np.float32)
        np.save(root / 'a.npy', vectors)
        tail = np.concatenate([vectors, vectors[:1]])
        np.save(root / 'b.npy', tail)
        assert compare_vectors(root / 'a.npy', root / 'b.npy', 2, 3)['exact']
        tail[1, 1] = np.nextafter(tail[1, 1], np.float32(0))
        np.save(root / 'b.npy', tail)
        result = compare_vectors(root / 'a.npy', root / 'b.npy', 2, 3)
        assert result['changed_rows'] == 1 and result['max_absolute_error'] > 0
        try:
            normalize({'run_id': 'wrong'}, {'run_id': 'short'})
        except RuntimeError:
            pass
        else:
            raise AssertionError('Mixed run accepted')
    print('Scope normalization, changed identity and one-ULP feature difference checks: PASSED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-cache-report', type=Path)
    parser.add_argument('--long-cache-report', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.short_cache_report is None or args.long_cache_report is None:
        parser.error('Both cache report paths are required')
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': digest}
        return path

    def load(label, path):
        cache_path = checked(label + ':cache_report', path)
        cache = json.loads(cache_path.read_text())
        require(cache.get('completed') is True and cache['protocol'] == 'frozen_detector_candidate_embeddings_v1',
                'Expected completed candidate cache')
        spec = cache['inputs']['pipeline_report']
        source_path = checked(label + ':pipeline_report', spec['path'], spec['sha256'])
        source = json.loads(source_path.read_text())
        require(source.get('completed') is True and source['protocol'] == 'scene_001_mtmc_sequential_fp32_v1'
                and source['run_id'] == cache['source_run_id'], 'Expected matching completed pipeline')
        rounds = source['summary']['rounds']
        require(cache['configuration']['frames'] == [0, rounds - 1]
                and cache['configuration']['rounds'] == rounds, 'Cache/source duration differs')
        files = {}
        for name in ('tracks', 'global_tracks', 'decisions', 'embeddings', 'mean_embeddings'):
            spec = source['artifacts'][name]
            files[name] = checked(label + ':' + name, source_path.parent / spec['path'], spec['sha256'])
        require(cache['inputs']['tracks']['sha256'] == source['artifacts']['tracks']['sha256']
                and cache['inputs']['embeddings']['sha256'] == source['artifacts']['embeddings']['sha256'],
                'Cache references different source artifacts')
        for name in ('detections.jsonl', 'embeddings.npy'):
            spec = cache['artifacts'][name]
            files[name] = checked(label + ':cache:' + name, cache_path.parent / spec['path'], spec['sha256'])
        return cache, source, files

    print('Verifying frozen reports and artifact checksums...', flush=True)
    short_cache, short, a = load('short', args.short_cache_report)
    long_cache, long, b = load('long', args.long_cache_report)
    rounds = short['summary']['rounds']
    require(0 < rounds < long['summary']['rounds'], 'Expected a strictly longer second run')
    require(short['configuration'] == long['configuration'] and short['configuration']['fps'] == 30,
            'Pipeline configurations differ')
    config_a = {k: v for k, v in short_cache['configuration'].items() if k not in ('frames', 'rounds')}
    config_b = {k: v for k, v in long_cache['configuration'].items() if k not in ('frames', 'rounds')}
    require(config_a == config_b, 'Cache configurations differ')
    require({k: v['sha256'] for k, v in short['inputs'].items()} ==
            {k: v['sha256'] for k, v in long['inputs'].items()}, 'Recorded source asset provenance differs')
    for name, digest in short['code_sha256'].items():
        require(long['code_sha256'].get(name) == digest, f'Existing pipeline source changed: {name}')
    require(short_cache['code_sha256'] == long_cache['code_sha256'], 'Candidate encoder source changed')
    result = {}
    scopes = [{'run_id': s['run_id'], 'identity_scope': s['run_id']} for s in (short, long)]
    for name in ('tracks', 'global_tracks', 'decisions'):
        result[name] = compare_jsonl(a[name], b[name], rounds, *scopes)
        print(name + ': ' + json.dumps(result[name]), flush=True)
    cache_scopes = [{'cache_run_id': c['run_id'], 'source_run_id': s['run_id']}
                    for c, s in ((short_cache, short), (long_cache, long))]
    result['candidates'] = compare_jsonl(a['detections.jsonl'], b['detections.jsonl'], rounds, *cache_scopes)
    print('candidates: ' + json.dumps(result['candidates']), flush=True)
    for name in ('embeddings', 'mean_embeddings'):
        result[name] = compare_vectors(a[name], b[name], short['summary']['total_embeddings'], long['summary']['total_embeddings'])
        print(name + ': ' + json.dumps(result[name]), flush=True)
    result['candidate_embeddings'] = compare_vectors(a['embeddings.npy'], b['embeddings.npy'],
                                                     short_cache['summary']['encoded'], long_cache['summary']['encoded'])
    print('candidate_embeddings: ' + json.dumps(result['candidate_embeddings']), flush=True)
    for spec in inputs.values():
        require(sha256(spec['path']) == spec['sha256'], 'Input changed during comparison')
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts/temporal_prefix' / run
    output.mkdir(parents=True, exist_ok=False)
    passed = all(item['exact'] for item in result.values())
    report = {'completed': True, 'passed': passed, 'protocol': 'frozen_temporal_prefix_exact_v1',
              'run_id': run, 'inputs': inputs, 'prefix_frames': [0, rounds - 1], 'comparisons': result,
              'short_source_run_id': short['run_id'], 'long_source_run_id': long['run_id'],
              'script_sha256': sha256(Path(__file__)),
              'limits': ['Run scope strings are normalized; all other frozen JSON fields are compared exactly.',
                         'Feature equality is exact, not a cosine tolerance; changes require inspection.',
                         'This checks repeatability of the shared prefix, not accuracy or the unseen suffix.']}
    path = output / 'report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Report: {path}')
    print('Temporal prefix comparison: ' + ('PASSED' if passed else 'DIFFERENCES FOUND'))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
