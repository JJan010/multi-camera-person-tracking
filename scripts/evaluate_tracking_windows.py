"""Evaluate disjoint time windows of frozen, continuously generated identities."""
import gzip
import json

import numpy as np

import evaluate_mtmc_sequence as sequence
from mtmc.reid.osnet import ObservationKey

VARIANTS = ('baseline', 'direct_iou', 'direct_appearance')
CAMERAS = (4, 5, 8)
require = sequence.require


def check_local_prefix(short_path, global_path, split, short_run, long_run):
    """Compare all local variant records, including accepted candidate provenance."""
    with gzip.open(short_path, 'rt') as short, gzip.open(global_path, 'rt') as long:
        for frame in range(split):
            a, b = short.readline(), long.readline()
            require(bool(a) and bool(b), 'Truncated local variant prefix')
            a, b = json.loads(a), json.loads(b)
            require(a['frame_index'] == b['frame_index'] == frame and a['run_id'] == short_run
                    and b['run_id'] == long_run, 'Local prefix scope differs')
            require(a['timestamp'] == b['timestamp'], 'Local prefix time differs')
            require(set(a['variants']) == set(b['variants']) == set(VARIANTS), 'Variant set differs')
            for name in VARIANTS:
                require(a['variants'][name] == b['variants'][name]['cameras'],
                        f'Frozen local prefix changed: variant={name}, frame={frame}')
        require(short.readline() == '', 'Short local experiment extends beyond split')


def evaluate_windows(path, ground, rounds, split):
    """Reset evaluation accumulators only. Never run or reset a tracker here."""
    import motmetrics as mm
    from importlib.metadata import version
    import evaluate_local_tracking as local_metric

    require(type(split) is int and 2 < split < rounds, 'Invalid split frame')
    require(version('motmetrics') == '1.4.0', 'Expected motmetrics 1.4.0')
    mm.lap.default_solver = 'scipy'
    intervals = {'first': [2, split - 1], 'second': [split, rounds - 1]}
    names = ['num_frames', 'num_objects', 'num_predictions', 'idtp', 'idfp', 'idfn',
             'idf1', 'idp', 'idr', 'precision', 'recall', 'num_switches', 'num_false_positives', 'num_misses']
    output = {w: {'frames': bounds, 'variants': {}} for w, bounds in intervals.items()}
    for variant in VARIANTS:
        counters = {w: sequence.metric.IdentityCounts() for w in intervals}
        references = {w: [] for w in intervals}
        local = {w: {c: mm.MOTAccumulator(auto_id=False) for c in CAMERAS} for w in intervals}
        with gzip.open(path, 'rt') as stream:
            for frame in range(rounds):
                line = stream.readline(); require(bool(line), 'Truncated global window trace')
                record = json.loads(line)
                require(record['frame_index'] == frame, 'Global window frame order differs')
                if frame < 2:
                    continue
                window = 'first' if frame < split else 'second'
                data = record['variants'][variant]
                assigned = {ObservationKey(**a['key']): a['global_id'] for a in data['identity']['assignments']}
                require(len(assigned) == len(data['identity']['assignments']), 'Duplicate global key')
                require(sorted(c['camera'] for c in data['cameras']) == list(CAMERAS), 'Window camera coverage differs')
                seen = set()
                for camera in data['cameras']:
                    c = camera['camera']; ids = camera['local_ids']
                    keys = tuple(ObservationKey(c, i, frame) for i in ids)
                    require(len(keys) == len(set(keys)), 'Duplicate local ID')
                    seen.update(keys)
                    raw_boxes = np.asarray(camera['xyxy'], np.float64).reshape(-1, 4)
                    gt, mask, _, _, _ = sequence.spatial_slot(ground[frame, c], keys, raw_boxes)
                    global_ids = [assigned[k] for k in keys]
                    counters[window].update(gt, global_ids, mask)
                    references[window].append((gt, global_ids, mask))
                    gt_boxes = local_metric.boxes_array([ground[frame, c][g] for g in gt])
                    pred_boxes = local_metric.boxes_array(camera['xyxy'])
                    distances = local_metric.distances(gt_boxes, pred_boxes)
                    require(np.array_equal(np.isfinite(distances), mask), 'Local/global spatial gates differ')
                    local[window][c].update(gt, ids, distances, frameid=frame)
                require(seen == set(assigned), 'Window local/global coverage differs')
            require(stream.readline() == '', 'Trailing global window records')
        for window, bounds in intervals.items():
            metrics, _ = counters[window].result()
            sequence.metric.check_reference(metrics, sequence.metric.reference_metrics(references[window]))
            table = mm.metrics.create().compute_many([local[window][c] for c in CAMERAS], metrics=names,
                names=[f'camera_{c:04d}' for c in CAMERAS], generate_overall=True)
            local_metrics = json.loads(table.to_json(orient='index'))
            total = local_metrics['OVERALL']
            require(total['num_objects'] == metrics['gt_observations']
                    and total['num_predictions'] == metrics['predicted_observations']
                    and total['num_frames'] == (bounds[1] - bounds[0] + 1) * len(CAMERAS),
                    'Window observation/frame denominators differ')
            output[window]['variants'][variant] = {'global': metrics, 'local': local_metrics}
        print(f'Window evaluation: {variant}; shared metrics/motmetrics: VERIFIED', flush=True)
    return output


def validate_windows(windows, full_global, full_local, short_global, short_local):
    """Check repeatability and additive counts, never subtract nonlinear ID scores."""
    for variant in VARIANTS:
        first = windows['first']['variants'][variant]
        second = windows['second']['variants'][variant]
        require(first['global'] == short_global[variant], f'First-minute global metrics changed: {variant}')
        for camera, metrics in first['local'].items():
            expected = short_local[variant][camera]
            require(set(metrics) == set(expected), 'Local metric schema differs')
            for name, actual in metrics.items():
                reference = expected[name]
                require(actual == reference or (actual is not None and reference is not None
                        and abs(actual - reference) <= 1e-9), f'First-minute local {name} changed: {variant}/{camera}')
        for name in ('camera_time_slots', 'gt_observations', 'predicted_observations'):
            require(first['global'][name] + second['global'][name] == full_global[variant][name],
                    f'Global interval count does not partition full run: {name}')
        for name in ('num_frames', 'num_objects', 'num_predictions'):
            require(first['local']['OVERALL'][name] + second['local']['OVERALL'][name] ==
                    full_local[variant]['OVERALL'][name], f'Local interval count does not partition full run: {name}')
    # IDTP/IDFP/IDFN and IDSW are intentionally not asserted additive: identity
    # matching and evaluator history are solved/reset separately in each window.
