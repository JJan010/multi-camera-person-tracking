"""Replay fixed global identity logic over the competitive local tracking experiment."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path

import numpy as np

import evaluate_mtmc_sequence as sequence
from evaluate_appearance_global import direct_observations, lifecycle
from check_frozen_bytetrack_replay import candidate_arrays
from mtmc.pipeline.core import IdentityStage
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, sha256
from run_mtmc import load_matrices, dump

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('staged', 'competitive_iou', 'competitive_appearance')
require = sequence.require


def check_staged(actual, reference, *, current_scope, reference_scope):
    """Compare every decision/state field; only the explicit run scope may differ."""
    require(actual['identity']['run_id'] == current_scope
            and reference['identity']['run_id'] == reference_scope, 'Staged identity scope differs')
    require(actual['cameras'] == reference['cameras'], 'Staged local candidate records differ')
    expected = {**reference['identity'], 'run_id': current_scope}
    require(actual['identity'] == expected, 'Staged global decisions/state differ')


def quality(path, ground, rounds):
    metrics, mapping, merge_counts = {}, {}, {}
    for variant in VARIANTS:
        counter = sequence.metric.IdentityCounts(); slots = []; categories = Counter()
        with gzip.open(path, 'rt') as stream:
            for frame in range(rounds):
                line = stream.readline(); require(bool(line), 'Truncated global experiment')
                record = json.loads(line); require(record['frame_index'] == frame, 'Global frame order differs')
                data = record['variants'][variant]
                assigned = {ObservationKey(**a['key']): a['global_id'] for a in data['identity']['assignments']}
                require(len(assigned) == len(data['identity']['assignments']), 'Duplicate global assignment')
                seen, unique = set(), {}
                for camera in data['cameras']:
                    c = camera['camera']; keys = tuple(ObservationKey(c, i, frame) for i in camera['local_ids'])
                    require(len(keys) == len(set(keys)), 'Duplicate local key')
                    seen.update(keys)
                    if frame < 2:
                        continue
                    boxes = np.asarray(camera['xyxy'], np.float64).reshape(-1, 4)
                    gt, mask, evidence, _, _ = sequence.spatial_slot(ground[frame, c], keys, boxes)
                    unique.update(evidence)
                    ids = [assigned[k] for k in keys]
                    counter.update(gt, ids, mask); slots.append((gt, ids, mask))
                require(seen == set(assigned), 'Global/local observation coverage differs')
                for event in data['identity']['merge_events']:
                    labels = [unique.get(ObservationKey(**m)) for m in event['members']]
                    category = ('unannotated_frame' if frame < 2 else 'unresolved' if not labels or any(x is None for x in labels)
                                else 'all_visible_members_same_gt' if len(set(labels)) == 1 else 'different_known_gt')
                    categories[category] += 1
            require(stream.readline() == '', 'Trailing global frames')
        metrics[variant], mapping[variant] = counter.result()
        sequence.metric.check_reference(metrics[variant], sequence.metric.reference_metrics(slots))
        merge_counts[variant] = dict(categories)
    return metrics, mapping, merge_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-report', type=Path, required=True)
    parser.add_argument('--reference-global-report', type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve()
        value = sha256(path)
        require(expected is None or value == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': value}
        return path

    print('Verifying competitive local traces and the frozen staged global reference...', flush=True)
    local_path = checked('local_report', args.local_report)
    local = json.loads(local_path.read_text())
    require(local.get('completed') is True and local['protocol'] == 'competitive_bytetrack_paired_v1'
            and all(local['checks'].get(k) is True for k in
                    ('staged_trace_exact','disabled_trace_exact','baseline_metrics_reproduced',
                     'all_candidates_accounted','frozen_predictions')), 'Expected verified competitive experiment')
    for name in ('cache_report','pipeline_report','tracks','detections.jsonl','embeddings.npy','ground_truth'):
        spec = local['inputs'][name]
        checked(name, spec['path'], spec['sha256'])
    spec = local['artifacts']['tracks']
    local_trace = checked('local_variants', local_path.parent / spec['path'], spec['sha256'])
    reference_path = checked('reference_global_report', args.reference_global_report)
    reference = json.loads(reference_path.read_text())
    require(reference.get('completed') is True and reference['protocol'] == 'appearance_local_to_global_paired_v1',
            'Expected completed appearance global experiment')
    spec = reference['inputs']['local_report']
    baseline_local_path = checked('baseline_local_report', spec['path'], spec['sha256'])
    baseline_local = json.loads(baseline_local_path.read_text())
    require(inputs['baseline_local_report']['sha256'] == local['inputs']['local_report']['sha256']
            and reference['local_experiment_run_id'] == baseline_local['run_id'], 'Reference belongs to another local baseline')
    for name in ('pipeline_report','cache_report','detections.jsonl','embeddings.npy','ground_truth'):
        require(reference['inputs'][name]['sha256'] == inputs[name]['sha256'], f'Mixed reference input: {name}')
    spec = reference['artifacts']['global_tracks']
    reference_trace = checked('reference_global_tracks', reference_path.parent / spec['path'], spec['sha256'])
    checked('code:scripts/evaluate_appearance_global.py', ROOT / 'scripts/evaluate_appearance_global.py', reference['script_sha256'])
    source = json.loads(Path(inputs['pipeline_report']['path']).read_text())
    cache = json.loads(Path(inputs['cache_report']['path']).read_text())
    require(source['run_id'] == local['source_run_id'] == reference['source_run_id']
            and cache['run_id'] == local['cache_run_id'], 'Mixed replay scope')
    rounds, cfg = source['summary']['rounds'], source['configuration']
    require(cfg['cameras'] == [4,5,8] and cfg['fps'] == 30
            and local['configuration']['runtime_frames'] == reference['configuration']['frames'] == [0,rounds-1]
            and local['configuration']['evaluation_frames'] == reference['configuration']['evaluation_frames'] == [2,rounds-1]
            and local['configuration']['variants'] == list(VARIANTS), 'Replay configuration differs')
    for key in ('history','appearance_variant','appearance_threshold','geometry','identity'):
        require(cfg[key] == reference['configuration'][key], f'Global settings changed: {key}')
    require(local['configuration']['appearance_threshold'] == reference['configuration']['local_appearance_threshold'],
            'Local appearance threshold changed')
    required_code = {'src/mtmc/pipeline/core.py','src/mtmc/reid/history.py',
                     *('src/mtmc/association/'+n+'.py' for n in
                       ('geometry','pairwise','grouping','global_identity','controlled_merge'))}
    require(required_code <= set(source['code_sha256']), 'Missing original global code provenance')
    for rel, digest in source['code_sha256'].items():
        if rel.startswith('src/mtmc/association/') or rel in ('src/mtmc/pipeline/core.py','src/mtmc/reid/history.py'):
            checked('code:'+rel, ROOT / rel, digest)
    for rel, digest in local['code_sha256'].items():
        checked('code:'+rel, ROOT / rel, digest)
    spec = source['inputs']['calibration']
    calibration_path = checked('calibration', spec['path'], spec['sha256'])
    require(reference['inputs']['calibration']['sha256'] == inputs['calibration']['sha256'], 'Reference calibration differs')
    matrices = load_matrices(calibration_path, cfg['cameras'])
    vectors = np.load(inputs['embeddings.npy']['path'], mmap_mode='r', allow_pickle=False)
    require(vectors.dtype == np.float32 and vectors.shape == (cache['summary']['encoded'],512), 'Invalid embedding cache')
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    histories, stages = {}, {}
    for name in VARIANTS:
        scope = run + '/' + name
        histories[name] = AppearanceHistory(scope, max_observations=cfg['history']['max_observations'],
                                            max_age=Fraction(cfg['history']['max_age_seconds']))
        stages[name] = IdentityStage(scope, matrices, cfg['coordinate_space'], variant=cfg['appearance_variant'],
            threshold=cfg['appearance_threshold'], **cfg['geometry'], identity_configuration=cfg['identity'])
    output = ROOT / 'artifacts/competitive_global' / run
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run_status.json').write_text(json.dumps({'completed':False,'run_id':run})+'\n')
    trace_path = output / 'global_tracks.jsonl.gz'
    counters = {n:Counter() for n in VARIANTS}; emitted = {n:set() for n in VARIANTS}
    next_row = candidate_count = outside = 0
    print('Phase 1: three causal global replays; unchanged history/geometry/identity; no GT input...', flush=True)
    with Path(inputs['tracks']['path']).open() as src, Path(inputs['detections.jsonl']['path']).open() as det, \
            gzip.open(local_trace,'rt') as loc, gzip.open(reference_trace,'rt') as ref, gzip.open(trace_path,'wt') as saved:
        for frame in range(rounds):
            lines = [f.readline() for f in (src,det,loc,ref)]
            require(all(lines), 'Truncated frozen input')
            original,candidates,variants,previous = map(json.loads, lines)
            require(variants['run_id'] == local['run_id'] and variants['source_run_id'] == source['run_id']
                    and variants['frame_index'] == previous['frame_index'] == frame
                    and previous['run_id'] == reference['run_id']
                    and variants['timestamp'] == previous['timestamp'] == str(Fraction(frame,30))
                    and set(variants['variants']) == set(VARIANTS), 'Frozen frame scope differs')
            arrays, _, next_row, missing = candidate_arrays(candidates, original, frame=frame,
                cache_run=cache['run_id'], source_run=source['run_id'], next_row=next_row)
            candidate_count += sum(len(a[0]) for a in arrays.values()); outside += missing
            current = {}
            for name in VARIANTS:
                cameras = variants['variants'][name]
                records, features = direct_observations(cameras,candidates,vectors,frame)
                history = histories[name].update(frame,Fraction(frame,30),features)
                descriptor = history.mean if cfg['appearance_variant']=='mean' else history.latest
                identity,_,_,_,_,_ = stages[name].update(frame,Fraction(frame,30),descriptor,records)
                current[name] = {'identity':json.loads(dump(asdict(identity))), 'cameras':cameras}
                lifecycle(current[name]['identity'],counters[name],emitted[name])
            check_staged(current['staged'],previous['variants']['direct_appearance'],
                         current_scope=run+'/staged', reference_scope=reference['run_id']+'/direct_appearance')
            saved.write(dump({'run_id':run,'frame_index':frame,'timestamp':str(Fraction(frame,30)),
                              'variants':current})+'\n')
            if (frame+1)%300==0 or frame==rounds-1:
                print(f'Replayed {frame+1}/{rounds}; staged global decisions/state EXACT',flush=True)
        require(all(f.readline()=='' for f in (src,det,loc,ref)), 'Trailing frozen input')
    require(next_row==len(vectors) and candidate_count==cache['summary']['detections']
            and outside==cache['summary']['fully_outside'], 'Candidate coverage differs')
    for name in VARIANTS:
        require(counters[name]['observations']==local['counts'][name]['observations'], 'Local/global observation count differs')
        counters[name]['ever_emitted_ids']=len(emitted[name])
    require(dict(counters['staged'])==reference['lifecycle']['direct_appearance'], 'Staged lifecycle differs')
    frozen = sha256(trace_path)
    print('Phase 2: offline shared identity evaluation against motmetrics...',flush=True)
    ground = sequence.load_ground_truth(Path(inputs['ground_truth']['path']),rounds)
    metrics,mapping,categories = quality(trace_path,ground,rounds)
    require(metrics['staged']==reference['metrics']['direct_appearance'], 'Staged global metrics differ')
    require(categories['staged']==reference['accepted_merge_diagnostics']['direct_appearance'], 'Staged merge diagnostics differ')
    for name in VARIANTS:
        m=metrics[name]; local_counts=local['metrics'][name]['OVERALL']
        require(m['predicted_observations']==local_counts['num_predictions']
                and m['gt_observations']==local_counts['num_objects'], 'Local/global evaluation denominators differ')
    require(sha256(trace_path)==frozen, 'Predictions changed during evaluation')
    for spec in inputs.values():
        require(sha256(Path(spec['path']))==spec['sha256'], 'Input changed during experiment')
    matching_path=output/'identity_matching.json'
    matching_path.write_text(json.dumps(mapping,indent=2,allow_nan=False)+'\n')
    report={'completed':True,'protocol':'competitive_local_to_global_paired_v1','run_id':run,
            'source_run_id':source['run_id'],'local_experiment_run_id':local['run_id'],'inputs':inputs,
            'configuration':{'frames':[0,rounds-1],'evaluation_frames':[2,rounds-1],
                             'global':{k:cfg[k] for k in ('history','appearance_variant','appearance_threshold','geometry','identity')},
                             'local':local['configuration']},
            'metrics':metrics,'lifecycle':{n:dict(c) for n,c in counters.items()},'accepted_merge_diagnostics':categories,
            'checks':{'staged_local_records_exact':True,'staged_global_records_exact_except_run_scope':True,
                      'staged_metrics_lifecycle_merges_exact':True,'all_variants_motmetrics_agreement':True,
                      'candidate_provenance':True,'local_global_denominators_match':True,'frozen_outputs':True},
            'artifacts':{'global_tracks':{'path':trace_path.name,'sha256':frozen},
                         'identity_matching':{'path':matching_path.name,'sha256':sha256(matching_path)}},
            'script_sha256':sha256(Path(__file__)),
            'limits':['Two-minute reused development sequence; not independent validation.',
                      'Each variant has independent local/global identity namespaces; numeric IDs are not cross-variant person labels.',
                      'One shared GT/global assignment over all cameras and evaluated frames for each variant.',
                      'Global histories keep the original all-encoded-observation update policy, including accepted weak observations.',
                      'Same-GT visible merge members do not establish purity of the whole retained identity.',
                      'No threshold tuning, runtime reset, model execution, or end-to-end performance claim.']}
    path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'run_status.json').write_text(json.dumps({'completed':True,'run_id':run})+'\n')
    print('Variant                   Global IDF1   IDP     IDR     IDTP    IDFP    IDFN   Delta/staged pp')
    for name in VARIANTS:
        m=metrics[name]
        print(f"{name:26} {100*m['idf1']:8.2f}% {100*m['idp']:6.2f}% {100*m['idr']:6.2f}% "
              f"{m['idtp']:7} {m['idfp']:7} {m['idfn']:7} {100*(m['idf1']-metrics['staged']['idf1']):+9.2f}")
        print('  Lifecycle:',json.dumps(dict(counters[name])))
        print('  Accepted merges:',json.dumps(categories[name]))
    print('Staged full records, metrics, lifecycle, merge diagnostics and all denominators: VERIFIED')
    print(f'Report: {path}')
    print('Competitive local-to-global evaluation: COMPLETED; runtime baseline unchanged')


if __name__=='__main__':
    main()


