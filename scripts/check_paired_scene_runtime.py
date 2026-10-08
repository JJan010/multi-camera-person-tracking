"""Prove scene-aware runtime equivalence to both frozen competitive variants."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from check_frozen_bytetrack_replay import candidate_arrays
from evaluate_appearance_global import lifecycle
from mtmc.data.scene import load_scene, require, sha256
from mtmc.pipeline.paired import CameraCandidates, CandidateRound, PairedAssociation, VARIANTS, validate_policy

ROOT = Path(__file__).resolve().parents[1]


def policy_from_reference(reference):
    """Copy the earlier experiment; selection here is fixed, not metric-driven."""
    local = reference['configuration']['local']
    return validate_policy({'schema_version': 1, 'variants': list(VARIANTS),
        'local': {'tracker': local['tracker'], 'appearance_threshold': local['appearance_threshold'],
                  'strong_history_size': 8, 'strong_history_max_age_frames': 30,
                  'weak_motion_min_iou': local['weak_motion_min_iou']},
        'global': reference['configuration']['global']})


def replay_exact(scene, policy, *, run_id, source, cache, local, reference,
                 source_trace, cache_trace, local_trace, global_trace, vectors):
    """Legacy cache reader is a regression bridge; runtime itself is scene-aware."""
    runtime = PairedAssociation(scene, policy, run_id=run_id, source_run_id=cache['run_id'])
    counts = {n: Counter() for n in VARIANTS}
    emitted = {n: set() for n in VARIANTS}
    refinement = {n: Counter() for n in VARIANTS}
    next_row = candidate_count = outside = updates = 0
    with Path(source_trace).open() as src, Path(cache_trace).open() as det, \
            gzip.open(local_trace, 'rt') as loc, gzip.open(global_trace, 'rt') as glob:
        for frame in range(scene.rounds):
            lines = [f.readline() for f in (src, det, loc, glob)]
            require(all(lines), 'Truncated frozen trace')
            original, cached, old_local, old_global = map(json.loads, lines)
            require(old_local['run_id'] == local['run_id']
                    and old_local['source_run_id'] == source['run_id']
                    and old_global['run_id'] == reference['run_id'], 'Mixed reference run scopes')
            for record in (old_local, old_global):
                require(type(record['frame_index']) is int and record['frame_index'] == frame
                        and record['timestamp'] == str(Fraction(frame, scene.fps)), 'Reference frame/time mismatch')
            arrays, _, next_row, missing = candidate_arrays(cached, original, frame=frame,
                cache_run=cache['run_id'], source_run=source['run_id'], next_row=next_row)
            cameras = tuple(CameraCandidates(c, boxes, scores, rows,
                tuple(vectors[i] if i is not None else None for i in rows))
                for c, (boxes, scores, rows) in sorted(arrays.items()))
            actual = runtime.step(CandidateRound(cache['run_id'], frame, Fraction(frame, scene.fps), cameras))
            for name in VARIANTS:
                current, expected = actual['variants'][name], old_global['variants'][name]
                require(current['cameras'] == old_local['variants'][name] == expected['cameras'],
                        f'Local outputs differ: {name}, frame {frame}')
                require(expected['identity']['run_id'] == reference['run_id'] + '/' + name,
                        'Reference global scope differs')
                require(current['identity'] == {**expected['identity'], 'run_id': run_id + '/' + name},
                        f'Global assignments/decisions/state differ: {name}, frame {frame}')
                require(actual['refinements'][name] == old_local['refinements'].get(name, []),
                        f'Refinement decisions differ: {name}, frame {frame}')
                refinement[name].update(actual['refinement_counters'][name])
                lifecycle(current['identity'], counts[name], emitted[name])
            candidate_count += sum(len(c.boxes) for c in cameras)
            outside += missing; updates += len(cameras)
            if (frame + 1) % 300 == 0 or frame + 1 == scene.rounds:
                print(f'Replayed {frame+1}/{scene.rounds}; both local/global records EXACT', flush=True)
        require(all(f.readline() == '' for f in (src, det, loc, glob)), 'Trailing frozen input')
    require(next_row == runtime.next_embedding_row == len(vectors)
            and candidate_count == cache['summary']['detections']
            and outside == cache['summary']['fully_outside'], 'Candidate/embedding accounting differs')
    for name in VARIANTS:
        counts[name]['ever_emitted_ids'] = len(emitted[name])
        require(dict(counts[name]) == reference['lifecycle'][name], f'Lifecycle differs: {name}')
        require(dict(refinement[name]) == local['refinement_counters'][name], f'Refinement counters differ: {name}')
        require(counts[name]['observations'] == local['counts'][name]['observations'], 'Local observation count differs')
    return {'rounds': scene.rounds, 'camera_updates_per_variant': updates,
            'candidate_observations': candidate_count, 'encoded_candidates': next_row,
            'fully_outside_candidates': outside, 'lifecycle': {n: dict(c) for n,c in counts.items()},
            'refinement_counters': {n: dict(c) for n,c in refinement.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-config', type=Path, required=True)
    parser.add_argument('--reference-global-report', type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve(); digest = sha256(path)
        require(expected is None or digest == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': digest}
        return path

    print('Verifying scene, frozen candidate cache and both reference variants...', flush=True)
    scene_inputs = load_scene(args.scene_config, project_root=ROOT)
    scene = scene_inputs.runtime
    require(scene.scene == 'MTMC_Tracking_2024/train/scene_001' and scene.camera_ids == (4,5,8)
            and scene.fps == 30 and all((c.width,c.height) == (1920,1080) for c in scene.cameras),
            'This regression bridge requires the legacy scene_001 reference')
    for name, spec in (('scene_config', scene_inputs.configuration), ('source_manifest', scene_inputs.source_manifest),
                       ('video_manifest', scene_inputs.video_manifest), ('calibration', scene.calibration)):
        checked(name, spec.path, spec.sha256)
    for c in scene.cameras:
        checked(f'video_{c.camera_id}', c.video.path, c.video.sha256)
    ref_path = checked('reference_global_report', args.reference_global_report)
    ref = json.loads(ref_path.read_text())
    required = ('staged_local_records_exact','staged_global_records_exact_except_run_scope',
                'staged_metrics_lifecycle_merges_exact','all_variants_motmetrics_agreement',
                'candidate_provenance','local_global_denominators_match','frozen_outputs')
    require(ref.get('completed') is True and ref['protocol'] == 'competitive_local_to_global_paired_v1'
            and all(ref['checks'].get(k) is True for k in required), 'Unverified global reference')
    for name in ('local_report','cache_report','pipeline_report','tracks','detections.jsonl','embeddings.npy','local_variants'):
        spec = ref['inputs'][name]; checked(name, spec['path'], spec['sha256'])
    local = json.loads(Path(inputs['local_report']['path']).read_text())
    cache = json.loads(Path(inputs['cache_report']['path']).read_text())
    source = json.loads(Path(inputs['pipeline_report']['path']).read_text())
    require(local.get('completed') is True and local['protocol'] == 'competitive_bytetrack_paired_v1'
            and all(local['checks'].get(k) is True for k in
                    ('staged_trace_exact','disabled_trace_exact','baseline_metrics_reproduced',
                     'all_candidates_accounted','frozen_predictions')), 'Unverified local reference')
    require(cache.get('completed') is True and cache['protocol'] == 'frozen_detector_candidate_embeddings_v1'
            and source.get('completed') is True, 'Incomplete source/cache')
    require(ref['source_run_id'] == local['source_run_id'] == source['run_id'] == cache['source_run_id']
            and local['cache_run_id'] == cache['run_id']
            and ref['local_experiment_run_id'] == local['run_id'], 'Mixed experiment scope')
    require(ref['configuration']['local'] == local['configuration']
            and ref['configuration']['frames'] == local['configuration']['runtime_frames'] == [0,scene.rounds-1]
            and ref['configuration']['evaluation_frames'] == local['configuration']['evaluation_frames']
                == [scene_inputs.evaluation.first_frame, scene_inputs.evaluation.last_frame]
            and scene_inputs.evaluation.gt_to_video_offset == 0 and scene_inputs.evaluation.min_iou == .5,
            'Reference evaluation/runtime interval differs')
    cfg = source['configuration']
    require(cfg['cameras'] == list(scene.camera_ids) and cfg['fps'] == scene.fps
            and source['summary']['rounds'] == scene.rounds and cfg['identity_start_frame'] == 0
            and cfg['coordinate_space'] == scene.coordinate_space, 'Reference scene timing/coordinate space differs')
    require(source['inputs']['calibration']['sha256'] == scene.calibration.sha256
            and source['inputs']['scene_source_manifest']['sha256'] == scene_inputs.source_manifest.sha256
            and source['inputs']['video_manifest']['sha256'] == scene_inputs.video_manifest.sha256
            and ref['inputs']['ground_truth']['sha256'] == scene_inputs.evaluation.ground_truth.sha256,
            'Reference scene assets differ')
    for c in scene.cameras:
        require(source['inputs'][f'video_{c.camera_id}']['sha256'] == c.video.sha256,
                'Reference video differs from scene configuration')
    require(cache['configuration']['frames'] == [0,scene.rounds-1]
            and cache['configuration']['rounds'] == scene.rounds
            and cache['configuration']['cameras'] == list(scene.camera_ids)
            and cache['configuration']['fps'] == scene.fps, 'Cache coverage differs')
    for name in ('pipeline_report','tracks'):
        require(cache['inputs'][name]['sha256'] == inputs[name]['sha256'], 'Cache source mismatch')
    for name in ('detections.jsonl','embeddings.npy'):
        spec = cache['artifacts'][name]
        checked('cache_artifact:'+name, Path(inputs['cache_report']['path']).parent / spec['path'], spec['sha256'])
        require(spec['sha256'] == inputs[name]['sha256'], 'Cache artifact mismatch')
    for name in ('cache_report','pipeline_report','tracks','detections.jsonl','embeddings.npy'):
        require(local['inputs'][name]['sha256'] == inputs[name]['sha256'], 'Local/global reference inputs differ')
    require(local['artifacts']['tracks']['sha256'] == inputs['local_variants']['sha256'], 'Local trace differs')
    spec = ref['artifacts']['global_tracks']
    global_trace = checked('global_variants', ref_path.parent / spec['path'], spec['sha256'])
    # Preserve the frozen global and experimental local implementations.
    required_code = {'src/mtmc/pipeline/core.py','src/mtmc/reid/history.py',
                     *('src/mtmc/association/'+n+'.py' for n in
                       ('geometry','pairwise','grouping','global_identity','controlled_merge'))}
    require(required_code <= set(source['code_sha256']), 'Missing frozen global code provenance')
    for rel in required_code:
        checked('code:'+rel, ROOT/rel, source['code_sha256'][rel])
    for rel, digest in local['code_sha256'].items():
        checked('code:'+rel, ROOT/rel, digest)
    baseline_spec = local['inputs']['local_report']
    baseline_path = checked('appearance_local_report', baseline_spec['path'], baseline_spec['sha256'])
    baseline = json.loads(baseline_path.read_text())
    require(baseline['configuration']['history_size'] == 8
            and baseline['configuration']['history_max_age_frames'] == 30
            and baseline['configuration']['appearance_threshold'] == local['configuration']['appearance_threshold']
            and baseline['configuration']['tracker'] == local['configuration']['tracker'] == cfg['tracker'],
            'Local policy lineage differs')
    for rel, digest in baseline['code_sha256'].items():
        checked('code:'+rel, ROOT/rel, digest)
    require(version('supervision') == '0.30.7', 'Expected pinned supervision 0.30.7')
    policy = policy_from_reference(ref)
    for name, value in policy['global'].items():
        require(cfg[name] == value, f'Global policy changed: {name}')
    vectors = np.load(inputs['embeddings.npy']['path'], mmap_mode='r', allow_pickle=False)
    require(vectors.dtype == np.float32 and vectors.shape == (cache['summary']['encoded'],512), 'Invalid feature cache')
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'artifacts/paired_scene_runtime_checks'/run
    output.mkdir(parents=True, exist_ok=False)
    (output/'run_status.json').write_text(json.dumps({'completed':False,'run_id':run})+'\n')
    print(f'Replaying {scene.rounds} rounds on CPU; no models, decoding or GT...', flush=True)
    summary = replay_exact(scene, policy, run_id=run, source=source, cache=cache, local=local, reference=ref,
        source_trace=inputs['tracks']['path'], cache_trace=inputs['detections.jsonl']['path'],
        local_trace=inputs['local_variants']['path'], global_trace=global_trace, vectors=vectors)
    for spec in inputs.values():
        require(sha256(spec['path']) == spec['sha256'], 'Input changed during replay')
    policy_path = ROOT/'configs/pipeline/paired_validation_policy.json'
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    if policy_path.exists():
        require(json.loads(policy_path.read_text()) == policy, 'Existing policy differs; refusing overwrite')
    else:
        policy_path.write_text(json.dumps(policy, indent=2, allow_nan=False)+'\n')
    checked('frozen_policy', policy_path)
    code = [Path(__file__), ROOT/'src/mtmc/pipeline/paired.py', ROOT/'src/mtmc/data/scene.py',
            ROOT/'src/mtmc/reid/crops.py', ROOT/'scripts/evaluate_appearance_global.py',
            ROOT/'scripts/check_frozen_bytetrack_replay.py']
    report = {'completed':True,'passed':True,'protocol':'paired_scene_runtime_parity_v1',
        'run_id':run, 'scene':scene.scene, 'configuration':policy, 'inputs':inputs, 'summary':summary,
        'checks':{'both_local_traces_exact':True,'both_global_records_exact_except_run_scope':True,
                  'refinement_decisions_and_counts_exact':True,'candidate_provenance_and_coverage':True,
                  'lifecycle_exact':True,'ground_truth_read':False},
        'reference_quality_not_recomputed':{n:ref['metrics'][n] for n in VARIANTS},
        'versions':{n:version(n) for n in ('supervision','numpy','scipy')},
        'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in code},
        'limits':['Integration parity on the reused development scene, not new validation quality.',
                  'GT files are not opened; reference metrics are copied and explicitly not recomputed.',
                  'The candidate cache reader in this checker is legacy-specific; the runtime is scene-aware.',
                  'Frozen local history retains the 30 FPS contract; other rates are rejected.',
                  'No video inference, GPU integration or end-to-end speed claim.']}
    path = output/'report.json'; path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'run_status.json').write_text(json.dumps({'completed':True,'run_id':run})+'\n')
    print('Both local traces, refinements, global assignments/decisions/state and lifecycle: EXACT')
    print(f'Policy: {policy_path}\nReport: {path}')
    print('Scene-aware paired runtime: PASSED; scene_041 inference/evaluation pending')


if __name__ == '__main__':
    main()
