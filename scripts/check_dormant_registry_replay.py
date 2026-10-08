"""Frozen two-policy parity with the recovery bridge explicitly disabled."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

import check_paired_scene_runtime as paired
from mtmc.association.recovery import RecoveryIdentityManager
from mtmc.data.scene import load_scene, require, sha256

ROOT=Path(__file__).resolve().parents[1]


def replay_disabled(scene,policy,**kwargs):
    """Inject only the disabled registry wrapper into the frozen replay checker.

    The dependency replacement is confined to this test call and restored even
    on failure. No source file or production runtime implementation is patched.
    """
    original=paired.PairedAssociation
    instances=[]
    class DisabledRecoveryRuntime(original):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw)
            for stage in self.stages.values():
                stage.manager=RecoveryIdentityManager(stage.manager,coordinate_space=scene.coordinate_space,
                    camera_ids=scene.camera_ids,enabled=False)
            instances.append(self)
    with patch.object(paired,'PairedAssociation',DisabledRecoveryRuntime):
        summary=paired.replay_exact(scene,policy,**kwargs)
    require(len(instances)==1,'Expected one isolated paired replay')
    runtime=instances[0];counters={}
    rename={'allocated_id_slots':'allocated_ids','expiration_events':'expired_ids','retained_ids':'retained_ids_at_end'}
    for name,stage in runtime.stages.items():
        manager=stage.manager;values=dict(manager.last_audit.lifecycle)
        require(not manager.archived_ids and manager._archive is None and not manager._snapshots
                and values['reactivation_events']==values['superseded_new_ids']==0,
                'Disabled control created archive state or a reactivation')
        expected=kwargs['reference']['lifecycle'][name]
        for field in ('observations','allocated_id_slots','ever_emitted_ids','absorbed_ids',
                      'expiration_events','retained_ids','merge_events'):
            require(values[field]==expected[rename.get(field,field)],'Disabled lifecycle counter differs: '+field)
        counters[name]=values
    return summary,counters


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity-report',type=Path,required=True)
    args=parser.parse_args();inputs={}
    def checked(name,path,expected=None):
        path=Path(path).resolve();digest=sha256(path)
        require(expected is None or digest==expected,'Changed frozen input: '+str(path))
        inputs[name]={'path':str(path),'sha256':digest};return path
    print('Verifying frozen runtime gate, sources, policy and code; no GT...',flush=True)
    gate_path=checked('runtime_parity_report',args.parity_report)
    gate=json.loads(gate_path.read_text())
    checks=('both_local_traces_exact','both_global_records_exact_except_run_scope',
            'refinement_decisions_and_counts_exact','candidate_provenance_and_coverage','lifecycle_exact')
    require(gate.get('completed') is True and gate.get('passed') is True
            and gate['protocol']=='paired_scene_runtime_parity_v1'
            and all(gate['checks'].get(k) is True for k in checks),'Invalid runtime parity gate')
    for name,spec in gate['inputs'].items():
        require(name!='ground_truth','Unexpected raw GT in parity inputs')
        checked(name,ROOT/name[5:] if name.startswith('code:') else spec['path'],spec['sha256'])
    for rel,digest in gate['code_sha256'].items(): checked('code:'+rel,ROOT/rel,digest)
    for package in ('supervision','numpy','scipy'):
        require(version(package)==gate['versions'][package],'Dependency changed: '+package)
    scene=load_scene(inputs['scene_config']['path'],project_root=ROOT).runtime
    require(scene.scene=='MTMC_Tracking_2024/train/scene_001' and scene.camera_ids==(4,5,8)
            and scene.fps==30 and scene.rounds==3600,'Expected pinned two-minute development scene')
    policy=json.loads(Path(inputs['frozen_policy']['path']).read_text())
    require(policy==gate['configuration'],'Frozen policy differs')
    source,cache,local,reference=[json.loads(Path(inputs[n]['path']).read_text()) for n in
        ('pipeline_report','cache_report','local_report','reference_global_report')]
    require(reference['source_run_id']==local['source_run_id']==cache['source_run_id']==source['run_id']
            and reference['local_experiment_run_id']==local['run_id'] and local['cache_run_id']==cache['run_id'],
            'Mixed source lineage')
    vectors=np.load(inputs['embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    require(vectors.dtype==np.float32 and vectors.shape==(cache['summary']['encoded'],512),'Invalid cached vectors')
    code=('src/mtmc/association/dormant.py','src/mtmc/association/recovery.py',
          'scripts/check_dormant_registry.py','scripts/check_dormant_registry_replay.py')
    for rel in code: checked('bridge_code:'+rel,ROOT/rel)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'artifacts/dormant_registry_checks'/run;output.mkdir(parents=True,exist_ok=False)
    status=output/'run_status.json';status.write_text(json.dumps({'completed':False,'run_id':run})+'\n')
    try:
        print('Replaying 3600 CPU rounds; recovery DISABLED; both frozen policies...',flush=True)
        summary,counters=replay_disabled(scene,policy,run_id=run,source=source,cache=cache,local=local,reference=reference,
            source_trace=inputs['tracks']['path'],cache_trace=inputs['detections.jsonl']['path'],
            local_trace=inputs['local_variants']['path'],global_trace=inputs['global_variants']['path'],vectors=vectors)
        require(summary==gate['summary'],'Original parity summary differs')
        for spec in inputs.values(): require(sha256(spec['path'])==spec['sha256'],'Input changed during replay')
        report={'completed':True,'passed':True,'protocol':'dormant_registry_disabled_parity_v1','run_id':run,
            'scene':scene.scene,'inputs':inputs,'configuration':{'recovery_enabled':False,'frozen_policy':policy},
            'summary':summary,'bridge_lifecycle':counters,
            'checks':{'both_local_traces_exact':True,'both_global_records_exact_except_run_scope':True,
                'refinement_decisions_and_counts_exact':True,'original_parity_summary_exact':True,
                'bridge_lifecycle_matches_baseline':True,'no_archive_or_reactivation_when_disabled':True,
                'frozen_inputs_unchanged':True},
            'versions':{n:version(n) for n in ('supervision','numpy','scipy')},
            'limits':['Recovery is disabled; this checks integration parity, not enabled tracking quality.',
                'No GT input, models, video decoding, threshold calibration or performance claim.',
                'Enabled recovery still needs real snapshot/query provenance and paired quality evaluation.']}
        path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps({'completed':True,'run_id':run})+'\n')
    except Exception as error:
        status.write_text(json.dumps({'completed':False,'run_id':run,'error_type':type(error).__name__,
            'error':str(error)})+'\n');raise
    print('Both frozen local/global traces, decisions and lifecycle: EXACT')
    print('Disabled archive state and reactivation counters: ZERO')
    print(f'Report: {path}')
    print('Dormant registry disabled replay: PASSED; enabled quality evaluation pending')


if __name__=='__main__': main()
