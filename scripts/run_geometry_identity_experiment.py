"""Paired causal geometry association replay followed by shared ID evaluation."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import audit_scene_geometry as calibration_io
import evaluate_appearance_history as history
import evaluate_global_identity as evaluation
import evaluate_pairwise_association as pair_io
import replay_global_identity as replay
import run_controlled_merge_experiment as merge_io
from mtmc.association import geometry, pairwise, grouping, controlled_merge, global_identity
from mtmc.reid.osnet import ObservationKey

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (4,5,8)
require = evaluation.require
plain = merge_io.to_plain


def edge_set(pairs):
    return {grouping.edge_key(m.left,m.right) for p in pairs for m in p.matches}


def partition_edges(groups):
    return {grouping.edge_key(a,b) for members in groups for a,b in combinations(members,2)}


def read_geometry_record(record, *, geometry_config, coordinate_space, **context):
    require(record.get('association_policy')==geometry.POLICY and record.get('geometry_configuration')==geometry_config
            and record.get('coordinate_space')==coordinate_space, 'Geometry output policy mismatch')
    return merge_io.read_controlled_record(record, **context)


class Lifecycle:
    def __init__(self):
        self.allocated,self.absorbed,self.expired,self.emitted = set(),set(),set(),set()
        self.retained=set()
        self.merge_events=0

    def update(self, result):
        fresh={a.global_id for a in result.base_assignments if a.reason=='new_identity'}
        require(not fresh & self.allocated,'Reused allocated identity')
        self.allocated.update(fresh)
        require(not self.expired.intersection(result.expired_global_ids),'Repeated expiry')
        self.expired.update(result.expired_global_ids)
        absorbed=[gid for event in result.merge_events for gid in event.absorbed_global_ids]
        require(len(absorbed)==len(set(absorbed)) and not set(absorbed)&(self.absorbed|self.expired),'Invalid absorption')
        self.absorbed.update(absorbed)
        self.retained={state.global_id for state in result.identities}
        require(not self.expired & self.absorbed and not self.retained & (self.expired|self.absorbed)
                and self.allocated==self.expired|self.absorbed|self.retained,'Lifecycle accounting mismatch')
        current={a.global_id for a in result.assignments}
        require(current <= self.retained,'Retired identity emitted')
        require(len({(a.global_id,a.key.camera_id) for a in result.assignments})==len(result.assignments),'Duplicate identity/camera slot')
        self.emitted.update(current)
        self.merge_events+=len(result.merge_events)

    def summary(self):
        return {'allocated_ids':len(self.allocated),'ever_emitted_ids':len(self.emitted),
                'absorbed_ids':len(self.absorbed),'expired_ids':len(self.expired),
                'retained_ids_at_end':len(self.retained),'merge_events':self.merge_events}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report',type=Path,required=True)
    parser.add_argument('--geometry-report',type=Path,required=True)
    parser.add_argument('--max-distance',type=float,required=True)
    parser.add_argument('--unavailable-policy',choices=('appearance_only','reject'),required=True)
    args=parser.parse_args()
    distance=geometry._positive_distance(args.max_distance)
    inputs={}
    def checked(name,path,expected=None):
        path=Path(path).resolve()
        digest=calibration_io.sha256(path)
        require(expected is None or digest==expected,f'Checksum mismatch: {path}')
        inputs[name]={'path':str(path),'sha256':digest}
        return path
    print('Verifying controlled baseline, calibration and frozen appearance histories...',flush=True)
    baseline_path=checked('controlled_baseline_report',args.baseline_report)
    audit_path=checked('geometry_audit',args.geometry_report)
    baseline=json.loads(baseline_path.read_text())
    audit=json.loads(audit_path.read_text())
    require(baseline.get('completed') and baseline['protocol']['name']=='scene_001_controlled_merge_paired_v1', 'Unexpected baseline')
    require(audit.get('completed') and audit['protocol']=='scene_001_geometry_audit_v1','Unexpected geometry audit')
    bp=baseline['protocol']
    require((bp['cameras'],bp['first_frame'],bp['last_frame'],bp['fps'],bp['policy'])==
            ([4,5,8],2,299,30,controlled_merge.POLICY),'Unsupported baseline scope')
    run,variant,tau=baseline['source_run_id'],bp['variant'],bp['threshold']
    for name in ('tracks','ground_truth','groups','grouping_report'):
        item=baseline['inputs'][name]; checked(name,item['path'],item['sha256'])
    item=baseline['artifacts']['assignments.jsonl.gz']
    baseline_assignments=checked('controlled_baseline_assignments',baseline_path.parent/item['path'],item['sha256'])
    item=audit['inputs']['calibration_2025_format.json']
    cal_path=checked('calibration',item['path'],item['sha256'])
    require(inputs['ground_truth']['sha256']==audit['inputs']['ground_truth.txt']['sha256'],'Different GT source')
    group_report=json.loads(Path(inputs['grouping_report']['path']).read_text())
    require(group_report.get('completed') and group_report['source_run_id']==run,'Different grouping source')
    require(group_report['artifacts']['groups.jsonl.gz']['sha256']==inputs['groups']['sha256'],'Grouping trace differs')
    item=group_report['inputs']['pairwise_report']
    pair_report_path=checked('pairwise_report',item['path'],item['sha256'])
    pair_report=json.loads(pair_report_path.read_text())
    require(pair_report.get('completed') and pair_report['source_run_id']==run,'Different pairwise source')
    item=pair_report['inputs']['history_report']
    history_path=checked('history_report',item['path'],item['sha256'])
    hist_report,source,trace,history_rows,latest,means,paths=history.load_inputs(history_path)
    require(source['run_id']==run,'Different embedding run')
    for name,path in paths.items():
        item=pair_report['inputs'][name]
        checked('history_'+name,path,item['sha256'])
    require(inputs['history_tracks']['sha256']==inputs['tracks']['sha256'],'Embedding row/track source differs')
    features={'latest':latest,'mean':means}[variant]
    frames={record['frame_index']:record for record in trace}
    calibration=json.loads(cal_path.read_text())
    matrices={}
    for camera in CAMERAS:
        sensors=[s for s in calibration['sensors'] if s['id'].lower()==f'camera_{camera:04d}']
        require(len(sensors)==1,'Missing/duplicate calibrated camera')
        s=sensors[0]
        k,e,p,h=[calibration_io.matrix(s,name,shape) for name,shape in
             [('intrinsicMatrix',(3,3)),('extrinsicMatrix',(3,4)),('cameraMatrix',(3,4)),('homography',(3,3))]]
        require(np.linalg.matrix_rank(h)==3 and calibration_io.proportional_error(p,k@e)<=1e-6
                and calibration_io.proportional_error(h,p[:,[0,1,3]])<=1e-6,'Unexpected projection convention')
        matrices[camera]=h
    selected_groups={}
    with gzip.open(inputs['groups']['path'],'rt') as handle:
        for line in handle:
            record=json.loads(line)
            if (record['variant'],record['threshold'])==(variant,tau):
                frame=record['frame_index']; require(frame not in selected_groups,'Duplicate source grouping frame')
                selected_groups[frame]=record
    require(set(selected_groups)==set(range(2,300)),'Incomplete source grouping timeline')
    expected_rows=[r for r in group_report['results'] if (r['variant'],r['threshold'])==(variant,tau)]
    require(len(expected_rows)==1,'Missing/duplicate grouping summary')
    expected=expected_rows[0]
    conf=bp['configuration']
    common=dict(run_id=run,max_idle=Fraction(conf['max_idle_seconds']),descriptor_variant=variant,min_similarity=tau,
                min_support_rounds=conf['min_support_rounds'],min_support_seconds=Fraction(conf['min_support_seconds']),
                max_evidence_gap=Fraction(conf['max_evidence_gap']))
    base_manager=controlled_merge.ControlledMergeIdentityManager(**common)
    manager=controlled_merge.ControlledMergeIdentityManager(**common)
    config={'max_distance':distance,'unavailable_policy':args.unavailable_policy}
    space=f'{run}/Z0/native/calibration={inputs["calibration"]["sha256"]}'
    experiment_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    scope=f'{experiment_id}/{geometry.POLICY}/{variant}/tau={tau!r}'
    output=ROOT/'artifacts/geometry_identity'/experiment_id
    output.mkdir(parents=True,exist_ok=False)
    assignment_path,decisions_path=output/'assignments.jsonl.gz',output/'decisions.jsonl.gz'
    baseline_maps,source_labels,rows_by_frame={},{},{}
    old_edges_by_frame,old_grouped_by_frame={},{ }
    counts,retained_counts=Counter(),Counter()
    gate_counts,eligible_gate_counts,merge_counts,reset_counts=Counter(),Counter(),Counter(),Counter()
    life=Lifecycle(); base_life=Lifecycle()
    by_frame=[]; unavailable=0; row_ids=set()
    print('Phase 1: causal replay without GT input; regenerating baseline and geometric assignments...',flush=True)
    with gzip.open(baseline_assignments,'rt') as old, gzip.open(assignment_path,'wt') as dst, gzip.open(decisions_path,'wt') as decisions:
        for frame in range(2,300):
            line=old.readline(); require(bool(line),'Truncated controlled baseline')
            saved_baseline=json.loads(line)
            saved=selected_groups[frame]
            original_groups,labels=replay.decode_round(saved,run_id=run,frame=frame,variant=variant,threshold=tau,fps=30,cameras=CAMERAS)
            group_counts,edges=replay.source_counts(saved,original_groups,labels)
            counts.update(group_counts); retained_counts.update(edges)
            source_labels[frame]=labels
            records=[r for r in frames[frame]['reid_observations'] if r['status']=='encoded']
            trace_rows={ObservationKey(r['camera'],r['local_id'],r['frame_index']):r['embedding_row'] for r in records}
            require(trace_rows=={k:v['embedding_row'] for k,v in labels.items()},'Frozen grouping/embedding row mismatch')
            require(not row_ids & set(trace_rows.values()),'Embedding row reused across frames')
            row_ids.update(trace_rows.values()); rows_by_frame[frame]=trace_rows
            appearances={c:pair_io.make_camera(run,c,frame,variant,records,features) for c in CAMERAS}
            grounds={}
            for c in CAMERAS:
                cameras=[x for x in frames[frame]['cameras'] if x['camera']==c]
                require(len(cameras)==1,'Missing/duplicate camera trace')
                camera=cameras[0]
                require(len(camera['local_ids'])==len(camera['xyxy']) and len(set(camera['local_ids']))==len(camera['local_ids']), 'Invalid tracked box mapping')
                boxes=dict(zip(camera['local_ids'],camera['xyxy']))
                positions=[]
                for key in appearances[c].observations.keys:
                    require(key.local_id in boxes,'Missing raw tracked box')
                    record=next(r for r in records if (r['camera'],r['local_id'])==(c,key.local_id))
                    require(np.array_equal(np.asarray(record['source_xyxy']),np.asarray(boxes[key.local_id])), 'Encoded and tracked boxes differ')
                    xy=geometry.project_box_foot(matrices[c],boxes[key.local_id])
                    positions.append(geometry.GroundObservation(key,xy)); unavailable+=int(xy is None)
                require(len(positions)==len(boxes),'Unencoded tracked observation; pair coverage would differ')
                grounds[c]=geometry.CameraGroundPositions(run,c,frame,Fraction(frame,30),space,tuple(positions))
            base_pairs=[pairwise.associate_camera_pair(appearances[a],appearances[b],min_similarity=tau) for a,b in combinations(CAMERAS,2)]
            regenerated=grouping.group_pair_associations(base_pairs)
            require(regenerated.groups==original_groups.groups and plain([asdict(d) for d in regenerated.decisions])==saved['decisions'],
                    'Recomputed appearance/grouping baseline differs from frozen source')
            base=base_manager.update(regenerated)
            require(plain(replay.encode_output(base,labels,baseline['identity_scope']))==saved_baseline,
                    'Full controlled baseline record not reproduced')
            base_life.update(base)
            baseline_maps[frame]={a.key:a.global_id for a in base.assignments}
            old_edges_by_frame[frame]=edge_set(base_pairs)
            old_grouped_by_frame[frame]=partition_edges(regenerated.groups)
            gated=[geometry.associate_camera_pair_with_geometry(appearances[a],appearances[b],grounds[a],grounds[b],
                     min_similarity=tau,**config) for a,b in combinations(CAMERAS,2)]
            grouped=geometry.group_geometry_associations(gated)
            result=manager.update(grouped)
            require({a.key for a in result.assignments}==set(labels),'Changed observation coverage')
            life.update(result)
            merge_counts.update(d.outcome for d in result.merge_decisions)
            reset_counts.update(d.reason for d in result.candidate_resets)
            for pair in gated:
                for candidate in pair.candidates:
                    gate_counts[candidate.geometry_decision]+=1
                    if candidate.appearance_eligible: eligible_gate_counts[candidate.geometry_decision]+=1
            record=replay.encode_output(result,labels,scope)
            record.update(association_policy=geometry.POLICY,geometry_configuration=config,coordinate_space=space)
            dst.write(json.dumps(record,default=replay.json_default,allow_nan=False)+'\n')
            decisions.write(json.dumps({'run_id':run,'frame_index':frame,'timestamp':str(Fraction(frame,30)),
                'geometry':config,'coordinate_space':space,'ground_positions':[asdict(grounds[c]) for c in CAMERAS],
                'pairs':[asdict(p) for p in gated],'groups':asdict(grouped)},default=replay.json_default,allow_nan=False)+'\n')
            after_edges=edge_set(p.association for p in gated)
            by_frame.append({'frame':frame,'observations':len(labels),'baseline_links':len(old_edges_by_frame[frame]),
                'geometry_links':len(after_edges),'removed_links':len(old_edges_by_frame[frame]-after_edges),
                'new_links':len(after_edges-old_edges_by_frame[frame]),'merge_events':len(result.merge_events)})
            if (frame+1)%60==0: print(f'Replayed through frame {frame}/299',flush=True)
        require(old.readline()=='','Trailing baseline records')
    require(dict(counts)==expected['groups'],'Source grouping counts differ')
    require(all(retained_counts[k]==expected['after'][k] for k in replay.grouping_eval.EDGE_FIELDS),'Source grouped-link counts differ')
    require(base_life.summary()==baseline['lifecycle'],'Controlled baseline lifecycle differs')
    frozen={p.name:calibration_io.sha256(p) for p in (assignment_path,decisions_path)}
    print('Frozen appearance groups and full controlled baseline records: REPRODUCED',flush=True)
    print('Phase 2: offline shared identity evaluation of frozen outputs...',flush=True)
    ground=history.load_ground_truth(Path(inputs['ground_truth']['path']))
    slots,frame_maps,excluded,outside=evaluation.build_slots(Path(inputs['tracks']['path']),ground,run)
    require(frame_maps==rows_by_frame,'Evaluation/replay row mappings differ')
    relabeled,_=history.label_trace(trace,ground)
    metric_labels={ObservationKey(r['camera'],r['local_id'],r['frame_index']):{'gt_id':r['gt_id']} for r in relabeled if r['frame_index']>=2}
    for frame,labels in source_labels.items():
        require(all(metric_labels[k]['gt_id']==v['diagnostic_gt_id'] for k,v in labels.items()),'Fixed GT labels changed')
    before,after=evaluation.IdentityCounts(),evaluation.IdentityCounts()
    edge_metrics={name:Counter() for name in ('baseline_pairs','geometry_pairs','baseline_groups','geometry_groups','removed_pairs','new_pairs')}
    merge_diagnostics=[]; merge_categories=Counter({k:0 for k in ('all_visible_members_same_gt','different_known_gt','unresolved')})
    with gzip.open(assignment_path,'rt') as assignments,gzip.open(decisions_path,'rt') as decisions:
        for frame in range(2,300):
            row=json.loads(assignments.readline()); detail=json.loads(decisions.readline())
            require(detail['frame_index']==frame and detail['run_id']==run,'Decision trace order differs')
            assigned=read_geometry_record(row,geometry_config=config,coordinate_space=space,source_run=run,scope=scope,
                         frame=frame,variant=variant,threshold=tau,expected_rows=frame_maps[frame])
            unique_gt={}
            for camera in CAMERAS:
                gt,keys,mask=slots[frame,camera]
                before.update(gt,[baseline_maps[frame][k] for k in keys],mask)
                after.update(gt,[assigned[k] for k in keys],mask)
                gd,pd=mask.sum(axis=1),mask.sum(axis=0)
                for i,j in zip(*np.nonzero(mask)):
                    if gd[i]==pd[j]==1: unique_gt[keys[j]]=gt[i]
            for event in row['merge_events']:
                category=merge_io.merge_label_category(event['members'],unique_gt)
                merge_categories[category]+=1
                merge_diagnostics.append({**event,'diagnostic_category':category,
                    'diagnostic_members':[{**k,'unique_overlap_gt_id':unique_gt.get(ObservationKey(**k))} for k in event['members']]})
            new_edges={grouping.edge_key(ObservationKey(**m['left']),ObservationKey(**m['right']))
                       for pair in detail['pairs'] for m in pair['association']['matches']}
            groups=tuple(tuple(ObservationKey(**k) for k in g) for g in detail['groups']['groups'])
            edge_sets={'baseline_pairs':old_edges_by_frame[frame],'geometry_pairs':new_edges,
                'baseline_groups':old_grouped_by_frame[frame],'geometry_groups':partition_edges(groups),
                'removed_pairs':old_edges_by_frame[frame]-new_edges,'new_pairs':new_edges-old_edges_by_frame[frame]}
            for name,edges in edge_sets.items():
                edge_metrics[name].update(replay.grouping_eval.count_edges(edges,metric_labels))
        require(assignments.readline()==decisions.readline()=='','Trailing output records')
    baseline_metrics,base_mapping=before.result()
    metrics,mapping=after.result()
    require(baseline_metrics==baseline['controlled_metrics'],'Controlled baseline metrics not reproduced')
    require(all(metrics[k]==baseline_metrics[k] for k in ('camera_time_slots','gt_observations','predicted_observations')),'Metric denominators differ')
    require(metrics['predicted_identities']==len(life.emitted),'Emitted identity count differs')
    require(sum(merge_categories.values())==life.merge_events,'Merge diagnostic accounting differs')
    for name,source in [('baseline_pairs','before'),('baseline_groups','after')]:
        require(all(edge_metrics[name][k]==expected[source][k] for k in replay.grouping_eval.EDGE_FIELDS),'Baseline link quality not reproduced')
    for item in inputs.values(): require(calibration_io.sha256(item['path'])==item['sha256'],'Source input changed')
    require(all(calibration_io.sha256(output/name)==digest for name,digest in frozen.items()),'Evaluation modified frozen output')
    history.write_csv(output/'comparison.csv',[{'variant':'appearance_controlled',**baseline_metrics},{'variant':'geometry_controlled',**metrics}])
    history.write_csv(output/'by_frame.csv',by_frame)
    (output/'merge_diagnostics.json').write_text(json.dumps(merge_diagnostics,indent=2,allow_nan=False)+'\n')
    (output/'identity_matching.json').write_text(json.dumps({'baseline':base_mapping,'geometry':mapping},indent=2)+'\n')
    report={'completed':True,'experiment_id':experiment_id,'source_run_id':run,'identity_scope':scope,
        'protocol':{'name':'scene_001_geometry_identity_paired_v1','association_policy':geometry.POLICY,
            'identity_policy':controlled_merge.POLICY,'appearance_variant':variant,'appearance_threshold':tau,
            'geometry_configuration':config,'coordinate_space':space,'identity_configuration':conf,
            'frames':[2,299],'cameras':list(CAMERAS),'fps':30,'runtime_gt_used':False,'past_assignments_rewritten':False,
            'role':'In-sample integration candidate; not independent threshold calibration','selected_deployment_threshold':None},
        'inputs':inputs,'baseline_metrics':baseline_metrics,'geometry_metrics':metrics,
        'delta_idf1_pp':100*(metrics['idf1']-baseline_metrics['idf1']), 'lifecycle':life.summary(),
        'link_metrics':{k:dict(v) for k,v in edge_metrics.items()},'geometry_candidate_decisions':dict(gate_counts),
        'geometry_decisions_above_appearance_threshold':dict(eligible_gate_counts),'unavailable_observations':unavailable,
        'merge_decisions':dict(merge_counts),'candidate_resets':dict(reset_counts),
        'merge_event_diagnostics':dict(merge_categories),'merge_examples':merge_diagnostics[:10],
        'excluded_zero_area_gt':excluded,'fully_outside_predictions':outside,
        'checks':{'regenerated_appearance_groups_match':True,'full_controlled_baseline_trace_matches':True,
                  'controlled_baseline_lifecycle_matches':True,'controlled_baseline_metrics_match':True,
                  'same_observation_rows_and_boxes':True,'frozen_output_unchanged_after_evaluation':True},
        'versions':{name:version(name) for name in ('numpy','scipy')},
        'code_sha256':{Path(m.__file__).name:calibration_io.sha256(Path(m.__file__)) for m in
            (calibration_io,history,history.history_io,history.snapshot,evaluation,pair_io,replay,merge_io,
             geometry,pairwise,grouping,controlled_merge,global_identity)},
        'artifacts':{p.name:{'path':p.name,'sha256':calibration_io.sha256(p)} for p in sorted(output.iterdir())},
        'limits':['Short reused training fragment and distance candidate chosen after inspecting it',
                  'Raw bottom-center projection does not detect occluded feet; coordinate units are native',
                  'Changed assignment may introduce new links; this is not post-filtering fixed baseline pairs',
                  'No splitting or retroactive repair of mixed identities; no geometry veto of existing local continuity',
                  'Merge-event GT checks describe visible members at acceptance, not complete history',
                  'Not a throughput benchmark; all pair candidates and detailed traces are recorded']}
    report['code_sha256'][Path(__file__).name]=calibration_io.sha256(Path(__file__))
    path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Variant                 IDF1      IDP      IDR    IDTP   IDFP   IDFN')
    for name,m in [('Appearance + merge',baseline_metrics),('Geometry + merge',metrics)]:
        print(f"{name:23} {m['idf1']:7.2%} {m['idp']:8.2%} {m['idr']:8.2%} {m['idtp']:7} {m['idfp']:6} {m['idfn']:6}")
    print(f"Delta IDF1: {report['delta_idf1_pp']:+.2f} pp")
    print('Link diagnostics:',json.dumps(report['link_metrics']))
    print('Lifecycle:',json.dumps(report['lifecycle']))
    print('Accepted merge diagnostics:',json.dumps(dict(merge_categories)))
    print('Unavailable projected observations:',unavailable)
    print('Baseline trace/metrics, fixed boxes/rows and lifecycle: VERIFIED')
    print(f'Report: {path}')
    print('Geometry + controlled identity experiment: COMPLETED')


if __name__=='__main__':
    main()
