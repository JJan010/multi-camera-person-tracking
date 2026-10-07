"""Offline distance distributions and hypothetical rejection of frozen pairs."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import audit_scene_geometry as geometry
import evaluate_global_identity as evaluation
import replay_global_identity as replay
from mtmc.reid.osnet import ObservationKey

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (4, 5, 8)
POPULATIONS = ('all_pairs', 'appearance_links', 'grouped_links')
STRATA = ('all', 'both_boxes_inside', 'both_labels_mutually_unique')
CLASSES = ('same', 'different', 'unknown')
CAMERA_PAIRS = ('4-5', '4-8', '5-8', 'ALL')


def category(a, b):
    return 'unknown' if a is None or b is None else 'same' if a == b else 'different'


def project_box(h, box):
    b = np.asarray(box, dtype=np.float64)
    geometry.require(b.shape == (4,) and np.isfinite(b).all() and np.all(b[2:] > b[:2]), 'Invalid tracked box')
    foot = np.array([(b[0]+b[2])/2, b[3]])
    world = geometry.project(np.linalg.inv(h), foot)
    inside = bool(b[0]>=0 and b[1]>=0 and b[2]<=1920 and b[3]<=1080)
    return foot, world, inside


class Bucket:
    def __init__(self):
        self.counts = Counter({c:0 for c in CLASSES})
        self.distances = {c:[] for c in CLASSES}

    def add(self, label, distance):
        geometry.require(label in CLASSES, 'Invalid pair label')
        geometry.require(distance is None or np.isfinite(distance) and distance >= 0, 'Invalid pair distance')
        self.counts[label] += 1
        if distance is not None:
            self.distances[label].append(distance)

    def sweep(self, threshold):
        result = {'distance_threshold': threshold}
        for label in CLASSES:
            values = np.asarray(self.distances[label])
            retained = int(np.count_nonzero(values <= threshold))
            result.update({f'{label}_total':self.counts[label], f'{label}_measurable':len(values),
                           f'{label}_retained':retained, f'{label}_rejected':len(values)-retained,
                           f'{label}_deferred':self.counts[label]-len(values)})
            result[f'{label}_rejected_fraction_measurable'] = (len(values)-retained)/len(values) if len(values) else None
        return result


def distance_summary(values):
    return {'measurable':len(values), **dict(zip(('min','p05','median','p95','p99','max'),
            [float(x) for x in np.quantile(values,[0,.05,.5,.95,.99,1])] if values else [None]*6))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-report', type=Path, required=True)
    parser.add_argument('--identity-report', type=Path, required=True)
    parser.add_argument('--distances', type=float, nargs='+', required=True)
    args = parser.parse_args()
    thresholds = args.distances
    geometry.require(thresholds == sorted(set(thresholds)) and all(np.isfinite(t) and t > 0 for t in thresholds),
                     'Distances must be finite, positive, unique and increasing')
    inputs = {}
    def checked(name, path, expected=None):
        path = Path(path).resolve()
        digest = geometry.sha256(path)
        geometry.require(expected is None or digest == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path':str(path),'sha256':digest}
        return path
    print('Verifying frozen tracks, grouping inputs and calibration...', flush=True)
    apath = checked('geometry_report',args.geometry_report)
    ipath = checked('identity_report',args.identity_report)
    audit, identity = (json.loads(p.read_text()) for p in (apath,ipath))
    geometry.require(audit.get('completed') and audit['protocol']=='scene_001_geometry_audit_v1', 'Unexpected geometry audit')
    geometry.require(identity.get('completed') and identity['protocol']['name']=='scene_001_controlled_merge_paired_v1', 'Unexpected identity report')
    run = identity['source_run_id']
    variant, appearance_threshold = identity['protocol']['variant'], identity['protocol']['threshold']
    for name in ('tracks','ground_truth','groups','grouping_report'):
        entry = identity['inputs'][name]
        checked(name,entry['path'],entry['sha256'])
    entry = audit['inputs']['calibration_2025_format.json']
    calibration_path = checked('calibration',entry['path'],entry['sha256'])
    geometry.require(inputs['ground_truth']['sha256'] == audit['inputs']['ground_truth.txt']['sha256'], 'Different GT sources')
    grouping = json.loads(Path(inputs['grouping_report']['path']).read_text())
    geometry.require(grouping.get('completed') and grouping['source_run_id']==run and
                     grouping['protocol']['name']=='scene_001_multicamera_grouping_diagnostic_v1', 'Unexpected grouping source')
    geometry.require(inputs['groups']['sha256']==grouping['artifacts']['groups.jsonl.gz']['sha256'], 'Different grouping trace')
    selected = [r for r in grouping['results'] if (r['variant'],r['threshold']) == (variant,appearance_threshold)]
    geometry.require(len(selected)==1, 'Missing/duplicate source setting')
    expected = selected[0]
    calibration = json.loads(calibration_path.read_text())
    models = {}
    for camera in CAMERAS:
        sensors = [s for s in calibration['sensors'] if s['id'].lower()==f'camera_{camera:04d}']
        geometry.require(len(sensors)==1, 'Missing/duplicate camera calibration')
        s = sensors[0]
        k,e,p,h = [geometry.matrix(s,name,shape) for name,shape in (
            ('intrinsicMatrix',(3,3)),('extrinsicMatrix',(3,4)),('cameraMatrix',(3,4)),('homography',(3,3)))]
        geometry.require(np.linalg.matrix_rank(h)==3 and geometry.proportional_error(p,k@e)<=1e-6 and
                         geometry.proportional_error(h,p[:,[0,1,3]])<=1e-6, 'Unexpected matrix convention')
        models[camera] = h
    ground = evaluation.history.load_ground_truth(Path(inputs['ground_truth']['path']))
    trace_path = Path(inputs['tracks']['path'])
    slots, frame_maps, _, _ = evaluation.build_slots(trace_path,ground,run)
    with trace_path.open() as handle:
        trace = [json.loads(line) for line in handle]
    labeled, _ = evaluation.history.label_trace(trace,ground)
    labels = {ObservationKey(r['camera'],r['local_id'],r['frame_index']):r for r in labeled if r['frame_index']>=2}
    geometry.require(len(labels)==sum(len(m) for m in frame_maps.values()), 'Different label/box coverage')
    unique = set()
    for (_, _), (_, keys, mask) in slots.items():
        gdegree, pdegree = mask.sum(axis=1), mask.sum(axis=0)
        for i,j in zip(*np.nonzero(mask)):
            if gdegree[i]==pdegree[j]==1:
                unique.add(keys[j])
    observations = {}
    for record in trace:
        frame = record['frame_index']
        if frame < 2:
            continue
        observations[frame] = {}
        for camera_record in record['cameras']:
            camera = camera_record['camera']
            for local,box in zip(camera_record['local_ids'],camera_record['xyxy']):
                key = ObservationKey(camera,local,frame)
                foot,world,inside = project_box(models[camera],box)
                label = labels[key]
                observations[frame][key] = {'camera':camera,'local_id':local,'frame':frame,
                    'embedding_row':label['embedding_row'],'raw_xyxy':box,'foot_uv':foot.tolist(),
                    'world_xy':None if world is None else world.tolist(),'box_fully_inside':inside,
                    'diagnostic_gt_id':label['gt_id'],'mutually_unique_gt_match':key in unique}
    geometry.require(set(observations)==set(range(2,300)), 'Incomplete timeline')
    groups_by_frame = {}
    with gzip.open(inputs['groups']['path'],'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if (record['variant'],record['threshold']) != (variant,appearance_threshold):
                continue
            frame = record['frame_index']
            geometry.require(frame in observations and frame not in groups_by_frame, 'Unexpected/duplicate grouping frame')
            groups_by_frame[frame] = record
    geometry.require(set(groups_by_frame)==set(observations), 'Missing grouping frame')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'artifacts/geometry_pairs'/run_id
    output.mkdir(parents=True,exist_ok=False)
    buckets = {key:Bucket() for key in product(POPULATIONS,STRATA,CAMERA_PAIRS)}
    group_counts, edge_counts = Counter(), Counter()
    total_possible = 0
    target_pair = None
    fields = ['frame','camera_a','local_a','row_a','gt_a','camera_b','local_b','row_b','gt_b',
              'category','distance','both_boxes_inside','both_labels_mutually_unique','appearance_link','grouped_link','cosine_similarity']
    print(f'Frames 2..299; appearance source {variant}/{appearance_threshold}; no assignments will change.',flush=True)
    with gzip.open(output/'pairs.csv.gz','wt',newline='') as handle, gzip.open(output/'observations.jsonl.gz','wt') as outobs:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for frame in range(2,300):
            obs = observations[frame]
            saved = groups_by_frame[frame]
            runtime, source_labels = replay.decode_round(saved,run_id=run,frame=frame,variant=variant,
                                         threshold=appearance_threshold,fps=30,cameras=CAMERAS)
            geometry.require(set(source_labels)==set(obs), 'Source grouping/track observations differ')
            for key,item in obs.items():
                geometry.require(source_labels[key]=={'embedding_row':item['embedding_row'], 'diagnostic_gt_id':item['diagnostic_gt_id']},
                                 'Recomputed GT label/source row differs from frozen grouping')
                outobs.write(json.dumps(item,allow_nan=False)+'\n')
            count, edges = replay.source_counts(saved,runtime,source_labels)
            group_counts.update(count)
            edge_counts.update(edges)
            accepted = {replay.grouping.edge_key(ObservationKey(**d['left']),ObservationKey(**d['right'])):d['cosine_similarity']
                        for d in saved['decisions']}
            retained = {replay.grouping.edge_key(a,b) for members in runtime.groups for a,b in combinations(members,2)}
            used_accepted, used_retained = set(),set()
            by_camera = {c:sorted((k for k in obs if k.camera_id==c),key=lambda k:k.local_id) for c in CAMERAS}
            for a,b in combinations(CAMERAS,2):
                total_possible += len(by_camera[a])*len(by_camera[b])
                for ka,kb in product(by_camera[a],by_camera[b]):
                    left,right = obs[ka],obs[kb]
                    edge = replay.grouping.edge_key(ka,kb)
                    distance = (float(np.linalg.norm(np.asarray(left['world_xy'])-right['world_xy']))
                                if left['world_xy'] is not None and right['world_xy'] is not None else None)
                    label = category(left['diagnostic_gt_id'],right['diagnostic_gt_id'])
                    inside = left['box_fully_inside'] and right['box_fully_inside']
                    mutual = left['mutually_unique_gt_match'] and right['mutually_unique_gt_match']
                    populations = ['all_pairs']
                    if edge in accepted:
                        populations.append('appearance_links'); used_accepted.add(edge)
                    if edge in retained:
                        populations.append('grouped_links'); used_retained.add(edge)
                    strata = ['all'] + (['both_boxes_inside'] if inside else []) + (['both_labels_mutually_unique'] if mutual else [])
                    for pop,stratum,pair in product(populations,strata,(f'{a}-{b}','ALL')):
                        buckets[pop,stratum,pair].add(label,distance)
                    row = dict(zip(fields,[frame,a,ka.local_id,left['embedding_row'],left['diagnostic_gt_id'],
                        b,kb.local_id,right['embedding_row'],right['diagnostic_gt_id'],label,distance,inside,mutual,
                        edge in accepted,edge in retained,accepted.get(edge)]))
                    writer.writerow(row)
                    if frame==149 and (a,ka.local_id,b,kb.local_id)==(5,9,8,17):
                        target_pair = row
            geometry.require(used_accepted==set(accepted) and used_retained==retained,'Source link missing from pair universe')
            if (frame+1)%60==0:
                print(f'Processed through frame {frame}/299',flush=True)
    geometry.require(dict(group_counts)==expected['groups'], 'Source group counters differ')
    geometry.require(all(edge_counts[k]==expected['after'][k] for k in replay.grouping_eval.EDGE_FIELDS), 'Retained counters differ')
    geometry.require(buckets['all_pairs','all','ALL'].counts['same']==expected['before']['available_positive_pairs'], 'Positive-pair denominator differs')
    geometry.require(sum(buckets['all_pairs','all','ALL'].counts.values())==total_possible, 'Cartesian pair count mismatch')
    for population,where in [('appearance_links','before'),('grouped_links','after')]:
        counts = buckets[population,'all','ALL'].counts
        geometry.require((counts['same'],counts['different'],counts['unknown'])==tuple(expected[where][name] for name in
                         ('correct_links','wrong_known_links','unresolved_gt_links')), 'Source link-label counters differ')
    distributions,sweeps = [],[]
    for (population,stratum,camera_pair),bucket in buckets.items():
        context = {'population':population,'stratum':stratum,'camera_pair':camera_pair}
        for label in CLASSES:
            distributions.append({**context,'category':label,'total':bucket.counts[label],
                'deferred':bucket.counts[label]-len(bucket.distances[label]),**distance_summary(bucket.distances[label])})
        for threshold in thresholds:
            sweeps.append({**context,**bucket.sweep(threshold)})
    evaluation.history.write_csv(output/'distance_summary.csv',distributions)
    evaluation.history.write_csv(output/'threshold_sweep.csv',sweeps)
    for item in inputs.values():
        geometry.require(geometry.sha256(item['path'])==item['sha256'],'Input changed during diagnostic')
    report = {'completed':True,'run_id':run_id,'source_run_id':run,
        'protocol':{'name':'scene_001_geometry_pairs_v1','frames':[2,299],'cameras':list(CAMERAS),
            'appearance_variant':variant,'appearance_threshold':appearance_threshold,'distance_grid':thresholds,
            'selected_distance_threshold':None,'distance_rule':'finite distance <= threshold retains; > rejects; unavailable defers',
            'units':'native dataset coordinates, physical scale not independently verified',
            'labels':'Existing maximum-cardinality then maximum-IoU one-to-one matching, clipped boxes, IoU >= 0.5',
            'strata':'Overlapping diagnostic subsets; not runtime visibility gates'},
        'inputs':inputs,'observation_count':sum(len(x) for x in observations.values()),'pair_count':total_possible,
        'invalid_projection_count':sum(o['world_xy'] is None for v in observations.values() for o in v.values()),
        'population_counts':{p:dict(buckets[p,'all','ALL'].counts) for p in POPULATIONS},
        'distance_summary':distributions,'threshold_sweep':sweeps,'frame149_wrong_pair':target_pair,
        'checks':{'frozen_gt_labels_and_rows_reproduced':True,'positive_pair_denominator_reproduced':True,
                  'appearance_and_grouped_counts_reproduced':True,'cartesian_pair_count_verified':True},
        'code_sha256':{Path(m.__file__).name:geometry.sha256(Path(m.__file__)) for m in
                      (geometry,evaluation,evaluation.history,evaluation.history.snapshot,replay,replay.grouping)},
        'numpy_version':np.__version__,
        'limits':['Offline filtering of fixed pairs is not reassignment, grouping replay, identity replay or IDF1 evaluation',
                  'No deployment threshold; short reused training fragment with correlated frame observations',
                  'All-pair negatives include easy cases; use appearance-linked subsets to study actual mistakes',
                  'Inside-image flags do not establish visible feet; uniquely matched boxes do not establish good geometry',
                  'Unknown GT labels are not negative pairs; unavailable projection is not automatic rejection',
                  'No embeddings, detection, tracking or global IDs are recomputed'],
        'artifacts':{p.name:{'sha256':geometry.sha256(p)} for p in sorted(output.iterdir())}}
    report['code_sha256'][Path(__file__).name]=geometry.sha256(Path(__file__))
    path = output/'report.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Population counts:',json.dumps(report['population_counts']))
    print('Distance distributions: population / class / n / median / p95 / max')
    for row in distributions:
        if row['stratum']=='all' and row['camera_pair']=='ALL':
            print(f"{row['population']:17} {row['category']:9} {row['measurable']:7} " +
                  ' '.join('n/a' if row[k] is None else f'{row[k]:.3f}' for k in ('median','p95','max')))
    print('Hypothetical rejection among finite appearance-linked pairs (ALL cameras, all boxes):')
    print('Distance  Same retained/rejected  Different retained/rejected  Unknown rejected')
    for row in sweeps:
        if (row['population'],row['stratum'],row['camera_pair'])==('appearance_links','all','ALL'):
            print(f"{row['distance_threshold']:8g} {row['same_retained']:7}/{row['same_rejected']:<7} "
                  f"{row['different_retained']:9}/{row['different_rejected']:<9} {row['unknown_rejected']}")
    print('Invalid projections:',report['invalid_projection_count'])
    print('Frozen source labels, pair/link denominators and grouping counters: VERIFIED')
    print('Threshold selection: NONE; identity assignments: UNCHANGED')
    print(f'Report: {path}')
    print('Geometry pair diagnostic: COMPLETED')


if __name__=='__main__':
    main()
