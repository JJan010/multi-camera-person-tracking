"""Download pinned scene annotations and audit a predetermined camera subset."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'nvidia/PhysicalAI-SmartSpaces'
REVISION = '2cbe9563cbe9f47f846e5c871ee994572bbbc60e'
NAMES = ('ground_truth.txt', 'calibration_2025_format.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hashes(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            sha.update(chunk); blob.update(chunk)
    return sha.hexdigest(), blob.hexdigest()


def audit(gt_path, calibration_path, cameras, frame_stop):
    calibration = json.loads(calibration_path.read_text())
    sizes = {}
    for sensor in calibration['sensors']:
        if sensor.get('type') != 'camera':
            continue
        camera = int(sensor['id'].rsplit('_',1)[1])
        require(camera not in sizes, 'Duplicate calibration camera')
        attrs = {a['name']: a['value'] for a in sensor['attributes']}
        width, height, fps = int(attrs['frameWidth']), int(attrs['frameHeight']), Fraction(attrs['fps'])
        require(width > 0 and height > 0 and fps > 0, 'Invalid camera metadata')
        sizes[camera] = {'width':width,'height':height,'fps':str(fps)}
    require(set(cameras) <= set(sizes), 'Requested cameras missing from calibration')
    print('Loading GT for schema/count/overlap audit; no model evaluation...', flush=True)
    gt = np.loadtxt(gt_path,dtype=np.float64,ndmin=2)
    require(len(gt)>0 and gt.shape[1]==9 and np.isfinite(gt).all(), 'Expected finite nonempty nine-column GT')
    require(np.all(gt[:,:3]>=0) and np.all(gt[:,:3]<2**63) and np.array_equal(gt[:,:3],np.floor(gt[:,:3])),
            'Camera/person/frame keys must be nonnegative integers')
    keys = gt[:,:3].astype(np.int64)
    require(set(np.unique(keys[:,0])) <= set(sizes), 'GT camera missing from calibration')
    print('First five GT rows:')
    for row in gt[:5]:
        print(' '.join(f'{v:g}' for v in row))
    counts, windows, observations = [], [], {}
    for camera, metadata in sorted(sizes.items()):
        mask = keys[:,0]==camera
        rows, local = gt[mask], keys[mask]
        width,height = metadata['width'],metadata['height']
        x,y,w,h=rows[:,3:7].T
        duplicate = len(local)-len(np.unique(local[:,1:3],axis=0))
        bad_boxes=int(np.sum((w<=0)|(h<=0)))
        wholly_outside=(x+w<=0)|(y+h<=0)|(x>=width)|(y>=height)
        item={'camera':camera,**metadata,'rows':len(rows),'persons':len(np.unique(local[:,1])),
              'annotated_frames':len(np.unique(local[:,2])),
              'frame_min':int(local[:,2].min()) if len(rows) else None,
              'frame_max':int(local[:,2].max()) if len(rows) else None,
              'duplicate_keys':duplicate,'nonpositive_boxes':bad_boxes,
              'outside_image':int(np.sum((x<0)|(y<0)|(x+w>width)|(y+h>height))),
              'fully_outside':int(wholly_outside.sum())}
        counts.append(item); print(json.dumps(item))
        require(duplicate==0 and bad_boxes==0, 'Invalid GT boxes or duplicate keys; stop before selection')
        if camera in cameras:
            use=(local[:,2]>=0)&(local[:,2]<frame_stop)&~wholly_outside
            chosen=local[use]
            observations[camera]=set(map(tuple,chosen[:,1:3].tolist()))
            windows.append({'camera':camera,'frame_range':[0,frame_stop-1],
                            'usable_observations':len(chosen),'persons':len(np.unique(chosen[:,1])),
                            'frames_with_usable_gt':len(np.unique(chosen[:,2]))})
    pairs=[]
    for a,b in combinations(cameras,2):
        common=observations[a]&observations[b]
        pairs.append({'camera_a':a,'camera_b':b,'same_frame_observations':len(common),
                      'same_frame_persons':len({p for p,f in common}),
                      'same_frame_count':len({f for p,f in common})})
    print('\nPredetermined camera subset; GT window [0, frame_stop):')
    for row in windows: print(json.dumps(row))
    print('Pair coverage (positive-area intersection with image; not verified visual visibility):')
    for row in pairs: print(json.dumps(row))
    reached={cameras[0]}
    while True:
        before=set(reached)
        for pair in pairs:
            if pair['same_frame_observations']>0 and ({pair['camera_a'],pair['camera_b']}&reached):
                reached.update((pair['camera_a'],pair['camera_b']))
        if reached==before: break
    return {'total_rows':len(gt),'per_camera':counts,'window':windows,'pairs':pairs,
            'positive_overlap_graph_connected':reached==set(cameras),
            'limits':['GT overlap is dataset coverage, not measured tracking performance.',
                      'No visible-person count, timestamp alignment, projection accuracy or video duration established.',
                      'Absent GT rows do not alone distinguish an empty scene from missing annotations.',
                      'Window is a frame-index range, not yet a verified two-minute video interval.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-report',type=Path,required=True)
    parser.add_argument('--camera-ids',type=int,nargs=3,required=True)
    parser.add_argument('--frame-stop',type=int,default=3600)
    args=parser.parse_args()
    require(len(set(args.camera_ids))==3 and min(args.camera_ids)>=0 and args.frame_stop>0,'Invalid audit settings')
    catalog_path=args.catalog_report.resolve()
    catalog_hash=hashes(catalog_path)[0]
    catalog=json.loads(catalog_path.read_text())
    require(catalog['dataset']==DATASET and catalog['revision']==REVISION,'Unexpected dataset revision')
    scene=catalog['candidate_scene']; remote=PurePosixPath(scene)
    require(not remote.is_absolute() and '..' not in remote.parts and len(remote.parts)==3
            and remote.parts[:2]==('MTMC_Tracking_2024','val') and remote.name.startswith('scene_'), 'Invalid candidate scene')
    scene_folders=sorted(x['path'] for x in catalog['catalog']['MTMC_Tracking_2024/val']
                         if x['type']=='directory' and PurePosixPath(x['path']).name.startswith('scene_'))
    require(scene_folders and scene==scene_folders[0], 'Candidate was not the first catalog scene')
    available={int(PurePosixPath(x['path']).name.split('_')[1]) for x in catalog['catalog'][scene]
               if x['type']=='directory' and PurePosixPath(x['path']).name.startswith('camera_')}
    require(set(args.camera_ids)<=available, 'Unknown candidate cameras')
    from huggingface_hub import HfApi, hf_hub_download
    entries={entry.path:entry for entry in HfApi(token=False).list_repo_tree(
        repo_id=DATASET,repo_type='dataset',revision=REVISION,path_in_repo=scene,recursive=False)}
    selected=[entries[f'{scene}/{name}'] for name in NAMES]
    print(f'Scene: {scene}; candidate cameras: {args.camera_ids}; no automatic camera ranking')
    print(f'Annotation download size: {sum(e.size for e in selected)/1024**2:.2f} MiB; no video downloads',flush=True)
    files=[]
    for entry in selected:
        path=Path(hf_hub_download(repo_id=DATASET,repo_type='dataset',revision=REVISION,
                                 filename=entry.path,local_dir=ROOT/'data/physicalai_smartspaces',token=False)).resolve()
        require(path.is_relative_to((ROOT/'data/physicalai_smartspaces').resolve()),'Unexpected download location')
        require(path.stat().st_size==entry.size,'Downloaded size differs from catalog')
        sha,blob=hashes(path)
        lfs=getattr(entry,'lfs',None)
        if lfs is not None:
            expected=lfs.get('sha256',lfs.get('oid')) if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
            require(isinstance(expected,str) and sha==expected.removeprefix('sha256:'),'LFS SHA-256 mismatch/missing')
            verification='LFS SHA-256'
        else:
            require(blob==entry.blob_id,'Git blob checksum mismatch')
            verification='Git blob SHA-1'
        files.append({'remote_path':entry.path,'local_path':path.relative_to(ROOT.resolve()).as_posix(),
                      'size_bytes':entry.size,'sha256':sha,'source_verification':verification})
        print(f'{path.name}: size and {verification} VERIFIED',flush=True)
    manifest={'dataset':DATASET,'revision':REVISION,'license':'cc-by-4.0','scene':scene,
              'role':'validation_candidate','files':files}
    manifest_path=ROOT/'configs/datasets'/f'{remote.name}_source.json'
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    if manifest_path.exists():
        require(json.loads(manifest_path.read_text())==manifest,'Existing manifest differs; it was not overwritten')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Source manifest: {manifest_path}')
    paths={Path(f['remote_path']).name:ROOT/f['local_path'] for f in files}
    result=audit(paths[NAMES[0]],paths[NAMES[1]],sorted(args.camera_ids),args.frame_stop)
    require(hashes(catalog_path)[0]==catalog_hash,'Catalog changed during audit')
    for f in files:
        require(hashes(ROOT/f['local_path'])[0]==f['sha256'],'Annotation changed during audit')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'artifacts/validation_annotations'/run;output.mkdir(parents=True,exist_ok=False)
    report={'completed':True,'protocol':'validation_annotation_audit_v1','run_id':run,'scene':scene,
            'inputs':{'catalog_report':{'path':str(catalog_path),'sha256':catalog_hash},
                      'source_manifest':{'path':str(manifest_path.resolve()),'sha256':hashes(manifest_path)[0]}},
            'candidate_cameras':sorted(args.camera_ids),'frame_stop_exclusive':args.frame_stop,
            'camera_selection_final':False,'audit':result,'script_sha256':hashes(Path(__file__))[0]}
    path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Report: {path}')
    print('Annotation audit: COMPLETED; video synchronization and geometry validation pending')


if __name__=='__main__':
    main()
