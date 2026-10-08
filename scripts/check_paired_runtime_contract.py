"""Small integration checks for candidate provenance and paired state ownership."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np

from mtmc.data.scene import CameraInput, FileRef, RuntimeScene
from mtmc.pipeline.paired import CameraCandidates, CandidateRound, PairedAssociation, VARIANTS


def policy():
    # Synthetic fixtures, not selection of validation/deployment settings.
    return {'schema_version':1,'variants':list(VARIANTS),
        'local':{'tracker':{'track_activation_threshold':.5,'lost_track_buffer':30,
                           'minimum_matching_threshold':.8,'frame_rate':30,'minimum_consecutive_frames':1},
                 'appearance_threshold':.6,'strong_history_size':8,
                 'strong_history_max_age_frames':30,'weak_motion_min_iou':.5},
        'global':{'history':{'max_observations':8,'max_age_seconds':'1'},
                  'appearance_variant':'mean','appearance_threshold':.7,
                  'geometry':{'max_distance':2.,'unavailable_policy':'appearance_only'},
                  'identity':{'max_idle_seconds':'1','min_support_rounds':3,
                              'min_support_seconds':'1/5','max_evidence_gap':'1/10'}}}


def scene():
    # These files deliberately do not exist: this runtime must not read them.
    unused = FileRef(Path('/nonexistent/paired-runtime-fixture'), '0'*64)
    return RuntimeScene('synthetic', 'synthetic', '0'*40, 30, 100,
        tuple(CameraInput(c,unused,w,h,np.eye(3)) for c,w,h in ((0,40,60),(11,80,100),(27,120,200))),
        unused, 'synthetic/Z0')


def batch(frame, start_row):
    a = np.zeros(512,np.float32); a[0]=1
    b = np.zeros(512,np.float32); b[1]=1
    # Camera 0 empty; camera 27 includes a fully outside candidate. Candidate
    # order deliberately differs from spatial order, and two appearance vectors differ.
    cameras = (
        CameraCandidates(0,np.empty((0,4),np.float32),np.empty(0,np.float32),(),()),
        CameraCandidates(11,np.array([[45,10,65,60],[10,10,30,60]],np.float32),
                         np.array([.9,.8],np.float32),(start_row,start_row+1),(b,a)),
        CameraCandidates(27,np.array([[10,10,30,60],[130,20,150,80]],np.float32),
                         np.array([.8,.9],np.float32),(start_row+2,None),(a,None)))
    return CandidateRound('candidates',frame,Fraction(frame,30),cameras)


def make():
    return PairedAssociation(scene(),policy(),run_id='synthetic',source_run_id='candidates')


def reject(fn):
    try:
        fn()
    except (ValueError,TypeError,RuntimeError):
        return
    raise AssertionError('Expected rejection')


def main():
    first, reordered = make(), make()
    for frame in range(3):
        current = batch(frame,frame*3)
        with patch('builtins.open',side_effect=AssertionError('Runtime attempted file access')):
            actual = first.step(current)
            other = reordered.step(replace(current,cameras=tuple(reversed(current.cameras))))
        assert actual == other
        for name in VARIANTS:
            cams = actual['variants'][name]['cameras']
            assert cams[0]['local_ids'] == []
            assert cams[1]['detection_indices'] == [0,1] and cams[1]['embedding_rows'] == [frame*3,frame*3+1]
            assert cams[2]['embedding_rows'] == [frame*3+2,None]
            assignments=actual['variants'][name]['identity']['assignments']
            by_key={(x['key']['camera_id'],x['key']['local_id']):x['global_id'] for x in assignments}
            assert len(by_key)==4 and by_key[11,2]==by_key[27,1]
            assert by_key[11,1]!=by_key[11,2] and by_key[27,2]!=by_key[27,1]
    assert first.next_embedding_row==9
    print('Arbitrary cameras/sizes, empty camera, outside candidate and exact feature-to-track mapping: OK')
    print('Camera input order preserves both variants; runtime performs no file/GT access: OK')

    empty=CandidateRound('candidates',3,Fraction(3,30),tuple(
        CameraCandidates(c,np.empty((0,4),np.float32),np.empty(0,np.float32),(),()) for c in (0,11,27)))
    out=first.step(empty)
    assert all(not out['variants'][n]['identity']['assignments'] for n in VARIANTS)
    assert first.next_frame==4 and first.next_embedding_row==9
    print('Empty rounds advance scene time without fabricating observations or embeddings: OK')

    runner=make(); runner.step(batch(0,0)); valid=batch(1,3)
    third=valid.cameras[2]
    bad_camera_cases=[replace(third,scores=third.scores.astype(np.float64)),
        replace(third,features=(np.zeros(512,np.float32),None)),
        replace(third,features=(None,None)), replace(third,embedding_rows=(0,None)),
        replace(third,embedding_rows=(5,6)),
        replace(third,boxes=np.array([[10,10,10,60],[130,20,150,80]],np.float32))]
    cases=[replace(valid,source_run_id='other'),replace(valid,frame_index=2),
           replace(valid,timestamp=1/30),replace(valid,cameras=valid.cameras[:-1]),
           replace(valid,cameras=(valid.cameras[0],valid.cameras[1],valid.cameras[1]))]
    cases += [replace(valid,cameras=(*valid.cameras[:2],c)) for c in bad_camera_cases]
    before=pickle.dumps(runner)
    for bad in cases:
        reject(lambda:runner.step(bad))
        assert pickle.dumps(runner)==before and not runner.failed
    runner.step(valid)
    print('All-camera preflight rejects malformed scope/time/rows/features before any state mutation: OK')

    config=policy(); isolated=PairedAssociation(scene(),config,run_id='synthetic',source_run_id='candidates')
    config['global']['appearance_threshold']=1.
    exported=isolated.policy; exported['local']['tracker']['lost_track_buffer']=100
    assert isolated.policy==policy()
    a=isolated.step(batch(0,0)); a['variants']['staged']['identity']['assignments'].clear()
    fresh=make(); fresh.step(batch(0,0))
    assert isolated.step(batch(1,3))==fresh.step(batch(1,3))
    reject(lambda:PairedAssociation(replace(scene(),fps=25),policy(),run_id='x',source_run_id='y'))
    reject(lambda:PairedAssociation(replace(scene(),cameras=(scene().cameras[0],)*2),policy(),run_id='x',source_run_id='y'))
    wrong=deepcopy(policy()); wrong['local']['strong_history_max_age_frames']=25
    reject(lambda:PairedAssociation(scene(),wrong,run_id='x',source_run_id='y'))
    print('Policy/output ownership and frozen 30 FPS/local-history requirements: OK')

    failed=make()
    with patch.object(failed.trackers['competitive_iou'][11],'update_candidates',side_effect=RuntimeError('Injected failure')):
        reject(lambda:failed.step(batch(0,0)))
    assert failed.failed and failed.trackers['staged'][11].frame_id==1
    reject(lambda:failed.step(batch(0,0)))
    print('An internal failure after partial work fails the paired run; retry is rejected: OK')
    print('Paired runtime contract: PASSED; fixtures do not measure tracking quality')


if __name__=='__main__':
    main()
