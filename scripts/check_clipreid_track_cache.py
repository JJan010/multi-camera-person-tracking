"""Known-answer streaming cache/provenance checks; no models, videos or GT."""
from copy import deepcopy
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from mtmc.data.scene import RuntimeScene, CameraInput, FileRef
from compare_reid_temporal import selected_crops
from cache_clipreid_tracks import collect, inventory, temporal_reference, verify_persisted


def rejected(call):
    try:
        call()
    except (ValueError, TypeError, StopIteration):
        return
    raise AssertionError('Malformed input accepted')


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); dummy = FileRef(root/'never_open', '0'*64)
        scene = RuntimeScene('fixture', 'fixture', '0'*40, 30, 5,
            tuple(CameraInput(c, dummy, w, 60, np.eye(3)) for c,w in ((0,80),(17,90))), dummy, 'fixture')
        rows = []; batches = []; source_offset = 0
        for frame in range(scene.rounds):
            cameras = []; packets = []
            for spec in scene.cameras:
                rgb = np.zeros((60,spec.width,3),np.uint8); rgb[10:30,:10,0] = 1; rgb[10:30,30:40,0] = 2
                packets.append(SimpleNamespace(camera_id=spec.camera_id, frame_index=frame,
                                               timestamp=Fraction(frame,30), rgb=rgb))
                cam = dict(camera=spec.camera_id, local_ids=[], xyxy=[], confidence=[], detection_indices=[], embedding_rows=[])
                if frame != 2 and spec.camera_id == 0:
                    # Unsorted input, ID zero, clipped border, fully outside; both weak and strong retained.
                    for identity, box, score in ((4,[30,10,40,30],.8),(0,[-2,10,10,30],.15),(9,[100,10,110,30],.7)):
                        index = source_offset if identity != 9 else None
                        for key,value in (('local_ids',identity),('xyxy',box),('confidence',score),('detection_indices',identity),('embedding_rows',index)):
                            cam[key].append(value)
                        source_offset += identity != 9
                cameras.append(cam)
            rows.append(dict(run_id='fixture',frame_index=frame,timestamp=str(Fraction(frame,30)),variants=dict(enabled=dict(cameras=cameras))))
            batches.append(SimpleNamespace(frame_index=frame,timestamp=Fraction(frame,30),frames=tuple(packets)))
        trace = root/'source.gz'
        def write_trace(values):
            with gzip.open(trace,'wt') as stream:
                for row in values: stream.write(json.dumps(row)+'\n')
        write_trace(rows); calls = []
        def encode(crops):
            assert crops and all(not c.rgb.flags.writeable for c in crops)
            calls.append(crops[0].key.frame_index)
            matrix = np.zeros((len(crops),1280),np.float32)
            for i,crop in enumerate(crops): matrix[i,int(crop.rgb[0,0,0])] = 1
            return tuple(c.key for c in crops), tuple(c.timestamp for c in crops), matrix
        samples = [0,2,4]; old_records = []; vectors = []
        for frame in samples:
            records,crops = selected_crops(scene,batches[frame],rows[frame],sum(len(v) for v in vectors))
            old_records.extend(json.loads(json.dumps(records)))
            if crops: vectors.append(encode(crops)[2])
        old = np.concatenate(vectors); reference = temporal_reference(old_records,old,samples)
        expected = inventory(scene,trace,'fixture',source_offset)
        assert expected['encoded']==8 and expected['observations']==12 and expected['fully_outside']==4
        calls.clear(); out=root/'good'; out.mkdir()
        summary = collect(scene,iter(batches),trace,'fixture',source_offset,encode,reference,old,samples,out,expected)
        assert calls==[0,1,3,4] and summary['temporal_reference_parity']['compared']==4
        assert summary['temporal_reference_parity']['max_absolute_error']==0
        with gzip.open(out/'observations.jsonl.gz','rt') as stream: saved=[json.loads(line) for line in stream]
        assert saved[2]['observations']==[]
        assert saved[0]['observations'][0]['local_id']==0 and saved[0]['observations'][0]['confidence']==.15
        assert saved[0]['observations'][0]['crop_xyxy_int']==[0,10,10,30]
        matrix=np.load(out/'clipreid_embeddings.npy'); assert matrix.shape==(8,1280)
        print('Full-rate streaming, original keys, arbitrary cameras, ID zero, weak/border crops, empty rounds and outside null rows: OK')
        print('Temporal pixel/vector parity and persisted source-to-vector mapping: OK')
        def bad_keys(crops):
            keys,times,features=encode(crops); return keys[::-1],times,features
        def bad_time(crops):
            keys,times,features=encode(crops); return keys,tuple(t+1 for t in times),features
        def bad_vector(crops):
            keys,times,features=encode(crops); features[0,0]=np.nan; return keys,times,features
        for i,encoder in enumerate((bad_keys,bad_time,bad_vector)):
            target=root/f'bad_{i}';target.mkdir()
            rejected(lambda:collect(scene,iter(batches),trace,'fixture',source_offset,encoder,reference,old,samples,target,expected))
        for field in ('rgb_sha256','source_embedding_row'):
            bad=deepcopy(reference); next(iter(bad.values()))[field]='changed'
            target=root/field;target.mkdir()
            rejected(lambda:collect(scene,iter(batches),trace,'fixture',source_offset,encode,bad,old,samples,target,expected))
        changed=np.roll(old,10,axis=1); target=root/'parity';target.mkdir()
        rejected(lambda:collect(scene,iter(batches),trace,'fixture',source_offset,encode,reference,changed,samples,target,expected))
        corrupted=matrix.copy();corrupted[0]=0;np.save(out/'clipreid_embeddings.npy',corrupted)
        rejected(lambda:verify_persisted(scene,trace,'fixture',source_offset,out,expected))
        np.save(out/'clipreid_embeddings.npy',matrix)
        saved[0]['observations'][0]['source_embedding_row']=999
        with gzip.open(out/'observations.jsonl.gz','wt') as stream:
            for row in saved:stream.write(json.dumps(row)+'\n')
        rejected(lambda:verify_persisted(scene,trace,'fixture',source_offset,out,expected))
        print('Changed encoder keys/times/vectors, crop pixels, source rows and persisted artifacts rejected: OK')
        bad=deepcopy(rows);bad[1]['variants']['enabled']['cameras'][0]['embedding_rows'][0]=0;write_trace(bad)
        rejected(lambda:inventory(scene,trace,'fixture',source_offset))
        write_trace(rows[:-1]);rejected(lambda:inventory(scene,trace,'fixture',source_offset))
        write_trace(rows+[rows[-1]]);rejected(lambda:inventory(scene,trace,'fixture',source_offset))
        write_trace(rows)
        # Valid all-empty collection must create a loadable (0,1280) matrix without invoking encoder.
        empty=deepcopy(rows)
        for row in empty:
            for cam in row['variants']['enabled']['cameras']:
                for key in ('local_ids','xyxy','confidence','detection_indices','embedding_rows'):cam[key]=[]
        write_trace(empty); expected_empty=inventory(scene,trace,'fixture',0); target=root/'empty';target.mkdir(); calls.clear()
        collect(scene,iter(batches),trace,'fixture',0,encode,{},np.empty((0,1280),np.float32),samples,target,expected_empty)
        assert not calls and np.load(target/'clipreid_embeddings.npy').shape==(0,1280)
        print('Duplicate source rows, truncated/trailing traces rejected; all-empty cache has no model calls: OK')
    print('CLIP-ReID track cache contract: PASSED; synthetic fixtures do not measure model quality')


if __name__=='__main__':main()
