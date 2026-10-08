"""Collect frozen RF-DETR/OSNet CUDA FP32 inputs for a configured scene."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np

from cache_detection_embeddings import encode_chunks
from mtmc.data.candidates import (cache_record, prepare_candidate_crops,
                                  validate_cache_round, validate_frame_batch)
from mtmc.data.scene import load_scene, require, sha256
from run_mtmc import EmbeddingArchive

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    x=np.asarray(values,np.float64)
    require(x.ndim==1 and len(x)>0 and np.isfinite(x).all() and np.all(x>=0),'Invalid timing samples')
    return {'mean_ms':float(x.mean()),'median_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95))}


def collect(scene, batches, detector, encoder, *, run_id, output, batch_size, warmup,
            synchronize, start_measurement=lambda:None):
    """Single-pass producer; injected models allow CPU contract testing.

    All candidate features are computed before any local tracking. No tracker,
    global manager, ground truth, crop-quality filter or confidence retuning.
    """
    require(type(warmup) is int and 0 <= warmup < scene.rounds,'Invalid timing warmup')
    archive=EmbeddingArchive(output/'embeddings.npy')
    samples=[]; counts=Counter(detections=0,encoded=0,fully_outside=0)
    counts.update({f'camera_{c}':0 for c in scene.camera_ids})
    measured_start=None
    try:
        with (output/'detections.jsonl').open('x') as saved, (output/'timings.jsonl').open('x') as timing_file:
            for frame in range(scene.rounds):
                if frame==warmup:
                    synchronize();start_measurement();measured_start=perf_counter_ns()
                t0=perf_counter_ns()
                batch=next(batches)
                validate_frame_batch(scene,batch,expected_frame=frame)
                t1=perf_counter_ns()
                detections=detector.detect(batch);synchronize()
                t2=perf_counter_ns()
                offset=archive.rows
                crops,records=prepare_candidate_crops(scene,batch,detections,row_offset=offset)
                t3=perf_counter_ns()
                vectors=encode_chunks(encoder,crops,batch_size)
                t4=perf_counter_ns()
                archive.append(vectors)
                saved.write(json.dumps(cache_record(run_id,scene,batch,records),allow_nan=False)+'\n')
                counts['detections']+=len(records);counts['encoded']+=len(crops)
                counts['fully_outside']+=len(records)-len(crops)
                for item in records:counts[f"camera_{item['camera_id']}"]+=1
                t5=perf_counter_ns()
                sample={'frame_index':frame,'measured':frame>=warmup,'candidates':len(records),'crops':len(crops),
                    'replay_ms':(t1-t0)/1e6,'detect_ms':(t2-t1)/1e6,'crop_ms':(t3-t2)/1e6,
                    'encode_ms':(t4-t3)/1e6,'record_ms':(t5-t4)/1e6,'core_ms':(t4-t0)/1e6,
                    'core_and_record_ms':(t5-t0)/1e6}
                timing_file.write(json.dumps(sample,allow_nan=False)+'\n');samples.append(sample)
                if (frame+1)%60==0 or frame+1==scene.rounds:
                    print(f'Processed {frame+1}/{scene.rounds}; candidates={counts["detections"]}; embeddings={archive.rows}',flush=True)
                del batch,detections,crops,records,vectors
            elapsed=(perf_counter_ns()-measured_start)/1e9
        final_start=perf_counter_ns();archive.finalize();final_ms=(perf_counter_ns()-final_start)/1e6
    finally:
        archive.close()
    require(archive.rows==counts['encoded'] and counts['detections']==counts['encoded']+counts['fully_outside'],
            'Producer count mismatch')
    # Read back the actual files. A matching in-memory count alone is insufficient.
    print('Verifying persisted candidate keys, crop geometry and every embedding row...',flush=True)
    vectors=np.load(output/'embeddings.npy',mmap_mode='r',allow_pickle=False)
    require(vectors.shape==(archive.rows,512),'Final embedding shape differs')
    checked=Counter(detections=0,encoded=0,fully_outside=0)
    checked.update({f'camera_{c}':0 for c in scene.camera_ids})
    row=0
    with (output/'detections.jsonl').open() as trace:
        for frame in range(scene.rounds):
            text=trace.readline();require(bool(text),'Truncated candidate archive')
            record=json.loads(text)
            row,missing,per_camera=validate_cache_round(record,scene,vectors,run_id=run_id,frame=frame,row_offset=row)
            checked['fully_outside']+=missing
            for c,n in per_camera.items():checked[f'camera_{c}']+=n;checked['detections']+=n
        require(trace.readline()=='','Trailing candidate frames')
    checked['encoded']=row
    require(dict(checked)==dict(counts),'Persisted candidate accounting differs')
    measured=[s for s in samples if s['measured']]
    phases=[k for k in measured[0] if k.endswith('_ms')]
    summary={'rounds':scene.rounds,'images':scene.rounds*len(scene.cameras),**dict(counts),
        'warmup_rounds':warmup,'measured_rounds':len(measured),'measured_frame_range':[warmup,scene.rounds-1],
        'measured_elapsed_s':elapsed,'rounds_per_second':len(measured)/elapsed,
        'images_per_second':len(measured)*len(scene.cameras)/elapsed,
        'crops_per_second':sum(s['crops'] for s in measured)/elapsed,
        'timings':{p:stats([s[p] for s in measured]) for p in phases},'embedding_finalization_ms':final_ms}
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-config',type=Path,required=True)
    parser.add_argument('--parity-report',type=Path,required=True)
    args=parser.parse_args();inputs={}
    def checked(name,path,expected=None):
        path=Path(path).resolve();digest=sha256(path)
        require(expected is None or digest==expected,f'Checksum mismatch: {path}')
        inputs[name]={'path':str(path),'sha256':digest};return path

    print('Verifying scene, passed paired-runtime gate and frozen model setup...',flush=True)
    gate_path=checked('paired_runtime_parity',args.parity_report)
    gate=json.loads(gate_path.read_text())
    require(gate.get('completed') is True and gate.get('passed') is True
            and gate['protocol']=='paired_scene_runtime_parity_v1'
            and all(gate['checks'].get(k) is True for k in
                    ('both_local_traces_exact','both_global_records_exact_except_run_scope',
                     'refinement_decisions_and_counts_exact','candidate_provenance_and_coverage','lifecycle_exact')),
            'Expected passed paired-runtime parity')
    for name in ('pipeline_report','cache_report','frozen_policy'):
        spec=gate['inputs'][name];checked(name,spec['path'],spec['sha256'])
    policy=json.loads(Path(inputs['frozen_policy']['path']).read_text())
    require(policy==gate['configuration'],'Policy differs from passed parity gate')
    for rel,digest in gate['code_sha256'].items():checked('code:'+rel,ROOT/rel,digest)
    source=json.loads(Path(inputs['pipeline_report']['path']).read_text())
    cache=json.loads(Path(inputs['cache_report']['path']).read_text())
    cfg=source['configuration'];cache_cfg=cache['configuration']
    require(source.get('completed') is True and source['protocol']=='scene_001_mtmc_sequential_fp32_v1'
            and cache.get('completed') is True and cache['protocol']=='frozen_detector_candidate_embeddings_v1'
            and cache['source_run_id']==source['run_id'],'Model/cache lineage differs')
    require(cfg['precision']=='FP32' and cfg['tf32'] is False and cfg['cudnn_benchmark'] is False
            and cfg['detector_threshold']==.1 and cfg['decoder_threads_per_camera']==1
            and cache_cfg['dtype']=='float32' and cache_cfg['feature_dim']==512,'Unsupported frozen inference setup')
    batch_size=cache_cfg['batch_size'];threads=cache_cfg['torch_threads'];warmup=source['summary']['warmup_rounds']
    require(type(batch_size) is int and batch_size>0 and type(threads) is int and threads>0
            and threads==cfg['torch_threads'],'Invalid/mixed frozen batching/thread settings')
    loaded=load_scene(args.scene_config,project_root=ROOT);scene=loaded.runtime
    require(scene.fps==cfg['fps']==policy['local']['tracker']['frame_rate']
            and 0<=warmup<scene.rounds,'Scene timing incompatible with frozen setup')
    for name,ref in (('scene_config',loaded.configuration),('source_manifest',loaded.source_manifest),
                     ('video_manifest',loaded.video_manifest),('calibration',scene.calibration)):
        checked(name,ref.path,ref.sha256)
    for c in scene.cameras:checked(f'video_{c.camera_id}',c.video.path,c.video.sha256)
    for name in ('detector_weights','osnet_configuration'):
        spec=source['inputs'][name];checked(name,spec['path'],spec['sha256'])
    osnet_config=Path(inputs['osnet_configuration']['path']);osnet=json.loads(osnet_config.read_text())
    require(osnet==cfg['osnet'],'OSNet setup differs from source pipeline')
    for name,spec in osnet['assets'].items():checked('osnet_asset:'+name,ROOT/spec['path'],spec['sha256'])
    frozen_code=('src/mtmc/detection/rfdetr.py','src/mtmc/reid/osnet.py','src/mtmc/reid/crops.py',
                 'src/mtmc/video/reader.py','src/mtmc/video/replay.py')
    for rel in frozen_code:checked('code:'+rel,ROOT/rel,source['code_sha256'][rel])
    for rel in ('scripts/cache_detection_embeddings.py','scripts/run_mtmc.py'):
        checked('code:'+rel,ROOT/rel,cache['code_sha256'][rel])
    for name,expected in source['runtime']['versions'].items():
        require(version(name)==expected,f'Frozen package version differs: {name}')
    os.environ['RF_HOME']=str(ROOT/'artifacts/models/rfdetr')
    os.environ['HF_HOME']=str(ROOT/'artifacts/models/huggingface')
    import torch
    import av
    from mtmc.detection.rfdetr import RFDETRPersonDetector
    from mtmc.reid.osnet import OSNetEncoder
    from mtmc.video.replay import synchronized_replay
    require(torch.cuda.is_available(),'CUDA is required')
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    print('Loading persistent RF-DETR Small and OSNet on CUDA, FP32...',flush=True)
    t0=perf_counter_ns()
    detector=RFDETRPersonDetector(Path(inputs['detector_weights']['path']),threshold=cfg['detector_threshold'])
    encoder=OSNetEncoder(osnet_config,project_root=ROOT);torch.cuda.synchronize()
    initialization_ms=(perf_counter_ns()-t0)/1e6
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'artifacts/scene_candidates'/run;output.mkdir(parents=True,exist_ok=False)
    status=output/'run_status.json';status.write_text(json.dumps({'completed':False,'run_id':run})+'\n')
    code=[Path(__file__),ROOT/'src/mtmc/data/candidates.py',ROOT/'src/mtmc/data/scene.py']
    for p in code:checked('code:'+p.relative_to(ROOT).as_posix(),p)
    try:
        with synchronized_replay(scene.sources,fps=scene.fps,threads=cfg['decoder_threads_per_camera']) as batches:
            summary=collect(scene,batches,detector,encoder,run_id=run,output=output,
                batch_size=batch_size,warmup=warmup,synchronize=torch.cuda.synchronize,
                start_measurement=torch.cuda.reset_peak_memory_stats)
        summary.update(peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                       peak_torch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        for spec in inputs.values():require(sha256(spec['path'])==spec['sha256'],'Input changed during collection')
        report={'completed':True,'protocol':'scene_candidate_collection_fp32_v1','run_id':run,
            'scene':scene.scene,'dataset':scene.dataset,'revision':scene.revision,'inputs':inputs,
            'configuration':{'frames':[0,scene.rounds-1],'rounds':scene.rounds,'cameras':list(scene.camera_ids),
                'fps':scene.fps,'image_sizes':{str(c.camera_id):[c.width,c.height] for c in scene.cameras},
                'detector_threshold':cfg['detector_threshold'],'batch_size':batch_size,'torch_threads':threads,
                'decoder_threads_per_camera':cfg['decoder_threads_per_camera'],'dtype':'float32','feature_dim':512,
                'tf32':False,'cudnn_benchmark':False,'ground_truth_used':False,
                'candidate_key':['cache_run_id','frame_index','camera_id','detection_index'],
                'sampling':'All person detections returned by the frozen detector; no crop-quality gate'},
            'summary':summary,'initialization_ms':initialization_ms,
            'runtime':{'python':platform.python_version(),'platform':platform.platform(),
                'gpu':torch.cuda.get_device_name(0),'cuda_build':torch.version.cuda,'ffmpeg_libraries':av.library_versions,
                'versions':{n:version(n) for n in (*source['runtime']['versions'],'transformers')},
                'torch_interop_threads':torch.get_num_interop_threads()},
            'checks':{'persisted_candidates_and_embeddings_verified':True,'all_frames_recorded':True,
                      'frozen_models_settings_and_code_verified':True},
            'artifacts':{n:{'path':n,'sha256':sha256(output/n)} for n in ('detections.jsonl','embeddings.npy','timings.jsonl')},
            'timing_protocol':{
                'clock':'Host perf_counter_ns; detector synchronized; OSNet returns CPU-ready vectors',
                'warmup':'First rounds still recorded; only their timings excluded from summaries',
                'encode_ms':'CPU preprocessing, transfers, OSNet CUDA forward and normalized CPU results',
                'record_ms':'Candidate JSON, buffered embedding writes and counters; excludes timing-log write',
                'throughput':'Warm collection loop including timing logs and progress, excluding final close/finalization/audit',
                'memory':'PyTorch allocator peaks after warmup reset; reserved is not total process VRAM',
                'cache':'Filesystem cache uncontrolled; source videos hashed before collection'},
            'limits':['Candidate collection only; no tracker, global identities or quality evaluation.',
                'Decode/RGB/crops/preprocessing on CPU; RF-DETR and OSNet inference on CUDA FP32.',
                'No throughput comparison to the previous full tracking loop: workload differs.',
                'Candidate index is frame-local, not a person ID; all duplicates and weak candidates are preserved.',
                'No identical-crop training-scene reference exists for this new scene; only contract/integrity checked.',
                'No independent-scene threshold tuning or model fine-tuning.']}
        path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps({'completed':True,'run_id':run})+'\n')
    except Exception as error:
        status.write_text(json.dumps({'completed':False,'run_id':run,'error_type':type(error).__name__,'error':str(error)})+'\n')
        raise
    print('Candidate counts:',json.dumps({k:summary[k] for k in ('rounds','images','detections','encoded','fully_outside')}))
    print('Per-camera candidates:',json.dumps({c:summary[f'camera_{c}'] for c in scene.camera_ids}))
    print(f'Report: {path}')
    print('Scene candidate collection: COMPLETED; persisted mapping VERIFIED; tracking/evaluation pending')


if __name__=='__main__':main()
