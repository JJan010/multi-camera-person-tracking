"""Scene evidence adapter for the experimental dormant registry bridge."""
from fractions import Fraction
import numpy as np

from mtmc.association.dormant import DormantIdentityArchive
from mtmc.association.geometry import project_box_foot
from mtmc.association.recovery import RecoveryIdentityManager, RecoveryEvidence, ObservationEvidence
from mtmc.pipeline.core import IdentityStage, require
from mtmc.reid.history import HistoryBatch
from mtmc.reid.osnet import ObservationKey, ReIDBatch


class _EvidenceManager:
    def __init__(self,registry):
        self.registry=registry;self.evidence=None

    def update(self,frame):
        require(self.evidence is not None,'Missing current evidence context')
        return self.registry.update(frame,self.evidence)


class RecoveryIdentityStage(IdentityStage):
    def __init__(self,*args,recovery_enabled,archive_settings=None,
                 history_max_observations,history_max_age,**kwargs):
        super().__init__(*args,**kwargs)
        require(self.variant=='mean','This experiment requires mean appearance')
        require(type(history_max_observations) is int and history_max_observations>0
                and isinstance(history_max_age,Fraction) and history_max_age>0,'Invalid history contract')
        self.history_max_observations=history_max_observations;self.history_max_age=history_max_age
        self.registry=RecoveryIdentityManager(self.manager,coordinate_space=self.coordinate_space,
            camera_ids=self.cameras,enabled=recovery_enabled,archive_settings=archive_settings)
        self.manager=_EvidenceManager(self.registry)
        self.failed=False;self.last_evidence=None

    def evidence_from_history(self,frame,timestamp,history,records):
        require(type(frame) is int and frame>=0 and isinstance(timestamp,Fraction) and timestamp>=0,'Invalid round')
        require(isinstance(history,HistoryBatch) and history.run_id==self.run_id,'Mixed history scope')
        require(isinstance(history.latest,ReIDBatch) and isinstance(history.mean,ReIDBatch),'Invalid history batches')
        keys=history.mean.keys;n=len(keys)
        require(isinstance(keys,tuple) and len(set(keys))==n and history.latest.keys==keys
                and history.latest.timestamps==history.mean.timestamps==(timestamp,)*n,'History mapping differs')
        for array in (history.latest.embeddings,history.mean.embeddings):
            require(isinstance(array,np.ndarray) and array.dtype==np.float32 and array.shape==(n,512)
                    and np.isfinite(array).all() and np.allclose(np.linalg.norm(array,axis=1),1,rtol=0,atol=1e-5),
                    'Invalid normalized history arrays')
        require(all(len(column)==n for column in (history.source_frames,history.source_times,
                history.mean_norms_before_normalization,history.used_latest_fallback)),'History provenance columns differ')
        descriptors={};scene_times={frame:timestamp}
        for i,key in enumerate(keys):
            require(isinstance(key,ObservationKey) and all(type(x) is int and x>=0 for x in
                    (key.camera_id,key.local_id,key.frame_index)) and key.frame_index==frame
                    and key.camera_id in self.cameras,'Invalid history observation key')
            frames,times=history.source_frames[i],history.source_times[i]
            require(isinstance(frames,tuple) and isinstance(times,tuple)
                    and 1<=len(frames)==len(times)<=self.history_max_observations
                    and all(type(f) is int and 0<=f<=frame for f in frames)
                    and tuple(sorted(set(frames)))==frames and frames[-1]==frame
                    and all(isinstance(t,Fraction) and 0<=t<=timestamp and timestamp-t<=self.history_max_age for t in times)
                    and tuple(sorted(set(times)))==times and times[-1]==timestamp,'Invalid/expired history provenance')
            for f,t in zip(frames,times):
                require(f not in scene_times or scene_times[f]==t,'Same source frame has inconsistent scene times')
                scene_times[f]=t
            norm=history.mean_norms_before_normalization[i];fallback=history.used_latest_fallback[i]
            require(type(norm) in (int,float) and np.isfinite(norm) and 0<=norm<=1.00001
                    and type(fallback) is bool and fallback==(norm<=1e-12),'Invalid history fallback provenance')
            if fallback:
                require(np.array_equal(history.mean.embeddings[i],history.latest.embeddings[i]),'Fallback differs from latest')
            effective_time=timestamp if fallback else times[0]
            descriptors[key]=(history.mean.embeddings[i].copy(),effective_time)
        rows=[];seen=set()
        for record in records:
            key=record.key
            require(isinstance(key,ObservationKey) and key not in seen and key.frame_index==frame
                    and all(type(x) is int and x>=0 for x in (key.camera_id,key.local_id,key.frame_index))
                    and key.camera_id in self.cameras,'Invalid/duplicate crop record key')
            seen.add(key)
            box=np.asarray(record.source_xyxy,np.float64)
            require(box.shape==(4,) and np.isfinite(box).all() and np.all(box[2:]>box[:2]),'Invalid raw tracked box')
            require((record.crop_xyxy_int is not None)==(key in descriptors),'Crop/descriptor coverage differs')
            descriptor,source_time=descriptors.get(key,(None,None))
            point=project_box_foot(self.matrices[key.camera_id],box) if descriptor is not None else None
            # Validate finite projection output before handing it to the registry.
            point=DormantIdentityArchive._point(point)
            rows.append(ObservationEvidence(key,descriptor,source_time,point,timestamp if point is not None else None))
        require(set(descriptors)<=seen,'History contains an untracked observation')
        return RecoveryEvidence(self.run_id,self.coordinate_space,frame,timestamp,tuple(rows))

    def update(self,*args,**kwargs):
        raise ValueError('Use update_with_history so archived descriptors retain their source times')

    def update_with_history(self,frame,timestamp,history,records):
        require(not self.failed,'Failed recovery stage cannot be retried; create a new run')
        records=tuple(records)
        evidence=self.evidence_from_history(frame,timestamp,history,records)
        self.manager.evidence=evidence
        try:
            result=super().update(frame,timestamp,history.mean,records)
            self.last_evidence=evidence
            return result
        except Exception:
            self.failed=True
            raise
        finally:
            self.manager.evidence=None
