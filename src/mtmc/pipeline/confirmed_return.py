"""Transactional gallery/archive bridge for the dimension-explicit identity stage."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
import numpy as np

from mtmc.association.confirmed_return import (ConfirmedReturnArchive, ConfirmationSettings, ConfirmationRound,
    YoungIdentity, GallerySample, gallery_reference)
from mtmc.association.dormant import ArchiveSettings
from mtmc.association.geometry import project_box_foot
from mtmc.reid.feature_history import validate_batch
from mtmc.reid.sample_quality import assess_sample
from .feature_identity import FeatureIdentityStage
from .core import require

POLICY='confirmed_retired_return_registry_v1'


def settings_from_dict(config):
    require(config['schema_version']==1 and config['experiment']=='clipreid_confirmed_retired_return_v1'
        and config['threshold_search'] is False,'Unexpected recovery configuration')
    a,c=config['archive'],config['confirmation']
    return ConfirmationSettings(ArchiveSettings(Fraction(a['max_age_seconds']),a['min_similarity'],a['min_margin'],
        a['max_speed_native_units_per_second'],a['position_slack_native_units'],a['max_identities']),
        Fraction(c['max_query_age_seconds']),Fraction(c['max_descriptor_age_seconds']),c['min_support_rounds'],
        Fraction(c['min_support_seconds']),Fraction(c['max_support_gap_seconds']),c['candidate_similarity'],c['max_queries_per_round'])


@dataclass(frozen=True)
class ReturnAudit:
    baseline: object
    archive: object
    reactivations: tuple
    lifecycle: dict


class ConfirmedReturnRegistry:
    def __init__(self,base,*,space,coordinate_space,camera_ids,enabled,configuration):
        require(base._last_frame is None and not base._bindings and not base._pending,'Expected fresh base registry')
        require(type(enabled) is bool,'Invalid enabled flag')
        cfg=settings_from_dict(configuration)
        require(type(configuration['max_live_gallery_identities']) is int and configuration['max_live_gallery_identities']>0,
            'Invalid live gallery budget')
        require(space.dimension==configuration['feature_dimension'],'Model dimension differs')
        require(configuration['query_admission']['all_visible_members_required'] is True
            and configuration['query_admission']['descriptor_source']=='current_accepted_raw_clip_features'
            and configuration['query_admission']['retry_after_rejected_first_observation'] is True,'Unsupported query policy')
        self.base=deepcopy(base);self.space=space;self.enabled=enabled;self.configuration=deepcopy(configuration)
        self.coordinates=coordinate_space;self.cameras=tuple(camera_ids);self.settings=cfg
        self.archive=ConfirmedReturnArchive(base.run_id,coordinate_space,self.cameras,space=space,settings=cfg) if enabled else None
        self.galleries={};self.last_seen={};self.births={};self.merged=set();self.absorbed=set();self.superseded=set()
        self.emitted=set();self.expirations=self.returns=self.merge_count=self.observations=0
        self.last_audit=None

    def update(self,grouped,evidence):
        keys={k for group in self.base._validate(grouped) for k in group}
        require(set(evidence)==keys,'Recovery evidence coverage differs')
        working=deepcopy(self)
        output=working._advance(grouped,evidence)
        self.__dict__.update(working.__dict__)
        return output

    def _advance(self,grouped,evidence):
        baseline=self.base.update(grouped)
        if not self.enabled:
            self.last_audit=ReturnAudit(baseline,None,(),{})
            return baseline
        now=grouped.timestamp;cfg=self.configuration;gc=cfg['gallery'];frame=grouped.frame_index
        retirements=[]
        for gid in baseline.expired_global_ids:
            require(gid in self.last_seen,'Expired identity missing last-seen state')
            reference,_=gallery_reference(gid,self.last_seen[gid],self.galleries.get(gid,()),now=now,space=self.space,
                max_age=Fraction(gc['max_sample_age_seconds']),max_per_camera=gc['max_per_camera'],
                min_confidence=gc['min_confidence'],border_fraction=gc['border_fraction'])
            retirements.append(reference)
        for event in baseline.merge_events:
            self.merged.add(event.canonical_global_id);self.absorbed.update(event.absorbed_global_ids)
        self.expirations+=len(baseline.expired_global_ids);self.merge_count+=len(baseline.merge_events)
        for a in baseline.base_assignments:
            if a.reason=='new_identity':
                require(a.global_id not in self.births or self.births[a.global_id]==now,'Allocated ID reused')
                self.births[a.global_id]=now
        live={s.global_id for s in baseline.identities}
        pending_merge_ids={g for p in self.base._pending.values() for g in p.global_ids}
        partitions={frozenset(g) for g in grouped.groups};queries=[]
        for state in baseline.identities:
            gid=state.global_id;born=self.births[gid]
            if not state.current_members or gid in self.merged or gid in pending_merge_ids or now-born>self.settings.max_query_age:continue
            values=[evidence[k] for k in state.current_members]
            qa=cfg['query_admission']
            passed=(frozenset(state.current_members) in partitions and all(s is not None and
                s.confidence>=qa['min_confidence'] and s.normalized_clearance>=qa['border_fraction'] for s in values))
            descriptor=point=None
            if passed:
                mean=np.mean([s.vector for s in values],axis=0,dtype=np.float64);norm=float(np.linalg.norm(mean))
                if norm>1e-12:descriptor=(mean/norm).astype(np.float32)
                if all(s.ground_xy is not None for s in values):point=tuple(map(float,np.mean([s.ground_xy for s in values],axis=0)))
            queries.append(YoungIdentity(gid,born,state.current_members,
                tuple((k.camera_id,k.local_id) for k in state.retained_local_tracks),descriptor,
                now if descriptor is not None else None,point,passed))
        result=self.archive.step(ConfirmationRound(self.base.run_id,self.coordinates,self.space,frame,now,
            tuple(retirements),tuple(queries),tuple(sorted(live|self.absorbed|self.superseded|pending_merge_ids))))
        remap={d.provisional_id:d.archived_id for d in result.decisions if d.outcome=='confirmed'}
        events=[]
        for fresh,old in remap.items():
            require(fresh in live and old not in live|self.absorbed|self.superseded|pending_merge_ids
                and fresh not in pending_merge_ids,'Unsafe archive claim')
            members=next(s.current_members for s in baseline.identities if s.global_id==fresh)
            events.append(dict(provisional_id=fresh,global_id=old,members=members))
        require(len(set(remap.values()))==len(remap),'Duplicate recovery claim')
        for key,binding in tuple(self.base._bindings.items()):
            if binding.global_id in remap:self.base._bindings[key]=replace(binding,global_id=remap[binding.global_id])
        output=replace(baseline,policy=POLICY,
            assignments=tuple(replace(a,global_id=remap[a.global_id],reason='confirmed_return') if a.global_id in remap else a for a in baseline.assignments),
            identities=tuple(sorted((replace(s,global_id=remap.get(s.global_id,s.global_id)) for s in baseline.identities),key=lambda s:s.global_id)))
        # Transfer only the young identity's real samples; archived means never become new samples.
        for fresh,old in remap.items():
            self.galleries[old]=self.galleries.pop(fresh,[]);self.superseded.add(fresh)
        self.returns+=len(remap);self.observations+=len(output.assignments)
        by_gid=defaultdict(list)
        for a in output.assignments:
            sample=evidence[a.key]
            if sample is not None and sample.confidence>=gc['min_confidence'] and sample.normalized_clearance>=gc['border_fraction']:
                by_gid[a.global_id].append(sample)
        active={s.global_id for s in output.identities}
        self.galleries={g:v for g,v in self.galleries.items() if g in active}
        self.last_seen={s.global_id:s.last_seen for s in output.identities}
        for g in active:
            candidates=[s for s in (*self.galleries.get(g,()),*by_gid[g]) if now-s.timestamp<=Fraction(gc['max_sample_age_seconds'])]
            per_camera=defaultdict(list)
            for sample in candidates:per_camera[sample.key.camera_id].append(sample)
            self.galleries[g]=[s for c in sorted(per_camera) for s in sorted(per_camera[c],key=lambda s:(s.timestamp,s.key.frame_index,s.key.local_id))[-gc['max_per_camera']:]]
        require(len(active)<=cfg['max_live_gallery_identities'],'Live gallery budget exceeded')
        slots=[(b.global_id,k.camera_id) for k,b in self.base._bindings.items()]
        require(len(slots)==len(set(slots)),'Recovered ID has a retained camera collision')
        require(set(self.archive.retained_ids).isdisjoint(active|self.absorbed|self.superseded),'Archive/live ownership collision')
        self.emitted.update(a.global_id for a in output.assignments)
        inactive=self.expirations-self.returns;allocated=self.base._next_id-1
        require(inactive>=0 and allocated==len(active)+len(self.absorbed)+len(self.superseded)+inactive,'Lifecycle accounting differs')
        lifecycle=dict(observations=self.observations,allocated_id_slots=allocated,ever_emitted_ids=len(self.emitted),
            absorbed_ids=len(self.absorbed),superseded_ids=len(self.superseded),expiration_events=self.expirations,
            reactivation_events=self.returns,currently_inactive_ids=inactive,retained_ids=len(active),
            merge_events=self.merge_count,archived_ids=len(self.archive.retained_ids),
            gallery_vectors=sum(len(v) for v in self.galleries.values()),pending_returns=self.archive.pending_count)
        self.last_audit=ReturnAudit(baseline,result,tuple(events),lifecycle)
        return output


class _Bridge:
    def __init__(self,registry):self.registry=registry;self.evidence=None
    def update(self,groups):
        require(self.evidence is not None,'Missing raw evidence')
        return self.registry.update(groups,self.evidence)


class ConfirmedReturnStage(FeatureIdentityStage):
    def __init__(self,*args,recovery_enabled,configuration,image_sizes,**kwargs):
        super().__init__(*args,**kwargs)
        require(self.variant=='mean','Expected mean cross-camera descriptor')
        require(set(image_sizes)==set(self.cameras),'Image sizes/cameras differ')
        self.sizes=dict(image_sizes);self.registry=ConfirmedReturnRegistry(self.manager,space=self.space,
            coordinate_space=self.coordinate_space,camera_ids=self.cameras,enabled=recovery_enabled,configuration=configuration)
        self.manager=_Bridge(self.registry);self.failed=False

    def update(self,*args,**kwargs):raise ValueError('Use update_with_raw for trustworthy gallery provenance')

    def update_with_raw(self,frame,timestamp,means,raw,records):
        require(not self.failed,'Failed stage cannot be retried; start a fresh run')
        validate_batch(raw,run_id=self.run_id,space=self.space,frame_index=frame,timestamp=timestamp)
        validate_batch(means,run_id=self.run_id,space=self.space,frame_index=frame,timestamp=timestamp)
        require(raw.keys==means.keys and raw.timestamps==means.timestamps,'Raw/mean mapping differs')
        rows={k:i for i,k in enumerate(raw.keys)};evidence={};records=tuple(records)
        for record in records:
            key=record.key;require(key not in evidence and key.camera_id in self.sizes,'Duplicate/invalid evidence key')
            width,height=self.sizes[key.camera_id]
            q=assess_sample(record,width,height,min_confidence=.5,border_fraction=.01)
            require((key in rows)==q.available,'Raw/crop coverage differs')
            evidence[key]=(GallerySample(key,timestamp,raw.embeddings[rows[key]].copy(),record.confidence,q.normalized_clearance,
                project_box_foot(self.matrices[key.camera_id],record.source_xyxy)) if q.available else None)
        self.manager.evidence=evidence
        try:return super().update(frame,timestamp,means,records)
        except Exception:self.failed=True;raise
        finally:self.manager.evidence=None
