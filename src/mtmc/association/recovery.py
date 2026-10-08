"""Transactional experimental bridge between the frozen registry and archive."""
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
import numpy as np

from mtmc.reid.osnet import ObservationKey
from .controlled_merge import ControlledMergeIdentityManager, ControlledGlobalFrame
from .dormant import (DormantIdentityArchive, ArchiveRound, ArchiveResult,
    InactiveIdentity, ReturnQuery, require)
from .grouping import key_order

POLICY='dormant_return_bridge_v1'


@dataclass(frozen=True)
class ObservationEvidence:
    key: ObservationKey
    descriptor: np.ndarray | None
    descriptor_time: Fraction | None
    ground_xy: tuple[float,float] | None
    ground_time: Fraction | None


@dataclass(frozen=True)
class RecoveryEvidence:
    run_id: str
    coordinate_space: str
    frame_index: int
    timestamp: Fraction
    observations: tuple[ObservationEvidence,...]


@dataclass(frozen=True)
class Reactivation:
    global_id: int
    superseded_new_id: int
    members: tuple[ObservationKey,...]


@dataclass(frozen=True)
class RecoveryAudit:
    baseline: ControlledGlobalFrame
    archive: ArchiveResult | None
    reactivations: tuple[Reactivation,...]
    snapshot_decisions: tuple[tuple[int,str],...]
    lifecycle: tuple[tuple[str,int],...]


class RecoveryIdentityManager:
    """Own a fresh registry; disabled update returns its original output exactly.

    Enabled mode runs the baseline on a private copy, then offers only wholly
    new, unanchored groups to the inactive archive. Recovery replaces a newly
    allocated provisional ID with an old inactive ID. The provisional number is
    burned, not emitted or reused. See last_audit for the pre-recovery record.
    Existing base_assignments/decisions describe that provisional baseline stage.
    Old lifecycle helpers are therefore not valid for enabled outputs.
    """
    def __init__(self,base,*,coordinate_space,camera_ids,enabled,archive_settings=None):
        require(type(base) is ControlledMergeIdentityManager and base._last_frame is None
                and not base._bindings and not base._pending and base._next_id==1,'Expected fresh controlled registry')
        require(type(enabled) is bool,'Invalid enabled flag')
        require(isinstance(coordinate_space,str) and bool(coordinate_space.strip())
                and isinstance(camera_ids,tuple) and len(camera_ids)>=2
                and all(type(c) is int and c>=0 for c in camera_ids)
                and len(set(camera_ids))==len(camera_ids),'Invalid scene scope')
        require(enabled or archive_settings is None,'Disabled recovery has no archive settings')
        self._base=deepcopy(base);self.enabled=enabled
        self.coordinate_space=coordinate_space;self.camera_ids=tuple(sorted(camera_ids))
        self._archive=(DormantIdentityArchive(base.run_id,coordinate_space,self.camera_ids,archive_settings)
                       if enabled else None)
        self._snapshots={};self._absorbed=set();self._emitted=set()
        self._expirations=self._returns=self._superseded=self._merge_events=self._observations=0
        self.last_audit=None

    @property
    def run_id(self): return self._base.run_id

    @property
    def archived_ids(self): return self._archive.retained_ids if self._archive else ()

    def _validate(self,frame,evidence):
        groups=self._base._validate(frame)
        keys={k for group in groups for k in group}
        require(all(k.camera_id in self.camera_ids for k in keys),'Unexpected camera')
        if evidence is None:
            require(not self.enabled,'Enabled recovery requires explicit evidence');return groups,{}
        require(isinstance(evidence,RecoveryEvidence) and evidence.run_id==self.run_id
                and evidence.coordinate_space==self.coordinate_space and evidence.frame_index==frame.frame_index
                and type(evidence.frame_index) is int and isinstance(evidence.timestamp,Fraction)
                and evidence.timestamp==frame.timestamp and isinstance(evidence.observations,tuple),'Invalid evidence scope')
        rows={}
        for row in evidence.observations:
            require(isinstance(row,ObservationEvidence) and isinstance(row.key,ObservationKey)
                    and all(type(v) is int and v>=0 for v in key_order(row.key))
                    and row.key in keys and row.key not in rows,'Invalid/duplicate evidence key')
            descriptor=DormantIdentityArchive._descriptor(row.descriptor)
            point=DormantIdentityArchive._point(row.ground_xy)
            for value,time in ((descriptor,row.descriptor_time),(point,row.ground_time)):
                require((value is None and time is None) or
                        (value is not None and isinstance(time,Fraction) and 0<=time<=frame.timestamp),
                        'Noncausal/missing evidence timestamp')
            rows[row.key]=replace(row,descriptor=descriptor,ground_xy=point)
        require(set(rows)==keys,'Evidence must cover every current observation, including unavailable ones')
        return groups,rows

    @staticmethod
    def _aggregate(members,rows,now):
        samples=[rows[k] for k in members]
        descriptor=descriptor_time=point=ground_time=None
        if all(s.descriptor is not None for s in samples):
            mean=np.mean(np.stack([s.descriptor for s in samples]),axis=0,dtype=np.float64)
            norm=float(np.linalg.norm(mean))
            if norm>1e-12:
                descriptor=(mean/norm).astype(np.float32)
                descriptor_time=min(s.descriptor_time for s in samples)
        if all(s.ground_xy is not None and s.ground_time==now for s in samples):
            points=np.array([s.ground_xy for s in samples],np.float64)
            xy=(points/len(samples)).sum(axis=0)
            require(np.isfinite(xy).all(),'Nonfinite aggregate ground point')
            point=tuple(float(v) for v in xy);ground_time=now
        return descriptor,descriptor_time,point,ground_time

    def update(self,frame,evidence=None):
        groups,rows=self._validate(frame,evidence)
        # Includes base registry, archive claims, snapshots and all counters.
        working=deepcopy(self)
        result=working._advance(frame,groups,rows)
        self.__dict__.update(working.__dict__)
        return result

    def _advance(self,frame,groups,rows):
        baseline=self._base.update(frame)
        active={s.global_id for s in baseline.identities}
        absorbed={g for event in baseline.merge_events for g in event.absorbed_global_ids}
        require(not (absorbed & self._absorbed),'An absorbed ID was reused')
        self._absorbed.update(absorbed)
        self._expirations+=len(baseline.expired_global_ids)
        self._merge_events+=len(baseline.merge_events)
        self._observations+=len(baseline.assignments)
        archive_result=None;events=[];snapshot_decisions=[];output=baseline
        if self.enabled:
            retirements=[]
            for gid in baseline.expired_global_ids:
                require(gid in self._snapshots,'Expired ID has no lifecycle snapshot')
                retirements.append(self._snapshots[gid])
            by_key={a.key:a for a in baseline.base_assignments}
            final_by_key={a.key:a.global_id for a in baseline.assignments}
            queries=[];provisional={}
            for group in groups:
                if not all(by_key[k].reason=='new_identity' for k in group): continue
                ids={final_by_key[k] for k in group}
                require(len(ids)==1,'New unanchored group has mixed IDs')
                gid=next(iter(ids))
                state=next(s for s in baseline.identities if s.global_id==gid)
                require(set(state.current_members)==set(group),'New group does not own its whole provisional ID')
                descriptor,time,point,_=self._aggregate(group,rows,frame.timestamp)
                if time is not None and frame.timestamp-time>self._archive.settings.max_age: descriptor=None
                queries.append(ReturnQuery(group,descriptor,point));provisional[group]=gid
            archive_result=self._archive.step(ArchiveRound(self.run_id,self.coordinate_space,
                frame.frame_index,frame.timestamp,tuple(retirements),tuple(queries),tuple(sorted(active|self._absorbed))))
            remap={}
            for decision in archive_result.decisions:
                if decision.global_id is None: continue
                fresh=provisional[decision.members];old=decision.global_id
                require(old not in active and old not in self._absorbed and fresh not in remap,'Invalid recovery claim')
                remap[fresh]=old;events.append(Reactivation(old,fresh,decision.members))
            require(len(set(remap.values()))==len(remap),'Duplicate reactivation')
            require(all(not (set(p.global_ids)&set(remap)) for p in self._base._pending.values()),
                    'Recovery would invalidate pending merge evidence')
            for key,binding in tuple(self._base._bindings.items()):
                if binding.global_id in remap:
                    self._base._bindings[key]=replace(binding,global_id=remap[binding.global_id])
            assignments=tuple(replace(a,global_id=remap[a.global_id],reason='dormant_reactivation')
                              if a.global_id in remap else a for a in baseline.assignments)
            identities=tuple(sorted((replace(s,global_id=remap.get(s.global_id,s.global_id))
                                     for s in baseline.identities),key=lambda s:s.global_id))
            output=replace(baseline,policy=POLICY,assignments=assignments,identities=identities)
            self._returns+=len(events);self._superseded+=len(remap)
            self._update_snapshots(output,groups,rows,snapshot_decisions)
        slots=[(b.global_id,k.camera_id) for k,b in self._base._bindings.items()]
        require(len(slots)==len(set(slots)),'Recovery created duplicate retained camera membership')
        require({a.key for a in output.assignments}=={a.key for a in baseline.assignments},'Observation coverage changed')
        self._emitted.update(a.global_id for a in output.assignments)
        allocated=self._base._next_id-1;inactive=self._expirations-self._returns
        require(inactive>=0 and allocated==len(output.identities)+len(self._absorbed)+self._superseded+inactive,
                'Lifecycle conservation failed')
        require(set(self.archived_ids).isdisjoint({s.global_id for s in output.identities}|self._absorbed),
                'Archived identity is active or absorbed')
        lifecycle={'observations':self._observations,'allocated_id_slots':allocated,
            'ever_emitted_ids':len(self._emitted),'absorbed_ids':len(self._absorbed),
            'expiration_events':self._expirations,'reactivation_events':self._returns,
            'superseded_new_ids':self._superseded,'currently_inactive_ids':inactive,
            'retained_ids':len(output.identities),'merge_events':self._merge_events,
            'archived_ids':len(self.archived_ids)}
        self.last_audit=RecoveryAudit(baseline,archive_result,tuple(events),tuple(snapshot_decisions),tuple(sorted(lifecycle.items())))
        return output

    def _update_snapshots(self,output,groups,rows,decisions):
        merged={event.canonical_global_id for event in output.merge_events}
        partitions={frozenset(g) for g in groups};next_snapshots={}
        for state in output.identities:
            gid=state.global_id
            previous=self._snapshots.get(gid) if gid not in merged else None
            snapshot=(replace(previous,last_seen=state.last_seen) if previous is not None else
                      InactiveIdentity(gid,state.last_seen,None,None,None,None))
            members=state.current_members
            if not members:
                status='absent_preserve_source_times'
            elif frozenset(members) not in partitions:
                status='unsupported_visible_partition'
            else:
                descriptor,time,point,ground_time=self._aggregate(members,rows,output.timestamp)
                if descriptor is None or point is None:
                    status='incomplete_or_cancelled_evidence'
                else:
                    snapshot=InactiveIdentity(gid,state.last_seen,descriptor,time,point,ground_time)
                    status='snapshot_updated'
            next_snapshots[gid]=snapshot;decisions.append((gid,status))
        self._snapshots=next_snapshots
