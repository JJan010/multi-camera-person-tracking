"""Model-scoped, time-confirmed archive claims; registry integration is external.

An owner must install a confirmed claim atomically into its registry. Queries
must cover a whole eligible young identity, not a subset of an established ID.
No GT, file access, identity relabeling or registry mutation occurs here.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
import math
import numpy as np

from mtmc.reid.feature_history import FeatureSpace
from mtmc.reid.osnet import ObservationKey
from .dormant import (DormantIdentityArchive, ArchiveSettings, ArchiveRound,
                      InactiveIdentity, ReturnQuery, require, integer, key_order)


@dataclass(frozen=True)
class ConfirmationSettings:
    archive: ArchiveSettings
    max_query_age: Fraction
    max_descriptor_age: Fraction
    min_support_rounds: int
    min_support_seconds: Fraction
    max_support_gap: Fraction
    candidate_similarity: float
    max_queries_per_round: int = 128


@dataclass(frozen=True)
class YoungIdentity:
    global_id: int
    born_at: Fraction
    members: tuple[ObservationKey,...]
    retained_members: tuple[tuple[int,int],...]
    descriptor: np.ndarray | None
    descriptor_time: Fraction | None
    ground_xy: tuple[float,float] | None
    quality_passed: bool


@dataclass(frozen=True)
class ConfirmationRound:
    run_id: str
    coordinate_space: str
    space: FeatureSpace
    frame_index: int
    timestamp: Fraction
    retirements: tuple[InactiveIdentity,...] = ()
    queries: tuple[YoungIdentity,...] = ()
    blocked_global_ids: tuple[int,...] = ()


@dataclass(frozen=True)
class ConfirmationDecision:
    provisional_id: int
    archived_id: int | None
    outcome: str
    support_rounds: int
    support_span: Fraction
    cosine: float | None


@dataclass(frozen=True)
class ConfirmationResult:
    frame_index: int
    timestamp: Fraction
    decisions: tuple[ConfirmationDecision,...]
    resets: tuple[tuple[int,str],...]
    retired: tuple[int,...]
    skipped_retirements: tuple[tuple[int,str],...]
    expired: tuple[int,...]
    capacity_evicted: tuple[int,...]
    blocked_removed: tuple[int,...]
    retained_ids: tuple[int,...]


@dataclass(frozen=True)
class _Pending:
    fingerprint: tuple
    first_time: Fraction
    last_time: Fraction
    count: int
    seed: np.ndarray


class _FeatureArchive(DormantIdentityArchive):
    def __init__(self,*args,space,**kwargs):
        self.space=space
        super().__init__(*args,**kwargs)

    def _descriptor(self,value):
        if value is None:return None
        require(isinstance(value,np.ndarray) and value.dtype==np.float32 and value.shape==(self.space.dimension,)
            and np.isfinite(value).all() and abs(float(np.linalg.norm(value.astype(np.float64)))-1)<=1e-5,
            'Invalid model-scoped descriptor')
        return value.copy()


class ConfirmedReturnArchive:
    """Causal archive with retries and confirmation for whole young identities.

    The reused archive handles aging/capacity/input validation. Its immediate
    claims are restored on a private copy; only confirmed claims leave storage.
    Ambiguity is evaluated over all geometry-feasible pairs, INCLUDING competitors
    below the acceptance threshold. Confirmation never relaxes any pair gate.
    """
    def __init__(self,run_id,coordinate_space,camera_ids,*,space,settings):
        require(isinstance(space,FeatureSpace) and isinstance(settings,ConfirmationSettings),'Invalid configuration')
        cfg=settings
        for value in (cfg.max_query_age,cfg.max_descriptor_age,cfg.min_support_seconds,cfg.max_support_gap):
            require(isinstance(value,Fraction) and value>0,'Expected positive scene-time duration')
        require(type(cfg.min_support_rounds) is int and cfg.min_support_rounds>=2
            and type(cfg.candidate_similarity) in (int,float) and math.isfinite(cfg.candidate_similarity)
            and -1<=cfg.candidate_similarity<=1,'Invalid confirmation settings')
        require(type(cfg.max_queries_per_round) is int and cfg.max_queries_per_round>0
            and cfg.min_support_seconds<=cfg.max_query_age,'Invalid query budget')
        self.space=space;self.settings=cfg
        self._archive=_FeatureArchive(run_id,coordinate_space,camera_ids,cfg.archive,space=space)
        self._pending={};self._claimed_provisional=set()

    @property
    def retained_ids(self):return self._archive.retained_ids

    @property
    def pending_count(self):return len(self._pending)

    @property
    def vector_payload_bytes(self):
        return self._archive.vector_payload_bytes+sum(p.seed.nbytes for p in self._pending.values())

    def step(self,event):
        require(isinstance(event,ConfirmationRound) and event.space==self.space,'Mixed model space')
        require(isinstance(event.queries,tuple) and len(event.queries)<=self.settings.max_queries_per_round,'Invalid query tuple/budget')
        queries=[];seen=set();filtered={}
        for q in event.queries:
            require(isinstance(q,YoungIdentity) and integer(q.global_id) and q.global_id not in seen
                and q.global_id not in self._claimed_provisional,'Duplicate or already superseded provisional ID')
            require(isinstance(event.timestamp,Fraction) and isinstance(q.born_at,Fraction)
                and 0<=q.born_at<=event.timestamp and type(q.quality_passed) is bool,'Invalid query metadata')
            require(q.global_id in event.blocked_global_ids,'Every query ID must be retained/blocked against archive recovery')
            require(isinstance(q.retained_members,tuple) and q.retained_members
                and all(isinstance(k,tuple) and len(k)==2 and all(integer(v) for v in k)
                        and k[0] in self._archive.cameras for k in q.retained_members)
                and len({k[0] for k in q.retained_members})==len(q.retained_members),'Invalid retained membership')
            require(isinstance(q.members,tuple) and all(isinstance(k,ObservationKey) for k in q.members)
                and {(k.camera_id,k.local_id) for k in q.members}<=set(q.retained_members),'Query is outside retained membership')
            descriptor=self._archive._descriptor(q.descriptor);point=self._archive._point(q.ground_xy)
            require((descriptor is None and q.descriptor_time is None) or
                (descriptor is not None and isinstance(q.descriptor_time,Fraction)
                 and q.born_at<=q.descriptor_time<=event.timestamp),'Invalid query descriptor time')
            reason=('query_too_old' if event.timestamp-q.born_at>self.settings.max_query_age else
                    'query_quality_rejected' if not q.quality_passed else
                    'missing_descriptor' if descriptor is None else
                    'stale_query_descriptor' if event.timestamp-q.descriptor_time>self.settings.max_descriptor_age else
                    'missing_geometry' if point is None else None)
            if reason:filtered[q.global_id]=reason
            queries.append(replace(q,members=tuple(sorted(q.members,key=key_order)),
                retained_members=tuple(sorted(q.retained_members)),descriptor=descriptor,ground_xy=point))
            seen.add(q.global_id)
        queries.sort(key=lambda q:q.global_id)
        raw=ArchiveRound(event.run_id,event.coordinate_space,event.frame_index,event.timestamp,event.retirements,
            tuple(ReturnQuery(q.members,None if q.global_id in filtered else q.descriptor,q.ground_xy) for q in queries),
            event.blocked_global_ids)
        # All retained state, counters and candidate evidence commit together.
        working=deepcopy(self)
        result=working._advance(event,raw,queries,filtered)
        self.__dict__.update(working.__dict__)
        return result

    def _advance(self,event,raw,queries,filtered):
        archive=self._archive;cfg=self.settings;now=event.timestamp
        validated,_,_=archive._validate(raw)
        possible={**archive._entries,**{e.global_id:e for e in validated}}
        base=archive.step(raw)
        for d in base.decisions:
            if d.global_id is not None:archive._entries[d.global_id]=possible[d.global_id]
        index={q.members:i for i,q in enumerate(queries)}
        # Do not exclude close below-threshold rivals from the ambiguity margin.
        pairs={(index[e.members],e.global_id):e.similarity for e in base.evidence if e.ground_distance<=e.distance_limit}
        def best(values):
            ranked=sorted(values,key=lambda x:(-x[1],x[0]))
            if not ranked:return None,False
            clear=len(ranked)==1 or (ranked[0][1]>ranked[1][1] and ranked[0][1]-ranked[1][1]>=cfg.archive.min_margin)
            return ranked[0][0],clear
        rows={i:best([(g,s) for (j,g),s in pairs.items() if j==i]) for i in range(len(queries))}
        cols={g:best([(i,s) for (i,h),s in pairs.items() if h==g]) for g in archive._entries}
        pending={};decisions=[];resets=[];claimed=set()
        for i,q in enumerate(queries):
            gid,clear=rows[i];similarity=pairs.get((i,gid));old=self._pending.get(q.global_id)
            reason=(filtered.get(q.global_id) or ('no_feasible_reference' if gid is None else
                'appearance_rejected' if similarity<=cfg.archive.min_similarity else
                'ambiguous_query' if not clear else 'ambiguous_identity' if not cols[gid][1] else
                'not_mutual_best' if cols[gid][0]!=i else None))
            if reason:
                if old:resets.append((q.global_id,reason))
                decisions.append(ConfirmationDecision(q.global_id,None,reason,0,Fraction(0),similarity));continue
            reference=archive._entries[gid]
            fingerprint=(gid,reference.last_seen,reference.descriptor_time,reference.ground_time,
                         q.born_at,q.retained_members,tuple((k.camera_id,k.local_id) for k in q.members))
            reset=('membership_or_reference_changed' if old and old.fingerprint!=fingerprint else
                   'support_gap' if old and now-old.last_time>cfg.max_support_gap else
                   'candidate_incoherent' if old and float(np.dot(old.seed.astype(np.float64),q.descriptor.astype(np.float64)))<cfg.candidate_similarity else None)
            if reset:resets.append((q.global_id,reset));old=None
            p=(_Pending(fingerprint,old.first_time,now,old.count+1,old.seed) if old else
               _Pending(fingerprint,now,now,1,q.descriptor.copy()))
            span=now-p.first_time
            confirmed=p.count>=cfg.min_support_rounds and span>=cfg.min_support_seconds
            decisions.append(ConfirmationDecision(q.global_id,gid if confirmed else None,
                'confirmed' if confirmed else 'pending',p.count,span,similarity))
            if confirmed:
                require(gid not in claimed,'Duplicate claim');claimed.add(gid);self._claimed_provisional.add(q.global_id)
            else:pending[q.global_id]=p
        for pid in set(self._pending)-{q.global_id for q in queries}:resets.append((pid,'absent_query'))
        for gid in claimed:del archive._entries[gid]
        self._pending=pending
        return ConfirmationResult(event.frame_index,now,tuple(decisions),tuple(sorted(resets)),base.retired,
            base.skipped_retirements,base.expired,base.capacity_evicted,base.blocked_removed,archive.retained_ids)


@dataclass(frozen=True)
class GallerySample:
    key: ObservationKey
    timestamp: Fraction
    vector: np.ndarray
    confidence: float
    normalized_clearance: float
    ground_xy: tuple[float,float] | None


def gallery_reference(global_id,last_seen,samples,*,now,space,max_age,max_per_camera=8,
                      min_confidence=.5,border_fraction=.01):
    """Build one owned reference from raw accepted samples, with oldest source time.

    Pure builder: caller owns bounded per-live-ID sample storage and lineage.
    Returns (InactiveIdentity, selected samples); no source time is refreshed.
    """
    require(integer(global_id) and isinstance(space,FeatureSpace) and isinstance(now,Fraction)
        and isinstance(last_seen,Fraction) and 0<=last_seen<=now and isinstance(max_age,Fraction) and max_age>0
        and type(max_per_camera) is int and max_per_camera>0,'Invalid gallery context')
    require(type(min_confidence) in (int,float) and 0<=min_confidence<=1
        and type(border_fraction) in (int,float) and 0<=border_fraction<=.5,'Invalid gallery admission')
    helper=_FeatureArchive('gallery','native',(0,1),ArchiveSettings(max_age,.9,.05,1.,2.,1),space=space)
    accepted={};seen=set()
    for sample in samples:
        require(isinstance(sample,GallerySample) and isinstance(sample.key,ObservationKey)
            and all(integer(v) for v in key_order(sample.key)) and sample.key not in seen
            and isinstance(sample.timestamp,Fraction) and 0<=sample.timestamp<=last_seen,'Invalid/duplicate gallery sample')
        seen.add(sample.key);vector=helper._descriptor(sample.vector);point=helper._point(sample.ground_xy)
        require(vector is not None and type(sample.confidence) in (int,float) and math.isfinite(sample.confidence)
            and 0<=sample.confidence<=1 and type(sample.normalized_clearance) in (int,float)
            and math.isfinite(sample.normalized_clearance),'Invalid sample quality')
        if now-sample.timestamp<=max_age and sample.confidence>=min_confidence and sample.normalized_clearance>=border_fraction:
            accepted.setdefault(sample.key.camera_id,[]).append(replace(sample,vector=vector,ground_xy=point))
    selected=tuple(sorted((s for values in accepted.values() for s in
        sorted(values,key=lambda s:(s.timestamp,key_order(s.key)))[-max_per_camera:]),key=lambda s:(s.timestamp,key_order(s.key))))
    vector=source_time=point=point_time=None
    if selected:
        mean=np.mean([s.vector for s in selected],axis=0,dtype=np.float64);norm=float(np.linalg.norm(mean))
        if norm>1e-12:vector=(mean/norm).astype(np.float32);source_time=min(s.timestamp for s in selected)
        latest=max(s.timestamp for s in selected);points=[s.ground_xy for s in selected if s.timestamp==latest]
        if all(p is not None for p in points):
            xy=np.mean(points,axis=0,dtype=np.float64);require(np.isfinite(xy).all(),'Invalid gallery projection mean')
            point=tuple(map(float,xy));point_time=latest
    return InactiveIdentity(global_id,last_seen,vector,source_time,point,point_time),selected
