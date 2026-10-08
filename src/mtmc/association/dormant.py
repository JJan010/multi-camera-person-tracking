"""Bounded inactive-identity archive; experimental causal matching contract.

This component returns claims, not complete global tracking outputs. The future
owner must apply claims atomically with its identity registry or fail the run.
It must not retire absorbed IDs or query groups already anchored to a live ID.
No existing manager implementation is changed here.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
import math
import numpy as np

from mtmc.reid.osnet import ObservationKey


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def key_order(key):
    return key.camera_id, key.local_id, key.frame_index


def group_order(keys):
    return tuple(key_order(k) for k in keys)


@dataclass(frozen=True)
class ArchiveSettings:
    max_age: Fraction
    min_similarity: float
    min_margin: float
    max_speed: float  # Native calibrated distance units / second, not assumed m/s.
    position_slack: float
    max_identities: int


@dataclass(frozen=True)
class InactiveIdentity:
    global_id: int
    last_seen: Fraction
    descriptor: np.ndarray | None
    descriptor_time: Fraction | None
    ground_xy: tuple[float, float] | None
    ground_time: Fraction | None


@dataclass(frozen=True)
class ReturnQuery:
    members: tuple[ObservationKey, ...]
    descriptor: np.ndarray | None
    ground_xy: tuple[float, float] | None


@dataclass(frozen=True)
class ArchiveRound:
    run_id: str
    coordinate_space: str
    frame_index: int
    timestamp: Fraction
    retirements: tuple[InactiveIdentity, ...] = ()
    queries: tuple[ReturnQuery, ...] = ()
    blocked_global_ids: tuple[int, ...] = ()  # ALL live/retained IDs plus absorbed IDs.


@dataclass(frozen=True)
class PairEvidence:
    members: tuple[ObservationKey, ...]
    global_id: int
    similarity: float
    ground_distance: float
    distance_limit: float
    outcome: str


@dataclass(frozen=True)
class ReturnDecision:
    members: tuple[ObservationKey, ...]
    global_id: int | None
    outcome: str


@dataclass(frozen=True)
class ArchiveResult:
    run_id: str
    coordinate_space: str
    frame_index: int
    timestamp: Fraction
    decisions: tuple[ReturnDecision, ...]
    evidence: tuple[PairEvidence, ...]
    retired: tuple[int, ...]
    skipped_retirements: tuple[tuple[int, str], ...]
    blocked_removed: tuple[int, ...]
    expired: tuple[int, ...]
    capacity_evicted: tuple[int, ...]
    retained_ids: tuple[int, ...]


class DormantIdentityArchive:
    """One scene/run; mutual unambiguous best matching with strict appearance gate.

    Geometry uses distance <= slack + speed * elapsed_since_ground_sample.
    Both source evidence ages and last-seen age must be <= max_age. Reuse never
    refreshes timestamps. No unavailable-geometry fallback is used for returns.
    Eligible pairs compete before any claim; both row and column need a unique
    best and at least min_margin over the next eligible candidate. Unmatched is
    explicit. This conservative rule does not optimize a global assignment sum.

    Claimed IDs leave this archive. The caller must install the claim into its
    live registry before the next round. The owner must verify that any later
    retirement of that ID is a new lifecycle event. Empty rounds advance expiry. All input
    validation precedes mutation; state is committed once per successful round.
    """
    def __init__(self, run_id, coordinate_space, camera_ids, settings):
        require(isinstance(run_id,str) and bool(run_id.strip()), 'Invalid run ID')
        require(isinstance(coordinate_space,str) and bool(coordinate_space.strip()), 'Invalid coordinate space')
        require(isinstance(camera_ids,tuple) and len(camera_ids)>=2
                and all(integer(c) for c in camera_ids) and len(set(camera_ids))==len(camera_ids), 'Invalid cameras')
        require(isinstance(settings,ArchiveSettings), 'Expected ArchiveSettings')
        require(isinstance(settings.max_age,Fraction) and settings.max_age>0, 'Invalid maximum age')
        for name in ('min_similarity','min_margin','max_speed','position_slack'):
            value=getattr(settings,name)
            require(type(value) in (int,float) and math.isfinite(value), 'Invalid '+name)
        require(-1<=settings.min_similarity<=1 and 0<settings.min_margin<=2
                and settings.max_speed>=0 and settings.position_slack>=0
                and type(settings.max_identities) is int and settings.max_identities>0, 'Invalid archive settings')
        self.run_id,self.coordinate_space=run_id,coordinate_space
        self.cameras=tuple(sorted(camera_ids));self.settings=settings
        self._entries={};self._last_frame=None;self._last_time=None

    @property
    def retained_ids(self):
        return tuple(sorted(self._entries))

    @property
    def vector_payload_bytes(self):
        return sum(e.descriptor.nbytes for e in self._entries.values())

    @staticmethod
    def _descriptor(value):
        if value is None: return None
        require(isinstance(value,np.ndarray) and value.dtype==np.float32 and value.shape==(512,)
                and np.isfinite(value).all()
                and abs(float(np.linalg.norm(value.astype(np.float64)))-1.)<=1e-5, 'Invalid descriptor')
        return value.copy()

    @staticmethod
    def _point(value):
        if value is None: return None
        require(isinstance(value,tuple) and len(value)==2
                and all(type(v) in (int,float) and math.isfinite(v) for v in value), 'Invalid ground point')
        return tuple(float(v) for v in value)

    def _validate(self, event):
        require(isinstance(event,ArchiveRound) and event.run_id==self.run_id
                and event.coordinate_space==self.coordinate_space, 'Mixed run/coordinate scope')
        require(integer(event.frame_index) and isinstance(event.timestamp,Fraction) and event.timestamp>=0
                and (self._last_frame is None or event.frame_index>self._last_frame)
                and (self._last_time is None or event.timestamp>self._last_time), 'Nonincreasing/invalid scene time')
        require(all(isinstance(v,tuple) for v in (event.retirements,event.queries,event.blocked_global_ids)), 'Expected immutable input tuples')
        blocked=event.blocked_global_ids
        require(all(integer(g) for g in blocked) and len(set(blocked))==len(blocked), 'Invalid blocked IDs')
        entries=[];seen=set()
        for entry in event.retirements:
            require(isinstance(entry,InactiveIdentity) and integer(entry.global_id)
                    and entry.global_id not in seen and entry.global_id not in blocked
                    and entry.global_id not in self._entries, 'Duplicate, blocked or already archived retirement')
            require(isinstance(entry.last_seen,Fraction) and 0<=entry.last_seen<event.timestamp,
                    'Invalid last_seen')
            descriptor=self._descriptor(entry.descriptor);point=self._point(entry.ground_xy)
            for value,time in ((descriptor,entry.descriptor_time),(point,entry.ground_time)):
                require((value is None and time is None) or
                        (value is not None and isinstance(time,Fraction) and 0<=time<=entry.last_seen),
                        'Invalid evidence source time')
            entries.append(replace(entry,descriptor=descriptor,ground_xy=point));seen.add(entry.global_id)
        queries=[];seen=set()
        for query in event.queries:
            require(isinstance(query,ReturnQuery) and isinstance(query.members,tuple) and query.members, 'Empty/invalid query')
            cameras=set()
            for key in query.members:
                require(isinstance(key,ObservationKey) and all(integer(v) for v in key_order(key))
                        and key.frame_index==event.frame_index and key.camera_id in self.cameras
                        and key.camera_id not in cameras and key not in seen, 'Invalid/duplicate query observation')
                cameras.add(key.camera_id);seen.add(key)
            queries.append(replace(query,members=tuple(sorted(query.members,key=key_order)),
                descriptor=self._descriptor(query.descriptor),ground_xy=self._point(query.ground_xy)))
        return sorted(entries,key=lambda e:e.global_id),sorted(queries,key=lambda q:group_order(q.members)),set(blocked)

    def step(self,event):
        entries,queries,blocked=self._validate(event)
        now=event.timestamp;cfg=self.settings
        state=dict(self._entries)
        removed=tuple(sorted(set(state)&blocked))
        for gid in removed: del state[gid]
        retired=[];skipped=[]
        for entry in entries:
            if entry.descriptor is None or entry.ground_xy is None:
                skipped.append((entry.global_id,'missing_evidence'));continue
            state[entry.global_id]=entry;retired.append(entry.global_id)
        expired=tuple(sorted(g for g,e in state.items() if
            any(now-t>cfg.max_age for t in (e.last_seen,e.descriptor_time,e.ground_time))))
        for gid in expired: del state[gid]
        overflow=max(0,len(state)-cfg.max_identities)
        evicted=tuple(sorted(state,key=lambda g:(state[g].last_seen,g))[:overflow])
        for gid in evicted: del state[gid]

        eligible={};evidence=[];unavailable={}
        for i,query in enumerate(queries):
            if query.descriptor is None or query.ground_xy is None:
                unavailable[i]='missing_descriptor' if query.descriptor is None else 'missing_geometry'
                continue
            for gid in sorted(state):
                entry=state[gid]
                similarity=float(np.clip(np.dot(query.descriptor.astype(np.float64),entry.descriptor.astype(np.float64)),-1,1))
                distance=math.hypot(query.ground_xy[0]-entry.ground_xy[0],query.ground_xy[1]-entry.ground_xy[1])
                limit=cfg.position_slack+cfg.max_speed*float(now-entry.ground_time)
                require(math.isfinite(limit) and math.isfinite(distance),'Nonfinite geometry calculation')
                outcome=('appearance_rejected' if similarity<=cfg.min_similarity else
                         'geometry_rejected' if distance>limit else 'eligible')
                evidence.append(PairEvidence(query.members,gid,similarity,distance,limit,outcome))
                if outcome=='eligible': eligible[i,gid]=similarity

        def best(items):
            ranked=sorted(items,key=lambda item:(-item[1],item[0]))
            if not ranked: return None,False
            clear=len(ranked)==1 or (ranked[0][1]>ranked[1][1]
                                     and ranked[0][1]-ranked[1][1]>=cfg.min_margin)
            return ranked[0][0],clear

        rows={i:best([(g,s) for (j,g),s in eligible.items() if i==j]) for i in range(len(queries))}
        cols={g:best([(i,s) for (i,h),s in eligible.items() if g==h]) for g in state}
        decisions=[];claimed=set()
        for i,query in enumerate(queries):
            gid,clear=rows[i]
            reason=(unavailable[i] if i in unavailable else 'no_eligible_identity' if gid is None else
                    'ambiguous_query' if not clear else 'ambiguous_identity' if not cols[gid][1] else
                    'not_mutual_best' if cols[gid][0]!=i else 'matched')
            match=gid if reason=='matched' else None
            decisions.append(ReturnDecision(query.members,match,reason))
            if match is not None:
                require(match not in claimed,'Duplicate identity claim');claimed.add(match)
        for gid in claimed: del state[gid]
        result=ArchiveResult(self.run_id,self.coordinate_space,event.frame_index,now,tuple(decisions),
            tuple(evidence),tuple(retired),tuple(skipped),removed,expired,evicted,tuple(sorted(state)))
        self._entries=state
        self._last_frame=event.frame_index;self._last_time=now
        return result
