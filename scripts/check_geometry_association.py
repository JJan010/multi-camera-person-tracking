"""Synthetic contract checks; distances and timing are not deployment settings."""
from dataclasses import replace
from fractions import Fraction
from itertools import product, combinations
import numpy as np

from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.association.pairwise import CameraAppearance, associate_camera_pair, solve_partial_assignment
from mtmc.association.geometry import (GroundObservation, CameraGroundPositions, project_box_foot,
    solve_geometry_assignment, associate_camera_pair_with_geometry, group_geometry_associations)
from mtmc.association.grouping import group_pair_associations


def camera(camera_id, rows):
    keys = tuple(ObservationKey(camera_id, local, 3) for local, _, _ in rows)
    embeddings = np.zeros((len(rows),512),dtype=np.float32)
    for i,(_,values,_) in enumerate(rows):
        embeddings[i,:len(values)] = values
        embeddings[i] /= np.linalg.norm(embeddings[i])
    times = (Fraction(1,10),)*len(rows)
    appearance = CameraAppearance('synthetic-geometry',camera_id,3,Fraction(1,10),'mean',ReIDBatch(keys,times,embeddings))
    ground = CameraGroundPositions(appearance.run_id,camera_id,3,appearance.timestamp,'synthetic-plane/native',
        tuple(GroundObservation(key,xy) for key,(_,_,xy) in zip(keys,rows)))
    return appearance,ground


def match(left,right,**settings):
    return associate_camera_pair_with_geometry(left[0],right[0],left[1],right[1],
        **({'min_similarity':.5,'max_distance':2.,'unavailable_policy':'appearance_only'}|settings))


def rejected(callback):
    try:
        callback()
    except (TypeError,ValueError):
        return
    raise AssertionError('Invalid input was accepted')


def main():
    left = camera(4,[(10,[1,0],(0.,0.))])
    right = camera(5,[(20,[.99,np.sqrt(1-.99**2)],(10.,0.)),(30,[.9,np.sqrt(1-.9**2)],(1.,0.))])
    baseline = associate_camera_pair(left[0],right[0],min_similarity=.5)
    result = match(left,right)
    assert baseline.matches[0].right.local_id==20 and result.association.matches[0].right.local_id==30
    assert [c.geometry_decision for c in result.candidates]==['outside_distance','within_distance']
    assert result.association.unmatched_right[0].reason=='geometry_blocked'
    print('Geometry before assignment selects a valid alternative; post-filtering the baseline would lose it: OK')
    scores=np.array([[.99,.8],[.85,.7]])
    mask=np.array([[False,True],[True,False]])
    assert set(solve_geometry_assignment(scores,min_similarity=.5,admissible=mask))=={(0,1),(1,0)}
    rng=np.random.default_rng(41)
    for n,m in product(range(4),repeat=2):
        for _ in range(5):
            scores=rng.uniform(-1,1,(n,m)); mask=rng.random((n,m))>.4
            threshold=float(rng.choice([-1.,-.2,.5,1.]))
            selected=solve_geometry_assignment(scores,min_similarity=threshold,admissible=mask)
            actual=sum(scores[i,j]-threshold for i,j in selected)
            best=0.
            for choices in product(range(-1,m),repeat=n):
                used=[j for j in choices if j>=0]
                if len(set(used))!=len(used):
                    continue
                if any(j>=0 and (not mask[i,j] or scores[i,j]<=threshold) for i,j in enumerate(choices)):
                    continue
                best=max(best,sum(scores[i,j]-threshold for i,j in enumerate(choices) if j>=0))
            assert abs(actual-best)<1e-10
            assert solve_geometry_assignment(scores,min_similarity=threshold,admissible=np.ones((n,m),bool)) == solve_partial_assignment(scores,min_similarity=threshold)
    print('Masked partial-assignment objective agrees with exhaustive small cases; all-admissible solver parity: OK')
    boundary = camera(5,[(20,[1,0],(2.,0.))])
    assert len(match(left,boundary).association.matches)==1
    assert len(match(left,boundary,max_distance=np.nextafter(2.,0.)).association.matches)==0
    assert len(match(left,boundary,min_similarity=1.).association.matches)==0
    far=match(left,right,max_distance=.1)
    assert not far.association.matches and far.association.unmatched_left[0].reason=='geometry_blocked'
    empty=camera(5,[])
    assert match(left,empty).association.unmatched_left[0].reason=='empty_opposite_camera'
    assert not match(camera(4,[]),empty).candidates
    print('Inclusive distance boundary, strict appearance boundary, all-blocked and empty cameras: OK')
    missing=camera(5,[(20,[1,0],None)])
    assert match(left,missing).association.matches
    blocked=match(left,missing,unavailable_policy='reject')
    assert not blocked.association.matches and blocked.candidates[0].geometry_decision=='unavailable_rejected'
    fallback=match(left,missing)
    assert fallback.candidates[0].distance is None and fallback.candidates[0].geometry_decision=='unavailable_appearance_fallback'
    assert fallback.association==associate_camera_pair(left[0],missing[0],min_similarity=.5)
    print('Unavailable geometry has explicit appearance fallback or rejection; malformed inputs do not fall back: OK')
    swapped=match(right,left)
    assert {(m.left,m.right) for m in result.association.matches}=={(m.right,m.left) for m in swapped.association.matches}
    permuted=camera(5,[(30,[.9,np.sqrt(1-.9**2)],(1.,0.)),(20,[.99,np.sqrt(1-.99**2)],(10.,0.))])
    assert match(left,permuted)==result
    assert match(left,(right[0],replace(right[1],observations=tuple(reversed(right[1].observations)))))==result
    tie_a=camera(4,[(2,[1,0],(0.,0.)),(1,[1,0],(0.,0.))])
    tie_b=camera(5,[(4,[1,0],(0.,0.)),(3,[1,0],(0.,0.))])
    assert match(tie_a,tie_b).association==associate_camera_pair(tie_a[0],tie_b[0],min_similarity=.5)
    assert {(m.left,m.right) for m in match(tie_a,tie_b).association.matches}=={(m.right,m.left) for m in match(tie_b,tie_a).association.matches}
    print('Camera orientation, key-based point mapping, input order and exact-score ties preserve canonical results: OK')
    three=[camera(c,[(1,[1,0],(0.,0.))]) for c in (4,5,8)]
    gated=[match(a,b) for a,b in combinations(three,2)]
    pairs=[r.association for r in gated]
    groups=group_geometry_associations(gated)
    assert len(groups.groups)==1 and len(groups.groups[0])==3
    assert groups==group_pair_associations(pairs)
    rejected(lambda: group_geometry_associations([replace(gated[0],max_distance=3.),*gated[1:]]))
    rejected(lambda: group_geometry_associations([replace(gated[0],unavailable_policy='reject'),*gated[1:]]))
    print('Grouping preserves the existing API and rejects mixed geometry settings: OK')
    h=np.array([[2.,0.,10.],[0.,3.,20.],[.1,0.,1.]])
    u,v=14/1.2,29/1.2
    box=[u-1,v-3,u+1,v]
    assert np.allclose(project_box_foot(h,box),(2,3))
    assert np.allclose(project_box_foot(-7*h,box),(2,3))
    assert project_box_foot(h,[19,5,21,10]) is None
    assert project_box_foot(np.eye(3),[-10,0,10,20])==(0.,20.)
    print('Raw box projection, homogeneous scaling and near-horizon unavailability: OK')
    for value in (0,-1,True,float('inf'),float('nan')):
        rejected(lambda v=value: match(left,right,max_distance=v))
    rejected(lambda: match(left,right,unavailable_policy='guess'))
    rejected(lambda: match(left,(right[0],replace(right[1],coordinate_space='another-plane'))))
    rejected(lambda: match(left,(right[0],replace(right[1],timestamp=Fraction(2,10)))))
    rejected(lambda: match(left,(right[0],replace(right[1],run_id='another-run'))))
    rejected(lambda: match(left,(right[0],replace(right[1],observations=()))))
    rejected(lambda: match(left,(right[0],replace(right[1],observations=(right[1].observations[0],)*2))))
    for xy in ((float('nan'),0.),(True,0.),[0.,0.]):
        bad=(replace(right[1].observations[0],xy=xy),right[1].observations[1])
        rejected(lambda b=bad: match(left,(right[0],replace(right[1],observations=b))))
    rejected(lambda: match(left,(replace(right[0],descriptor_variant='latest'),right[1])))
    rejected(lambda: solve_geometry_assignment([[float('nan')]],min_similarity=.5,admissible=np.array([[False]])))
    rejected(lambda: solve_geometry_assignment([[.9]],min_similarity=.5,admissible=np.array([[1]])))
    rejected(lambda: project_box_foot(np.zeros((3,3)),[0,0,1,1]))
    rejected(lambda: project_box_foot(np.eye(3),[0,0,0,1]))
    print('Mixed scopes, incomplete/duplicate geometry, invalid coordinates, masks and matrices rejected: OK')
    print('Synthetic distances are test fixtures; no deployment threshold is selected.')
    print('Geometry association smoke test: PASSED')


if __name__=='__main__':
    main()
