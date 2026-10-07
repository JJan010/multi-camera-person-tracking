"""Known-answer checks for projection and offline pair rejection accounting."""
import numpy as np
from evaluate_geometry_pairs import Bucket, category, distance_summary, project_box


def main():
    assert category(None,None)=='unknown'
    assert category(1,None)=='unknown'
    assert category(1,1)=='same' and category(1,2)=='different'
    bucket = Bucket()
    for label,distance in [('same',.5),('same',.50001),('same',None),('different',.25),('different',2.),('unknown',3.),('unknown',None)]:
        bucket.add(label,distance)
    result = bucket.sweep(.5)
    assert (result['same_total'],result['same_retained'],result['same_rejected'],result['same_deferred'])==(3,1,1,1)
    assert (result['different_retained'],result['different_rejected'])==(1,1)
    assert (result['unknown_rejected'],result['unknown_deferred'])==(1,1)
    for label in ('same','different','unknown'):
        assert result[f'{label}_total']==sum(result[f'{label}_{s}'] for s in ('retained','rejected','deferred'))
    assert result['same_rejected_fraction_measurable']==.5
    assert Bucket().sweep(1.)['same_rejected_fraction_measurable'] is None
    assert distance_summary([])['median'] is None
    print('Inclusive distance boundary, empty sets, unknown labels and unavailable-geometry deferral: OK')
    h=np.array([[2.,0.,10.],[0.,3.,20.],[.1,0.,1.]])
    u,v=14/1.2,29/1.2
    foot,world,inside=project_box(h,[u-1,v-3,u+1,v])
    assert np.allclose(world,[2,3]) and inside
    assert np.allclose(project_box(-7*h,[u-1,v-3,u+1,v])[1],world)
    assert project_box(h,[19,5,21,10])[1] is None
    foot,world,inside=project_box(np.eye(3),[-10,0,10,20])
    assert np.array_equal(foot,[0,20]) and np.array_equal(world,[0,20]) and not inside
    print('Known ground projection, projective scale, horizon and raw un-clipped footpoint: OK')
    for box in ([0,0,0,1],[0,0,float('nan'),1],[1,2,3]):
        try:
            project_box(np.eye(3),box)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid box accepted')
    print('Invalid tracked boxes rejected: OK')
    print('Geometry pair diagnostic checks: PASSED')


if __name__=='__main__':
    main()
