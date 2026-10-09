"""Known-answer paired ranking, AP, eligibility and mapping checks."""
import numpy as np
from compare_clipreid_snapshot import retrieval, paired_transitions, features_valid


def main():
    labels=[dict(camera=4,local_id=0,gt_id=0),
            dict(camera=4,local_id=1,gt_id=1),
            dict(camera=4,local_id=2,gt_id=2),
            dict(camera=4,local_id=3,gt_id=None),
            dict(camera=5,local_id=0,gt_id=0),
            dict(camera=5,local_id=1,gt_id=1),
            dict(camera=5,local_id=2,gt_id=None)]
    # Unknown gallery observations remain distractors, including a stable tie.
    old=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                  [0,1,0,0],[1,0,0,0],[1,0,0,0]],np.float32)
    new=old.copy();new[[4,5]]=new[[5,4]]
    a, ar=retrieval(old,labels,[4,5]);b,br=retrieval(new,labels,[4,5])
    direction=a['directions'][0]
    assert direction['evaluated_queries']==2 and direction['unmatched_queries']==1
    assert direction['no_positive_in_gallery']==1 and direction['rank1_hits']==0
    assert direction['mAP_single_positive']==(1/3+1/2)/2
    assert b['pooled']['rank1']==1 and b['pooled']['mAP_single_positive']==1
    transitions,changes=paired_transitions(ar,br)
    assert transitions==dict(both_correct=0,improved=4,worsened=0,both_wrong=0) and len(changes)==4
    padded=np.pad(new,((0,0),(0,3)))
    c,cr=retrieval(padded,labels,[4,5]);assert c==b and cr==br
    # No eligible positives must be reported as undefined, not zero accuracy.
    unknown=[dict(r,gt_id=None) for r in labels]
    empty,er=retrieval(new,unknown,[4,5])
    assert empty['pooled']['rank1'] is None and empty['pooled']['mAP_single_positive'] is None
    assert paired_transitions(er,er)[0]==dict(both_correct=0,improved=0,worsened=0,both_wrong=0)
    try:paired_transitions(ar,br[::-1])
    except ValueError:pass
    else:raise AssertionError('Changed query mapping accepted')
    features_valid(new,7,4)
    for value in (new.astype(np.float64),new*2,np.full_like(new,np.nan)):
        try:features_valid(value,7,4)
        except ValueError:pass
        else:raise AssertionError('Invalid features accepted')
    print('Paired ranking/AP, GT zero, unknown distractors, absent positives, stable ties, independent dimensions and mapping rejection: PASSED')


if __name__=='__main__':main()
