"""Known-answer end-to-end geometry/grouping/identity/evaluation contract check."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import numpy as np

import run_geometry_identity_experiment as experiment
from mtmc.association import geometry, pairwise, grouping, controlled_merge
from mtmc.reid.osnet import ObservationKey


def rejected(callback):
    try:
        callback()
    except (ValueError,TypeError):
        return
    raise AssertionError('Invalid serialized metadata accepted')


def main():
    run='synthetic-geometry-identity'
    config={'max_distance':2.,'unavailable_policy':'appearance_only'}
    space='synthetic-plane/native'
    common=dict(run_id=run,max_idle=Fraction(1),descriptor_variant='mean',min_similarity=.5,
                min_support_rounds=2,min_support_seconds=Fraction(1,30),max_evidence_gap=Fraction(1,10))
    before_manager=controlled_merge.ControlledMergeIdentityManager(**common)
    after_manager=controlled_merge.ControlledMergeIdentityManager(**common)
    before_life,after_life=experiment.Lifecycle(),experiment.Lifecycle()
    before,after=experiment.evaluation.IdentityCounts(),experiment.evaluation.IdentityCounts()
    features=np.zeros((8,512),np.float32);features[:,0]=1
    snapshots=[]
    for frame in range(4):
        records=[{'camera':c,'local_id':i+1,'frame_index':frame,'embedding_row':frame*2+i} for i,c in enumerate((4,5))]
        cameras={c:experiment.pair_io.make_camera(run,c,frame,'mean',records,features) for c in (4,5)}
        positions={c:geometry.CameraGroundPositions(run,c,frame,Fraction(frame,30),space,
            (geometry.GroundObservation(cameras[c].observations.keys[0],(0.,0.) if c==4 else (10.,0.)),)) for c in (4,5)}
        baseline=pairwise.associate_camera_pair(cameras[4],cameras[5],min_similarity=.5)
        gated=geometry.associate_camera_pair_with_geometry(cameras[4],cameras[5],positions[4],positions[5],
                    min_similarity=.5,**config)
        old_groups=grouping.group_pair_associations([baseline])
        new_groups=geometry.group_geometry_associations([gated])
        old=before_manager.update(old_groups);new=after_manager.update(new_groups)
        before_life.update(old);after_life.update(new)
        labels={ObservationKey(r['camera'],r['local_id'],frame):{'embedding_row':r['embedding_row'],'diagnostic_gt_id':r['local_id']}
                for r in records}
        record=experiment.replay.encode_output(new,labels,'candidate-scope')
        record.update(association_policy=geometry.POLICY,geometry_configuration=config,coordinate_space=space)
        plain=experiment.plain(record)
        context=dict(geometry_config=config,coordinate_space=space,source_run=run,scope='candidate-scope',frame=frame,
                     variant='mean',threshold=.5,expected_rows={k:v['embedding_row'] for k,v in labels.items()})
        assigned=experiment.read_geometry_record(plain,**context)
        for assignment in old.assignments:
            before.update([labels[assignment.key]['diagnostic_gt_id']],[assignment.global_id],np.ones((1,1),bool))
        for key,gid in assigned.items():
            after.update([labels[key]['diagnostic_gt_id']],[gid],np.ones((1,1),bool))
        assert len(experiment.edge_set([baseline]))==1 and not experiment.edge_set([gated.association])
        assert len(experiment.partition_edges(old_groups.groups))==1 and not experiment.partition_edges(new_groups.groups)
        snapshots.append((plain,deepcopy(plain)))
    assert before.result()[0]['idf1']==.5 and after.result()[0]['idf1']==1.
    assert before_life.summary()['allocated_ids']==1 and after_life.summary()['allocated_ids']==2
    assert all(original==copy for original,copy in snapshots)
    print('Known-answer appearance -> grouping -> controlled identity -> global evaluation: 50% versus 100%: OK')
    print('Same observation coverage, identity lifecycle and immutable earlier outputs: OK')
    assert plain['policy']==controlled_merge.POLICY and plain['association_policy']==geometry.POLICY
    assert plain['timestamp']=='1/10'
    wrong=deepcopy(plain);wrong['association_policy']='appearance_only'
    rejected(lambda:experiment.read_geometry_record(wrong,**context))
    wrong=deepcopy(plain);wrong['geometry_configuration']['max_distance']=3.
    rejected(lambda:experiment.read_geometry_record(wrong,**context))
    wrong=deepcopy(plain);wrong['coordinate_space']='another-plane'
    rejected(lambda:experiment.read_geometry_record(wrong,**context))
    wrong=deepcopy(plain);wrong['assignments'][0]['embedding_row']+=100
    rejected(lambda:experiment.read_geometry_record(wrong,**context))
    print('Serialized identity and association policies, exact time, source rows and geometry metadata: OK')
    print('Scores and distances above are synthetic known answers, not real-scene results.')
    print('Geometry identity experiment checks: PASSED')


if __name__=='__main__':
    main()
