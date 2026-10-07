"""Check that the staged control permits only an explicit run-scope change."""
from copy import deepcopy
from evaluate_competitive_global import check_staged


def main():
    reference = {'cameras':[{'camera':8,'local_ids':[44],'detection_indices':[20]}],
                 'identity':{'run_id':'old/direct_appearance','frame_index':1441,
                             'assignments':[{'global_id':112,'key':{'camera_id':8,'local_id':44,'frame_index':1441}}],
                             'merge_events':[], 'identities':[{'global_id':112}]}}
    actual = deepcopy(reference); actual['identity']['run_id']='new/staged'
    before = deepcopy((actual,reference))
    check_staged(actual,reference,current_scope='new/staged',reference_scope='old/direct_appearance')
    assert (actual,reference)==before
    print('Only the declared run scope may differ; inputs remain unchanged: OK')
    for field in ('scope','frame','global_id','candidate','merge','state'):
        bad=deepcopy(actual)
        if field=='scope': bad['identity']['run_id']='unexpected'
        if field=='frame': bad['identity']['frame_index']+=1
        if field=='global_id': bad['identity']['assignments'][0]['global_id']+=1
        if field=='candidate': bad['cameras'][0]['detection_indices'][0]+=1
        if field=='merge': bad['identity']['merge_events'].append({'canonical_global_id':112})
        if field=='state': bad['identity']['identities']=[]
        try:
            check_staged(bad,reference,current_scope='new/staged',reference_scope='old/direct_appearance')
        except ValueError:
            pass
        else:
            raise AssertionError(f'Changed {field} accepted')
    print('Changed scopes, frames, assigned IDs, candidate provenance, merges and retained state rejected: OK')
    print('Competitive global input checks: PASSED')


if __name__=='__main__':
    main()
