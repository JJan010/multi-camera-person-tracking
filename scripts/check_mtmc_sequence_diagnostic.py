"""Known answers for local/global constraints and temporal evidence summaries."""
import numpy as np

import diagnose_mtmc_sequence as diagnostic
from mtmc.reid.osnet import ObservationKey


def main():
    # Same correct local ID in each camera, but no shared cross-camera ID.
    local = {c:diagnostic.metric.IdentityCounts() for c in (4,5)}
    global_counts = diagnostic.metric.IdentityCounts()
    for c,gid in ((4,1),(5,2)):
        local[c].update([7],[1],np.ones((1,1),bool))
        global_counts.update([7],[gid],np.ones((1,1),bool))
    pooled,per_camera = diagnostic.pooled_local(local)
    global_metrics,_ = global_counts.result()
    parts = diagnostic.original.decompose(global_metrics,2)
    assert pooled["idf1"] == 1 and global_metrics["idf1"] == .5
    assert parts["framewise_spatial_f1_ceiling"] == 1 and parts["shared_identity_assignment_gap"] == 1
    print("Perfect per-camera identity and boxes can coexist with global IDF1=50%: OK")
    # Pool counts rather than averaging camera percentages: supports 1 and 3.
    local = {c:diagnostic.metric.IdentityCounts() for c in (4,5)}
    local[4].update([7],[1],np.ones((1,1),bool))
    for _ in range(3):
        local[5].update([7],[1],np.zeros((1,1),bool))
    pooled,_ = diagnostic.pooled_local(local)
    assert pooled["idf1"] == .25
    print("Pooled local IDF1 sums counts, not per-camera percentages: OK")
    e = diagnostic.Evidence()
    e.update(2,[(ObservationKey(4,1,2),7,10),(ObservationKey(5,1,2),7,11)])
    e.update(3,[(ObservationKey(4,1,3),7,10),(ObservationKey(5,1,3),8,10)])
    e.update(4,[])  # Missing evidence breaks contiguous segments.
    e.update(5,[(ObservationKey(4,1,5),7,10)])
    gt,global_rows,local_rows = e.tables()
    assert next(r for r in gt if r["gt_id"]==7)["simultaneous_split_frames"] == 1
    assert next(r for r in global_rows if r["global_id"]==10)["simultaneous_mix_frames"] == 1
    assert next(r for r in local_rows if r["camera"]==5)["distinct_gt_ids"] == 2
    assert len(e.transitions) == 1 and e.transitions[0]["previous_gt"] == 7 and e.transitions[0]["current_gt"] == 8
    segments = e.all_segments()
    assert sum(r["observations"] for r in segments) == 5
    camera4 = [r for r in segments if r["camera"]==4]
    assert [(r["first_frame"],r["last_frame"],r["observations"]) for r in camera4] == [(2,3,2),(5,5,1)]
    assert e.all_segments() == segments
    print("Global fragmentation, simultaneous mixing, local label transitions and gap-separated segments: OK")
    mask = np.array([[True,True],[True,False]])
    keys = [ObservationKey(4,1,2),ObservationKey(4,2,2)]
    maximum,evidence,stats = diagnostic.original.inspect_slot([7,8],keys,[1,2],mask)
    assert maximum == 2 and not evidence
    print("Ambiguous overlaps contribute to spatial ceiling but not identity evidence: OK")
    print("MTMC sequence diagnostic checks: PASSED")


if __name__ == "__main__":
    main()
