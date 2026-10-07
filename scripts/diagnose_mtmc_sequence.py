"""Separate fixed-box limitations from shared-ID errors in a frozen MTMC run."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path

import diagnose_global_identity as original
import evaluate_mtmc_sequence as sequence
from mtmc.reid.osnet import sha256

ROOT = Path(__file__).resolve().parents[1]
metric,require = sequence.metric,sequence.require


def pooled_local(counts):
    """Sum counts after independent per-camera identity matchings, never percentages."""
    per_camera = {str(c): counter.result()[0] for c,counter in sorted(counts.items())}
    totals = {k:sum(row[k] for row in per_camera.values())
              for k in ("gt_observations","predicted_observations","idtp","idfp","idfn")}
    ng,npred,tp = (totals[k] for k in ("gt_observations","predicted_observations","idtp"))
    totals.update(idf1=2*tp/(ng+npred) if ng+npred else None,
                  idp=tp/npred if npred else None,idr=tp/ng if ng else None)
    return totals,per_camera


class Evidence:
    """Mutually unique spatial evidence; not a ground-truth runtime assignment."""
    def __init__(self):
        self.gt_global,self.global_gt,self.local_gt = defaultdict(Counter),defaultdict(Counter),defaultdict(Counter)
        self.gt_simultaneous,self.global_simultaneous = Counter(),Counter()
        self.last_local = {}
        self.transitions = []
        self.open_segments = {}
        self.segments = []

    def update(self,frame,items):
        frame_gt,frame_global = defaultdict(set),defaultdict(set)
        for key,gt,gid in items:
            require(key.frame_index == frame,"Evidence frame differs")
            local = key.camera_id,key.local_id
            self.gt_global[gt][gid] += 1
            self.global_gt[gid][gt] += 1
            self.local_gt[local][gt] += 1
            frame_gt[gt].add(gid);frame_global[gid].add(gt)
            previous = self.last_local.get(local)
            if previous is not None and previous[1] != gt:
                self.transitions.append({"camera":key.camera_id,"local_id":key.local_id,
                    "previous_evidence_frame":previous[0],"frame":frame,"gap_frames":frame-previous[0],
                    "previous_gt":previous[1],"current_gt":gt,"previous_global_id":previous[2],"current_global_id":gid})
            self.last_local[local] = frame,gt,gid
            segment = self.open_segments.get(local)
            if segment and segment["gt_id"] == gt and segment["global_id"] == gid and segment["last_frame"] == frame-1:
                segment["last_frame"] = frame
                segment["observations"] += 1
            else:
                if segment:
                    self.segments.append(segment)
                self.open_segments[local] = {"camera":key.camera_id,"local_id":key.local_id,"gt_id":gt,
                    "global_id":gid,"first_frame":frame,"last_frame":frame,"observations":1}
        self.gt_simultaneous.update(gt for gt,ids in frame_gt.items() if len(ids)>1)
        self.global_simultaneous.update(gid for gid,ids in frame_global.items() if len(ids)>1)

    def tables(self):
        gt_rows,global_rows,local_rows = [],[],[]
        for gt,hist in sorted(self.gt_global.items()):
            major,n = sorted(hist.items(),key=lambda x:(-x[1],x[0]))[0]
            gt_rows.append({"gt_id":gt,"distinct_global_ids":len(hist),"evidence_observations":sum(hist.values()),
                "largest_global_id":major,"evidence_outside_largest_id":sum(hist.values())-n,
                "simultaneous_split_frames":self.gt_simultaneous[gt],"support_by_global_id":dict(sorted(hist.items()))})
        for gid,hist in sorted(self.global_gt.items()):
            major,n = sorted(hist.items(),key=lambda x:(-x[1],x[0]))[0]
            global_rows.append({"global_id":gid,"distinct_gt_ids":len(hist),"evidence_observations":sum(hist.values()),
                "largest_gt_id":major,"evidence_outside_largest_gt":sum(hist.values())-n,
                "simultaneous_mix_frames":self.global_simultaneous[gid],"support_by_gt_id":dict(sorted(hist.items()))})
        for (c,lid),hist in sorted(self.local_gt.items()):
            major,n = sorted(hist.items(),key=lambda x:(-x[1],x[0]))[0]
            local_rows.append({"camera":c,"local_id":lid,"distinct_gt_ids":len(hist),
                "evidence_observations":sum(hist.values()),"largest_gt_id":major,
                "evidence_outside_largest_gt":sum(hist.values())-n,"support_by_gt_id":dict(sorted(hist.items()))})
        gt_rows.sort(key=lambda r:(-r["evidence_outside_largest_id"],r["gt_id"]))
        global_rows.sort(key=lambda r:(-r["evidence_outside_largest_gt"],r["global_id"]))
        local_rows.sort(key=lambda r:(-r["evidence_outside_largest_gt"],r["camera"],r["local_id"]))
        return gt_rows,global_rows,local_rows

    def all_segments(self):
        return sorted([*self.segments,*self.open_segments.values()],key=lambda r:(r["camera"],r["local_id"],r["first_frame"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report",type=Path,required=True)
    args = parser.parse_args()
    inputs = {}
    def checked(name,path,digest=None):
        path = Path(path).resolve()
        actual = sha256(path)
        require(digest is None or actual == digest,f"Checksum mismatch: {path}")
        inputs[name] = {"path":str(path),"sha256":actual}
        return path
    evaluation_path = checked("sequence_evaluation",args.evaluation_report)
    evaluated = json.loads(evaluation_path.read_text())
    require(evaluated.get("completed") and evaluated["protocol"]["name"] == "scene_001_sequence_global_2d_identity_v1",
            "Expected completed sequence evaluation")
    for name in ("pipeline_report","tracks","global_tracks","ground_truth"):
        item = evaluated["inputs"][name]
        checked(name,item["path"],item["sha256"])
    source = json.loads(Path(inputs["pipeline_report"]["path"]).read_text())
    rounds = source["summary"]["rounds"]
    cfg = source["configuration"]
    require(source.get("completed") and source["run_id"] == evaluated["source_run_id"]
            and evaluated["protocol"]["frames"] == [2,rounds-1]
            and source["protocol"] == "scene_001_mtmc_sequential_fp32_v1", "Mixed source scope")
    for name in ("tracks","global_tracks"):
        require(source["artifacts"][name]["sha256"] == inputs[name]["sha256"],"Source artifact/evaluation differs")
    print("Verifying frozen run; loading GT without modifying runtime outputs...",flush=True)
    ground = sequence.load_ground_truth(inputs["ground_truth"]["path"],rounds)
    global_counts,control_counts = metric.IdentityCounts(),metric.IdentityCounts()
    local_counts = {c:metric.IdentityCounts() for c in sequence.CAMERAS}
    local_control = {}
    next_row = total_observations = spatial_tp = 0
    spatial_stats = Counter()
    evidence = Evidence()
    decision_counts,decision_examples = defaultdict(Counter),[]
    prefixes = []
    expected_prefixes = {p["runtime_rounds"]:p for p in evaluated["prefixes"]}
    require(len(expected_prefixes) == len(evaluated["prefixes"]) and rounds in expected_prefixes
            and all(type(n) is int and 3 <= n <= rounds for n in expected_prefixes), "Invalid evaluated prefix coverage")
    emitted = set()
    with Path(inputs["tracks"]["path"]).open() as tracks,Path(inputs["global_tracks"]["path"]).open() as globals_file:
        for frame in range(rounds):
            a,b = tracks.readline(),globals_file.readline()
            require(bool(a) and bool(b),"Truncated diagnostic source")
            local,record = json.loads(a),json.loads(b)
            cameras,assignments,next_row = sequence.validate_round(local,record,frame=frame,run=source["run_id"],config=cfg,next_row=next_row)
            total_observations += len(assignments);emitted.update(assignments.values())
            if frame < 2:
                continue
            all_evidence = []
            for camera in sequence.CAMERAS:
                keys,boxes = cameras[camera]
                gt,mask,_,_,_ = sequence.spatial_slot(ground[frame,camera],keys,boxes)
                gids = [assignments[k] for k in keys]
                control_ids = []
                for key in keys:
                    pair = key.camera_id,key.local_id
                    if pair not in local_control:
                        local_control[pair] = len(local_control)+1
                    control_ids.append(local_control[pair])
                global_counts.update(gt,gids,mask)
                control_counts.update(gt,control_ids,mask)
                local_counts[camera].update(gt,[k.local_id for k in keys],mask)
                ceiling,items,stats = original.inspect_slot(gt,keys,gids,mask)
                spatial_tp += ceiling;spatial_stats.update(stats);all_evidence.extend(items)
            evidence.update(frame,all_evidence)
            unique_gt = {key:gt for key,gt,gid in all_evidence}
            for decision in record["merge_decisions"]:
                category = original.classify_conflict(decision["members"],unique_gt)
                decision_counts[decision["outcome"]][category] += 1
                if len(decision_examples) < 20 and decision["outcome"].startswith("blocked") and category == "all_members_same_gt":
                    decision_examples.append({"frame":frame,"timestamp":record["timestamp"],**decision,
                        "diagnostic_members":[{**k,"unique_overlap_gt_id":unique_gt.get(sequence.observation_key(k,frame))}
                                              for k in decision["members"]]})
            if frame+1 in expected_prefixes:
                computed = sequence.prefix_result(global_counts,control_counts,frame+1)
                require(computed == expected_prefixes[frame+1],"Published prefix metrics differ")
                local_metric,_ = pooled_local(local_counts)
                current,_ = global_counts.result()
                parts = original.decompose(current,spatial_tp)
                prefixes.append({**computed,"local_pooled_idf1":local_metric["idf1"],
                    "spatial_f1_ceiling":parts["framewise_spatial_f1_ceiling"],
                    "minimum_fn_with_fixed_boxes":parts["minimum_fn_with_fixed_boxes"],
                    "minimum_fp_with_fixed_boxes":parts["minimum_fp_with_fixed_boxes"],
                    "shared_identity_assignment_gap":parts["shared_identity_assignment_gap"]})
                print(f"Through frame {frame}: global={current['idf1']:.2%}; local pooled={local_metric['idf1']:.2%}; "
                      f"spatial ceiling={parts['framewise_spatial_f1_ceiling']:.2%}",flush=True)
        require(tracks.readline() == globals_file.readline() == "","Trailing diagnostic source rows")
    summary = source["summary"]
    require(next_row == summary["total_embeddings"] and total_observations == summary["observations"]
            and len(emitted) == summary["ever_emitted_global_ids"],"Source summary counts differ")
    metrics,_ = global_counts.result()
    control_metric,_ = control_counts.result()
    require(metrics == evaluated["pipeline_metrics"] and control_metric == evaluated["no_cross_camera_control"],
            "Published global/control metrics differ")
    decomposition = original.decompose(metrics,spatial_tp)
    local_metric,per_camera = pooled_local(local_counts)
    gt_rows,global_rows,local_rows = evidence.tables()
    segments = evidence.all_segments()
    require(sum(r["observations"] for r in segments) == spatial_stats["mutually_unique_pairs"],"Segment evidence accounting differs")
    now = datetime.now(timezone.utc)
    output = ROOT/"artifacts/mtmc_sequence_diagnostic"/now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in (("prefix_diagnostics.csv",prefixes),("gt_fragmentation.csv",gt_rows),
                      ("global_mixing.csv",global_rows),("local_mixing.csv",local_rows),
                      ("evidence_segments.csv",segments),("local_label_transitions.csv",evidence.transitions)):
        if rows:
            csv_rows = [{k:json.dumps(v,sort_keys=True) if isinstance(v,dict) else v for k,v in row.items()} for row in rows]
            metric.history.write_csv(output/name,csv_rows)
    (output/"blocked_merge_examples.json").write_text(json.dumps(decision_examples,indent=2)+"\n")
    for item in inputs.values():
        metric.checked(item["path"],item["sha256"])
    report = {"completed":True,"created_utc":now.isoformat(),"source_run_id":source["run_id"],
        "protocol":{"name":"scene_001_sequence_failure_diagnostic_v1","frames":[2,rounds-1],
            "cameras":list(sequence.CAMERAS),"min_iou":.5,"predictions_modified":False,
            "evidence":"Mutually unique same-camera/frame IoU overlaps; not manual identity labels",
            "local_pooled":"Sum IDTP/IDFP/IDFN after independent per-camera identity matchings",
            "selected_deployment_threshold":None},
        "inputs":inputs,"global_metrics_reproduced":metrics,"no_cross_camera_control_reproduced":control_metric,
        "decomposition":decomposition,"local_pooled_metrics":local_metric,"per_camera_local_metrics":per_camera,
        "prefixes":prefixes,"spatial_diagnostics":dict(spatial_stats),
        "merge_decisions_by_listed_member_evidence":{k:dict(v) for k,v in sorted(decision_counts.items())},
        "evidence_summary":{"gt_identities_with_evidence":len(gt_rows),"global_ids_with_evidence":len(global_rows),
            "gt_identities_with_multiple_global_ids":sum(r["distinct_global_ids"]>1 for r in gt_rows),
            "global_ids_with_multiple_gt_ids":sum(r["distinct_gt_ids"]>1 for r in global_rows),
            "local_tracks_with_multiple_gt_ids":sum(r["distinct_gt_ids"]>1 for r in local_rows),
            "local_evidence_label_transitions":len(evidence.transitions)},
        "top_gt_fragmentation":gt_rows[:5],"top_global_mixing":[r for r in global_rows if r["distinct_gt_ids"]>1][:5],
        "top_local_mixing":[r for r in local_rows if r["distinct_gt_ids"]>1][:5],
        "checks":{"global_control_and_all_prefix_metrics_reproduced":True,"decomposition_accounting_verified":True,
                  "evidence_segment_coverage_verified":True,"frozen_inputs_unchanged":True},
        "code_sha256":{str(Path(m.__file__).relative_to(ROOT)):sha256(Path(m.__file__)) for m in
                       (sequence,sequence.baseline,original,metric,metric.history,metric.history.snapshot)},
        "artifacts":{p.name:{"path":p.name,"sha256":sha256(p)} for p in sorted(output.iterdir())},
        "limits":["Framewise spatial ceiling is GT-assisted and not a promised causal identity score",
                  "Shared-ID gap includes global mapping constraints and competing boxes; not a causal attribution to one module",
                  "Local pooled and global IDF1 use different identity constraints; their difference is not an error decomposition",
                  "Unique IoU evidence can be wrong and excludes ambiguous overlaps; confirm selected examples visually",
                  "Multiple emitted global IDs for one GT also arise from correct causal merges that do not rewrite history",
                  "Local label transitions are not official CLEAR ID switches; gaps and matching errors can contribute",
                  "Blocked-merge labels cover listed candidate members, not necessarily all absent retained or other visible members",
                  "Single training scene; no thresholds changed; no future-assisted runtime repair"]}
    report["code_sha256"][str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    path = output/"report.json";path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"Reproduced global IDF1: {metrics['idf1']:.2%}")
    print(f"Independent-camera pooled local IDF1: {local_metric['idf1']:.2%}")
    print(f"Framewise spatial F1 ceiling: {decomposition['framewise_spatial_f1_ceiling']:.2%}")
    print(f"IDFN {metrics['idfn']} = fixed-box minimum {decomposition['minimum_fn_with_fixed_boxes']} + shared-ID gap {decomposition['shared_identity_assignment_gap']}")
    print(f"IDFP {metrics['idfp']} = fixed-box minimum {decomposition['minimum_fp_with_fixed_boxes']} + shared-ID gap {decomposition['shared_identity_assignment_gap']}")
    print("Mutually unique overlap evidence:",json.dumps(report["evidence_summary"]))
    print("Merge decisions by listed-member evidence:",json.dumps(report["merge_decisions_by_listed_member_evidence"]))
    print("Top GT fragmentation: GT / global IDs / evidence outside largest ID / simultaneous-split frames")
    for row in gt_rows[:5]:
        print(row["gt_id"],row["distinct_global_ids"],row["evidence_outside_largest_id"],row["simultaneous_split_frames"])
    print("Top global mixing: global ID / GT IDs / evidence outside largest GT / simultaneous-mix frames")
    for row in report["top_global_mixing"]:
        print(row["global_id"],row["distinct_gt_ids"],row["evidence_outside_largest_gt"],row["simultaneous_mix_frames"])
    print("Top local mixing: camera / local ID / GT IDs / evidence outside largest GT")
    for row in report["top_local_mixing"]:
        print(row["camera"],row["local_id"],row["distinct_gt_ids"],row["evidence_outside_largest_gt"])
    print(f"Report: {path}")
    print("MTMC sequence failure diagnostic: COMPLETED; runtime unchanged")


if __name__ == "__main__":
    main()
