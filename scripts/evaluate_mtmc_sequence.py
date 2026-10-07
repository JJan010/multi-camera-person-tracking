"""Evaluate growing prefixes of a completed scene_001 MTMC run; CPU only."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

import evaluate_mtmc_pipeline as baseline
from evaluate_mtmc_pipeline import CAMERAS, require, validate_round, spatial_slot, observation_key
from mtmc.association import geometry, controlled_merge
from mtmc.reid.osnet import sha256

metric = baseline.metric
ROOT = Path(__file__).resolve().parents[1]


def load_ground_truth(path, rounds):
    """Same audited camera subset and spatial convention, variable prefix end.

    Cameras 4/5/8 each have annotated observations on every frame 2..23993 in
    pinned scene_001. This completeness requirement is scene-specific, not a
    general assumption that an empty GT frame is always invalid.
    """
    require(type(rounds) is int and 3 <= rounds <= 23994, "Unsupported GT duration")
    ground = defaultdict(dict)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            require(len(fields) == 9, "Expected nine GT columns")
            camera, identity, frame = map(int, fields[:3])
            if camera not in CAMERAS or not 2 <= frame < rounds:
                continue
            x,y,width,height = map(float,fields[3:7])
            require(identity >= 0 and np.isfinite([x,y,width,height]).all()
                    and width > 0 and height > 0, "Invalid GT identity/box")
            require(identity not in ground[frame,camera], "Duplicate GT identity in camera/frame")
            ground[frame,camera][identity] = [x,y,x+width,y+height]
    require(all(ground[f,c] for f in range(2,rounds) for c in CAMERAS),
            "Incomplete GT for the audited scene_001 camera subset")
    return ground


def prefix_result(full, control, rounds):
    current,_ = full.result()
    reference,_ = control.result()
    require(all(current[k] == reference[k] for k in ("camera_time_slots","gt_observations","predicted_observations")),
            "Prefix denominators differ")
    return {"runtime_rounds":rounds,"video_duration_s":rounds/30,"first_evaluated_frame":2,
            "last_evaluated_frame":rounds-1,"gt_observations":current["gt_observations"],
            "predicted_observations":current["predicted_observations"],
            "pipeline_idf1":current["idf1"],"pipeline_idp":current["idp"],"pipeline_idr":current["idr"],
            "pipeline_idtp":current["idtp"],"pipeline_idfp":current["idfp"],"pipeline_idfn":current["idfn"],
            "pipeline_predicted_identities":current["predicted_identities"],
            "control_idf1":reference["idf1"],"control_idtp":reference["idtp"],
            "delta_control_pp":100*(current["idf1"]-reference["idf1"])}


def evaluate(run_report, *, prefix_stride=300):
    inputs = {}
    def checked(name,path,expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected,f"Checksum mismatch: {path}")
        inputs[name] = {"path":str(path),"sha256":digest}
        return path
    report_path = checked("pipeline_report",run_report)
    source = json.loads(report_path.read_text())
    cfg = source["configuration"]
    rounds = source["summary"]["rounds"]
    require(type(rounds) is int and 3 <= rounds <= 23994, "Unsupported scene duration")
    require(type(prefix_stride) is int and prefix_stride >= 3, "Invalid prefix stride")
    checkpoints = set(range(prefix_stride, rounds+1, prefix_stride)) | {rounds}
    prefixes = []
    require(source.get("completed") is True and source["protocol"] == "scene_001_mtmc_sequential_fp32_v1"
            and cfg["fps"] == 30 and cfg["cameras"] == [4,5,8]
            and cfg["identity_start_frame"] == 0 and cfg["ground_truth_used"] is False
            and cfg["association_policy"] == geometry.POLICY and cfg["identity_policy"] == controlled_merge.POLICY,
            "Expected completed scene_001 full pipeline, identities starting at frame zero")
    print("Verifying pipeline report, frozen outputs and pinned GT...",flush=True)
    paths = {}
    for name in ("tracks","global_tracks","embeddings","mean_embeddings","decisions"):
        item = source["artifacts"][name]
        paths[name] = checked(name,report_path.parent/item["path"],item["sha256"])
    item = source["inputs"]["scene_source_manifest"]
    manifest_path = checked("scene_source_manifest",item["path"],item["sha256"])
    manifest = json.loads(manifest_path.read_text())
    require(manifest["dataset"] == "nvidia/PhysicalAI-SmartSpaces"
            and manifest["revision"] == "2cbe9563cbe9f47f846e5c871ee994572bbbc60e"
            and manifest["scene"] == "MTMC_Tracking_2024/train/scene_001", "Unexpected dataset revision/scene")
    entries = [x for x in manifest["files"] if Path(x["local_path"]).name == "ground_truth.txt"]
    require(len(entries) == 1,"Missing/duplicate ground truth entry")
    gt_path = checked("ground_truth",ROOT/entries[0]["local_path"],entries[0]["sha256"])
    print(f"Loading GT for evaluation frames 2..{rounds-1}; warmup predictions remain included...",flush=True)
    ground = load_ground_truth(gt_path, rounds)
    full,control = metric.IdentityCounts(),metric.IdentityCounts()
    full_slots,control_slots = [],[]
    local_ids = {}
    emitted = set()
    next_row = total_observations = 0
    excluded,outside = Counter({c:0 for c in CAMERAS}),Counter({c:0 for c in CAMERAS})
    categories = Counter()
    merge_diagnostics = []
    with paths["tracks"].open() as tracks,paths["global_tracks"].open() as globals_file:
        for frame in range(rounds):
            a,b = tracks.readline(),globals_file.readline()
            require(bool(a) and bool(b),"Truncated pipeline traces")
            local,record = json.loads(a),json.loads(b)
            cameras,assignments,next_row = validate_round(local,record,frame=frame,run=source["run_id"],config=cfg,next_row=next_row)
            total_observations += len(assignments)
            emitted.update(assignments.values())
            unique_gt = {}
            if frame >= 2:
                for c in CAMERAS:
                    keys,boxes = cameras[c]
                    gt,mask,unique,ng,npred = spatial_slot(ground[frame,c],keys,boxes)
                    unique_gt.update(unique)
                    excluded[c] += ng; outside[c] += npred
                    global_ids,control_ids = [],[]
                    for k in keys:
                        key = (k.camera_id,k.local_id)
                        if key not in local_ids:
                            local_ids[key] = len(local_ids)+1
                        global_ids.append(assignments[k]); control_ids.append(local_ids[key])
                    full.update(gt,global_ids,mask);control.update(gt,control_ids,mask)
                    full_slots.append((gt,global_ids,mask));control_slots.append((gt,control_ids,mask))
            for event in record["merge_events"]:
                members = [observation_key(k,frame) for k in event["members"]]
                require(event["frame_index"] == frame and Fraction(event["timestamp"]) == Fraction(frame,30),"Merge event time differs")
                labels = [unique_gt.get(k) for k in members]
                category = ("unannotated_frame" if frame < 2 else "unresolved" if not labels or any(x is None for x in labels)
                            else "all_visible_members_same_gt" if len(set(labels)) == 1 else "different_known_gt")
                categories[category] += 1
                merge_diagnostics.append({**event,"diagnostic_category":category,
                    "diagnostic_members":[{**k,"unique_overlap_gt_id":label} for k,label in zip(event["members"],labels)]})
            if frame+1 in checkpoints:
                prefix = prefix_result(full, control, frame+1)
                prefixes.append(prefix)
                print(f"Prefix 0..{frame}: pipeline IDF1={prefix['pipeline_idf1']:.2%}; "
                      f"control={prefix['control_idf1']:.2%}",flush=True)
        require(tracks.readline() == globals_file.readline() == "","Unexpected trailing trace rows")
    summary = source["summary"]
    require(next_row == summary["total_embeddings"] and total_observations == summary["observations"]
            and len(emitted) == summary["ever_emitted_global_ids"] and len(merge_diagnostics) == summary["merge_events"],
            "Source summary/trace counts differ")
    for name in ("embeddings","mean_embeddings"):
        array = np.load(paths[name],mmap_mode="r",allow_pickle=False)
        require(array.shape == (next_row,512) and array.dtype == np.float32,"Embedding archive shape/type differs")
        del array
    metrics,mapping = full.result()
    control_metrics,control_mapping = control.result()
    require(all(metrics[k] == control_metrics[k] for k in ("camera_time_slots","gt_observations","predicted_observations")),
            "Metric denominators differ")
    print("Checking both shared identity metrics against motmetrics...",flush=True)
    metric.check_reference(metrics,metric.reference_metrics(full_slots))
    metric.check_reference(control_metrics,metric.reference_metrics(control_slots))
    for item in inputs.values():
        metric.checked(item["path"],item["sha256"])
    now = datetime.now(timezone.utc)
    output = ROOT/"artifacts/mtmc_sequence_evaluation"/now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True,exist_ok=False)
    metric.history.write_csv(output/"comparison.csv",[{"variant":"no_cross_camera_control",**control_metrics},
                                                      {"variant":"full_mtmc_pipeline",**metrics}])
    metric.history.write_csv(output/"prefixes.csv",prefixes)
    (output/"identity_matching.json").write_text(json.dumps({"pipeline":mapping,"control":control_mapping},indent=2)+"\n")
    (output/"merge_diagnostics.json").write_text(json.dumps(merge_diagnostics,indent=2)+"\n")
    report = {"completed":True,"created_utc":now.isoformat(),"source_run_id":source["run_id"],
        "protocol":{"name":"scene_001_sequence_global_2d_identity_v1","frames":[2,rounds-1],"cameras":list(CAMERAS),
            "fps":30,"min_iou":.5,"width":1920,"height":1080,"runtime_identity_start_frame":0,
            "timing_warmup_excluded_from_quality":False,"runtime_predictions_modified":False,
            "spatial_gate":"Same camera and same frame; all IoU >= 0.5 candidates retained",
            "box_policy":"Clip both; exclude zero-area GT; retain all predictions, including unencoded/outside",
            "identity_matching":"One shared one-to-one GT/global identity assignment over all cameras and evaluated frames",
            "control":"Camera-scoped original local IDs on identical boxes; no extra expiry or relabeling",
            "selected_deployment_threshold":None,
            "prefix_stride_rounds":prefix_stride, "prefix_rule":"Evaluate frames 2..end with one shared identity mapping per growing prefix; no runtime state reset"},
        "inputs":inputs,"prefixes":prefixes,"pipeline_metrics":metrics,"no_cross_camera_control":control_metrics,
        "delta_idf1_vs_control_pp":100*(metrics["idf1"]-control_metrics["idf1"]),
        "excluded_zero_area_gt":dict(excluded),"fully_outside_predictions":dict(outside),
        "merge_event_diagnostics":dict(categories),"runtime_summary":summary,
        "control_local_id_map":[{"camera":c,"local_id":i,"control_id":v} for (c,i),v in sorted(local_ids.items())],
        "checks":{"all_rounds_and_prediction_keys_verified":True,"embedding_rows_verified":True,
                  "both_metrics_agree_with_motmetrics":True,"shared_denominators":True,"frozen_inputs_unchanged":True},
        "versions":{name:version(name) for name in ("numpy","scipy","motmetrics","pandas")},
        "code_sha256":{str(Path(m.__file__).relative_to(ROOT)):sha256(Path(m.__file__)) for m in
                       (baseline,metric,metric.history,metric.history.snapshot,metric.history.history_io)},
        "artifacts":{p.name:{"path":p.name,"sha256":sha256(p)} for p in sorted(output.iterdir())},
        "limits":["Single training-scene temporal extension; not independent validation or official AI City evaluation",
                  "Identity state starts at frame 0; old frozen association experiments started at frame 2",
                  "New model inference may change local boxes and embeddings; old 86.02% is not a paired control",
                  "Merge-event checks describe uniquely spatially matched visible members at acceptance, not whole trajectories",
                  "Ever-emitted IDs over all runtime frames can differ from IDs in annotated evaluation frames",
                  "No CLEAR ID switches or MOTA inferred from the artificial camera/time ordering"]}
    report["code_sha256"][str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    path = output/"report.json"
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("Variant                  IDF1      IDP      IDR    IDTP   IDFP   IDFN")
    for name,m in (("No-cross-camera control",control_metrics),("Full MTMC pipeline",metrics)):
        print(f"{name:24} {m['idf1']:7.2%} {m['idp']:8.2%} {m['idr']:8.2%} {m['idtp']:7} {m['idfp']:6} {m['idfn']:6}")
    print(f"Delta/control: {report['delta_idf1_vs_control_pp']:+.2f} pp")
    print("GT observations:",metrics["gt_observations"],"Prediction observations:",metrics["predicted_observations"])
    print("Accepted merge diagnostics:",json.dumps(dict(categories)))
    print("Both metrics/motmetrics, all prediction keys and frozen checksums: VERIFIED")
    print(f"Report: {path}")
    print("MTMC sequence evaluation: COMPLETED")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--prefix-stride", type=int, default=300)
    args = parser.parse_args()
    evaluate(args.run_report, prefix_stride=args.prefix_stride)


if __name__ == "__main__":
    main()
