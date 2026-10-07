"""Evaluate recorded full-pipeline identities on scene_001 frames 2..299; CPU only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

import evaluate_global_identity as metric
from mtmc.association import geometry, controlled_merge
from mtmc.reid.osnet import ObservationKey, sha256

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (4,5,8)
require = metric.require


def observation_key(value, frame):
    key = ObservationKey(**value)
    require(all(type(v) is int and v >= 0 for v in (key.camera_id,key.local_id,key.frame_index))
            and key.camera_id in CAMERAS and key.frame_index == frame, "Invalid observation key")
    return key


def validate_round(local, global_record, *, frame, run, config, next_row):
    """Join on keys, not array positions; keep unencoded predictions too."""
    for row in (local,global_record):
        require(row["run_id"] == run and type(row["frame_index"]) is int and row["frame_index"] == frame
                and Fraction(row["timestamp"]) == Fraction(frame,30), "Trace scope/order/time mismatch")
    r = global_record
    require(r["identity_scope"] == run and r["policy"] == controlled_merge.POLICY
            and r["association_policy"] == geometry.POLICY
            and r["descriptor_variant"] == config["appearance_variant"]
            and r["min_similarity"] == config["appearance_threshold"]
            and r["geometry_configuration"] == config["geometry"]
            and r["coordinate_space"] == config["coordinate_space"], "Mixed identity settings/scope")
    require(sorted(c["camera"] for c in local["cameras"]) == list(CAMERAS), "Camera coverage differs")
    boxes, cameras = {}, {}
    for camera in local["cameras"]:
        c = camera["camera"]
        ids = camera["local_ids"]
        require(all(type(i) is int and i >= 0 for i in ids) and len(ids) == len(set(ids)), "Invalid local IDs")
        array = np.asarray(camera["xyxy"],dtype=np.float64).reshape(-1,4)
        require(array.shape == (len(ids),4) and np.isfinite(array).all()
                and np.all(array[:,2:] > array[:,:2]), "Invalid tracked boxes")
        keys = tuple(ObservationKey(c,i,frame) for i in ids)
        boxes.update(zip(keys,array))
        cameras[c] = (keys,array)
    rows, skipped = {}, set()
    for item in local["reid_observations"]:
        key = observation_key(dict(camera_id=item["camera"],local_id=item["local_id"],frame_index=item["frame_index"]),frame)
        require(key in boxes and key not in rows and np.array_equal(boxes[key],item["source_xyxy"]),
                "Crop and tracked box/key differ")
        if item["status"] == "encoded":
            require(type(item["embedding_row"]) is int and item["embedding_row"] == next_row
                    and item["crop_xyxy_int"] is not None, "Embedding rows are not contiguous")
            next_row += 1
        else:
            require(item["status"] == "skipped_fully_outside" and item["embedding_row"] is None
                    and item["crop_xyxy_int"] is None, "Unknown unencoded observation policy")
            clipped = metric.history.snapshot.clip_boxes([boxes[key]],1920,1080)
            require(np.any(clipped[0,2:] <= clipped[0,:2]), "Skipped box is not fully outside")
            skipped.add(key)
        rows[key] = item["embedding_row"]
    require(set(rows) == set(boxes), "Crop records do not cover local tracks")
    assignments, used_slots = {}, set()
    for item in r["assignments"]:
        key = observation_key(item["key"],frame)
        gid = item["global_id"]
        require(key in rows and key not in assignments and item["embedding_row"] == rows[key], "Global key/row mismatch")
        require((rows[key] is None and item["embedding_row"] is None)
                or (rows[key] is not None and type(item["embedding_row"]) is int), "Invalid global embedding-row type")
        require(type(item["has_current_embedding"]) is bool
                and item["has_current_embedding"] == (rows[key] is not None), "Embedding availability flag differs")
        require(type(gid) is int and gid > 0 and (gid,key.camera_id) not in used_slots,
                "Invalid global ID or duplicate same-camera identity")
        assignments[key] = gid
        used_slots.add((gid,key.camera_id))
    require(set(assignments) == set(boxes), "Global outputs do not cover every local prediction")
    missing = [observation_key(k,frame) for k in r["unencoded_singletons"]]
    require(len(missing) == len(set(missing)) and set(missing) == skipped, "Unencoded singleton list differs")
    return cameras,assignments,next_row


def spatial_slot(ground, keys, predictions):
    gt_ids = sorted(ground)
    gt_boxes = metric.history.snapshot.clip_boxes([ground[g] for g in gt_ids],1920,1080)
    visible = np.all(gt_boxes[:,2:] > gt_boxes[:,:2],axis=1)
    gt_ids = [g for g, keep in zip(gt_ids,visible) if keep]
    predicted = metric.history.snapshot.clip_boxes(predictions,1920,1080)
    mask = metric.history.snapshot.pairwise_iou(gt_boxes[visible],predicted) >= .5
    unique = {}
    gt_degree,pred_degree = mask.sum(axis=1),mask.sum(axis=0)
    for i,j in zip(*np.nonzero(mask)):
        if gt_degree[i] == pred_degree[j] == 1:
            unique[keys[j]] = gt_ids[i]
    return gt_ids,mask,unique,int((~visible).sum()),int(np.any(predicted[:,2:] <= predicted[:,:2],axis=1).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report",type=Path,required=True)
    args = parser.parse_args()
    inputs = {}
    def checked(name,path,expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected,f"Checksum mismatch: {path}")
        inputs[name] = {"path":str(path),"sha256":digest}
        return path
    report_path = checked("pipeline_report",args.run_report)
    source = json.loads(report_path.read_text())
    cfg = source["configuration"]
    require(source.get("completed") is True and source["protocol"] == "scene_001_mtmc_sequential_fp32_v1"
            and source["summary"]["rounds"] == 300 and cfg["fps"] == 30 and cfg["cameras"] == [4,5,8]
            and cfg["identity_start_frame"] == 0 and cfg["ground_truth_used"] is False
            and cfg["association_policy"] == geometry.POLICY and cfg["identity_policy"] == controlled_merge.POLICY,
            "Expected completed 300-round full pipeline, identities starting at frame zero")
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
    print("Loading GT for evaluation frames 2..299; warmup predictions remain included...",flush=True)
    ground = metric.history.load_ground_truth(gt_path)
    full,control = metric.IdentityCounts(),metric.IdentityCounts()
    full_slots,control_slots = [],[]
    local_ids = {}
    emitted = set()
    next_row = total_observations = 0
    excluded,outside = Counter({c:0 for c in CAMERAS}),Counter({c:0 for c in CAMERAS})
    categories = Counter()
    merge_diagnostics = []
    with paths["tracks"].open() as tracks,paths["global_tracks"].open() as globals_file:
        for frame in range(300):
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
            if (frame+1)%60 == 0:
                print(f"Evaluated through frame {frame}/299",flush=True)
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
    output = ROOT/"artifacts/mtmc_pipeline_evaluation"/now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True,exist_ok=False)
    metric.history.write_csv(output/"comparison.csv",[{"variant":"no_cross_camera_control",**control_metrics},
                                                      {"variant":"full_mtmc_pipeline",**metrics}])
    (output/"identity_matching.json").write_text(json.dumps({"pipeline":mapping,"control":control_mapping},indent=2)+"\n")
    (output/"merge_diagnostics.json").write_text(json.dumps(merge_diagnostics,indent=2)+"\n")
    report = {"completed":True,"created_utc":now.isoformat(),"source_run_id":source["run_id"],
        "protocol":{"name":"scene_001_full_mtmc_global_2d_identity_v1","frames":[2,299],"cameras":list(CAMERAS),
            "fps":30,"min_iou":.5,"width":1920,"height":1080,"runtime_identity_start_frame":0,
            "timing_warmup_excluded_from_quality":False,"runtime_predictions_modified":False,
            "spatial_gate":"Same camera and same frame; all IoU >= 0.5 candidates retained",
            "box_policy":"Clip both; exclude zero-area GT; retain all predictions, including unencoded/outside",
            "identity_matching":"One shared one-to-one GT/global identity assignment over all cameras and evaluated frames",
            "control":"Camera-scoped original local IDs on identical boxes; no extra expiry or relabeling",
            "selected_deployment_threshold":None},
        "inputs":inputs,"pipeline_metrics":metrics,"no_cross_camera_control":control_metrics,
        "delta_idf1_vs_control_pp":100*(metrics["idf1"]-control_metrics["idf1"]),
        "excluded_zero_area_gt":dict(excluded),"fully_outside_predictions":dict(outside),
        "merge_event_diagnostics":dict(categories),"runtime_summary":summary,
        "control_local_id_map":[{"camera":c,"local_id":i,"control_id":v} for (c,i),v in sorted(local_ids.items())],
        "checks":{"all_300_rounds_and_prediction_keys_verified":True,"embedding_rows_verified":True,
                  "both_metrics_agree_with_motmetrics":True,"shared_denominators":True,"frozen_inputs_unchanged":True},
        "versions":{name:version(name) for name in ("numpy","scipy","motmetrics","pandas")},
        "code_sha256":{str(Path(m.__file__).relative_to(ROOT)):sha256(Path(m.__file__)) for m in
                       (metric,metric.history,metric.history.snapshot,metric.history.history_io)},
        "artifacts":{p.name:{"path":p.name,"sha256":sha256(p)} for p in sorted(output.iterdir())},
        "limits":["Same short reused training fragment; not independent validation or official AI City evaluation",
                  "Identity state starts at frame 0; old frozen association experiments started at frame 2",
                  "New model inference may change local boxes and embeddings; old 86.02% is not a paired control",
                  "Merge-event checks describe uniquely spatially matched visible members at acceptance, not whole trajectories",
                  "Ever-emitted IDs over 0..299 can differ from predicted identities evaluated over 2..299",
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
    print("Full MTMC pipeline evaluation: COMPLETED")


if __name__ == "__main__":
    main()
