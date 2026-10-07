"""Global identity metrics over all camera/time slots with one shared ID assignment."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import evaluate_appearance_history as history
from mtmc.association.global_identity import POLICY
from mtmc.reid.osnet import ObservationKey, sha256

ROOT = Path(__file__).resolve().parents[1]
require = history.require
CAMERAS, FIRST, LAST, FPS = (4, 5, 8), 2, 299, 30


class IdentityCounts:
    """Count every admissible spatial overlap, then assign whole identities once.

    update() consumes ONE camera at ONE time. IDs remain shared across cameras.
    For fixed total GT/prediction observations, minimizing IDFN+IDFP is equivalent
    to maximizing IDTP. Zero-weight assignments are ignored; unmatched identities
    contribute their full observation counts. No framewise identity preassignment.
    """

    def __init__(self):
        self.ground, self.predicted, self.potential = Counter(), Counter(), Counter()
        self.slots = 0

    def update(self, ground_ids, predicted_ids, admissible):
        ground_ids, predicted_ids = tuple(ground_ids), tuple(predicted_ids)
        require(all(type(x) is int and x >= 0 for x in (*ground_ids, *predicted_ids)), "Invalid identity IDs")
        require(len(set(ground_ids)) == len(ground_ids) and len(set(predicted_ids)) == len(predicted_ids),
                "An identity occurs twice in one camera/time slot")
        mask = np.asarray(admissible)
        require(mask.dtype == np.bool_ and mask.shape == (len(ground_ids), len(predicted_ids)), "Invalid admissibility matrix")
        self.ground.update(ground_ids)
        self.predicted.update(predicted_ids)
        for i, j in zip(*np.nonzero(mask)):
            self.potential[ground_ids[i], predicted_ids[j]] += 1
        self.slots += 1

    def result(self):
        ground, predicted = sorted(self.ground), sorted(self.predicted)
        matrix = np.array([[self.potential[g, p] for p in predicted] for g in ground], dtype=np.int64).reshape(len(ground), len(predicted))
        selected = []
        if ground and predicted:
            rows, cols = linear_sum_assignment(-matrix)
            selected = [{"gt_id": ground[i], "global_id": predicted[j], "idtp": int(matrix[i, j])}
                        for i, j in zip(rows, cols) if matrix[i, j] > 0]
        tp = sum(x["idtp"] for x in selected)
        ng, npred = sum(self.ground.values()), sum(self.predicted.values())
        require(0 <= tp <= min(ng, npred), "Invalid identity accounting")
        metrics = {"camera_time_slots": self.slots, "gt_observations": ng, "predicted_observations": npred,
                   "gt_identities": len(ground), "predicted_identities": len(predicted),
                   "idtp": tp, "idfp": npred - tp, "idfn": ng - tp,
                   "idp": tp / npred if npred else None, "idr": tp / ng if ng else None,
                   "idf1": 2 * tp / (ng + npred) if ng + npred else None}
        return metrics, selected


def reference_metrics(slots):
    """Independent reference engine, using camera/time only as observation slots.

    Only ID metrics are requested: artificial slot ordering is not used to report
    CLEAR switches, temporal fragments, MOTA or latency.
    """
    import motmetrics as mm
    require(version("motmetrics") == "1.4.0", "Expected existing motmetrics==1.4.0")
    mm.lap.default_solver = "scipy"
    accumulator = mm.MOTAccumulator(auto_id=False)
    for index, (gt, pred, mask) in enumerate(slots):
        accumulator.update(gt, pred, np.where(mask, 0.0, np.nan), frameid=index)
    names = ["idtp", "idfp", "idfn", "idf1", "idp", "idr"]
    result = mm.metrics.create().compute(accumulator, metrics=names, name="global")
    return {key: float(result.loc["global", key]) for key in names}


def check_reference(ours, reference):
    for key in ("idtp", "idfp", "idfn", "idf1", "idp", "idr"):
        if ours[key] is None:
            require(np.isnan(reference[key]), f"Undefined metric differs: {key}")
        else:
            require(np.isclose(ours[key], reference[key], atol=1e-12, rtol=0), f"Reference mismatch: {key}")


def checked(path, expected_hash):
    path = Path(path).resolve()
    require(sha256(path) == expected_hash, f"Checksum mismatch: {path}")
    return path


def build_slots(trace_path, ground, run_id):
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    require([r["frame_index"] for r in trace] == list(range(300)), "Expected frozen frames 0..299")
    slots, frame_maps, seen_rows = {}, {}, set()
    excluded, outside = Counter({c: 0 for c in CAMERAS}), Counter({c: 0 for c in CAMERAS})
    for frame in trace:
        number = frame["frame_index"]
        require(frame["run_id"] == run_id and Fraction(frame["timestamp"]) == Fraction(number, FPS), "Source scope/time mismatch")
        if number < FIRST:
            continue
        require(sorted(c["camera"] for c in frame["cameras"]) == list(CAMERAS), "Unexpected source cameras")
        encoded = {}
        for item in frame["reid_observations"]:
            require(item["status"] == "encoded", "Protocol requires an assigned global ID for every local prediction")
            key = ObservationKey(item["camera"], item["local_id"], item["frame_index"])
            row = item["embedding_row"]
            require(key.frame_index == number and key not in encoded and type(row) is int and row >= 0 and row not in seen_rows,
                    "Duplicate or invalid source key/row")
            seen_rows.add(row)
            encoded[key] = item
        all_keys = set()
        for camera in frame["cameras"]:
            c = camera["camera"]
            ids = camera["local_ids"]
            require(len(ids) == len(camera["xyxy"]) and len(set(ids)) == len(ids), "Invalid source local predictions")
            keys = tuple(ObservationKey(c, local_id, number) for local_id in ids)
            for key, box in zip(keys, camera["xyxy"]):
                require(key in encoded and np.array_equal(box, encoded[key]["source_xyxy"]), "Embedding observation differs from tracked box")
            all_keys.update(keys)
            predictions = history.snapshot.clip_boxes(camera["xyxy"], 1920, 1080)
            outside[c] += int(np.any(predictions[:, 2:] <= predictions[:, :2], axis=1).sum())
            gt_ids = sorted(ground[number, c])
            boxes = history.snapshot.clip_boxes([ground[number, c][g] for g in gt_ids], 1920, 1080)
            visible = np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
            excluded[c] += int((~visible).sum())
            gt_ids = [g for g, keep in zip(gt_ids, visible) if keep]
            mask = history.snapshot.pairwise_iou(boxes[visible], predictions) >= 0.5
            slots[number, c] = (gt_ids, keys, mask)
        require(all_keys == set(encoded), "Local tracks and encoded observations do not have equal coverage")
        frame_maps[number] = {key: item["embedding_row"] for key, item in encoded.items()}
    return slots, frame_maps, dict(excluded), dict(outside)


def read_assignments(record, *, run_id, number, variant, threshold, identity_scope, expected_rows):
    require((record["run_id"], record["frame_index"], record["descriptor_variant"], record["min_similarity"],
             record["policy"], record["identity_scope"]) == (run_id, number, variant, threshold, POLICY, identity_scope),
            "Unexpected identity record context/order")
    require(Fraction(record["timestamp"]) == Fraction(number, FPS), "Identity timestamp mismatch")
    assignments, slots = {}, set()
    for a in record["assignments"]:
        key = ObservationKey(**a["key"])
        gid = a["global_id"]
        require(key in expected_rows and a["embedding_row"] == expected_rows[key] and key not in assignments,
                "Identity/source key or row mismatch")
        require(type(gid) is int and gid > 0 and (gid, key.camera_id) not in slots, "Invalid or same-camera duplicate global ID")
        assignments[key] = gid
        slots.add((gid, key.camera_id))
    require(set(assignments) == set(expected_rows), "Identity assignments do not cover all source predictions")
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.identity_report.resolve()
    report_hash = sha256(report_path)
    source = json.loads(report_path.read_text(encoding="utf-8"))
    p = source["protocol"]
    require(source.get("completed") is True and p["name"] == "scene_001_global_identity_replay_v1" and p["policy"] == POLICY,
            "Expected completed global identity replay")
    require((p["cameras"], p["first_frame"], p["last_frame"], p["fps"]) == (list(CAMERAS), FIRST, LAST, FPS),
            "Unexpected evaluation scope")
    settings = [(v, t) for v in p["variants"] for t in p["thresholds"]]
    expected = {(r["variant"], r["threshold"]): r for r in source["results"]}
    require(len(settings) == len(set(settings)) == len(source["results"]) and set(settings) == set(expected), "Invalid settings")
    inputs = {"identity_report": {"path": str(report_path), "sha256": report_hash}}
    print("Verifying identity assignments, frozen tracked boxes and ground truth...", flush=True)
    entry = source["artifacts"]["assignments.jsonl.gz"]
    input_path = checked(report_path.parent / entry["path"], entry["sha256"])
    inputs["assignments"] = {"path": str(input_path), "sha256": entry["sha256"]}
    earlier = source["inherited_earlier_inputs_not_reverified"]
    for name in ("tracks", "ground_truth"):
        path = checked(earlier[name]["path"], earlier[name]["sha256"])
        inputs[name] = {"path": str(path), "sha256": earlier[name]["sha256"]}
    print("Loading GT once and building camera-specific IoU gates...", flush=True)
    ground = history.load_ground_truth(Path(inputs["ground_truth"]["path"]))
    slots, frame_maps, excluded, outside = build_slots(Path(inputs["tracks"]["path"]), ground, source["source_run_id"])
    local_keys = sorted({(key.camera_id, key.local_id) for mapping in frame_maps.values() for key in mapping})
    local_ids = {key: i + 1 for i, key in enumerate(local_keys)}
    control = IdentityCounts()
    reference_slots = []
    for (number, camera), (gt, keys, mask) in sorted(slots.items()):
        pred = [local_ids[k.camera_id, k.local_id] for k in keys]
        control.update(gt, pred, mask)
        reference_slots.append((gt, pred, mask))
    control_metrics, control_mapping = control.result()
    check_reference(control_metrics, reference_metrics(reference_slots))
    print("No-cross-camera control agrees with motmetrics: VERIFIED", flush=True)
    counters = {s: IdentityCounts() for s in settings}
    with gzip.open(input_path, "rt", encoding="utf-8") as handle:
        for number in range(FIRST, LAST + 1):
            for setting in settings:
                line = handle.readline()
                require(bool(line), "Truncated identity trace")
                record = json.loads(line)
                assignments = read_assignments(record, run_id=source["source_run_id"], number=number,
                    variant=setting[0], threshold=setting[1], identity_scope=expected[setting]["identity_scope"],
                    expected_rows=frame_maps[number])
                for camera in CAMERAS:
                    gt, keys, mask = slots[number, camera]
                    counters[setting].update(gt, [assignments[k] for k in keys], mask)
            if (number + 1) % 60 == 0:
                print(f"Evaluated through frame {number}/299", flush=True)
        require(handle.readline() == "", "Unexpected trailing identity records")
    now = datetime.now(timezone.utc)
    output = ROOT / "artifacts/global_identity_evaluation" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    summaries, mappings = [], []
    for setting in settings:
        metrics, mapping = counters[setting].result()
        require(metrics["predicted_observations"] == expected[setting]["observations"], "Prediction count mismatch")
        require(metrics["predicted_identities"] == expected[setting]["distinct_global_ids"], "Identity count mismatch")
        require(metrics["gt_observations"] == control_metrics["gt_observations"] and
                metrics["predicted_observations"] == control_metrics["predicted_observations"], "Evaluation denominators changed")
        summaries.append({"variant": setting[0], "threshold": setting[1], **metrics,
                          "delta_idf1_vs_no_cross_camera_pp": 100 * (metrics["idf1"] - control_metrics["idf1"])})
        mappings.extend({"variant": setting[0], "threshold": setting[1], **m} for m in mapping)
    for item in inputs.values():
        checked(item["path"], item["sha256"])
    history.write_csv(output / "summary.csv", summaries)
    if mappings:
        history.write_csv(output / "identity_matching.csv", mappings)
    report = {
        "completed": True, "created_utc": now.isoformat(), "source_replay_id": source["replay_id"],
        "protocol": {"name": "scene_001_global_2d_identity_v1", "cameras": CAMERAS, "first_frame": FIRST,
            "last_frame": LAST, "fps": FPS, "min_iou": 0.5, "width": 1920, "height": 1080,
            "box_policy": "Clip both; exclude zero-area GT; keep every prediction including zero-area",
            "spatial_matching": "Same camera and same frame only; every IoU >= 0.5 is admissible",
            "identity_matching": "One shared one-to-one GT/predicted identity assignment across all cameras and frames",
            "preassigned_gt_labels_used": False, "selected_threshold": None, "max_idle_seconds": p["max_idle_seconds"],
            "idtp": "Admissible observation pairs under the single whole-clip identity assignment",
            "idfp": "All prediction observations minus IDTP", "idfn": "All evaluated GT observations minus IDTP",
            "idf1": "2*IDTP/(GT observations + predicted observations)",
            "control": "Frozen local IDs scoped by camera; no cross-camera association, no extra expiry or relabeling",
            "scope": "Project integration subset; not official AI City evaluation"},
        "inputs": inputs, "excluded_zero_area_gt": excluded, "fully_outside_predictions": outside,
        "no_cross_camera_control": control_metrics,
        "control_local_id_map": [{"camera": c, "local_id": lid, "control_id": value} for (c, lid), value in local_ids.items()],
        "control_identity_matching": control_mapping, "results": summaries,
        "checks": {"checksums_verified": True, "all_predictions_covered": True,
            "shared_denominators": True, "no_cross_camera_control_motmetrics_parity": True},
        "versions": {name: version(name) for name in ("numpy", "scipy", "motmetrics", "pandas")},
        "code_sha256": {q.name: sha256(q) for q in (Path(__file__), Path(history.__file__), Path(history.snapshot.__file__))},
        "artifacts": {q.name: {"path": q.name, "sha256": sha256(q)} for q in sorted(output.iterdir()) if q.is_file()},
        "limits": ["Only 298 frames from one reused training-scene fragment; no final threshold calibration",
            "Includes localization, missing/false predictions and identity errors; not pure association accuracy",
            "No CLEAR ID-switch/MOTA figures are inferred from serialized camera/time slots",
            "The GT-to-predicted identity matching is offline evaluation only; runtime assignments are unchanged",
            "The local-ID control is evaluated globally and is not the earlier pooled per-camera IDF1",
            "Equal-cost identity matchings may have different pairings but identical aggregate ID metrics"],
    }
    target = output / "report.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"No-cross-camera control: IDF1={control_metrics['idf1']:.2%}; predicted IDs={control_metrics['predicted_identities']}")
    print("Variant Tau    Global IDF1   IDP      IDR       IDTP   IDFP   IDFN   Delta/control pp")
    for r in summaries:
        print(f"{r['variant']:<7} {r['threshold']:4.2f}    {r['idf1']:8.2%} {r['idp']:8.2%} {r['idr']:8.2%} "
              f"{r['idtp']:7} {r['idfp']:6} {r['idfn']:6} {r['delta_idf1_vs_no_cross_camera_pp']:+12.2f}")
    print("GT observations:", control_metrics["gt_observations"], "Prediction observations:", control_metrics["predicted_observations"])
    print("Identity matching: ONE shared assignment across all cameras and frames")
    print("Threshold selection: NONE")
    print(f"Report: {target}")
    print("Global identity evaluation: COMPLETED")


if __name__ == "__main__":
    main()
