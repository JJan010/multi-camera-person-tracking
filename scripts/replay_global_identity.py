"""Replay frozen frame groups through isolated causal identity managers on CPU."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path

from mtmc.association import global_identity, grouping
from mtmc.association.pairwise import validate_threshold
from mtmc.reid.osnet import ObservationKey, sha256
import evaluate_multicamera_grouping as grouping_eval

ROOT = Path(__file__).resolve().parents[1]
require = global_identity.require
FIELDS = (
    "rounds", "observations", "local_continuity_observations", "group_attachment_observations",
    "new_identity_observations", "new_global_ids", "expired_local_bindings", "expired_global_ids",
    "groups_new_identity", "groups_continued", "groups_attached", "groups_rejected_multiple_existing_ids",
    "groups_rejected_reserved_camera", "groups_rejected_competing_attachments",
)


def decode_round(record, *, run_id, frame, variant, threshold, fps, cameras):
    """Construct GT-free runtime input; diagnostic metadata stays outside it."""
    require((record["run_id"], record["frame_index"], record["variant"], record["threshold"], record["policy"])
            == (run_id, frame, variant, threshold, grouping.POLICY), "Unexpected grouping context or order")
    require(Fraction(record["timestamp"]) == Fraction(frame, fps), "Invalid exact scene timestamp")
    labels, groups = {}, []
    for members in record["groups"]:
        require(1 <= len(members) <= len(cameras), "Invalid group size")
        keys, seen_cameras = [], set()
        for member in members:
            camera, local_id, frame_index, row = (member[k] for k in
                ("camera_id", "local_id", "frame_index", "embedding_row"))
            require(all(type(x) is int and x >= 0 for x in (camera, local_id, frame_index, row)), "Invalid observation/row")
            require(camera in cameras and camera not in seen_cameras and frame_index == frame, "Invalid camera/frame in group")
            key = ObservationKey(camera, local_id, frame_index)
            require(key not in labels, "Duplicate observation across groups")
            gt = member["diagnostic_gt_id"]
            require(gt is None or type(gt) is int, "Invalid diagnostic label")
            labels[key] = {"embedding_row": row, "diagnostic_gt_id": gt}
            keys.append(key)
            seen_cameras.add(camera)
        groups.append(tuple(keys))
    require(len({r["embedding_row"] for r in labels.values()}) == len(labels), "Duplicate embedding row")
    # Edge diagnostics are not identity input. The input partition comes from
    # the verified grouping artifact; support is additionally checked below.
    runtime = grouping.FrameGroups(run_id, frame, Fraction(frame, fps), variant, threshold,
                                   grouping.POLICY, tuple(groups), ())
    return runtime, labels


def source_counts(record, frame, labels):
    """Verify saved clique support and reconstruct counts from the grouping report."""
    counts = Counter({key: 0 for key in grouping_eval.GROUP_FIELDS})
    counts.update(frames=1, observations=len(labels), groups=len(frame.groups))
    retained = set()
    for group in frame.groups:
        counts[{1: "singletons", 2: "two_member_groups", 3: "three_member_groups"}[len(group)]] += 1
        retained.update(grouping.edge_key(a, b) for a, b in combinations(group, 2))
        if len(group) > 1:
            gt = [labels[k]["diagnostic_gt_id"] for k in group]
            known = {x for x in gt if x is not None}
            category = "different_known_gt" if len(known) > 1 else "unresolved_gt" if None in gt else "all_known_same_gt"
            counts["linked_groups_" + category] += 1
    edges, kept = set(), set()
    for decision in record["decisions"]:
        a, b = (ObservationKey(**decision[k]) for k in ("left", "right"))
        edge = grouping.edge_key(a, b)
        require(a in labels and b in labels and a.camera_id != b.camera_id and edge not in edges, "Invalid audit edge")
        score = decision["cosine_similarity"]
        require(type(score) in (float, int) and frame.min_similarity < score <= 1, "Invalid accepted edge score")
        outcome = decision["outcome"]
        require(outcome in grouping_eval.GROUP_FIELDS[9:], "Invalid grouping outcome")
        counts[outcome] += 1
        edges.add(edge)
        if outcome in ("merged", "already_in_same_group"):
            kept.add(edge)
    require(retained == kept and retained <= edges, "Group partition and saved edge decisions disagree")
    metric_labels = {k: {"gt_id": r["diagnostic_gt_id"]} for k, r in labels.items()}
    return counts, grouping_eval.count_edges(retained, metric_labels)


def json_default(value):
    if isinstance(value, Fraction):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def encode_output(result, labels, scope):
    record = asdict(result)
    record["identity_scope"] = scope
    record["assignments"] = [{**asdict(a), **labels[a.key]} for a in result.assignments]
    return record


def add_counts(counts, result):
    counts.update(rounds=1, observations=len(result.assignments), expired_local_bindings=len(result.expired_local_tracks),
                  expired_global_ids=len(result.expired_global_ids))
    created = {a.global_id for a in result.assignments if a.reason == "new_identity"}
    counts["new_global_ids"] += len(created)
    for assignment in result.assignments:
        key = assignment.reason + "_observations"
        require(key in FIELDS, "Unknown assignment reason")
        counts[key] += 1
    for decision in result.decisions:
        key = "groups_" + decision.outcome
        require(key in FIELDS, "Unknown registry decision")
        counts[key] += 1
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouping-report", type=Path, required=True)
    parser.add_argument("--max-idle-seconds", required=True, help="Explicit positive seconds or rational, e.g. 1 or 1/2")
    args = parser.parse_args()
    idle = Fraction(args.max_idle_seconds)
    require(idle > 0, "Idle duration must be positive")
    source_path = args.grouping_report.resolve()
    source_hash = sha256(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    p = source["protocol"]
    require(source.get("completed") is True and p["name"] == "scene_001_multicamera_grouping_diagnostic_v1",
            "Expected a completed grouping diagnostic")
    require(p["policy"] == grouping.POLICY, "Unsupported grouping policy")
    require((p["cameras"], p["first_frame"], p["last_frame"], p["fps"]) == ([4, 5, 8], 2, 299, 30), "Unexpected source timeline")
    require(p["variants"] == ["latest", "mean"], "Unexpected descriptor variants")
    thresholds = [validate_threshold(t) for t in p["thresholds"]]
    require(thresholds and thresholds == sorted(set(thresholds)), "Invalid threshold grid")
    settings = [(v, t) for v in p["variants"] for t in thresholds]
    expected = {(r["variant"], r["threshold"]): r for r in source["results"]}
    require(set(expected) == set(settings) and len(expected) == len(source["results"]), "Source summary settings mismatch")
    entry = source["artifacts"]["groups.jsonl.gz"]
    input_path = (source_path.parent / entry["path"]).resolve()
    print("Verifying frozen grouping trace and provenance...", flush=True)
    require(sha256(input_path) == entry["sha256"], "Grouping trace checksum mismatch")
    now = datetime.now(timezone.utc)
    replay_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "artifacts/global_identity" / replay_id
    output.mkdir(parents=True, exist_ok=False)
    scopes = {(v, t): f"{replay_id}/{v}/tau={t!r}/idle={idle}" for v, t in settings}
    managers = {s: global_identity.GlobalIdentityManager(source["source_run_id"], max_idle=idle,
                descriptor_variant=s[0], min_similarity=s[1]) for s in settings}
    totals = {s: Counter({k: 0 for k in FIELDS}) for s in settings}
    original_counts, original_edges = defaultdict(Counter), defaultdict(Counter)
    peaks = {s: Counter() for s in settings}
    all_ids, local_tracks = {s: set() for s in settings}, {s: set() for s in settings}
    final_states, used_rows, by_frame = {}, set(), []
    print(f"Replay: {len(settings)} isolated managers; idle={idle}s; frames 2..299; CPU only", flush=True)
    with gzip.open(input_path, "rt", encoding="utf-8") as handle, \
            gzip.open(output / "assignments.jsonl.gz", "wt", encoding="utf-8") as destination:
        for number in range(p["first_frame"], p["last_frame"] + 1):
            reference = None
            for variant, threshold in settings:
                setting = variant, threshold
                line = handle.readline()
                require(bool(line), "Truncated grouping trace")
                record = json.loads(line)
                runtime, labels = decode_round(record, run_id=source["source_run_id"], frame=number,
                    variant=variant, threshold=threshold, fps=p["fps"], cameras=p["cameras"])
                if reference is None:
                    reference = labels
                    rows = {r["embedding_row"] for r in labels.values()}
                    require(not rows & used_rows, "Source row reused across frames")
                    used_rows.update(rows)
                else:
                    require(labels == reference, "Observation/row/label mapping differs across settings")
                counts, edges = source_counts(record, runtime, labels)
                original_counts[setting].update(counts)
                original_edges[setting].update(edges)
                result = managers[setting].update(runtime)  # no diagnostic labels passed
                require({a.key for a in result.assignments} == set(labels), "Identity output coverage mismatch")
                frame_counts = Counter({k: 0 for k in FIELDS})
                created = add_counts(frame_counts, result)
                require(not created & all_ids[setting], "Global ID was allocated twice")
                all_ids[setting].update(created)
                require(all(a.global_id in all_ids[setting] for a in result.assignments), "Unknown output global ID")
                totals[setting].update(frame_counts)
                local_tracks[setting].update((a.key.camera_id, a.key.local_id) for a in result.assignments)
                live = sum(s.status == "visible" for s in result.identities)
                lost = sum(s.status == "lost" for s in result.identities)
                retained = sum(len(s.retained_local_tracks) for s in result.identities)
                require(live + lost == len(result.identities), "Unknown visibility state")
                for key, value in (("visible_global_ids", live), ("lost_global_ids", lost), ("retained_local_bindings", retained)):
                    peaks[setting]["peak_" + key] = max(peaks[setting]["peak_" + key], value)
                final_states[setting] = len(result.identities)
                by_frame.append({"frame_index": number, "variant": variant, "threshold": threshold,
                    **frame_counts, "visible_global_ids": live, "lost_global_ids": lost, "retained_local_bindings": retained})
                destination.write(json.dumps(encode_output(result, labels, scopes[setting]), default=json_default,
                                             allow_nan=False) + "\n")
            if (number + 1) % 60 == 0:
                print(f"Processed through frame {number}/299", flush=True)
        require(handle.readline() == "", "Unexpected trailing grouping records")
    results = []
    for setting in settings:
        counts = totals[setting]
        require(dict(original_counts[setting]) == expected[setting]["groups"], "Source grouping counters differ")
        require(all(original_edges[setting][k] == expected[setting]["after"][k] for k in grouping_eval.EDGE_FIELDS),
                "Source retained-edge counts differ")
        require(counts["rounds"] == 298 and counts["observations"] == len(used_rows) == source["checks"]["source_row_count"],
                "Incomplete identity replay")
        require(counts["observations"] == sum(counts[k] for k in
                ("local_continuity_observations", "group_attachment_observations", "new_identity_observations")),
                "Assignment reason accounting mismatch")
        require(counts["new_global_ids"] == len(all_ids[setting]) == counts["expired_global_ids"] + final_states[setting],
                "Global identity lifetime accounting mismatch")
        results.append({"variant": setting[0], "threshold": setting[1], "identity_scope": scopes[setting],
            **counts, "distinct_local_tracks": len(local_tracks[setting]), "distinct_global_ids": len(all_ids[setting]),
            "retained_global_ids_at_end": final_states[setting], **peaks[setting]})
    require(len(by_frame) == source["checks"]["groups_records"], "Source/output record count mismatch")
    require(sha256(source_path) == source_hash and sha256(input_path) == entry["sha256"], "Inputs changed during replay")
    grouping_eval.baseline.history_eval.write_csv(output / "summary.csv", results)
    grouping_eval.baseline.history_eval.write_csv(output / "by_frame.csv", by_frame)
    report = {
        "completed": True, "replay_id": replay_id, "created_utc": now.isoformat(), "source_run_id": source["source_run_id"],
        "protocol": {"name": "scene_001_global_identity_replay_v1", "policy": global_identity.POLICY,
            "cameras": p["cameras"], "first_frame": 2, "last_frame": 299, "fps": 30, "variants": p["variants"],
            "thresholds": thresholds, "max_idle_seconds": str(idle), "selected_threshold": None,
            "ground_truth_used_by_manager": False, "persistent_global_ids_created": True,
            "identity_key": "(identity_scope, global_id); scopes separate variants, thresholds, idle durations and replay runs",
            "initial_state": "Empty at frame 2; no warmup, no lookahead, no retrospective changes",
            "idle_role": "Explicit integration setting, not deployment calibration",
            "conflict_counting": "Group decisions summed over frames; repeated conflicts are counted repeatedly"},
        "inputs": {"grouping_report": {"path": str(source_path), "sha256": source_hash},
                   "groups": {"path": str(input_path), "sha256": entry["sha256"]}},
        "inherited_grouping_inputs_not_reverified": source["inputs"],
        "inherited_earlier_inputs_not_reverified": source.get("inherited_source_inputs_not_reverified", {}),
        "code_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(global_identity.__file__),
                          Path(grouping.__file__), Path(grouping_eval.__file__))},
        "versions": {name: version(name) for name in ("numpy", "scipy")},
        "checks": {"source_checksums_verified": True, "source_group_and_retained_edge_counts_reproduced": True,
            "source_row_count": len(used_rows), "output_records": len(by_frame), "all_observations_assigned": True,
            "same_observation_mapping_for_all_settings": True, "camera_uniqueness_enforced": True,
            "identity_lifetime_accounting_verified": True},
        "results": results,
        "artifacts": {p.name: {"path": p.name, "sha256": sha256(p)} for p in sorted(output.iterdir()) if p.is_file()},
        "limits": ["No identity quality metrics or global IDF1 computed in this replay",
            "Distinct global IDs are not a count of true unique people",
            "GT labels are inherited diagnostic metadata and never enter the manager",
            "No-merge policy can preserve fragmentation and propagate wrong initial associations or local-ID switches",
            "All managers begin empty at frame 2; a full-video run from frame zero may differ",
            "No synthetic end-of-video expiry rounds are added; retained identities remain retained at clip end",
            "This is not a throughput benchmark or independent validation set"],
    }
    target = output / "report.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Variant Tau   GlobalIDs Attached  MultiIDConf ReservedCam Competing ExpiredIDs")
    for r in results:
        print(f"{r['variant']:<7} {r['threshold']:4.2f} {r['distinct_global_ids']:10} "
              f"{r['group_attachment_observations']:8} {r['groups_rejected_multiple_existing_ids']:12} "
              f"{r['groups_rejected_reserved_camera']:11} {r['groups_rejected_competing_attachments']:9} "
              f"{r['expired_global_ids']:10}")
    print("Source grouping counters reproduced: VERIFIED")
    print("All observations assigned; camera uniqueness and lifetime accounting: VERIFIED")
    print("GT used by manager: NO; threshold selection: NONE; quality metrics: NOT YET")
    print(f"Report: {target}")
    print(f"Assignments: {output / 'assignments.jsonl.gz'}")
    print("Global identity replay: COMPLETED")


if __name__ == "__main__":
    main()
