"""Run scene_001 video -> CUDA FP32 models -> causal global identities."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np

import run_local_reid as local_baseline
from mtmc.pipeline.core import IdentityStage, MTMCPipeline, require
from mtmc.reid.osnet import sha256
from mtmc.association import geometry, controlled_merge

ROOT = Path(__file__).resolve().parents[1]


def json_default(value):
    if isinstance(value, Fraction):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(value):
    return json.dumps(value, default=json_default, allow_nan=False)


def checked(path, expected=None):
    path = Path(path).resolve()
    digest = sha256(path)
    require(expected is None or digest == expected, f"Checksum mismatch: {path}")
    return {"path": str(path), "sha256": digest}


def read_reference(path):
    """Read configuration and calibration only; never load ground truth."""
    inputs = {"geometry_reference": checked(path)}
    ref = json.loads(Path(path).read_text(encoding="utf-8"))
    p = ref["protocol"]
    require(ref.get("completed") and p["name"] == "scene_001_geometry_identity_paired_v1"
            and p["association_policy"] == geometry.POLICY and p["identity_policy"] == controlled_merge.POLICY
            and p["cameras"] == [4, 5, 8] and p["fps"] == 30, "Unexpected reference protocol")
    for name, key in (("calibration", "calibration"), ("history_reference", "history_history_report")):
        item = ref["inputs"][key]
        inputs[name] = checked(item["path"], item["sha256"])
    history = json.loads(Path(inputs["history_reference"]["path"]).read_text(encoding="utf-8"))["configuration"]
    matrices = load_matrices(Path(inputs["calibration"]["path"]), p["cameras"])
    config = {
        "appearance_variant": p["appearance_variant"], "appearance_threshold": p["appearance_threshold"],
        "geometry": p["geometry_configuration"], "identity": p["identity_configuration"],
        "history": {"max_observations": history["max_observations"], "max_age_seconds": history["max_age_seconds"]},
    }
    return ref, config, matrices, inputs


def load_matrices(path, cameras):
    calibration = json.loads(path.read_text(encoding="utf-8"))
    matrices = {}
    for camera in cameras:
        sensors = [s for s in calibration["sensors"] if s["id"].lower() == f"camera_{camera:04d}"]
        require(len(sensors) == 1, "Missing or duplicate camera calibration")
        sensor = sensors[0]
        values = []
        for name, shape in (("intrinsicMatrix", (3,3)), ("extrinsicMatrix", (3,4)),
                            ("cameraMatrix", (3,4)), ("homography", (3,3))):
            a = np.asarray(sensor[name], dtype=np.float64)
            require(a.shape == shape and np.isfinite(a).all(), f"Invalid {name}")
            values.append(a)
        k, e, p, h = values
        for a, b in ((p, k @ e), (h, p[:, [0,1,3]])):
            require(np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0, "Zero projective matrix")
            a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
            require(min(np.linalg.norm(a-b), np.linalg.norm(a+b)) <= 1e-6, "Unexpected ground -> image convention")
        require(np.linalg.matrix_rank(h) == 3, "Singular homography")
        matrices[camera] = h
    return matrices


def make_stage(run, matrices, space, config):
    return IdentityStage(run, matrices, space, variant=config["appearance_variant"],
                         threshold=config["appearance_threshold"], **config["geometry"],
                         identity_configuration=config["identity"])


class EmbeddingArchive:
    """Append CPU arrays; finalize as NPY in bounded chunks, not a growing RAM list."""
    def __init__(self, path):
        self.path = Path(path)
        self.raw = self.path.with_suffix(".f32.partial")
        require(not self.path.exists(), "Embedding archive already exists")
        self.handle = self.raw.open("xb")
        self.rows = 0

    def append(self, array):
        require(array.dtype == np.float32 and array.ndim == 2 and array.shape[1] == 512
                and np.isfinite(array).all(), "Invalid embedding archive batch")
        self.handle.write(np.ascontiguousarray(array).tobytes())
        self.rows += len(array)

    def close(self):
        self.handle.close()

    def finalize(self):
        self.close()
        require(self.raw.stat().st_size == self.rows * 512 * 4, "Truncated embedding spool")
        if self.rows:
            source = np.memmap(self.raw, dtype=np.float32, mode="r", shape=(self.rows, 512))
            destination = np.lib.format.open_memmap(self.path, mode="w+", dtype=np.float32, shape=source.shape)
            for start in range(0, self.rows, 8192):
                destination[start:start+8192] = source[start:start+8192]
            destination.flush()
            del destination, source
        else:
            np.save(self.path, np.empty((0,512), np.float32), allow_pickle=False)
        self.raw.unlink()


def identity_record(round_result, row_offset, stage):
    record = asdict(round_result.identities)
    rows = {key: row_offset+i for i, key in enumerate(round_result.features.keys)}
    for item, assignment in zip(record["assignments"], round_result.identities.assignments):
        item["embedding_row"] = rows.get(assignment.key)
        item["has_current_embedding"] = assignment.key in rows
    record.update(identity_scope=stage.run_id, association_policy=geometry.POLICY,
                  geometry_configuration=stage.geometry_configuration, coordinate_space=stage.coordinate_space,
                  unencoded_singletons=[asdict(key) for key in round_result.unencoded])
    return record


def decision_record(result, row_offset):
    return {"run_id": result.identities.run_id, "frame_index": result.batch.frame_index,
            "timestamp": result.batch.timestamp, "groups": asdict(result.groups),
            "ground_positions": [asdict(result.grounds[c]) for c in sorted(result.grounds)],
            "pairs": [{"left_camera": p.association.left_camera, "right_camera": p.association.right_camera,
                       "matches": [asdict(m) for m in p.association.matches],
                       "unmatched_left": [asdict(m) for m in p.association.unmatched_left],
                       "unmatched_right": [asdict(m) for m in p.association.unmatched_right],
                       "candidate_counts": dict(Counter(c.geometry_decision for c in p.candidates))} for p in result.pairs],
            "history": [{"key": asdict(key), "embedding_row": row_offset+i,
                         "source_frames": result.history.source_frames[i],
                         "source_times": result.history.source_times[i],
                         "used_latest_fallback": result.history.used_latest_fallback[i]}
                        for i, key in enumerate(result.features.keys)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.rounds <= 23994 or args.torch_threads < 1:
        parser.error("Require 0 <= warmup < rounds <= 23994 and positive torch-threads")
    reference, config, matrices, inputs = read_reference(args.reference_report.resolve())
    video_path = ROOT / "configs/datasets/scene_001_video_manifest.json"
    inputs["video_manifest"] = checked(video_path)
    manifest = json.loads(video_path.read_text())
    require(manifest["dataset"] == "nvidia/PhysicalAI-SmartSpaces"
            and manifest["revision"] == "2cbe9563cbe9f47f846e5c871ee994572bbbc60e"
            and manifest["scene"] == "MTMC_Tracking_2024/train/scene_001", "Different dataset generation")
    require(sorted(item["camera"] for item in manifest["files"]) == [4,5,8], "Unexpected camera set")
    source_manifest_path = ROOT / "configs/datasets/scene_001_source.json"
    inputs["scene_source_manifest"] = checked(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    require(all(source_manifest[k] == manifest[k] for k in ("dataset", "revision", "scene")), "Scene manifests differ")
    calibration_entries = [f for f in source_manifest["files"] if Path(f["local_path"]).name == "calibration_2025_format.json"]
    require(len(calibration_entries) == 1 and calibration_entries[0]["sha256"] == inputs["calibration"]["sha256"],
            "Reference calibration does not belong to pinned scene")
    print("Verifying configuration, calibration, videos and model assets...", flush=True)
    sources = {}
    for item in manifest["files"]:
        path = ROOT / item["local_path"]
        require(path.stat().st_size == item["size_bytes"], "Video size differs")
        inputs[f"video_{item['camera']}"] = checked(path, item["sha256"])
        sources[item["camera"]] = path
    detector_path = ROOT / "docs/detector/first_preview.json"
    inputs["detector_reference"] = checked(detector_path)
    detector_reference = json.loads(detector_path.read_text())
    hashes = [h for h in (detector_reference.get("weights_sha256"),
                          detector_reference.get("configuration", {}).get("weights_sha256")) if h is not None]
    require(bool(hashes) and len(set(hashes)) == 1, "Missing or conflicting detector hash")
    weights = ROOT / detector_reference["weights_path"]
    inputs["detector_weights"] = checked(weights, hashes[0])
    osnet_path = ROOT / "configs/models/osnet_x1_0_msmt17.json"
    inputs["osnet_configuration"] = checked(osnet_path)
    os.environ["RF_HOME"] = str(ROOT / "artifacts/models/rfdetr")
    os.environ["HF_HOME"] = str(ROOT / "artifacts/models/huggingface")
    import torch
    import av
    from mtmc.detection.rfdetr import RFDETRPersonDetector
    from mtmc.reid.osnet import OSNetEncoder
    from mtmc.tracking.bytetrack import CameraTracker, TRACKER_SETTINGS
    from mtmc.video.replay import synchronized_replay
    torch.set_num_threads(args.torch_threads)
    require(torch.cuda.is_available(), "CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    now = datetime.now(timezone.utc)
    run = now.strftime("%Y%m%dT%H%M%S%fZ")
    space = f"scene_001/Z0/native/calibration={inputs['calibration']['sha256']}"
    stage = make_stage(run, matrices, space, config)
    print("Loading persistent CUDA FP32 models...", flush=True)
    t0 = perf_counter_ns()
    detector = RFDETRPersonDetector(weights, threshold=0.1)
    t1 = perf_counter_ns()
    encoder = OSNetEncoder(osnet_path, project_root=ROOT)
    torch.cuda.synchronize()
    t2 = perf_counter_ns()
    pipeline = MTMCPipeline(detector, {c: CameraTracker(c) for c in sources}, encoder, stage,
                            synchronize=torch.cuda.synchronize,
                            history_max_observations=config["history"]["max_observations"],
                            history_max_age=Fraction(config["history"]["max_age_seconds"]))
    output = ROOT / "artifacts/mtmc_pipeline" / run
    output.mkdir(parents=True, exist_ok=False)
    latest = EmbeddingArchive(output / "embeddings.npy")
    means = EmbeddingArchive(output / "mean_embeddings.npy")
    (output / "run_status.json").write_text(dump({"completed": False, "run_id": run, "configuration": config})+"\n")
    samples, emitted = [], set()
    counters = Counter()
    merges, resets = Counter(), Counter()
    peaks = {"history_tracks": 0, "history_vectors": 0}
    try:
        with synchronized_replay(sources, fps=30, threads=1) as batches, \
                (output / "tracks.jsonl").open("w") as trace, \
                (output / "global_tracks.jsonl").open("w") as globals_file, \
                (output / "decisions.jsonl").open("w") as decisions_file:
            for index in range(args.rounds):
                if index == args.warmup:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    measured_start = perf_counter_ns()
                result = pipeline.step(batches)
                record_start = perf_counter_ns()
                offset = latest.rows
                trace.write(dump(local_baseline.trace_record(run, result.batch, result.detections, result.tracks,
                                      result.prepared, result.features, offset))+"\n")
                globals_file.write(dump(identity_record(result, offset, stage))+"\n")
                decisions_file.write(dump(decision_record(result, offset))+"\n")
                latest.append(result.features.embeddings)
                means.append(result.history.mean.embeddings)
                emitted.update(a.global_id for a in result.identities.assignments)
                counters["observations"] += len(result.identities.assignments)
                counters["unencoded_observations"] += len(result.unencoded)
                counters["merge_events"] += len(result.identities.merge_events)
                counters["absorbed_ids"] += sum(len(e.absorbed_global_ids) for e in result.identities.merge_events)
                counters["expired_ids"] += len(result.identities.expired_global_ids)
                counters["allocated_ids"] += len({a.global_id for a in result.identities.base_assignments
                                                  if a.reason == "new_identity"})
                counters["unavailable_projected_observations"] += sum(o.xy is None for g in result.grounds.values() for o in g.observations)
                counters["retained_ids_at_end"] = len(result.identities.identities)
                require(counters["allocated_ids"] == counters["absorbed_ids"]+counters["expired_ids"]+counters["retained_ids_at_end"],
                        "Identity lifecycle accounting differs")
                merges.update(d.outcome for d in result.identities.merge_decisions)
                resets.update(d.reason for d in result.identities.candidate_resets)
                peaks["history_tracks"] = max(peaks["history_tracks"], pipeline.history.active_tracks)
                peaks["history_vectors"] = max(peaks["history_vectors"], pipeline.history.stored_vectors)
                timing = dict(result.timings)
                replay_by_camera = {str(f.camera_id): {"read_decode_ms": f.read_decode_ms,
                                                       "rgb_conversion_ms": f.rgb_conversion_ms}
                                    for f in result.batch.frames}
                timing["read_decode_ms"] = sum(x["read_decode_ms"] for x in replay_by_camera.values())
                timing["rgb_conversion_ms"] = sum(x["rgb_conversion_ms"] for x in replay_by_camera.values())
                timing["record_ms"] = (perf_counter_ns()-record_start)/1e6
                timing["core_and_record_ms"] = timing["core_ms"]+timing["record_ms"]
                samples.append({"frame_index": index, "measured": index >= args.warmup,
                                "crops": len(result.features.keys), "replay_by_camera": replay_by_camera, **timing})
                # Release RGB buffers, crop views and current candidate audit data.
                del result
                if index+1 == args.rounds:
                    measured_end = perf_counter_ns()
                    peak_allocated = torch.cuda.max_memory_allocated()/1024**2
                    peak_reserved = torch.cuda.max_memory_reserved()/1024**2
                if (index+1) % 60 == 0 or index+1 == args.rounds:
                    print(f"Processed {index+1}/{args.rounds}; embeddings={latest.rows}; emitted global IDs={len(emitted)}", flush=True)
        finalize_start = perf_counter_ns()
        require(latest.rows == means.rows, "Latest/mean archive coverage differs")
        latest.finalize()
        means.finalize()
        finalize_ms = (perf_counter_ns()-finalize_start)/1e6
    except Exception as error:
        latest.close(); means.close()
        (output / "run_status.json").write_text(dump({"completed": False, "run_id": run,
            "error_type": type(error).__name__, "error": str(error)})+"\n")
        raise
    measured = [s for s in samples if s["measured"]]
    elapsed = (measured_end-measured_start)/1e9
    phases = [key for key in measured[0] if key.endswith("_ms")]
    summary = {"rounds": args.rounds, "images": args.rounds*3, "warmup_rounds": args.warmup,
               "measured_frame_range": [args.warmup, args.rounds-1], "measured_rounds": len(measured),
               "total_embeddings": latest.rows, "measured_elapsed_s": elapsed,
               "rounds_per_second": len(measured)/elapsed, "images_per_second": len(measured)*3/elapsed,
               "crops_per_second": sum(s["crops"] for s in measured)/elapsed,
               "crops_per_measured_round": {"min": min(s["crops"] for s in measured),
                                            "mean": float(np.mean([s["crops"] for s in measured])),
                                            "max": max(s["crops"] for s in measured)},
               "timings": {p: local_baseline.stats([s[p] for s in measured]) for p in phases},
               "replay_per_camera": {str(c): {p: local_baseline.stats([s["replay_by_camera"][str(c)][p] for s in measured])
                                                for p in ("read_decode_ms", "rgb_conversion_ms")} for c in (4,5,8)},
               "peak_torch_allocated_mib": peak_allocated, "peak_torch_reserved_mib": peak_reserved,
               "ever_emitted_global_ids": len(emitted), **dict(counters), "history_peaks": peaks}
    artifacts = {key: {"path": name, "sha256": sha256(output/name)} for key, name in
                 (("tracks", "tracks.jsonl"), ("global_tracks", "global_tracks.jsonl"), ("decisions", "decisions.jsonl"),
                  ("embeddings", "embeddings.npy"), ("mean_embeddings", "mean_embeddings.npy"))}
    for name in ("embeddings", "mean_embeddings"):
        artifacts[name].update(shape=[latest.rows, 512], dtype="float32")
    paths = sorted((ROOT/"src/mtmc").rglob("*.py")) + [Path(__file__), ROOT/"scripts/run_local_reid.py"]
    report = {"schema_version": 1, "completed": True, "run_id": run, "created_utc": now.isoformat(),
        "protocol": "scene_001_mtmc_sequential_fp32_v1", "summary": summary,
        "configuration": {**config, "cameras": [4,5,8], "fps": 30, "detector_threshold": 0.1,
            "tracker": TRACKER_SETTINGS, "osnet": encoder.configuration, "precision": "FP32", "tf32": False,
            "cudnn_benchmark": False, "identity_start_frame": 0, "ground_truth_used": False,
            "past_assignments_rewritten": False, "dropped_frames": 0, "rendering": False,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "decoder_threads_per_camera": 1, "association_policy": geometry.POLICY,
            "identity_policy": controlled_merge.POLICY, "coordinate_space": space,
            "unencoded_policy": "Current singleton, retained local binding can preserve identity"},
        "inputs": inputs, "merge_decisions": dict(merges), "candidate_resets": dict(resets),
        "git_commit": local_baseline.git_output("rev-parse", "HEAD"),
        "git_status": local_baseline.git_output("status", "--porcelain"),
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "initialization_ms": {"detector_constructor": (t1-t0)/1e6, "osnet_constructor": (t2-t1)/1e6},
        "embedding_finalization_ms": finalize_ms,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda,
            "ffmpeg_libraries": av.library_versions,
            "versions": {p: version(p) for p in ("torch","torchvision","rfdetr","supervision","av","numpy","scipy","pillow")}},
        "timing_protocol": {
            "clock": "Host perf_counter_ns; detector boundary synchronized; encoder returns CPU-ready features",
            "core_ms": "Replay, models, tracking, crops, history, projection, assignment, grouping, global identities",
            "association_block_ms": "Whole identity stage, including validation and four component timings; do not add both",
            "read_decode_ms_and_rgb_conversion_ms": "Subtimings inside replay_ms, also reported per camera; do not add to replay_ms",
            "record_ms": "JSON serialization, buffered trace and feature writes, counters; no durability guarantee",
            "throughput": "Warm loop wall time, including recording and intermediate progress; 1 round = 3 images",
            "excluded": "Checksums, constructors, replay open, final stream close/flush, NPY finalization, reports",
            "warmup": "First rounds processed and recorded normally; no state reset at measurement start",
            "filesystem_cache": "Uncontrolled and warmed by full video checksum reads",
            "memory": "PyTorch allocator peak after warmup reset; reserved includes caches retained from warmup; not total process VRAM",
        },
        "samples": samples, "artifacts": artifacts,
        "limits": ["Single sequential instrumentation run, not a final performance benchmark",
            "Geometry/appearance/merge settings are inherited integration candidates, not independent calibration",
            "Global identity now starts at frame 0; earlier frozen experiments started at frame 2",
            "New detector/tracker/embedding inference; prior 86.02% IDF1 is not asserted for this run",
            "No ground-truth evaluation or rendering in this command; evaluate recorded output separately",
            "Raw bottom-center geometry does not detect occluded feet; local continuity can preserve incorrect IDs"]}
    (output/"report.json").write_text(json.dumps(report, indent=2, default=json_default, allow_nan=False)+"\n")
    (output/"run_status.json").write_text(dump({"completed": True, "run_id": run})+"\n")
    print(json.dumps(summary, indent=2))
    print(f"Report: {output/'report.json'}")
    print(f"Global tracks: {output/'global_tracks.jsonl'}")
    print("Full MTMC pipeline: COMPLETED; quality evaluation pending")


if __name__ == "__main__":
    main()
