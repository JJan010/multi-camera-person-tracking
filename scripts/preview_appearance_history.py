"""Replay frozen appearance observations into a causal history, on CPU only."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, ReIDBatch, sha256

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_frozen_run(report_path):
    report = read_json(report_path)
    require(report["completed"] is True, "Source pipeline did not complete")
    tracks_path = report_path.parent / report["artifacts"]["tracks"]["path"]
    features_path = report_path.parent / report["artifacts"]["embeddings"]["path"]
    require(sha256(tracks_path) == report["artifacts"]["tracks"]["sha256"], "Trace checksum mismatch")
    require(sha256(features_path) == report["artifacts"]["embeddings"]["sha256"], "Embedding checksum mismatch")
    features = np.load(features_path, allow_pickle=False, mmap_mode="r")
    require(features.dtype == np.float32 and features.shape == (report["summary"]["total_embeddings"], 512)
            and list(features.shape) == report["artifacts"]["embeddings"]["shape"]
            and np.isfinite(features).all()
            and np.allclose(np.linalg.norm(features, axis=1), 1, rtol=0, atol=1e-5),
            "Invalid source feature matrix")
    with tracks_path.open(encoding="utf-8") as handle:
        trace = [json.loads(line) for line in handle]
    require([row["frame_index"] for row in trace] == list(range(report["summary"]["rounds"])),
            "Trace must contain every source round in order")
    expected_row = 0
    for row in trace:
        require(row["run_id"] == report["run_id"], "Mixed run IDs")
        timestamp = Fraction(row["timestamp"])
        require(timestamp == Fraction(row["frame_index"], report["configuration"]["fps"])
                and math.isclose(float(timestamp), row["timestamp_seconds"], rel_tol=0, abs_tol=1e-9),
                "Source timeline mismatch")
        require(sorted(item["camera"] for item in row["cameras"]) == sorted(report["configuration"]["cameras"]),
                "Camera coverage mismatch")
        for item in row["reid_observations"]:
            require(item["frame_index"] == row["frame_index"]
                    and item["camera"] in report["configuration"]["cameras"], "Observation frame/camera mismatch")
            if item["status"] == "encoded":
                require(type(item["embedding_row"]) is int and item["embedding_row"] == expected_row,
                        "Embedding rows must be unique, contiguous and in source order")
                expected_row += 1
            else:
                require(item["status"] == "skipped_fully_outside" and item["embedding_row"] is None,
                        "Unknown skip policy or invalid embedding row")
    require(expected_row == len(features), "Trace does not cover the complete feature matrix")
    return report, trace, features, tracks_path, features_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reid/history_baseline.json")
    args = parser.parse_args()
    report_path, config_path = args.run_report.resolve(), args.config.resolve()
    config = read_json(config_path)
    print("Verifying frozen trace and embedding checksums...", flush=True)
    source, trace, embeddings, tracks_path, embeddings_path = read_frozen_run(report_path)
    memory = AppearanceHistory(source["run_id"], max_observations=config["max_observations"],
                               max_age=Fraction(config["max_age_seconds"]))
    now = datetime.now(timezone.utc)
    output = ROOT / "artifacts/appearance_history" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    mean_path = output / "mean_embeddings.npy"
    means = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32, shape=embeddings.shape)
    observations_path = output / "observations.jsonl"
    history_sizes, source_row_by_key = Counter(), {}
    update_ms, cosines, examples = [], [], []
    observations, fallbacks, max_tracks, max_vectors = 0, 0, 0, 0
    print(f"Replaying {len(trace)} rounds, {len(embeddings)} observations on CPU...", flush=True)
    with observations_path.open("w", encoding="utf-8") as handle:
        for record in trace:
            frame = record["frame_index"]
            timestamp = Fraction(record["timestamp"])
            selected = [item for item in record["reid_observations"] if item["status"] == "encoded"]
            rows = [item["embedding_row"] for item in selected]
            keys = tuple(ObservationKey(item["camera"], item["local_id"], frame) for item in selected)
            inputs = ReIDBatch(keys, (timestamp,) * len(keys), np.asarray(embeddings[rows]))
            for key, row in zip(keys, rows):
                require(key not in source_row_by_key, "Duplicate observation key")
                source_row_by_key[key] = row
            started = perf_counter_ns()
            result = memory.update(frame, timestamp, inputs)
            update_ms.append((perf_counter_ns() - started) / 1e6)
            require(result.latest.keys == keys and result.mean.keys == keys
                    and result.latest.timestamps == inputs.timestamps and result.mean.timestamps == inputs.timestamps,
                    "History changed observation mapping")
            require(np.array_equal(result.latest.embeddings, inputs.embeddings), "Latest baseline changed")
            require(np.isfinite(result.mean.embeddings).all()
                    and np.allclose(np.linalg.norm(result.mean.embeddings, axis=1), 1, rtol=0, atol=1e-5),
                    "Mean descriptors are not normalized")
            means[rows] = result.mean.embeddings
            similarity = np.clip(np.sum(inputs.embeddings * result.mean.embeddings, axis=1), -1, 1)
            cosines.extend(float(v) for v in similarity)
            for index, (key, row) in enumerate(zip(keys, rows)):
                frames, times = result.source_frames[index], result.source_times[index]
                require(1 <= len(frames) <= memory.max_observations and frames[-1] == frame
                        and all(f <= frame for f in frames)
                        and all(Fraction(0) <= timestamp - t <= memory.max_age for t in times),
                        "History violates count, causality or age limits")
                history_sizes[len(frames)] += 1
                source_rows = [source_row_by_key[ObservationKey(key.camera_id, key.local_id, f)] for f in frames]
                details = {"run_id": source["run_id"], "camera": key.camera_id, "local_id": key.local_id,
                           "frame_index": frame, "timestamp": str(timestamp), "embedding_row": row,
                           "history_size": len(frames), "source_frames": frames,
                           "source_embedding_rows": source_rows,
                           "history_span_seconds": float(times[-1] - times[0]),
                           "latest_mean_cosine": float(similarity[index]),
                           "mean_norm_before_normalization": result.mean_norms_before_normalization[index],
                           "used_latest_fallback": result.used_latest_fallback[index]}
                handle.write(json.dumps(details, allow_nan=False) + "\n")
                # Fixed examples around the known local ID swap; labels are NOT inputs.
                if key.camera_id == 8 and key.local_id in (10, 12) and frame in (210, 213, 214, 217, 221):
                    examples.append(details)
            observations += len(keys)
            fallbacks += sum(result.used_latest_fallback)
            max_tracks = max(max_tracks, memory.active_tracks)
            max_vectors = max(max_vectors, memory.stored_vectors)
            if (frame + 1) % 60 == 0 or frame + 1 == len(trace):
                print(f"Processed {frame + 1}/{len(trace)}; descriptors: {observations}", flush=True)
    means.flush()
    del means
    require(observations == len(embeddings), "Not every source observation received a descriptor")
    cosine_summary = ({"minimum": float(np.min(cosines)), "median": float(np.median(cosines)),
                       "p05": float(np.percentile(cosines, 5))} if cosines else None)
    summary = {"rounds": len(trace), "observations": observations,
               "max_observations": memory.max_observations, "max_age_seconds": str(memory.max_age),
               "history_size_counts": dict(sorted(history_sizes.items())),
               "latest_matches_source_exactly": True, "causality_and_age_checks": True,
               "mean_descriptors_normalized": True, "cancellation_fallbacks": fallbacks,
               "peak_cached_tracks": max_tracks, "peak_cached_vectors": max_vectors,
               "peak_vector_payload_mib": max_vectors * 512 * 4 / 1024**2,
               "latest_vs_mean_cosine": cosine_summary,
               "update_ms": {"mean": float(np.mean(update_ms)), "median": float(np.median(update_ms)),
                             "p95": float(np.percentile(update_ms, 95))}}
    report = {"schema_version": 1, "created_utc": now.isoformat(), "completed": True,
              "source_run_id": source["run_id"], "summary": summary,
              "source_report": {"path": str(report_path), "sha256": sha256(report_path)},
              "source_tracks": {"path": str(tracks_path), "sha256": sha256(tracks_path)},
              "source_embeddings": {"path": str(embeddings_path), "sha256": sha256(embeddings_path)},
              "configuration": config, "config_sha256": sha256(config_path),
              "module_sha256": sha256(ROOT / "src/mtmc/reid/history.py"), "script_sha256": sha256(Path(__file__)),
              "numpy": version("numpy"), "scope": "CPU descriptor-history construction, no models and no GT",
              "row_mapping": "Output mean row i describes the same current observation as source embedding row i",
              "artifacts": {"mean_embeddings": {"path": mean_path.name, "sha256": sha256(mean_path),
                                                 "shape": list(embeddings.shape), "dtype": "float32"},
                            "observations": {"path": observations_path.name, "sha256": sha256(observations_path)}},
              "diagnostic_examples": examples, "update_samples_ms": update_ms,
              "limits": ["No retrieval or global association quality measured yet",
                         "Latest/mean cosine measures descriptor change, not identity correctness",
                         "Payload memory excludes Python metadata, temporary arrays and offline trace loading",
                         "Update times are one CPU replay diagnostic, not full-pipeline timing",
                         "A local ID switch can mix identities in the history"]}
    target = output / "report.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Report: {target}")
    print(f"Mean embeddings: {mean_path}")
    print("Appearance history replay: COMPLETED")


if __name__ == "__main__":
    main()
