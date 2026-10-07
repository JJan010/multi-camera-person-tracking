"""Check composition and optionally exact frozen-replay parity; CPU only."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS

import numpy as np

import run_mtmc as runner
from mtmc.pipeline.core import MTMCPipeline, require
from mtmc.reid.crops import CropRecord
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, ReIDBatch


def plain(value):
    return json.loads(runner.dump(value))


def synthetic_checks():
    config = {"appearance_variant": "mean", "appearance_threshold": .7,
              "geometry": {"max_distance": 1., "unavailable_policy": "appearance_only"},
              "identity": {"max_idle_seconds": "1", "min_support_rounds": 2,
                           "min_support_seconds": "1/30", "max_evidence_gap": "1/10"}}
    matrices = {c: np.eye(3) for c in (4,5,8)}

    class Detector:
        def detect(self, batch):
            result = []
            for camera in (8,5,4):  # Deliberately different camera order.
                boxes = []
                if camera != 8 and batch.frame_index < 4:
                    x = 7 if camera == 5 and batch.frame_index == 0 else 1
                    if camera == 5 and batch.frame_index == 3:
                        x = 20  # Fully outside image: no crop, same local binding.
                    boxes = [[x,1,x+1,4]]
                n = len(boxes)
                result.append(NS(camera_id=camera, frame_index=batch.frame_index, timestamp=batch.timestamp,
                                 local_ids=np.ones(n, dtype=np.int64), xyxy=np.array(boxes,np.float32).reshape(-1,4),
                                 confidence=np.full(n,.9,np.float32)))
            return tuple(result)

    class Tracker:
        def update(self, detection):
            return detection

    class Encoder:
        def __init__(self):
            self.calls = []
        def encode(self, crops):
            self.calls.append(len(crops))
            vectors = np.zeros((len(crops),512),np.float32)
            vectors[:,0] = 1
            return ReIDBatch(tuple(c.key for c in crops),tuple(c.timestamp for c in crops),vectors)

    def batches():
        for index in range(36):
            time = Fraction(index,30)
            yield NS(frame_index=index,timestamp=time,
                     frames=tuple(NS(camera_id=c,frame_index=index,timestamp=time,rgb=np.ones((10,10,3),np.uint8))
                                  for c in (5,4,8)))

    encoder = Encoder()
    stage = runner.make_stage("synthetic-composition",matrices,"synthetic-plane",config)
    pipeline = MTMCPipeline(Detector(),{c:Tracker() for c in matrices},encoder,stage,synchronize=lambda:None)
    stream = batches()
    snapshots, offset, past = [], 0, None
    with tempfile.TemporaryDirectory() as directory:
        archive = runner.EmbeddingArchive(Path(directory)/"vectors.npy")
        all_features = []
        for index in range(36):
            result = pipeline.step(stream)
            record = plain(runner.identity_record(result,offset,stage))
            decision = plain(runner.decision_record(result,offset))
            assert decision["frame_index"] == index and decision["timestamp"] == str(Fraction(index,30))
            if index == 0:
                assert len({a.global_id for a in result.identities.assignments}) == 2
                past = (record,deepcopy(record))
            if index == 1:
                assert len(result.identities.pending_candidates) == 1 and not result.identities.merge_events
            if index == 2:
                assert len(result.identities.merge_events) == 1
                assert len({a.global_id for a in result.identities.assignments}) == 1
            if index == 3:
                assert len(result.features.keys) == 1 and len(result.identities.assignments) == 2
                assert len(result.unencoded) == 1
                assert len({a.global_id for a in result.identities.assignments}) == 1
                a = next(a for a in record["assignments"] if a["key"]["camera_id"] == 5)
                assert a["embedding_row"] is None and a["has_current_embedding"] is False
            if index == 35:
                assert not result.identities.identities and pipeline.history.stored_vectors == 0
            assert all(np.isfinite(t) and t >= 0 for t in result.timings.values())
            archive.append(result.features.embeddings)
            all_features.append(result.features.embeddings)
            offset += len(result.features.keys)
            snapshots.append(record)
        archive.finalize()
        assert np.array_equal(np.load(archive.path),np.concatenate(all_features))
        assert not archive.raw.exists()
        empty = runner.EmbeddingArchive(Path(directory)/"empty.npy")
        empty.append(np.empty((0,512),np.float32)); empty.finalize()
        assert np.load(empty.path).shape == (0,512)
    assert past[0] == past[1] and encoder.calls[:4] == [2,2,2,1]
    assert encoder.calls[4:] == [0]*32
    print("Adapters -> crops -> history -> geometry -> confirmed global merge: OK")
    print("Camera/key mapping, unencoded singleton continuity, empty rounds and expiry: OK")
    print("Earlier outputs unchanged; exact scene time; streamed embedding row parity including empty output: OK")
    try:
        pipeline.step(iter([NS(frame_index=2,timestamp=Fraction(2,30),frames=())]))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid replay order accepted")
    try:
        pipeline.step(batches())
    except RuntimeError:
        pass
    else:
        raise AssertionError("Failed state reused")
    print("A failed round stops the run; partially advanced adapters cannot be retried: OK")


def reference_check(path):
    # Reuse the existing checksum/row validators. No GT labels enter the stage.
    import evaluate_appearance_history as history_io
    reference, config, matrices, inputs = runner.read_reference(path)
    run = reference["source_run_id"]
    history_report, source, trace, history_rows, latest, means, paths = history_io.load_inputs(
        Path(inputs["history_reference"]["path"]))
    require(source["run_id"] == run, "Different frozen source run")
    require(runner.sha256(paths["tracks"]) == reference["inputs"]["tracks"]["sha256"], "Frozen trace differs")
    frozen_path = path.parent/reference["artifacts"]["assignments.jsonl.gz"]["path"]
    runner.checked(frozen_path, reference["artifacts"]["assignments.jsonl.gz"]["sha256"])
    p = reference["protocol"]
    first, last = p["frames"]
    stage = runner.make_stage(run,matrices,p["coordinate_space"],config)
    history = AppearanceHistory(run,max_observations=config["history"]["max_observations"],
                                max_age=Fraction(config["history"]["max_age_seconds"]))
    count = 0
    print("Verifying frozen history and full identity-record parity on CPU...",flush=True)
    with gzip.open(frozen_path,"rt") as saved:
        for row in trace:
            frame = row["frame_index"]
            if frame > last:
                break
            time = Fraction(row["timestamp"])
            encoded = [r for r in row["reid_observations"] if r["status"] == "encoded"]
            keys = tuple(ObservationKey(r["camera"],r["local_id"],frame) for r in encoded)
            indices = [r["embedding_row"] for r in encoded]
            batch = ReIDBatch(keys,(time,)*len(keys),np.asarray(latest[indices]).copy())
            h = history.update(frame,time,batch)
            require(np.array_equal(h.mean.embeddings,means[indices]),"Recomputed history differs")
            if frame < first:
                continue
            records = tuple(CropRecord(ObservationKey(r["camera"],r["local_id"],frame),r["confidence"],
                                tuple(r["source_xyxy"]),None if r["crop_xyxy_int"] is None else tuple(r["crop_xyxy_int"]),
                                r["inside_image_fraction"]) for r in row["reid_observations"])
            descriptors = h.mean if config["appearance_variant"] == "mean" else h.latest
            result, groups, pairs, grounds, missing, timing = stage.update(frame,time,descriptors,records)
            line = saved.readline(); require(bool(line),"Truncated frozen assignment trace")
            old = json.loads(line)
            current = plain(asdict(result))
            for name, value in current.items():
                expected = old[name]
                if name == "assignments":
                    expected = [{k:a[k] for k in ("key","global_id","reason")} for a in expected]
                require(value == expected,f"Frozen identity field differs at frame {frame}: {name}")
            count += 1
        require(saved.readline() == "" and count == last-first+1,"Frozen timeline coverage differs")
    print(f"Recomputed mean descriptors and every identity/merge/state record: {count}/{count} EXACT")
    print("This check uses the reference identity start frame; video execution starts identity state at frame 0.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report",type=Path)
    args = parser.parse_args()
    synthetic_checks()
    if args.reference_report is not None:
        reference_check(args.reference_report.resolve())
    print("MTMC pipeline composition checks: PASSED (CPU; CUDA adapters require the video run)")


if __name__ == "__main__":
    main()
