"""Check variable-duration GT, prefix semantics and optional 300-round parity."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

import evaluate_mtmc_sequence as sequence
import run_mtmc_sequence as workflow


def rejected(callback):
    try:
        callback()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Invalid input accepted")


def unit_checks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root/"ground_truth.txt"
        lines = [f"{c} 7 {f} 10 10 20 40 0 0" for f in range(2,605) for c in (4,5,8)]
        path.write_text("\n".join(lines)+"\n")
        short = sequence.load_ground_truth(path,300)
        long = sequence.load_ground_truth(path,601)
        assert len(short) == 298*3 and len(long) == 599*3
        assert long[600,8][7] == [10.,10.,30.,50.] and (601,8) not in long
        assert all(long[k] == v for k,v in short.items())
        path.write_text("\n".join(line for line in lines if not line.startswith("8 7 600 "))+"\n")
        rejected(lambda:sequence.load_ground_truth(path,601))
        path.write_text("\n".join(lines+[lines[0]])+"\n")
        rejected(lambda:sequence.load_ground_truth(path,601))
        print("Variable GT duration, exact last-frame boundary, missing slot and duplicate labels: OK")
        # One GT identity changes predicted ID. Prefix mappings are recomputed
        # globally; window-perfect results must not be averaged into 100%.
        full,control = sequence.metric.IdentityCounts(),sequence.metric.IdentityCounts()
        for _ in range(2):
            full.update([7],[1],np.ones((1,1),bool))
            control.update([7],[10],np.ones((1,1),bool))
        first = sequence.prefix_result(full,control,4)
        assert first["pipeline_idf1"] == 1.
        for _ in range(2):
            full.update([7],[2],np.ones((1,1),bool))
            control.update([7],[10],np.ones((1,1),bool))
        second = sequence.prefix_result(full,control,6)
        assert second["pipeline_idf1"] == .5 and second["control_idf1"] == 1.
        assert first["pipeline_idf1"] == 1. and first["predicted_observations"] == 2
        print("Growing prefix uses one shared mapping; an identity split gives 100% -> 50%, not an average of window scores: OK")
        folder = root/"artifacts/mtmc_pipeline/synthetic-run"
        folder.mkdir(parents=True)
        report_path = folder/"report.json"
        report_path.write_text(json.dumps({"completed":True,"protocol":"scene_001_mtmc_sequential_fp32_v1",
                                          "run_id":"synthetic-run","summary":{"rounds":1800}}))
        line = f"Report: {report_path}\n"
        assert workflow.checked_report([line],root,1800) == report_path
        rejected(lambda:workflow.checked_report([],root,1800))
        rejected(lambda:workflow.checked_report([line,line],root,1800))
        rejected(lambda:workflow.checked_report([line],root,300))
        print("Workflow evaluates only its explicitly emitted completed report, never a latest-directory guess: OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-evaluation",type=Path)
    args = parser.parse_args()
    unit_checks()
    if args.baseline_evaluation:
        path = args.baseline_evaluation.resolve()
        original = json.loads(path.read_text())
        sequence.require(original.get("completed") is True and
                         original["protocol"]["name"] == "scene_001_full_mtmc_global_2d_identity_v1",
                         "Expected original 300-round pipeline evaluation")
        item = original["inputs"]["pipeline_report"]
        source_path = sequence.metric.checked(item["path"],item["sha256"])
        source = json.loads(source_path.read_text())
        sequence.require(source["summary"]["rounds"] == 300,"Reference must contain 300 rounds")
        result_path = sequence.evaluate(source_path)
        new = json.loads(result_path.read_text())
        sequence.require(new["source_run_id"] == original["source_run_id"],"Different reference run")
        for name in ("pipeline_metrics","no_cross_camera_control","excluded_zero_area_gt",
                     "fully_outside_predictions","merge_event_diagnostics","control_local_id_map"):
            sequence.require(new[name] == original[name],f"300-round baseline differs: {name}")
        prefix = new["prefixes"][-1]
        sequence.require(prefix["runtime_rounds"] == 300 and prefix["pipeline_idf1"] == original["pipeline_metrics"]["idf1"],
                         "Final prefix differs from original metric")
        print("Original 300-round metrics, denominators, merge diagnostics and control map: EXACT")
    print("MTMC sequence evaluation checks: PASSED")


if __name__ == "__main__":
    main()
