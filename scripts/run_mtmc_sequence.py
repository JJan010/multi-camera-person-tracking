"""Run the unchanged pipeline, then evaluate exactly the report it produced."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def checked_report(lines, root, rounds):
    reports = [line[len("Report: "):].strip() for line in lines if line.startswith("Report: ")]
    if len(reports) != 1:
        raise RuntimeError("Pipeline must print exactly one completed report path")
    path = Path(reports[0]).resolve()
    if path.name != "report.json" or path.parent.parent != (root/"artifacts/mtmc_pipeline").resolve():
        raise RuntimeError("Unexpected pipeline report destination")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("completed") is not True or report.get("protocol") != "scene_001_mtmc_sequential_fp32_v1"
            or report["summary"]["rounds"] != rounds or report["run_id"] != path.parent.name):
        raise RuntimeError("Pipeline output is incomplete or has a different scope")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1800)
    args = parser.parse_args()
    if not 300 <= args.rounds <= 23994:
        parser.error("Require 300 <= rounds <= 23994")
    reference = args.reference_report.resolve()
    if not reference.is_file():
        parser.error(f"Reference report does not exist: {reference}")
    evaluator = ROOT/"scripts/evaluate_mtmc_sequence.py"
    if not evaluator.is_file():
        parser.error("Sequence evaluator is missing")
    command = [sys.executable, "-u", str(ROOT/"scripts/run_mtmc.py"),
               "--reference-report", str(reference), "--rounds", str(args.rounds),
               "--warmup", "30", "--torch-threads", "1"]
    print("Phase 1: unchanged CUDA FP32 pipeline; warmup=30, torch-threads=1...", flush=True)
    reports = []
    # Relay progress immediately and capture only the declared report path.
    # Never infer a result by selecting the newest directory from another run.
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as child:
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                if line.startswith("Report: "):
                    reports.append(line)
            code = child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            raise
    if code:
        raise SystemExit(code)
    report_path = checked_report(reports, ROOT, args.rounds)
    print(f"Phase 2: CPU evaluation of this exact run: {report_path}", flush=True)
    result = subprocess.run([sys.executable, "-u", str(evaluator), "--run-report", str(report_path)], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)
    print("MTMC temporal extension: COMPLETED; no threshold tuning or runtime state reset")


if __name__ == "__main__":
    main()
