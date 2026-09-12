#!/usr/bin/env python3
"""Two fixed-length, paired evaluator screens using one frozen executable."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from nnue_dataset import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.exists() and any(run.iterdir()):
        raise SystemExit("Use a new empty evidence directory")
    experiment = json.loads((args.training_dir / "experiment.json").read_text())
    if experiment["state"] != "complete":
        raise SystemExit("Training experiment is not complete")
    networks = {}
    for variant in experiment["variants"]:
        path = Path(variant["network"]["path"])
        if sha256_file(path) != variant["network"]["sha256"]:
            raise SystemExit("Network checksum mismatch")
        report = json.loads(path.with_suffix(".manifest.json").read_text())
        if report["cppVerification"]["exactMatch"] is not True:
            raise SystemExit("Network did not pass C++ export verification")
        networks[variant["id"]] = str(path)
    state = {"state": "running", "current": "network", "startedAt": datetime.now(timezone.utc).isoformat(),
             "engineSha256": sha256_file(args.engine), "training": str(args.training_dir.resolve()),
             "stages": [{"id": "network", "name": "shared vs original NNUE", "state": "queued"},
                        {"id": "classical", "name": "shared NNUE vs classical", "state": "queued"}]}
    run.mkdir(parents=True, exist_ok=True)
    status = run / "series.json"
    try:
        for index, stage in enumerate(state["stages"]):
            state["current"] = stage["id"]
            stage["state"] = "running"
            command = [sys.executable, "-u", str(ROOT / "scripts/compare_engines.py"),
                       "--candidate-exe", str(args.engine.resolve()), "--baseline-exe", str(args.engine.resolve()),
                       "--candidate-name", "NNUE-shared", "--baseline-name", "NNUE-original" if index == 0 else "Classical",
                       "--candidate-eval-file", networks["shared"],
                       "--games", "400", "--tc", "30+0.3", "--threads", "1", "--hash", "128",
                       "--concurrency", "8", "--seed", str(20260912 + index),
                       "--output-dir", str(run / stage["id"])]
            if index == 0:
                command += ["--baseline-eval-file", networks["control"]]
            else:
                command += ["--baseline-option", "Use NNUE=false"]
            if args.quick:
                command.append("--quick")
            stage["command"] = command
            write_json_atomic(status, state)
            with (run / f"{stage['id']}.runner.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{stage['name']} runner failed: exit {result.returncode}")
            report = json.loads((run / stage["id"] / "result.json").read_text())
            stage["result"] = report
            if not report.get("completed") or any(report.get("failures", {}).values()):
                raise RuntimeError(f"{stage['name']} incomplete or technical failure; inspect preserved results")
            stage["state"] = "complete"
            write_json_atomic(status, state)
        state["state"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        state.update(state="failed", error=str(error))
        raise
    finally:
        write_json_atomic(status, state)


if __name__ == "__main__":
    main()
