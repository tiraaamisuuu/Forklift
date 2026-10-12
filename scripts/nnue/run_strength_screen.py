#!/usr/bin/env python3
"""Two fixed-length, paired evaluator screens using one frozen executable."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from nnue_dataset import sha256_file, write_json_atomic
from promotion_contracts import ORIGINAL_CONTRACT_ID, RECOVERY_CONTRACT_ID, RECOVERY_CONTRACT
from series_lifecycle import coordinator_heartbeat

ROOT = Path(__file__).resolve().parents[2]


def screen_plan(confirm: bool, overnight: bool = False, recover_long: bool = False):
    if recover_long:
        original = screen_plan(False, True)
        return [original[0], ("classical", "fresh long-control recovery vs classical",
                             1200, "90+0.9", RECOVERY_CONTRACT["classical"]["seed"], False)]
    if overnight:
        return [("network", "repaired NNUE vs classical confirmation", 800, "30+0.3", 202609140, False),
                ("classical", "repaired NNUE vs classical long control", 1200, "90+0.9", 202609141, False)]
    if confirm:
        return [("network", "NNUE vs classical confirmation", 800, "30+0.3", 202609130, False),
                ("classical", "NNUE vs classical longer control", 400, "60+0.6", 202609131, False)]
    return [("network", "shared vs original NNUE", 400, "30+0.3", 20260912, True),
            ("classical", "shared NNUE vs classical", 400, "30+0.3", 20260913, False)]


def prepare_long_recovery(source_dir: Path, engine: Path, training_dir: Path) -> tuple[dict, dict]:
    from verify_promotion import PromotionVerifier, read_object

    source_dir = source_dir.resolve()
    source_path = source_dir / "series.json"
    source = read_object(source_path)
    report = PromotionVerifier(source_dir).verify()
    if report["observedSeriesState"] != "interrupted":
        raise RuntimeError("recovery requires an interrupted series; complete or live runs cannot be retried")
    if source.get("contractId", ORIGINAL_CONTRACT_ID) != ORIGINAL_CONTRACT_ID:
        raise RuntimeError("recovery supports the original frozen promotion contract only")
    required = [
        check for check in report["checks"]
        if check["code"].startswith(("network.", "engine.", "training."))
        or check["code"] in {"series.stages", "series.contract_id"}
    ]
    if not required or any(check["status"] != "pass" for check in required):
        raise RuntimeError("source short stage, export, or frozen-engine provenance failed verification")
    if sha256_file(engine) != source.get("engineSha256"):
        raise RuntimeError("recovery must use the original frozen executable")
    if str(training_dir.resolve()) != source.get("training"):
        raise RuntimeError("recovery must use the original training experiment")
    if (source_dir / "classical" / "result.json").exists():
        raise RuntimeError("a terminal long-stage result cannot be discarded for a new attempt")
    network_stage = next(stage for stage in source["stages"] if stage["id"] == "network")
    if network_stage["state"] != "complete":
        raise RuntimeError("only a completed short stage can be reused")
    recovery = {
        "sourceSeries": str(source_path),
        "sourceSeriesSha256": sha256_file(source_path),
        "reusedStage": "network",
        "reusedResultSha256": sha256_file(source_dir / "network" / "result.json"),
        "excludedStage": "classical",
        "excludedLogSha256": sha256_file(source_dir / "classical" / "match.log"),
        "excludedPgnSha256": sha256_file(source_dir / "classical" / "games.pgn"),
        "reason": "interrupted before the predeclared sample completed; partial games excluded",
    }
    return network_stage, recovery


def stage_command(args, stage, networks, run):
    stage_id, _, games, tc, seed, neural_baseline = stage
    command = [sys.executable, "-u", str(ROOT / "scripts/compare_engines.py"),
               "--candidate-exe", str(args.engine.resolve()), "--baseline-exe", str(args.engine.resolve()),
               "--candidate-name", "NNUE-shared", "--baseline-name", "NNUE-original" if neural_baseline else "Classical",
               "--candidate-eval-file", networks["shared"],
               "--candidate-option", "NNUE Weight=100",
               "--games", str(games), "--tc", tc, "--threads", "1", "--hash", "128",
               "--concurrency", "6" if args.overnight or args.recover_long_from else "8",
               "--seed", str(seed), "--output-dir", str(run / stage_id)]
    if neural_baseline:
        command += ["--baseline-eval-file", networks["control"], "--baseline-option", "NNUE Weight=100"]
    else:
        command += ["--baseline-option", "Use NNUE=false"]
    if args.quick:
        command.append("--quick")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--confirm-classical", action="store_true",
                        help="800-game replication and 400-game longer-control classical test")
    mode.add_argument("--overnight", action="store_true",
                        help="Fresh repaired-engine confirmation, then 1200 longer games; no automatic promotion")
    mode.add_argument("--recover-long-from", type=Path,
                      help="Reuse a verified completed short stage and replace the interrupted long stage with fresh evidence")
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
    plan = screen_plan(args.confirm_classical, args.overnight, bool(args.recover_long_from))
    state = {"schemaVersion": 2, "state": "running", "current": "network",
             "startedAt": datetime.now(timezone.utc).isoformat(),
             "engineSha256": sha256_file(args.engine), "training": str(args.training_dir.resolve()),
             "trainingManifestSha256": sha256_file(args.training_dir / "experiment.json"),
             "runnerCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "runnerDirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
             "stages": [{"id": item[0], "name": item[1], "state": "queued",
                         "command": stage_command(args, item, networks, run)} for item in plan]}
    if args.overnight:
        state["contractId"] = ORIGINAL_CONTRACT_ID
    if args.recover_long_from:
        reused, recovery = prepare_long_recovery(args.recover_long_from, args.engine, args.training_dir)
        state.update(contractId=RECOVERY_CONTRACT_ID, recovery=recovery, current="classical")
        state["stages"][0] = reused
    run.mkdir(parents=True, exist_ok=True)
    status = run / "series.json"
    write_json_atomic(status, state)
    try:
        with coordinator_heartbeat(run):
            for stage in state["stages"]:
                if stage["state"] == "complete":
                    continue
                state["current"] = stage["id"]
                stage["state"] = "running"
                write_json_atomic(status, state)
                with (run / f"{stage['id']}.runner.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(stage["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{stage['name']} runner failed: exit {result.returncode}")
                report = json.loads((run / stage["id"] / "result.json").read_text())
                stage["result"] = report
                if not report.get("completed") or any(report.get("failures", {}).values()):
                    raise RuntimeError(f"{stage['name']} incomplete or technical failure; inspect preserved results")
                stage["state"] = "complete"
                write_json_atomic(status, state)
        state["state"] = "complete"
        state["completedAt"] = datetime.now(timezone.utc).isoformat()
    except (Exception, KeyboardInterrupt) as error:
        state.update(state="failed", error=str(error))
        for stage in state["stages"]:
            if stage["state"] == "running":
                stage["state"] = "failed"
        raise
    finally:
        write_json_atomic(status, state)


if __name__ == "__main__":
    main()
