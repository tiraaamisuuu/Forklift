#!/usr/bin/env python3
"""Train a predeclared NNUE feature-sharing candidate and matched control."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from nnue_dataset import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cpp-tools", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--resume-failed", action="store_true",
                        help="Resume a failed experiment from its last saved epoch")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.exists() and any(run.iterdir()) and not args.resume_failed:
        raise SystemExit("Use a new run directory; existing evidence is preserved.")
    if not args.cpp_tools.is_file():
        raise SystemExit("C++ verification executable is missing")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise SystemExit("Commit the experiment implementation before training")
    dataset = args.dataset_dir.resolve()
    source_manifest = dataset / "dataset.manifest.json"
    provenance = json.loads(source_manifest.read_text(encoding="utf-8-sig"))
    inputs = []
    for filename, split in (("train.nnuebin", "training"), ("validation.nnuebin", "validation")):
        path = dataset / filename
        checksum = sha256_file(path)
        expected = next(x for x in provenance["outputs"] if x["split"] == split)
        if checksum != expected["sha256"] or path.stat().st_size != expected["sizeBytes"]:
            raise SystemExit(f"Dataset provenance mismatch: {path}")
        inputs.append({"path": str(path), "sha256": checksum, "split": split})
    run.mkdir(parents=True, exist_ok=True)
    experiment = {
        "schemaVersion": 1, "name": "NNUE v2 · shared piece-square learning",
        "state": "running", "pid": os.getpid(),
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "data": inputs, "datasetManifestSha256": sha256_file(source_manifest),
        "hypothesis": "Sharing piece-square weights across king squares improves sparse-feature learning.",
        "contract": {"epochs": args.epochs, "seed": args.seed, "hidden": 256,
                     "batchSize": 2048, "resultWeight": 0, "loss": "wdl", "patience": 5,
                     "workers": 2, "cpuThreads": 2, "selection": "minimum validation loss"},
        "variants": [{"id": "shared", "name": "shared features", "state": "queued"},
                     {"id": "control", "name": "original features", "state": "queued"}],
    }
    manifest = run / "experiment.json"
    if args.resume_failed:
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("state") != "failed":
            raise SystemExit("Only a failed experiment can be resumed")
        if previous.get("contract") != experiment["contract"] or previous.get("data") != inputs:
            raise SystemExit("Resume settings or dataset differ from the original experiment")
        if [v.get("id") for v in previous.get("variants", [])] != ["shared", "control"]:
            raise SystemExit("Unexpected experiment variants")
        attempt = len(previous.get("recoveries", [])) + 1
        write_json_atomic(run / f"experiment.failed-{attempt}.json", previous)
        previous.setdefault("recoveries", []).append({
            "at": experiment["startedAt"], "commit": experiment["commit"],
            "reason": previous.get("error"),
        })
        previous.update(state="running", pid=os.getpid())
        previous.pop("error", None)
        experiment = previous
    write_json_atomic(manifest, experiment)
    # Keep training awake while allowing displays to sleep.
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for variant in experiment["variants"]:
            if variant["state"] == "complete":
                continue
            directory = run / variant["id"]
            directory.mkdir(exist_ok=args.resume_failed)
            output = directory / "model.nnue"
            command = [sys.executable, "-u", str(ROOT / "scripts/nnue/train.py"),
                       "--data", str(dataset / "train.nnuebin"),
                       "--validation-data", str(dataset / "validation.nnuebin"),
                       "--output", str(output), "--hidden", "256", "--epochs", str(args.epochs),
                       "--seed", str(args.seed), "--batch-size", "2048", "--workers", "2",
                       "--cpu-threads", "2", "--device", "cuda", "--loss", "wdl",
                       "--result-weight", "0", "--wdl-scale", "400", "--target-scale", "600",
                       "--output-scale", "32", "--hidden-scale", "1024",
                       "--learning-rate", "0.001", "--minimum-learning-rate", "0.00001",
                       "--scheduler", "cosine", "--early-stopping-patience", "5",
                       "--verify-samples", "512", "--checkpoint-every", "4",
                       "--cpp-tools", str(args.cpp_tools.resolve())]
            if variant["id"] == "shared":
                command.append("--feature-factorization")
            checkpoint = output.with_suffix(".checkpoint.pt")
            if args.resume_failed and checkpoint.is_file():
                command.extend(["--resume", str(checkpoint)])
            variant.pop("exitCode", None)
            variant.update(state="running", command=command)
            write_json_atomic(manifest, experiment)
            print(f"Training {variant['name']}", flush=True)
            with (directory / "train.log").open("a" if args.resume_failed else "w", encoding="utf-8") as log:
                if args.resume_failed:
                    log.write("\n--- recovery attempt ---\n")
                    log.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                variant.update(state="failed", exitCode=result.returncode)
                raise RuntimeError(f"{variant['name']} failed; inspect {directory / 'train.log'}")
            report = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            if report.get("cppVerification", {}).get("exactMatch") is not True:
                raise RuntimeError("Export did not pass exact C++ verification")
            variant.update(state="complete", validation=report["bestValidation"],
                           quantization=report["quantization"], network=report["outputs"]["network"])
            write_json_atomic(manifest, experiment)
        experiment.update(state="complete", completedAt=datetime.now(timezone.utc).isoformat())
        control, shared = experiment["variants"][1], experiment["variants"][0]
        experiment["comparison"] = {
            "validationLossChangePercent": 100 * (shared["validation"]["loss"] /
                                                    control["validation"]["loss"] - 1),
            "playingStrength": "not tested; offline improvement alone is not promotion",
        }
        write_json_atomic(manifest, experiment)
        print("Training and C++ export checks complete. Review validation slices before matches.", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        experiment.update(state="failed", error=str(error) or type(error).__name__)
        write_json_atomic(manifest, experiment)
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
