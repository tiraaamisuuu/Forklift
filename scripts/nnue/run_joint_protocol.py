#!/usr/bin/env python3
"""Execute only the frozen October 16 joint-NNUE protocol, without promotion.

The child has a 2,700-second cooperative study budget, including its own audits.
This wrapper allows 30 additional seconds for cleanup, then terminates only the
process tree rooted at its own child. Wrapper preflight time is recorded apart
from the study budget. There are no training-hyperparameter overrides or resume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research" / "experiments" / "2026-10-16-nnue-joint-protocol.json"
FROZEN_PROTOCOL_CANONICAL_SHA256 = "ab9f79dbf9017fc2df4d2c554dc9175c5a9ec5bd4eb68e6823c17ed3cfa8b6d9"
WATCHDOG_GRACE_SECONDS = 30
SHARD_HEADER = struct.Struct("<8sIIQ")
SHARD_RECORD_BYTES = 42
DEPENDENCY_FILES = (
    "run_joint_protocol.py", "joint_finetune.py", "head_cache.py", "head_study.py",
    "initialization.py", "nnue_dataset.py", "train.py", "dataset_paths.py",
    "dataset_diagnostics.py",
)


@dataclass(frozen=True)
class RunInputs:
    train: Path
    validation: Path
    cache: Path
    source_weights: Path
    source_network: Path
    cpp_tools: Path
    output: Path
    protocol: Path = PROTOCOL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_record(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    contents = path.read_bytes()
    protocol = json.loads(contents)
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != FROZEN_PROTOCOL_CANONICAL_SHA256:
        raise ValueError("protocol differs from the frozen October 16 file; no job launched")
    if not protocol.get("frozenBeforeTraining") or protocol.get("schemaVersion") != 1:
        raise ValueError("protocol is not a frozen schema-1 study")
    return protocol


def _compact_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(SHARD_HEADER.size)
    if len(header) != SHARD_HEADER.size:
        raise ValueError(f"truncated compact shard: {path}")
    magic, version, record_bytes, count = SHARD_HEADER.unpack(header)
    if magic != b"TNNDS1\0\0" or version != 1 or record_bytes != SHARD_RECORD_BYTES:
        raise ValueError(f"unsupported compact shard: {path}")
    if path.stat().st_size != SHARD_HEADER.size + count * record_bytes:
        raise ValueError(f"compact shard size/count mismatch: {path}")
    return count


def preflight(inputs: RunInputs, protocol: dict[str, Any]) -> dict[str, Any]:
    """Validate pinned inputs before importing Torch or creating a child job."""
    expected = {
        "train": protocol["trainSha256"],
        "validation": protocol["parentValidationSha256"],
        "cache": protocol["metadataCacheSha256"],
        "source_weights": protocol["sourceWeightsSha256"],
        "source_network": protocol["sourceNetworkSha256"],
    }
    artifacts = {}
    for name, expected_hash in expected.items():
        path = getattr(inputs, name).resolve(strict=True)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"preflight {name} hash differs from frozen protocol")
        artifacts[name] = {"path": str(path), "sha256": actual_hash,
                           "bytes": path.stat().st_size}
    tools = inputs.cpp_tools.resolve(strict=True)
    if not tools.is_file():
        raise ValueError("C++ verification tools must be an existing executable file")
    artifacts["cpp_tools"] = {"path": str(tools), "sha256": sha256(tools),
                              "bytes": tools.stat().st_size}
    train_count = _compact_count(inputs.train)
    parent_count = _compact_count(inputs.validation)
    if train_count != protocol["training"]["fullTrainingPositions"]:
        raise ValueError("training source does not contain the frozen 5M positions")
    cache = json.loads(inputs.cache.read_text(encoding="utf-8"))
    if cache.get("schemaVersion") != 1 or cache.get("status") != "complete":
        raise ValueError("metadata cache is not a completed schema-1 cache")
    if cache.get("sourceWeights", {}).get("sha256") != protocol["sourceWeightsSha256"]:
        raise ValueError("metadata cache source weights disagree with protocol")
    for split, expected_hash, expected_count in (
            ("train", protocol["trainSha256"], train_count),
            ("validation", protocol["parentValidationSha256"], parent_count)):
        descriptor = cache.get("sources", {}).get(split, {})
        if descriptor.get("sha256") != expected_hash or descriptor.get("count") != expected_count:
            raise ValueError(f"cache parent {split} source disagrees with pinned compact shard")
    configuration = cache.get("configuration", {})
    for field, expected_value in (
            ("hidden", protocol["architecture"]["hidden"]),
            ("featureFactorization", protocol["architecture"]["featureFactorization"]),
            ("targetScale", protocol["training"]["targetScale"]),
            ("wdlScale", protocol["training"]["wdlScale"]), ("resultWeight", 0)):
        if configuration.get(field) != expected_value:
            raise ValueError(f"cache has incompatible {field}")
    for split in ("validation", "test"):
        expected_count = protocol["selection"]["validationPositions"]
        descriptor = cache["splits"][split]
        if descriptor.get("count") != expected_count:
            raise ValueError(f"old cache {split} membership is not the frozen 32k positions")
        for field, shape in (("boards", [expected_count, 32]), ("turns", [expected_count]),
                             ("game_ids", [expected_count]), ("source_indices", [expected_count]),
                             ("identity", [expected_count, 32])):
            array = descriptor["files"][field]
            path = (inputs.cache.resolve().parent / array["path"]).resolve(strict=True)
            if inputs.cache.resolve().parent not in path.parents:
                raise ValueError("membership path escapes metadata cache")
            if array.get("shape") != shape or sha256(path) != array["sha256"]:
                raise ValueError(f"old cache {split}/{field} membership differs from manifest")
    return {"artifacts": artifacts, "trainingPositions": train_count,
            "parentValidationPositions": parent_count,
            "selectionPositions": protocol["selection"]["validationPositions"],
            "diagnosticPositions": protocol["diagnostic"]["positions"]}


def child_command(inputs: RunInputs, protocol: dict[str, Any]) -> list[str]:
    training, architecture, export = (protocol[name] for name in ("training", "architecture", "export"))
    command = [sys.executable, "-u", str(ROOT / "scripts" / "nnue" / "joint_finetune.py")]
    paths = {"train": inputs.train, "validation": inputs.validation, "cache": inputs.cache,
             "source-weights": inputs.source_weights, "output": inputs.output / "study",
             "cpp-tools": inputs.cpp_tools}
    for name, path in paths.items():
        command.extend(("--" + name, str(path.resolve())))
    options = {
        "epochs": training["epochsPerArm"], "batch-size": training["batchSize"],
        "transformer-lr": training["transformerLearningRate"], "head-lr": training["headLearningRate"],
        "weight-decay": training["weightDecay"], "device": "cuda", "cpu-threads": training["cpuThreads"],
        "workers": training["loaderWorkers"], "max-seconds": training["wallCapSeconds"],
        "hidden": architecture["hidden"], "diagnostic-count": protocol["diagnostic"]["positions"],
        "diagnostic-seed": protocol["diagnostic"]["seed"], "hidden-scale": export["hiddenScale"],
        "output-scale": export["outputScale"], "residual-weight-scale": export["residualWeightScale"],
        "target-scale": training["targetScale"], "wdl-scale": training["wdlScale"],
        "bootstrap-resamples": 1000, "expected-source-sha256": protocol["sourceWeightsSha256"],
    }
    for name, value in options.items():
        command.extend(("--" + name, str(value)))
    command.extend(("--seeds", *map(str, training["seeds"])))
    command.extend(("--residual-widths", *map(str, architecture["residualWidths"])))
    command.append("--feature-factorization" if architecture["featureFactorization"] else "--no-feature-factorization")
    return command


def code_hashes() -> dict[str, str]:
    paths = [ROOT / "scripts" / "nnue" / name for name in DEPENDENCY_FILES]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}


def _git(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=ROOT, text=True, encoding="utf-8",
                            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False, timeout=10)
    if result.returncode:
        raise RuntimeError(f"unable to record Git provenance: {result.stderr.strip()}")
    return result.stdout.strip()


def capture_environment(protocol: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise ValueError("frozen protocol requires CUDA; CPU fallback is prohibited")
    gpu = torch.cuda.get_device_properties(0)
    if re.search(r"(?:^| )RTX 3070$", gpu.name) is None:
        raise ValueError(f"frozen protocol requires RTX 3070, not {gpu.name}")
    return {"python": sys.version, "pythonExecutable": str(Path(sys.executable).resolve()),
            "pythonExecutableSha256": sha256(Path(sys.executable)), "platform": platform.platform(),
            "numpyVersion": str(np.__version__), "torchVersion": str(torch.__version__),
            "cudaVersion": torch.version.cuda, "cudnnVersion": torch.backends.cudnn.version(),
            "cudaDeviceIndex": 0, "gpuName": gpu.name, "gpuCapability": [gpu.major, gpu.minor],
            "gpuMemoryBytes": gpu.total_memory, "gitCommit": _git("rev-parse", "HEAD"),
            "gitStatusPorcelain": _git("status", "--porcelain=v1"),
            "sourceDependencySha256": code_hashes(),
            "inheritedCudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpuThreads": protocol["training"]["cpuThreads"]}


def stop_timed_out_child(child: subprocess.Popen) -> None:
    """Called only after a wait timeout; never search for or kill unrelated jobs."""
    if child.poll() is not None:
        return  # It exited during the timeout race; do not target a recycled PID.
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                check=False, timeout=15)
        if result.returncode and child.poll() is None:
            raise RuntimeError(f"unable to stop timed-out child PID {child.pid}: {result.stderr}")
    else:
        # POSIX child starts in its own session, so this group contains only its descendants.
        import signal
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=15)


def _validate_completed_study(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("state") != "complete":
        raise ValueError("child did not publish a complete comparable study")
    training, architecture = protocol["training"], protocol["architecture"]
    expected = {(seed, width) for seed in training["seeds"] for width in architecture["residualWidths"]}
    arms = report.get("arms", [])
    if len(arms) != len(expected) or {(arm["seed"], arm["residualHidden"]) for arm in arms} != expected:
        raise ValueError("completed study has unexpected or missing arms")
    updates = math.ceil(training["fullTrainingPositions"] / training["batchSize"]) * training["epochsPerArm"]
    positions = training["fullTrainingPositions"] * training["epochsPerArm"]
    for arm in arms:
        if arm.get("updates") != updates or arm.get("trainingPositions") != positions:
            raise ValueError("completed arm does not have the frozen equal update budget")
        if arm.get("diagnostic", {}).get("samples") != protocol["diagnostic"]["positions"]:
            raise ValueError("completed arm diagnostic count differs from frozen protocol")
    return {"path": str(path.resolve()), "sha256": sha256(path), "arms": len(arms),
            "updatesPerArm": updates, "trainingPositionsPerArm": positions,
            "strengthEvidence": False, "promotionAuthorized": False}


def run_protocol(inputs: RunInputs) -> dict[str, Any]:
    started = time.monotonic()
    output = inputs.output.resolve()
    if output.exists():
        raise ValueError("execution output must not already exist; no overwrites or resume")
    for name in ("train", "validation", "cache", "source_weights", "source_network", "cpp_tools", "protocol"):
        source = getattr(inputs, name).resolve()
        if source == output or output in source.parents:
            raise ValueError("execution output must not contain an input")
    output.mkdir(parents=True, exist_ok=False)
    execution_path = output / "execution.json"
    record: dict[str, Any] = {"schemaVersion": 1, "state": "preflight",
                              "startedAt": datetime.now(timezone.utc).isoformat(),
                              "protocolPath": str(inputs.protocol.resolve()),
                              "wrapperCommand": [sys.executable, *sys.argv],
                              "childStarted": False, "automaticResume": False,
                              "promotionAuthorized": False}
    _write_record(execution_path, record)
    child = None
    child_started = None
    try:
        protocol = load_protocol(inputs.protocol)
        record["protocolSha256"] = sha256(inputs.protocol)
        record["protocolCanonicalSha256"] = FROZEN_PROTOCOL_CANONICAL_SHA256
        record["preflight"] = preflight(inputs, protocol)
        threads = str(protocol["training"]["cpuThreads"])
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                           OPENBLAS_NUM_THREADS=threads, CUBLAS_WORKSPACE_CONFIG=":4096:8")
        # Set before the environment probe imports NumPy/Torch as well as for the child.
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"):
            os.environ[key] = environment[key]
        record["environment"] = capture_environment(protocol)
        command = child_command(inputs, protocol)
        record.update(command=command, commandWindowsDisplay=subprocess.list2cmdline(command),
                      childEnvironmentOverrides={key: environment[key] for key in
                          ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")},
                      preflightElapsedSeconds=time.monotonic() - started, state="launching",
                      budget={"studySeconds": protocol["training"]["wallCapSeconds"],
                              "watchdogGraceSeconds": WATCHDOG_GRACE_SECONDS,
                              "externalChildTimeoutSeconds": protocol["training"]["wallCapSeconds"] + WATCHDOG_GRACE_SECONDS,
                              "wrapperPreflightIncludedInStudyBudget": False,
                              "policy": "45-minute cooperative study; 30-second external cleanup safeguard; no extension or resume"})
        # This record and the complete command exist BEFORE Popen can start training.
        _write_record(execution_path, record)
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            child_started = time.monotonic()
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                     start_new_session=os.name != "nt")
            record.update(state="running", childStarted=True, childPid=child.pid,
                          childStartedAt=datetime.now(timezone.utc).isoformat())
            _write_record(execution_path, record)
            try:
                code = child.wait(timeout=record["budget"]["externalChildTimeoutSeconds"])
            except subprocess.TimeoutExpired:
                record["watchdogTriggered"] = True
                stop_timed_out_child(child)
                raise TimeoutError("study exceeded 2700 seconds plus the 30-second cleanup safeguard")
            except KeyboardInterrupt:
                # A terminal interrupt normally also reaches the inherited child.
                # Give its own finally blocks time to stop DataLoader workers.
                try:
                    child.wait(timeout=WATCHDOG_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    stop_timed_out_child(child)
                raise
            record.update(childExitCode=code, childElapsedSeconds=time.monotonic() - child_started)
        if code:
            raise RuntimeError(f"joint study exited {code}; see stderr.log and study/progress.json")
        if code_hashes() != record["environment"]["sourceDependencySha256"]:
            raise ValueError("NNUE source/dependency files changed during the child study")
        if sha256(inputs.protocol) != record["protocolSha256"]:
            raise ValueError("frozen protocol changed during the child study")
        if sha256(inputs.source_network) != record["preflight"]["artifacts"]["source_network"]["sha256"]:
            raise ValueError("protected source network changed during the child study")
        record["completedStudy"] = _validate_completed_study(output / "study" / "study.json", protocol)
        record["state"] = "complete"
        return record
    except BaseException as error:
        record.update(state="incomplete", error=str(error) or type(error).__name__,
                      comparable=False, automaticExtension=False)
        if child is not None:
            record["childExitCode"] = child.poll()
        raise
    finally:
        if child_started is not None:
            record["childElapsedSeconds"] = time.monotonic() - child_started
        record.update(finishedAt=datetime.now(timezone.utc).isoformat(),
                      totalElapsedSeconds=time.monotonic() - started)
        _write_record(execution_path, record)


def parse_args() -> RunInputs:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "cache", "source_weights", "source_network", "cpp_tools", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    return RunInputs(**vars(parser.parse_args()))


if __name__ == "__main__":
    try:
        arguments = parse_args()
        result = run_protocol(arguments)
        print(json.dumps({"state": result["state"], "execution": str(arguments.output / "execution.json")}))
    except (Exception, KeyboardInterrupt) as error:
        print(f"protocol execution failed: {error}", file=sys.stderr)
        raise SystemExit(1)
