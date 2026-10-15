#!/usr/bin/env python3
"""Bounded, paired build-profile benchmark with strict identical-tree checks.

No builds, training, network downloads or engine-setting changes are performed.
Use an otherwise idle host. A faster fixed-depth result is not an Elo result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time


POSITION = re.compile(
    r"^(.+?)\s+best=(\S+) depth=(\d+) score=(-?\d+) nodes=(\d+) qnodes=(\d+)"
    r".*? time=(\d+)ms", re.MULTILINE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(output: str, depth: int, ceiling: int) -> dict:
    header = dict(re.findall(r"(\w+)=(\S+)", output.splitlines()[0])) if output else {}
    required = dict(depth=str(depth), timeLimit=f"{ceiling}ms", threads="1",
                    evaluator="nnue", nnueWeight="100", nnueMode="incremental",
                    probCut="off", singular="off", singularBudget="0", positions="4")
    if any(header.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"unexpected benchmark settings: {header}")
    positions = []
    for name, move, reached, score, nodes, qnodes, elapsed in POSITION.findall(output):
        positions.append(dict(name=name.strip(), best=move, depth=int(reached), score=int(score),
                              nodes=int(nodes), qnodes=int(qnodes), milliseconds=int(elapsed)))
    if len(positions) != 4 or any(row["depth"] != depth for row in positions):
        raise RuntimeError("all four positions must complete the requested fixed depth")
    elapsed = sum(row["milliseconds"] for row in positions)
    if elapsed <= 0:
        raise RuntimeError("benchmark timing resolution is insufficient")
    return dict(header=header, positions=positions, totalMilliseconds=elapsed, rawOutput=output)


def tree_signature(run: dict) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "milliseconds"}
            for row in run["positions"]]


def build_metadata(path: Path) -> dict:
    build = path.parent.parent
    cache = build / "CMakeCache.txt"
    settings = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^(CHESS_NATIVE_ARCH|CHESS_ENABLE_IPO|CHESS_BUILD_GUI|BUILD_TESTING|"
                             r"CMAKE_CXX_FLAGS_RELEASE|CMAKE_GENERATOR|CMAKE_HOME_DIRECTORY):[^=]*=(.*)$", line)
            if match:
                settings[match[1]] = match[2]
    uci = path.with_name("chess-engine-uci" + path.suffix)
    return dict(path=str(path), sha256=sha256(path), cmakeSettings=settings,
                uci={"path": str(uci), "sha256": sha256(uci)} if uci.is_file() else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tools", required=True, type=Path)
    parser.add_argument("--native-tools", required=True, type=Path)
    parser.add_argument("--ipo-tools", required=True, type=Path)
    parser.add_argument("--native-ipo-tools", required=True, type=Path)
    parser.add_argument("--nnue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True, help="declared common binary source revision")
    parser.add_argument("--cpu-label", default=platform.processor())
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--ceiling-ms", type=int, default=30000)
    parser.add_argument("--trials", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--max-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if args.depth < 1 or args.ceiling_ms < 1 or not 0 < args.max_seconds <= 1800:
        parser.error("depth/ceiling must be positive and wall budget in (0,1800] seconds")
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; preserve prior results")
    paths = {name: path.resolve(strict=True) for name, path in (
        ("portable", args.baseline_tools), ("native", args.native_tools),
        ("ipo", args.ipo_tools), ("native-ipo", args.native_ipo_tools))}
    network = args.nnue.resolve(strict=True)
    if any(not path.is_file() for path in (*paths.values(), network)):
        parser.error("all inputs must be files")
    if output in (*paths.values(), network):
        parser.error("output must not overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new evidence file before starting any work; never overwrite.
    with output.open("x", encoding="utf-8") as destination:
        destination.write("{}\n")
    started = time.monotonic()
    report = dict(schemaVersion=1, state="running", startedAt=datetime.now(timezone.utc).isoformat(),
                  binarySourceCommitDeclared=args.source_commit,
                  host=dict(platform=platform.platform(), processor=args.cpu_label),
                  protocol=dict(depth=args.depth, timeCeilingPerPositionMs=args.ceiling_ms,
                                trialsPerCandidate=args.trials, threads=1, hashMiB=128,
                                nnueWeight=100, incremental=True, probCut=False, singular=False,
                                warmupsPerBinary=1, maximumWallSeconds=args.max_seconds,
                                ordering="reverse candidate order on odd trials; alternate within-pair order",
                                hostCondition="root reports no concurrent project workloads; otherwise idle host",
                                interpretation="exploratory fixed-depth elapsed cost, not Elo or portable-release validation"),
                  network=dict(path=str(network), sha256=sha256(network)),
                  binaries={name: build_metadata(path) for name, path in paths.items()},
                  warmups=[], trials=[])
    expected = None

    def run(name: str, trial: int, comparison: str | None = None) -> None:
        nonlocal expected
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("build-profile benchmark exceeded its fixed wall budget")
        command = [str(paths[name]), "--bench", "--bench-depth", str(args.depth),
                   "--bench-time", str(args.ceiling_ms), "--bench-tt", "128", "--threads", "1",
                   "--nnue", str(network), "--nnue-weight", "100", "--no-probcut", "--no-singular"]
        process_started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, timeout=remaining, check=False)
        record = dict(binary=name, trial=trial, comparison=comparison, command=command,
                      processExitCode=result.returncode, stderr=result.stderr,
                      processWallSeconds=time.monotonic() - process_started)
        if result.returncode:
            record["rawOutput"] = result.stdout
            report["trials" if trial >= 0 else "warmups"].append(record)
            raise RuntimeError(f"{name} exited {result.returncode}: {result.stderr}")
        record.update(parse_run(result.stdout, args.depth, args.ceiling_ms))
        report["trials" if trial >= 0 else "warmups"].append(record)
        current = tree_signature(record)
        if expected is None:
            expected = current
        elif current != expected:
            raise RuntimeError(f"{name} produced a different tree/move/score/depth; speed comparison invalid")
        print(f"trial={trial} comparison={comparison} binary={name} time={record['totalMilliseconds']}ms", flush=True)

    try:
        for name in paths:
            run(name, -1)
        candidates = ("native", "ipo", "native-ipo")
        for trial in range(args.trials):
            order = candidates if trial % 2 == 0 else candidates[::-1]
            for index, name in enumerate(order):
                pair = ("portable", name) if (trial + index) % 2 == 0 else (name, "portable")
                for binary in pair:
                    run(binary, trial, name)
        for name, path in paths.items():
            if sha256(path) != report["binaries"][name]["sha256"]:
                raise RuntimeError(f"binary changed during benchmark: {name}")
        if sha256(network) != report["network"]["sha256"]:
            raise RuntimeError("network changed during benchmark")
        report["allTreesIdentical"] = True
        report["treeSignature"] = expected
        report["medianMilliseconds"] = {
            name: statistics.median(row["totalMilliseconds"] for row in report["trials"]
                                    if row["binary"] == name) for name in paths}
        report["medianPairedCandidateOverPortable"] = {
            name: statistics.median(
                next(row["totalMilliseconds"] for row in report["trials"]
                     if row["comparison"] == name and row["trial"] == trial and row["binary"] == name) /
                next(row["totalMilliseconds"] for row in report["trials"]
                     if row["comparison"] == name and row["trial"] == trial and row["binary"] == "portable")
                for trial in range(args.trials)) for name in candidates}
        report["state"] = "complete"
    except BaseException as error:
        report.update(state="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        report["wallSeconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "state", "medianMilliseconds", "medianPairedCandidateOverPortable", "wallSeconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
