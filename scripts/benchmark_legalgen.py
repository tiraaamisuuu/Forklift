#!/usr/bin/env python3
"""Bounded same-tree legality benchmark; no builds, training or Elo inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

from benchmark_build_profiles import build_metadata, sha256


ROW = re.compile(r"^(.+?)\s+best=(\S+) depth=(\d+) score=(-?\d+) nodes=(\d+) qnodes=(\d+)(.*)$",
                 re.MULTILINE)


def parse_run(output: str, depth: int, ceiling: int, evaluator: str, count: int,
              diagnostics: bool = False) -> dict:
    header = dict(re.findall(r"(\w+)=(\S+)", output.splitlines()[0])) if output else {}
    required = dict(depth=str(depth), timeLimit=f"{ceiling}ms", threads="1", evaluator=evaluator,
                    nnueWeight="100" if evaluator == "nnue" else "0",
                    nnueMode="incremental" if evaluator == "nnue" else "n/a",
                    probCut="off", singular="off", singularBudget="0", positions=str(count))
    if any(header.get(key) != value for key, value in required.items()):
        raise ValueError(f"unexpected benchmark settings: {header}")
    positions = []
    for name, move, reached, score, nodes, qnodes, tail in ROW.findall(output):
        fields = dict(re.findall(r"(\w+)=(\S+)", tail))
        if "time" not in fields or not fields["time"].endswith("ms"):
            raise ValueError("missing elapsed search time")
        elapsed = int(fields.pop("time")[:-2])
        fields.pop("nps", None)
        if set(fields) != {"probcut", "singular", "singularPathMax", "singularProofNodes", "singularSkipped"}:
            raise ValueError("missing or unexpected search counters")
        positions.append(dict(name=name.strip(), best=move, depth=int(reached), score=int(score),
                              nodes=int(nodes), qnodes=int(qnodes), searchCounters=fields,
                              milliseconds=elapsed))
    if len(positions) != count or any(row["depth"] != depth for row in positions):
        raise ValueError("every position must complete the requested fixed depth")
    total = sum(row["milliseconds"] for row in positions)
    if total <= 0:
        raise ValueError("search timing resolution is insufficient")
    result = dict(header=header, positions=positions, totalMilliseconds=total)
    matches = re.findall(r"^Legalgen diagnostics: (.+)$", output, re.MULTILINE)
    if diagnostics:
        if len(matches) != 1:
            raise ValueError("missing or duplicate diagnostic counters")
        counters = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", matches[0])}
        if set(counters) != {"calls", "pseudo", "probes", "fast", "checked", "legalgenNs", "harnessNs"}:
            raise ValueError("unexpected legality diagnostic fields")
        if counters["pseudo"] != counters["probes"] + counters["fast"]:
            raise ValueError("legality diagnostic accounting does not balance")
        result["diagnostics"] = counters
    elif matches:
        raise ValueError("instrumented binary cannot be used as a speed result")
    return result


def signature(run: dict) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "milliseconds"}
            for row in run["positions"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference_tools", "candidate_tools", "reference_diagnostics", "candidate_diagnostics",
                 "nnue", "book", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--breadth-depth", type=int, default=10)
    parser.add_argument("--breadth-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--trials", type=int, default=7, choices=(7,))
    parser.add_argument("--ceiling-ms", type=int, default=30000)
    parser.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args()
    if not (1 <= args.trials <= 15 and 1 <= args.breadth_count <= 64
            and min(args.depth, args.breadth_depth, args.ceiling_ms) > 0
            and 0 < args.max_seconds <= 1800):
        parser.error("invalid fixed sample, depth or wall budget")
    output = args.output.resolve()
    if output.exists():
        parser.error("output exists; preserve prior evidence")
    paths = {name: getattr(args, name + "_tools").resolve(strict=True)
             for name in ("reference", "candidate")}
    paths.update({name + "-diagnostics": getattr(args, name + "_diagnostics").resolve(strict=True)
                  for name in ("reference", "candidate")})
    network, book = args.nnue.resolve(strict=True), args.book.resolve(strict=True)
    if any(not path.is_file() for path in (*paths.values(), network, book)):
        parser.error("inputs must be files")
    if output in (*paths.values(), network, book):
        parser.error("output must not replace an input")
    all_fens = [line.strip() for line in book.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(fen.split()) != 6 for fen in all_fens):
        parser.error("breadth book must contain complete six-field FENs, one per line")
    indices = random.Random(args.seed).sample(range(len(all_fens)), args.breadth_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write("{}\n")
    started = time.monotonic()
    report = dict(schemaVersion=1, state="running", startedAt=datetime.now(timezone.utc).isoformat(),
                  binarySourceCommitDeclared=args.source_commit,
                  binaries={name: build_metadata(path) for name, path in paths.items()},
                  network=dict(path=str(network), sha256=sha256(network)),
                  book=dict(path=str(book), sha256=sha256(book), totalPositions=len(all_fens)),
                  protocol=dict(trialsPerEvaluator=args.trials, defaultSuiteDepth=args.depth,
                                breadthDepth=args.breadth_depth, breadthCount=args.breadth_count,
                                breadthSeed=args.seed, breadthIndices=indices,
                                breadthFens=[all_fens[index] for index in indices],
                                threads=1, hashMiB=128, nnueWeight=100, incrementalNnue=True,
                                probCut=False, singular=False, timeCeilingPerPositionMs=args.ceiling_ms,
                                warmupsPerBinaryAndEvaluator=1, maximumWallSeconds=args.max_seconds,
                                order="alternate evaluators and within-pair binary order; fixed seven paired trials",
                                interpretation="same-tree elapsed cost, not Elo; diagnostic timers are attribution only",
                                hostCondition="quiet host, no concurrent project build/training/test workloads"),
                  runs=[])
    expected: dict[tuple, list[dict]] = {}

    def persist() -> None:
        report["wallSeconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def run(binary: str, evaluator: str, phase: str, trial: int, fen: str | None = None) -> None:
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("fixed legality benchmark wall cap exhausted")
        depth = args.breadth_depth if fen else args.depth
        command = [str(paths[binary]), "--bench", "--bench-depth", str(depth),
                   "--bench-time", str(args.ceiling_ms), "--bench-tt", "128", "--threads", "1",
                   "--no-probcut", "--no-singular"]
        if evaluator == "nnue":
            command += ["--nnue", str(network), "--nnue-weight", "100"]
        if fen:
            command += ["--fen", fen]
        record = dict(binary=binary, evaluator=evaluator, phase=phase, trial=trial, fen=fen, command=command)
        report["runs"].append(record)
        process_started = time.monotonic()
        process = subprocess.run(command, capture_output=True, text=True, timeout=remaining, check=False)
        record.update(processExitCode=process.returncode, rawOutput=process.stdout, stderr=process.stderr,
                      processWallSeconds=time.monotonic() - process_started)
        if process.returncode:
            raise RuntimeError(f"{binary} exited {process.returncode}: {process.stderr}")
        record.update(parse_run(process.stdout, depth, args.ceiling_ms, evaluator, 1 if fen else 4,
                                binary.endswith("-diagnostics")))
        key = (evaluator, fen)
        current = signature(record)
        if key in expected and expected[key] != current:
            raise RuntimeError(f"search move/score/depth/tree/counters changed for {key}; speed comparison invalid")
        expected.setdefault(key, current)
        print(f"{phase} trial={trial} {evaluator} {binary}: {record['totalMilliseconds']}ms", flush=True)
        persist()

    try:
        for evaluator in ("nnue", "classical"):
            for binary in ("reference", "candidate"):
                run(binary, evaluator, "warmup", -1)
        for trial in range(args.trials):
            evaluators = ("nnue", "classical") if trial % 2 == 0 else ("classical", "nnue")
            for evaluator in evaluators:
                fixed_index = int(evaluator == "classical")
                pair = (("reference", "candidate") if (trial + fixed_index) % 2 == 0
                        else ("candidate", "reference"))
                for binary in pair:
                    run(binary, evaluator, "timing", trial)
        for index, book_index in enumerate(indices):
            for evaluator in ("nnue", "classical"):
                pair = ("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")
                for binary in pair:
                    run(binary, evaluator, "breadth", index, all_fens[book_index])
        for evaluator in ("nnue", "classical"):
            for binary in ("reference-diagnostics", "candidate-diagnostics"):
                run(binary, evaluator, "diagnostics", 0)
        for name, path in paths.items():
            if sha256(path) != report["binaries"][name]["sha256"]:
                raise RuntimeError(f"binary changed during benchmark: {name}")
        if sha256(network) != report["network"]["sha256"] or sha256(book) != report["book"]["sha256"]:
            raise RuntimeError("network or book changed during benchmark")
        report["sameTreeComparisonsPassed"] = True
        report["results"] = {}
        for evaluator in ("nnue", "classical"):
            times = {(row["trial"], row["binary"]): row["totalMilliseconds"] for row in report["runs"]
                     if row["phase"] == "timing" and row["evaluator"] == evaluator}
            ratios = [times[(trial, "candidate")] / times[(trial, "reference")]
                      for trial in range(args.trials)]
            ratio = statistics.median(ratios)
            faster = sum(value < 1 for value in ratios)
            report["results"][evaluator] = dict(pairedCandidateOverReference=ratios, medianTimeRatio=ratio,
                                                  fasterPairs=faster,
                                                  medianTimeReductionPercent=100 * (1 - ratio),
                                                  impliedSameTreeNpsGainPercent=100 * (1 / ratio - 1))
        primary = report["results"]["nnue"]
        report["preregisteredSpeedGatePassed"] = (primary["medianTimeReductionPercent"] >= 2
                                                  and primary["fasterPairs"] >= 6)
        report["state"] = "complete"
    except BaseException as error:
        report.update(state="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        persist()
    print(json.dumps(report["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
