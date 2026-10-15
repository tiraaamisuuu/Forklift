#!/usr/bin/env python3
"""Bounded, interleaved old/new residual inference cost diagnostic, not Elo.

Zero-output heads preserve the search tree, but both binaries must execute the
actual dense arithmetic (no zero-output skipping). Models are temporary; hashes,
raw output, exact tree checks and timings are retained in the JSON report.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

import torch

from initialization import load_initial_weights
from nnue_dataset import sha256_file
from train import HalfKpV1, export_network

torch.set_num_threads(1)

POSITION = re.compile(
    r"^(.+?)\s+best=(\S+) depth=(\d+) score=(-?\d+) nodes=(\d+) qnodes=(\d+)"
    r".*? time=(\d+)ms", re.MULTILINE)


def parse_run(output: str, depth: int) -> dict:
    positions = []
    for match in POSITION.finditer(output):
        name, move, reached, score, nodes, qnodes, elapsed = match.groups()
        positions.append(dict(name=name.strip(), best=move, depth=int(reached),
                              score=int(score), nodes=int(nodes), qnodes=int(qnodes),
                              milliseconds=int(elapsed)))
    if len(positions) != 4 or any(row["depth"] != depth for row in positions):
        raise RuntimeError("benchmark did not complete all four positions at fixed depth")
    kernel = re.search(r"nnueOutput=(\S+)", output)
    if kernel is None:
        raise RuntimeError("benchmark omitted kernel diagnostic")
    header = dict(re.findall(r"(\w+)=(\S+)", output.splitlines()[0]))
    required = dict(depth=str(depth), timeLimit="5000ms", threads="1", evaluator="nnue",
                    nnueWeight="100", nnueMode="incremental", probCut="off",
                    singular="off", singularBudget="0", positions="4")
    if any(header.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"unexpected benchmark configuration: {header}")
    elapsed = sum(row["milliseconds"] for row in positions)
    if elapsed <= 0:
        raise RuntimeError("timing resolution too coarse; increase fixed depth")
    return dict(kernel=kernel[1], header=header, positions=positions,
                totalMilliseconds=elapsed,
                rawOutput=output)


def signature(run: dict) -> list:
    return [{key: value for key, value in row.items() if key != "milliseconds"}
            for row in run["positions"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tools", required=True, type=Path)
    parser.add_argument("--candidate-tools", required=True, type=Path)
    parser.add_argument("--initialize-from", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-residual-kernel", default="residual64",
                        choices=("residual64", "residual-sse2"),
                        help="Expected baseline kernel; SSE2 permits isolated scratch-layout comparisons")
    parser.add_argument("--depth", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--trials", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("report already exists; preserve it and choose a fresh output")
    for path in (args.baseline_tools, args.candidate_tools, args.initialize_from):
        if not path.is_file() or path.resolve() == args.output.resolve():
            parser.error(f"invalid/protected input: {path}")
    started = time.monotonic()
    deadline = started + 180
    report = dict(schemaVersion=1, evidence="shared-host cost only, not strength",
                  sourceWeightsSha256=sha256_file(args.initialize_from),
                  binaries={name: dict(path=str(path.resolve()), sha256=sha256_file(path))
                            for name, path in (("baseline", args.baseline_tools),
                                               ("candidate", args.candidate_tools))},
                  protocol=dict(depth=args.depth, threads=1, hashMiB=128,
                                timeCeilingPerPositionMs=5000, trials=args.trials,
                                hidden=256, hiddenScale=1024, outputScale=32,
                                targetScale=600, residualWeightScale=256,
                                zeroOutputHeads=True, seed=20261015,
                                expectedBaselineResidualKernel=args.baseline_residual_kernel),
                  models={}, warmups=[], trials=[])
    expected = None
    with tempfile.TemporaryDirectory(prefix="forklift-residual-cost-") as directory:
        models = {}
        for width in (0, 4, 8):
            torch.manual_seed(20261015)
            model = HalfKpV1(256, feature_factorization=True, residual_hidden=width)
            initialization = load_initial_weights(model, args.initialize_from)
            path = Path(directory) / f"r{width}.nnue"
            export_network(model, path, hidden_scale=1024, output_scale=32,
                           target_scale=600, residual_weight_scale=256)
            models[width] = path
            report["models"][str(width)] = dict(sha256=sha256_file(path),
                                                 initialization=initialization)
        variants = [(binary, width) for width in (0, 4, 8)
                    for binary in ("baseline", "candidate")]
        for trial in range(-1, args.trials):
            # Alternate both whole order and old/new order across measured trials.
            order = variants if trial % 2 == 0 else list(reversed(variants))
            for binary, width in order:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("180-second diagnostic ceiling reached")
                executable = args.baseline_tools if binary == "baseline" else args.candidate_tools
                result = subprocess.run(
                    [str(executable.resolve()), "--bench", "--bench-depth", str(args.depth),
                     "--bench-time", "5000", "--bench-tt", "128", "--threads", "1",
                     "--nnue", str(models[width]), "--nnue-weight", "100",
                     "--no-probcut", "--no-singular"], capture_output=True, text=True,
                    check=True, timeout=min(25, remaining))
                run = dict(binary=binary, residual=width, trial=trial,
                           **parse_run(result.stdout, args.depth))
                expected_kernel = ("sse2" if width == 0 else
                                   "sse2+" + args.baseline_residual_kernel if binary == "baseline" else
                                   "sse2+residual-sse2")
                if run["kernel"] != expected_kernel:
                    raise RuntimeError(f"unexpected kernel: {run['kernel']}, expected {expected_kernel}")
                current = signature(run)
                if expected is None:
                    expected = current
                if current != expected:
                    raise RuntimeError("tree, move, score or depth mismatch; cost comparison invalid")
                report["warmups" if trial < 0 else "trials"].append(run)
                print(f"trial={trial} {binary} R{width}: {run['totalMilliseconds']}ms "
                      f"kernel={run['kernel']}", flush=True)
    report["allTreesIdentical"] = True
    report["treeSignature"] = expected
    report["medianMilliseconds"] = {
        f"{binary}-r{width}": statistics.median(
            run["totalMilliseconds"] for run in report["trials"]
            if run["binary"] == binary and run["residual"] == width)
        for binary, width in variants}
    report["medianPairedCandidateOverBaseline"] = {
        str(width): statistics.median(
            next(run["totalMilliseconds"] for run in report["trials"]
                 if run["binary"] == "candidate" and run["residual"] == width and run["trial"] == trial) /
            next(run["totalMilliseconds"] for run in report["trials"]
                 if run["binary"] == "baseline" and run["residual"] == width and run["trial"] == trial)
            for trial in range(args.trials)) for width in (0, 4, 8)}
    report["wallSeconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(json.dumps({key: report[key] for key in (
        "medianMilliseconds", "medianPairedCandidateOverBaseline", "wallSeconds")}, indent=2))


if __name__ == "__main__":
    main()
