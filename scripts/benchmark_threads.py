#!/usr/bin/env python3
"""Measure repeatable search throughput across thread counts.

This is a hardware-scaling diagnostic, not a chess-strength or Elo test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SUMMARY_PATTERN = re.compile(
    r"^Benchmark summary: nodes=(?P<nodes>\d+) time=(?P<time_ms>\d+)ms nps=(?P<nps>\d+)$",
    re.MULTILINE,
)
POSITION_PATTERN = re.compile(
    r"^(?P<name>.+?)\s+best=(?P<best>\S+) depth=(?P<depth>\d+) score=(?P<score>-?\d+) "
    r"nodes=(?P<nodes>\d+) qnodes=(?P<qnodes>\d+)"
    r"(?: probcut=\d+/\d+)?(?: singular=\d+/\d+ singularPathMax=\d+)?"
    r"(?: singularProofNodes=\d+ singularSkipped=\d+)?"
    r" time=(?P<time_ms>\d+)ms nps=(?P<nps>\d+)$",
    re.MULTILINE,
)


def parse_thread_counts(value: str) -> list[int]:
    counts: list[int] = []
    for item in value.split(","):
        try:
            count = int(item.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"Invalid thread count: {item}") from error
        if count < 1 or count > 64:
            raise argparse.ArgumentTypeError("Thread counts must be between 1 and 64")
        if count not in counts:
            counts.append(count)
    if not counts:
        raise argparse.ArgumentTypeError("At least one thread count is required")
    return counts


def parse_benchmark_output(output: str) -> dict[str, object]:
    summary_matches = list(SUMMARY_PATTERN.finditer(output))
    if not summary_matches:
        raise RuntimeError("Engine output did not contain a benchmark summary")
    summary = {key: int(value) for key, value in summary_matches[-1].groupdict().items()}
    positions = []
    for match in POSITION_PATTERN.finditer(output):
        values = match.groupdict()
        positions.append({
            "name": values["name"].strip(),
            "best": values["best"],
            **{key: int(values[key]) for key in ("depth", "score", "nodes", "qnodes", "time_ms", "nps")},
        })
    summary["positions"] = positions
    return summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarise(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["threads"])].append(record)
    baseline_threads = 1 if 1 in grouped else min(grouped)
    baseline_nps = statistics.median(int(item["nps"]) for item in grouped[baseline_threads])
    results = []
    for threads in sorted(grouped):
        samples = grouped[threads]
        nps_values = [int(item["nps"]) for item in samples]
        time_values = [int(item["time_ms"]) for item in samples]
        depths = [
            int(position["depth"])
            for item in samples
            for position in item.get("positions", [])
            if isinstance(position, dict) and "depth" in position
        ]
        median_nps = statistics.median(nps_values)
        speedup = median_nps / baseline_nps if baseline_nps else 0.0
        results.append({
            "threads": threads,
            "runs": len(samples),
            "medianNps": round(median_nps),
            "minNps": min(nps_values),
            "maxNps": max(nps_values),
            "medianTimeMs": round(statistics.median(time_values)),
            "medianDepth": statistics.median(depths) if depths else None,
            "speedup": round(speedup, 3),
            "efficiency": round(speedup / threads, 3),
        })
    return {
        "baselineThreads": baseline_threads,
        "warning": (
            "Node throughput and depth are scaling diagnostics, not Elo. "
            "Use equal-time paired matches before changing the default thread count."
        ),
        "results": results,
    }


def run_benchmark(
    engine: Path,
    threads: int,
    depth: int,
    time_ms: int,
    hash_mb: int,
    timeout_seconds: int,
) -> tuple[dict[str, object], str]:
    command = [
        str(engine),
        "--bench",
        "--bench-depth", str(depth),
        "--bench-time", str(time_ms),
        "--bench-tt", str(hash_mb),
        "--threads", str(threads),
    ]
    completed = subprocess.run(
        command,
        cwd=engine.parent,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    output = completed.stdout
    if completed.stderr:
        output += ("\n" if output else "") + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for {threads} thread(s) with exit code {completed.returncode}\n{output}"
        )
    return parse_benchmark_output(output), output


def write_outputs(
    output_dir: Path,
    configuration: dict[str, object],
    records: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    manifest = {"configuration": configuration, "runs": records}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "threads", "runs", "medianNps", "minNps", "maxNps",
                "medianTimeMs", "medianDepth", "speedup", "efficiency",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["results"])

    lines = [
        "# Thread scaling report",
        "",
        f"- Engine: `{configuration['engine']['path']}`",
        f"- SHA-256: `{configuration['engine']['sha256']}`",
        f"- Benchmark: depth {configuration['depth']}, {configuration['timeMs']} ms per position, "
        f"{configuration['hashMB']} MB hash",
        f"- Repetitions: {configuration['repetitions']}",
        "",
        "| Threads | Median NPS | Median depth | Speedup | Efficiency |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        depth = result["medianDepth"] if result["medianDepth"] is not None else "—"
        lines.append(
            f"| {result['threads']} | {result['medianNps']:,} | {depth} | "
            f"{result['speedup']:.3f}x | {result['efficiency']:.1%} |"
        )
    lines.extend(["", f"> {summary['warning']}", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="Path to chess-engine-tools")
    parser.add_argument("--threads", type=parse_thread_counts, default=parse_thread_counts("1,2,4,6,12"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--time-ms", type=int, default=1000)
    parser.add_argument("--hash", type=int, default=256, dest="hash_mb")
    parser.add_argument("--timeout", type=int, default=120, dest="timeout_seconds")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    engine = args.engine.expanduser().resolve()
    if not engine.is_file():
        raise RuntimeError(f"Engine tools executable not found: {engine}")
    if args.repetitions < 1 or args.repetitions > 100:
        raise RuntimeError("--repetitions must be between 1 and 100")
    if args.depth < 1 or args.time_ms < 1 or args.hash_mb < 1 or args.timeout_seconds < 1:
        raise RuntimeError("Depth, time, hash, and timeout must be positive")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or Path("artifacts") / "perf" / f"thread-scaling-{timestamp}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration: dict[str, object] = {
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "engine": {"path": str(engine), "sha256": sha256(engine)},
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logicalCpuCount": os.cpu_count(),
        "threads": args.threads,
        "repetitions": args.repetitions,
        "depth": args.depth,
        "timeMs": args.time_ms,
        "hashMB": args.hash_mb,
    }
    records: list[dict[str, object]] = []
    total = len(args.threads) * args.repetitions
    current = 0
    for repetition in range(1, args.repetitions + 1):
        for threads in args.threads:
            current += 1
            print(f"[{current}/{total}] threads={threads} repetition={repetition}", flush=True)
            result, output = run_benchmark(
                engine, threads, args.depth, args.time_ms, args.hash_mb, args.timeout_seconds
            )
            record = {"threads": threads, "repetition": repetition, **result}
            records.append(record)
            log_dir = output_dir / f"threads-{threads:02d}"
            log_dir.mkdir(exist_ok=True)
            (log_dir / f"run-{repetition:03d}.log").write_text(output, encoding="utf-8")

    summary = summarise(records)
    write_outputs(output_dir, configuration, records, summary)
    print(f"Thread scaling report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
