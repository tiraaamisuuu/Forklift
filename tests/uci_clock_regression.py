#!/usr/bin/env python3
"""Exercise raw UCI clocks and retain wall-time evidence (standard library only)."""

import argparse
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    engine = args.engine.resolve()
    process = subprocess.Popen([str(engine)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True, bufsize=1)
    lines = queue.Queue()

    def read_output():
        for line in process.stdout:
            lines.put(line.strip())
        lines.put(None)

    threading.Thread(target=read_output, daemon=True).start()

    def send(command):
        process.stdin.write(command + "\n")
        process.stdin.flush()

    def receive(prefix, timeout=5):
        deadline = time.perf_counter() + timeout
        received = []
        while True:
            line = lines.get(timeout=max(0.001, deadline - time.perf_counter()))
            if line is None:
                raise RuntimeError(f"engine exited: {process.poll()}")
            received.append(line)
            if line.startswith(prefix):
                return received

    samples = []
    try:
        send("uci")
        receive("uciok")
        send("setoption name Hash value 16")
        send("ucinewgame")
        send("isready")
        receive("readyok")  # exclude engine startup / hash allocation from move timing
        for threads in (1, 2):
            send(f"setoption name Threads value {threads}")
            for repetition in range(args.repetitions):
                for clock in (0, 1, 20, 50, 100, 300, 325):
                    # Exact checked endgame from gate game 798; no restriction on search.
                    send("position fen 2N5/4kp2/1p4p1/4P3/p4P2/4P3/2K4p/4r3 b - - 3 58")
                    start = time.perf_counter()
                    send(f"go wtime {clock} btime {clock} winc 300 binc 300")
                    output = receive("bestmove ")
                    elapsed = (time.perf_counter() - start) * 1000
                    # Wide CI tolerance at zero; the old 1000/1500 ms fallback fails.
                    limit = max(200, clock + 100)
                    samples.append({"threads": threads, "repetition": repetition,
                                    "clockMs": clock, "wallMs": round(elapsed, 3),
                                    "passed": elapsed < limit and output[-1].split()[1] in
                                    {"e7d8", "e7e8", "e7f8", "e7d7", "e7f6", "e7e6"},
                                    "output": output})
    finally:
        if process.poll() is None:
            send("quit")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    result = {"engine": str(engine), "sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
              "samples": samples, "passed": bool(samples) and all(s["passed"] for s in samples)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "samples": len(samples),
                      "failures": [s for s in samples if not s["passed"]]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
