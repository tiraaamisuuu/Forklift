#!/usr/bin/env python3
"""Replay positions from an abandoned PGN game through real UCI worker threads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import chess.engine
import chess.pgn
from nnue_dataset import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--pgn", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve existing replay evidence; choose a new output")
    if args.rounds < 1 or args.seconds <= 0:
        raise SystemExit("Positive rounds and seconds required")
    with args.pgn.open() as source:
        while (game := chess.pgn.read_game(source)) is not None:
            if game.headers.get("Termination") == "abandoned":
                break
    if game is None:
        raise SystemExit("No abandoned game in PGN")
    boards = [node.board() for node in game.mainline() if node.board().turn == chess.WHITE]
    report = {"state": "running", "startedAt": datetime.now(timezone.utc).isoformat(),
              "engineSha256": sha256_file(args.engine), "networkSha256": sha256_file(args.network),
              "sourcePgnSha256": sha256_file(args.pgn), "seconds": args.seconds,
              "rounds": args.rounds, "total": len(boards) * args.rounds * 2, "cases": []}

    def replay(round_index, threads):
        engine = None
        current_fen = None
        results = []
        try:
            engine = chess.engine.SimpleEngine.popen_uci(str(args.engine.resolve()), timeout=60)
            engine.configure({"Hash": 128, "Threads": threads,
                              "EvalFile": str(args.network.resolve()), "Use NNUE": True})
            engine.ping()
            for board in boards:
                current_fen = board.fen()
                started = time.monotonic()
                answer = engine.play(board, chess.engine.Limit(time=args.seconds))
                if answer.move not in board.legal_moves:
                    raise RuntimeError("Illegal UCI best move")
                results.append({"round": round_index, "threads": threads, "fen": board.fen(),
                                "move": answer.move.uci(), "elapsed": time.monotonic() - started})
        except Exception as error:
            results.append({"round": round_index, "threads": threads, "fen": current_fen,
                            "error": repr(error)})
        finally:
            if engine:
                try:
                    engine.quit()
                except Exception:
                    engine.close()
        return results

    write_json_atomic(args.output, report)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(replay, i, threads) for i in range(args.rounds) for threads in (1, 2)]
        for future in as_completed(futures):
            report["cases"].extend(future.result())
            write_json_atomic(args.output, report)
            print(f"replay cases={len(report['cases'])}/{report['total']}", flush=True)
    failed = any("error" in case for case in report["cases"])
    report.update(state="failed" if failed else "complete", completedAt=datetime.now(timezone.utc).isoformat())
    write_json_atomic(args.output, report)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
