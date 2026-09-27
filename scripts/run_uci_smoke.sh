#!/usr/bin/env bash
set -euo pipefail

ENGINE_BIN="${1:-./build/chess-engine-uci}"

if [[ ! -x "$ENGINE_BIN" ]]; then
  echo "Error: engine binary not found or not executable: $ENGINE_BIN" >&2
  echo "Build first, e.g. cmake -S . -B build && cmake --build build -j8" >&2
  exit 1
fi

OUT_FILE="$(mktemp)"
trap 'rm -f "$OUT_FILE"' EXIT

{
  printf 'uci\n'
  printf 'isready\n'
  printf 'setoption name Hash value 128\n'
  printf 'setoption name ProbCut value true\n'
  printf 'ucinewgame\n'
  printf 'position startpos moves e2e4 e7e5 g1f3 b8c6\n'
  printf 'go movetime 300\n'
  sleep 0.6
  printf 'position startpos\n'
  printf 'go searchmoves a2a3 movetime 200\n'
  sleep 0.4
  printf 'position startpos\n'
  printf 'go searchmoves b2b3 wtime 20 btime 20 winc 20 binc 20\n'
  sleep 0.3
  printf 'quit\n'
} | "$ENGINE_BIN" --uci > "$OUT_FILE"

if ! grep -Eq '^uciok$' "$OUT_FILE"; then
  echo "[FAIL] Missing uciok" >&2
  cat "$OUT_FILE" >&2
  exit 1
fi

if ! grep -Eq '^readyok$' "$OUT_FILE"; then
  echo "[FAIL] Missing readyok" >&2
  cat "$OUT_FILE" >&2
  exit 1
fi

if ! grep -Eq '^bestmove [a-h][1-8][a-h][1-8][qrbn]?$' "$OUT_FILE"; then
  echo "[FAIL] Missing or invalid bestmove" >&2
  cat "$OUT_FILE" >&2
  exit 1
fi

if ! grep -Eq '^bestmove a2a3$' "$OUT_FILE"; then
  echo "[FAIL] UCI searchmoves restriction was not respected" >&2
  cat "$OUT_FILE" >&2
  exit 1
fi

if ! grep -Eq '^bestmove b2b3$' "$OUT_FILE"; then
  echo "[FAIL] Low-clock UCI search did not complete with its restricted move" >&2
  cat "$OUT_FILE" >&2
  exit 1
fi

if ! grep -Eq '^option name Clear Hash type button$' "$OUT_FILE"; then
  echo "[FAIL] Clear Hash option was not advertised" >&2
  cat "$OUT_FILE" >&2
  exit 1
fi

if ! grep -Eq '^option name Move Overhead type spin default 25 min 0 max 5000$' "$OUT_FILE"; then
  echo "[FAIL] Move Overhead option was not advertised" >&2
  cat "$OUT_FILE" >&2
  exit 1
fi

if ! grep -Eq '^option name EvalFile type string default <empty>$' "$OUT_FILE" ||
   ! grep -Eq '^option name Use NNUE type check default false$' "$OUT_FILE"; then
  echo "[FAIL] NNUE UCI options were not advertised" >&2
  cat "$OUT_FILE" >&2
  exit 1
fi

if ! grep -Eq '^option name ProbCut type check default false$' "$OUT_FILE"; then
  echo "[FAIL] ProbCut option was not advertised" >&2
  cat "$OUT_FILE" >&2
  exit 1
fi

echo "UCI smoke: PASS"
