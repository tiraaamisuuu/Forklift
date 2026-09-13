# NNUE disconnect: stack exhaustion and recovery

## Evidence

The 800-game confirmation completed 405 wins / 219 draws / 176 losses
(+102.3 ±21.1 reported relative Elo), but failed its technical gate. Game 97
ended in a white NNUE disconnect. The longer-control stage did not start.

Windows Application events 1000/1001 at 2026-09-13 20:52:55–58 local time identify
`chess-engine-uci.exe` from `build-clock-safe`, exception **0xc00000fd (stack
overflow)**, offset `0x1fcf7`, report `bc79c30d-6670-40eb-9b9f-4fba85cd8032`.
This is an engine process crash, not a LAN connection failure.

PGN source:
`E:\Dev\Forklift-Research\matches\nnue-v2-confirmation-20260913\network\games.pgn`.
The final position after 12...Rg8 was:

```text
r3kbr1/pn2n2p/2p2p2/6p1/2N5/2P2PB1/5PPP/3QR1K1 w q - 2 13
```

A fresh-process three-second replay on the old binary returned a legal move.
However, a subsequent 15-second-per-position replay over the abandoned game's
white-to-move positions crashed the old binary with both one and two search
threads: exit code 3221225725, the same 0xc00000fd stack overflow. This reproduces
the failure class with the actual network/game history, although not necessarily
the exact original overflowing instruction or final position.

## Repair

Recursive negamax/quiescence carried multiple 320-entry move arrays; the staged
picker added two more arrays. The existing 127-ply ceiling therefore did not
ensure safe stack consumption, especially with same-ply IID/verification calls.
Move lists and picker metadata now live in reusable heap frames owned by each
search context. Frames are indexed by active call nesting, not chess ply, so
same-ply recursion cannot overwrite a parent's state. RAII releases frames on
all normal/exception returns, and allocated frames are reused between searches.
No ply limit or pruning/evaluation rule was changed to hide the crash.

The dashboard now counts completed-game failure records in the live log rather
than waiting for final result JSON. Summary lines are excluded to avoid double
counting. A failed series is explicitly shown as failed, even if all games in
its last match finished. The coordinator marks that stage failed, not running.

## Verification

- Optimized MSVC Release: core and UCI smoke checks passed.
- Windows AddressSanitizer Release: core and UCI checks passed.
- Six-position perft suite passed through depth 4.
- New tests cover nested-frame separation, stable parent references on pool
  growth, scope unwinding, reuse, and a small staged-picker stack footprint.
- Depth-10 NNUE diagnostic on four positions preserved every move, score, main
  node and quiescence node count. Both versions searched 321,287 main nodes.
  One sample took 872 ms before / 824 ms after under background load; this is
  **not** a controlled performance-gain claim.
- Failed-game replay: 72/72 five-second searches passed on Release, using one
  and two threads across three rounds of its 12 white-to-move positions. Another
  24/24 two-second searches passed under AddressSanitizer.
- The identical 15-second replay that crashed both old-binary workers completed
  **24/24 searches** on the repaired Release binary, with no illegal moves or
  process deaths. Evidence: `nnue-stack-old-replay-20260914.json` versus
  `nnue-stack-long-replay-20260914.json` under the same matches directory.
- 79 Python tests, the real-engine web smoke, and eight quick paired games passed.
  Quick games are workflow checks and are excluded from strength evidence.

Replay evidence is checksummed JSON under
`E:\Dev\Forklift-Research\matches\nnue-stack-{replay,asan-replay}-20260914.json`.
Use `scripts/nnue/replay_reliability.py` with explicit executable, network,
source PGN and fresh output path. It retains the game's move history and tests
actual UCI worker threads rather than only a top-level evaluation call.

## Overnight contract

Fresh repaired-engine evidence, separate from the failed confirmation:
`E:\Dev\Forklift-Research\matches\nnue-stack-overnight-20260914`.
The same fixed shared network faces classical on the **same repaired executable**.
Run 800 paired games at 30+0.3, seed 202609140; then 1,200 at 90+0.9,
seed 202609141. Six concurrent games, one thread and 128 MiB hash per side,
pure NNUE weight 100. Source revisions, binary/network/opening hashes, PGNs,
logs and results are preserved by the runner. The longer stage runs only after
the first completes without technical failures. This is a substantial overnight
batch; wall time depends on game length and is not guaranteed to equal 10–12 hours.

These are fixed-length tests, not a procedure to add games until a positive
result appears. Require zero technical failures and a positive reported lower
Elo bound in both before considering default integration. No automatic release
or default change. A recurrence remains a blocker requiring investigation.
