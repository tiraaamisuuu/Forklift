# Staged move picker champion gate

Date: 2026-09-10

## First gate result

The staged move picker at `d8a456f7f4e4983dfe6393177b7b450aa8691704`
was tested against frozen champion
`071b4b624629ead2326b8ff03cace89b0520aa1f`. The contract allowed 10,000
paired games at `30+0.3`, one thread, 256 MiB hash, concurrency six, opening
seed 6001 and SPRT Elo0 = 0 / Elo1 = 5.

- Scored games: 2,093
- Candidate W-D-L: `691-848-554`
- Candidate score: 53.3%
- Relative Elo: `+22.8 +/- 11.5`
- SPRT: H1 accepted, LLR `2.96` above the `2.94` boundary
- Time forfeits, crashes and illegal moves: zero
- Candidate disconnects: one

The automated gate correctly classified this as `technical_failure`, so the
candidate was not promoted despite passing the strength boundary.

## Crash investigation

Game 1906 ended after three plies when the candidate process exited. Windows
Application Error and Windows Error Reporting events at the same timestamp
identified exception `0xc00000fd`, a stack overflow in `chess-engine-uci.exe`.
The event named the immutable candidate binary, ruling out a dashboard or
tournament-runner disconnect.

The search had no absolute ply boundary, and quiescence checked threefold
repetition only when the side to move was not in check. A pathological chain
of check evasions could therefore keep recursing through a repeated checking
cycle until the process stack was exhausted.

## Remediation

Commit `e90bfdbd476f5dce3fe744b5a5489dd582c3cffc`:

- recognises rule draws in quiescence even when the side to move is in check,
  while preserving checkmate precedence;
- bounds both principal and quiescence search at ply 127;
- adds regression tests for checked repetitions, checkmate precedence and the
  principal/quiescence ply boundary.

Release core tests and UCI smoke tests pass. A native Windows AddressSanitizer
build also passes both suites. At fixed depth 12 the four-position diagnostic
kept every best move and score; it searched 703,600 nodes versus 703,416
(`+0.026%`), with equivalent throughput.

## Clean rerun

A new immutable gate is running at
`E:\Dev\Forklift-Research\matches\champion-gates\staged-move-picker-stack-safe-20260910`
with opening seed 6002 and the same `30+0.3` resource contract. Promotion
remains blocked until that gate accepts H1 with no technical termination.
