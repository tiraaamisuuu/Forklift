# Staged move picker

Date: 2026-08-30

## Hypothesis

A staged move picker should improve search quality by trying the most useful
move classes in a deliberate order while avoiding the cost of fully sorting
moves that will never be searched after a cutoff.

## Immutable revisions

- Validated baseline: `bcf262d041e723771e89fd833d1ed78b08743fea`
- Candidate: `d8a456f7f4e4983dfe6393177b7b450aa8691704`

The candidate changes `src/search.hpp` and focused invariants in
`tests/core_tests.cpp`. Recursive alpha-beta search now selects moves in these
stages: transposition-table move, good tactical moves, killer/counter moves,
remaining quiet moves, then bad tactical moves. Root search and quiescence
search remain unchanged.

## Correctness invariants

- The transposition-table move is returned first when legal.
- Winning captures precede quiet moves.
- Killer and counter moves precede ordinary quiet moves.
- Losing captures are deferred until after quiet moves.
- Every legal move is returned exactly once.
- Release build, core tests and UCI smoke test: passed.

## Fixed-depth diagnostics

The picker is not presented as a raw node-count optimisation. On the four-position
diagnostic set it changed the explored tree and principal variation:

- Depth 10: 186,618 baseline nodes versus 272,988 candidate nodes (`+46.3%`).
- Depth 12: 626,297 baseline nodes versus 703,416 candidate nodes (`+12.3%`).
- Nodes per second remained broadly similar.

This makes direct match testing essential: a larger tree can still be stronger
if the new ordering spends effort on tactically useful branches, but no speed
or efficiency gain is claimed from these measurements.

## Short promotion screen

Contract: 200 paired games, `2+0.02`, one thread and 256 MiB hash per engine,
six concurrent games, opening seed 5901.

- Candidate W-D-L: `83-49-68`
- Candidate score: 53.7%
- Relative Elo: `+26.1 +/- 42.1`
- LOS: 88.9%
- Technical failures: zero

The screen is too small for a strength claim, but the positive point estimate
and clean stability record qualify the candidate for confirmation.

## Powered confirmation

The preregistered confirmation completed under
`E:\Dev\Forklift-Research\matches\staged-move-picker-confirmation-20260830`.
Its contract was a maximum 5,000 games at `10+0.1`, one thread, 256 MiB hash,
six concurrent games, opening seed 5902 and SPRT Elo0 = 0 / Elo1 = 5.

- Decisive scored games: 2,593
- Candidate W-D-L: `906-931-756`
- Candidate score: 52.9%
- Relative Elo: `+20.1 +/- 10.7`
- LOS: 100.0%
- SPRT: H1 accepted, LLR `2.96` above the `2.94` boundary

## Technical-termination audit

The match recorded 35 time forfeits, requiring investigation under the
registered rule. They were balanced between the engines: 18 by the candidate
and 17 by the baseline. Excluding every time-forfeit game leaves
`889-931-738` from the candidate perspective, a 52.95% score and approximately
`+20.5` Elo. The five unscored games were the concurrently running games
terminated after the SPRT boundary had already been crossed, rather than
engine crashes or illegal moves. Crashes, illegal moves and disconnects were
all zero.

## Decision

Retained. The candidate crossed the preregistered H1 boundary, and the
technical audit found no asymmetric failure signal capable of explaining the
gain. It advances to the separate frozen-champion promotion gate; the result
is not yet a champion-registry update.
