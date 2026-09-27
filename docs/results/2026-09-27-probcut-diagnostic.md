# ProbCut search diagnostic

## Candidate

Forklift now has an experimental ProbCut path for non-PV nodes. It considers
only SEE-qualified captures and queen promotions, requires a wider beta margin,
preflights each candidate through quiescence, and then performs a reduced
verification search. ProbCut is disabled inside that verification search to
prevent a speculative recursive cascade. Successful proofs are stored as
lower-bound transposition-table entries.

The implementation is instrumented with attempt/cutoff counters and can be
enabled independently through the UCI `ProbCut` check option. The option is
**false by default**; the SFML app and ordinary UCI play therefore retain the
established search unless an experiment explicitly enables it.

## Correctness gates

- Optimized MSVC Release build passed.
- Core and UCI smoke tests passed.
- Windows AddressSanitizer core and UCI tests passed with the MSVC ASAN runtime
  directory on `PATH`.
- A targeted fixture verifies a winning capture, beta cutoff, lower-bound TT
  entry, exact board restoration, and the disabled path.
- The Windows and POSIX UCI smoke scripts advertise and exercise the option.

## Fixed-depth benchmark

The same optimized binary searched the four committed benchmark positions with
one thread and a 256 MiB TT. Disabling ProbCut provides the within-binary
control.

At depth 10, the tightened candidate was roughly neutral: 269,432 main nodes
versus 272,988, but slightly more total work and 760 ms versus 745 ms in the
recorded pass.

At depth 12, the candidate searched 661,197 main plus 393,228 quiescence nodes,
versus 703,600 plus 408,440 with ProbCut off. Combined work fell from 1,112,040
to 1,054,425 nodes (5.18%), and wall time fell from 1,839 to 1,751 ms (4.79%).
Three of four best moves matched. The second middlegame move and several scores
changed, as expected for selective search, so this is efficiency evidence rather
than a strength claim.

## Paired match

The first diagnostic used the same executable on both sides, differing only in
`ProbCut=false` versus `ProbCut=true`:

```powershell
py -3 scripts/compare_engines.py `
  --baseline-exe .\build-probcut\Release\chess-engine-uci.exe `
  --candidate-exe .\build-probcut\Release\chess-engine-uci.exe `
  --baseline-name "ProbCut off" --candidate-name "ProbCut on" `
  --baseline-option "ProbCut=false" --candidate-option "ProbCut=true" `
  --games 100 --tc 2+0.02 --threads 1 --hash 128 --concurrency 6 `
  --seed 20260927 `
  --output-dir artifacts\elo\probcut-diagnostic-20260927
```

The candidate scored `32-35-33` (48.5%), reported as -10.4 +/- 56.2 Elo with
35.7% LOS. There were zero time forfeits, crashes, illegal moves, or
disconnects. The wide interval includes both useful and harmful outcomes.

## Decision

Retain the bounded implementation as an opt-in research path, but do not enable
it by default and do not claim a playing-strength gain. Before promotion, tune
one variable at a time and require a positive slower paired result through the
frozen-champion process. The initial fast result is fixed evidence; it must not
be extended opportunistically until it turns positive.
