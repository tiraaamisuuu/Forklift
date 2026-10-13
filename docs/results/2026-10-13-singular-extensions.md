# Conservative singular-extension prototype — 2026-10-13

## Hypothesis and scope

Search should spend an extra ply on a move supported by a reasonably deep
transposition-table (TT) lower/exact bound when a reduced search of **every
other legal move at that node** fails below a lower threshold. This may help
forced tactical lines, but the verification and extension cost can reduce
ordinary search depth at equal time. Neither extra nodes nor changed moves
prove a strength improvement.

This is an established technique, not a novelty claim. See the
[official Stockfish search source](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
for the general excluded-move approach and its caveats. Forklift's implementation
and deliberately conservative policy are written for its own search; this
experiment does not copy Stockfish's tuned parameter system.

## Implemented policy

- Opt-in only: UCI `Singular Extensions`, default `false`; benchmark
  `--singular`, default off. Both single-thread and root-parallel search
  propagate the option.
- Eligible non-root nodes have remaining depth at least 8, are not in check,
  and have room before the maximum search ply.
- A TT entry must be lower-bound or exact, with depth at least `depth - 3`,
  a non-mate score in `(-MATE/2, MATE/2)`, and a move resolved against the
  actual legal list. The compact TT does not retain capture/castle/EP flags,
  so the legal move supplies these before exclusion.
- Verification uses depth `(depth - 1)/2` and a zero-width integer window
  `[ttScore - 2*depth - 1, ttScore - 2*depth]`, excluding the selected move.
- The excluded root cannot reuse ordinary TT bounds, run IID/null-move/
  ProbCut/razoring/reverse-futility, skip quiet alternatives, or reduce them
  with LMR. Descendants use ordinary search, including its selective pruning;
  this is a search heuristic, not an exhaustive mathematical proof.
- A successful verification grants one extra ply to that move, never stacked
  with the existing check extension on the same edge, and never LMR-reduced.
- There are at most four added singular plies along an active line. Singular
  verification cannot recursively trigger more singular verification.
- Excluded-root scores are never stored under the ordinary position hash;
  excluded-root cutoffs do not update killer/history tables. Descendant TT
  entries and history updates still describe ordinary legal searches.
- An empty set of alternatives returns the fail-low window edge, not a
  fabricated checkmate or stalemate. Normal chess terminal rules remain intact.
- Scope guards restore verification mode and the path budget; same-ply
  verification restores the parent's current move/static-evaluation slots.
  Existing repetition, NNUE, scratch, maximum-ply and stop handling remain in use.

Counters expose verification attempts, applied extensions, and peak singular
plies on a line in benchmark output. The thread-benchmark parser accepts
these counters while retaining compatibility with earlier output formats.

## Validation so far

The Release headless build passes six CTests. Dedicated assertions cover
eligibility, upper/shallow/mate/malformed TT rejection, illegal TT moves,
canonical castle/EP/underpromotion identity, exclusion without TT cutoff or
root TT pollution, an empty alternative set, successful extension of a
winning capture, refusal when alternatives compete, path-state unwinding,
exact board/repetition/scratch restoration, and aborted search.

The NNUE fixture now performs incremental-versus-full-rebuild depth-9 searches
with singular verification active, requiring identical moves, scores, main
nodes, quiescence nodes, verification attempts and extension counts. Large
test contexts are heap allocated to preserve Windows stack safety. UCI smoke
enables the option before `ucinewgame` and exercises ordinary, restricted-root
and emergency-clock searches. Existing expired-deadline/budget assertions
also run with the feature enabled.

An initial real-network depth-12 diagnostic verified actual activation (68
extensions across 644 verification attempts). Its tree and some moves changed;
this is expected and **not** a strength result. A separate off-mode comparison
matched the previous executable's moves/scores/main/quiescence nodes at every
fixed position, retaining 969,129 main nodes overall. Timing is exploratory
while the independent model confirmation occupies the same host.

## Final implementation evidence

The implementation is committed as
`7a148b8ecdb6182348644dafbed4e0008b2f61f4`. A fresh diagnostic on that source
again exercised 68 extensions / 644 attempts at depth 12 with the shared
network. It visited 1,302,166 main nodes versus 969,129 with the option off
(34.36% more main nodes). Single exploratory reported times were 2,881 ms on
and 2,098 ms off. This overhead is why equal-time strength must be assessed;
the result is not comparable to the earlier identical-tree inference speedup.

Real-network incremental and full-rebuild depth-10 runs matched every move,
score, main-node/quiescence count, attempt/extension count and path maximum,
with 351,727 main nodes overall and 15 extensions / 157 attempts. A four-thread
depth-10 diagnostic also activated verification/extensions. Its tree and
some scores differed from the single-thread search, as expected from the
existing shared-TT scheduling; no parallel-strength claim follows. The
classical depth-12 run reached the four-extension path cap without exceeding it.

[CI run 38373963982](https://github.com/tiraaamisuuu/Forklift/actions/runs/38373963982)
passed all six jobs on the implementation commit: Ubuntu/macOS/Windows
headless, ASAN/UBSAN, web GUI, and Windows desktop build/test/package checks.
Local CTest passed all six tests, and the thread-benchmark parser passed all
four unit tests.

[Machine-readable evidence](../../research/experiments/2026-10-13-singular-extensions.json)
contains the policy, exact commit, binary/network hashes, raw and parsed
outputs for all six final diagnostics, CI evidence, limitations, and the
requirements for a future strength comparison. No new match was launched.
The existing confirmation executable retains its frozen SHA-256
`b97bfa1199e5c3ff13b3d53cb713a3d21ff2f47cfae42c9035e3581ec050c2ff`.

## Reproduce

```powershell
cmake -S . -B build-singular -DCHESS_BUILD_GUI=OFF -DBUILD_TESTING=ON
cmake --build build-singular --config Release --parallel 2
ctest --test-dir build-singular -C Release --output-on-failure
.\build-singular\Release\chess-engine-tools.exe --bench `
  --bench-depth 12 --bench-time 60000 --bench-tt 128 --threads 1 --singular `
  --nnue E:\Dev\Forklift-Research\models\nnue-v2-factorization-20260912\shared\model.nnue
```

In any UCI client, set `Singular Extensions=true` to use the candidate. Changing
the option stops active search and resets its TT/history, preventing an
on/off comparison from inheriting entries computed under the other policy.

## Decision and next evidence

Retain as an experimental implementation; do **not** enable it by default or
claim an Elo gain. Finish coding and targeted diagnostics without starting
another unchanged-engine match campaign. A later equal-time comparison must
compare this exact option on/off using identical binaries, evaluation weights,
threads, hash, paired openings and clocks. Freeze the match contract before
launch, and do not tune thresholds in response to partial scores.

The running shared-NNUE model confirmation still uses its older frozen binary;
none of those games is evidence for this search feature. Preserve the model
promotion decision separately from this implementation's future strength test.
