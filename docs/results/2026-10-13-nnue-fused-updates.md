# Fused NNUE accumulator updates — 2026-10-13

## Engine change

`NnueNetwork::updateAfterMove` previously copied both parent accumulator
vectors, then traversed each destination again for the moving piece removal,
optional captured piece removal, and resulting piece addition. The new path
combines these operations into one lane-wise write per perspective for
non-king moves:

```text
quiet:   child[h] = parent[h] - from[h] + to[h]
capture: child[h] = parent[h] - from[h] - captured[h] + to[h]
```

Capture detection uses `Undo::captured`, not just the move flag. En passant
removes the pawn from its actual capture square. Promotion removes a pawn
feature and adds the promoted piece feature, including underpromotion and
capture promotion. Each subtraction/addition retains the old per-lane order
and sign-extends the stored 16-bit weights into the 32-bit accumulator.

King moves still refresh the moving king's perspective. Its opposite
perspective retains the existing capture and castling-rook delta path, but
the discarded copy before a full refresh is now avoided. An invalid parent
or a parent with incompatible vector lengths falls back to a full refresh.
The format, weights, evaluation, search policy, and UCI defaults do not change.
This is portable C++17; no explicit SIMD instructions or CPU requirement were
introduced. Compiler auto-vectorization is permitted.

## Correctness evidence

The Release headless build passed all four CTests (core, production benchmark
defaults, ProbCut opt-in, UCI smoke). Expanded core assertions compare **every
accumulator lane** with a full rebuild, not only the clipped final score, at
widths 1, 4, 17, 256 (production), and 1,024 (format maximum). The fixture
contains varying per-lane weights and both extreme `int16_t` values.

The suite exercises both colors' quiet moves, captures, en passant, all four
promotion types with and without capture, both castling directions, king
moves/captures, in-place updates, dirty reused destinations, invalid-parent
fallbacks, seeded 160-ply playouts and their complete unmake, and null-move
copies. The width-4 fixture also compares incremental versus full-rebuild
search moves, scores, main nodes, and quiescence nodes.

With the real shared-feature network, the baseline and candidate depth-10
four-position benchmark produced identical moves, scores, main nodes, and
quiescence nodes (321,287 main nodes total). A separate candidate depth-8
incremental/rebuild comparison also matched exactly (108,697 main nodes).

## Timing protocol

The baseline is clean commit `695495434a335687b481e47d7001c3a4e2338e07`, built
before editing engine source. Both baseline and candidate use MSVC
19.50.35725.0 Release, C++17, native architecture **off**, IPO **off**, the
same Ryzen 9 5900X host, and separate build directories. The network SHA-256
is `ff25cf447dc6137591ecc2522d23f14e2f9bee0a641dda36ec2cd9c67a762c7c`.

The depth-10 pilot took 791 ms baseline and 662 ms candidate. This is a
single exploratory timing, **not** the adoption result. The subsequent
protocol, fixed before its repeated runs, is:

- The existing four fixed benchmark positions; depth 12, 60,000 ms ceiling
  per position, one thread, 128 MiB TT, NNUE weight 100, ProbCut off.
- One unrecorded warm-up per executable, then ten paired trials.
- Alternate baseline/candidate order by pair; never run them concurrently.
- Require every position to reach depth 12 and every move, score, main-node
  count and quiescence-node count to match throughout the entire sample.
- Retain all raw stdout, executable hashes, order and per-trial timings.
- Report median total time and median paired throughput ratio. A reduction
  below 3% is treated as inconclusive for retaining the performance change;
  a regression is a reason to revise or reject it.

The independent long NNUE confirmation is still running on its frozen binary.
Timing on this shared host is therefore **preliminary under contention**.
Even repeatable results here do not establish isolated-hardware performance,
an equal-time Elo gain, or Stockfish-class strength. Repeat on an idle host
before publishing a performance estimate. The confirmation games do not
test this new optimization and must not be attributed to it.

Reproduce one trial (use the same command on the baseline executable):

```powershell
.\build-nnue-fusion\Release\chess-engine-tools.exe `
  --bench --bench-depth 12 --bench-time 60000 --bench-tt 128 --threads 1 `
  --nnue E:\Dev\Forklift-Research\models\nnue-v2-factorization-20260912\shared\model.nnue
```

## Repeated measurement result

Candidate commit `a0a35e2217dacb12496acd4fea3a3ced8bffa5f9` completed the
predeclared ten pairs from 08:40:46 to 08:41:40 UTC. All twenty runs reached
depth 12 at every position and had identical moves, scores, main nodes and
quiescence nodes. Each run visited 969,129 main nodes and 218,408 quiescence
nodes across the four positions.

Median total reported search time was **2,467 ms baseline versus 2,048.5 ms
candidate**, a **16.96% reduction**. The median paired throughput ratio was
**1.2047** (approximately **20.5% higher throughput**); all ten pairs favored
the candidate, with ratios from 1.1782 to 1.2211. Reduction in time and increase
in throughput are different percentages and must not be conflated.

The change is retained: exact parity passed and the observed reduction
exceeded the documented 3% threshold. The timing remains preliminary under
concurrent match load; an idle-host replication and equal-time matches are
still needed to quantify broader performance and any playing-strength gain.
No Elo gain is claimed for this optimization.

[Machine-readable measurements](../../research/experiments/2026-10-13-nnue-fused-update.json)
preserve the protocol, hashes, every trial's full benchmark stdout, paired
ratios, and correctness/CI evidence. The tools executable hashes are:

- Baseline: `6417568aeb89bbf486e7f09114615f3d83d529b170a730d8609b5f60607e957a`
- Candidate: `49dd7070405ebb0287eafae51836aedb9905b23404ff4944c9fbcd0b3f15c79e`

## Cross-platform validation and isolation

[CI run 38370773757](https://github.com/tiraaamisuuu/Forklift/actions/runs/38370773757)
passed all six jobs for the engine commit: Ubuntu/macOS/Windows headless,
address/undefined-behavior sanitizers, web GUI, and the Windows desktop
build/test/package checks. No extra CPU-specific build flag was needed.

After the repeated benchmark, the independent confirmation executable still
had SHA-256
`b97bfa1199e5c3ff13b3d53cb713a3d21ff2f47cfae42c9035e3581ec050c2ff`,
matching its frozen contract. It was not rebuilt, replaced or restarted.

Next engineering steps are to profile the remaining inference costs before
choosing explicit SIMD work, and separately prototype singular extensions
with their own controlled search/strength experiment. This result alone is
not evidence that either follow-on change will help.
