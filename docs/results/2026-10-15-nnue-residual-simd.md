# Exact, bounded residual-head inference optimisation

Date: 2026-10-15. Engine implementation: `969e33f`, following the
[initial residual implementation and validation](2026-10-15-nnue-residual-head.md).
User authorised modest CPU testing and tiny training while the PC remained in
shared use. No GPU training, teacher generation, strength matches, default
changes or background schedules were started.

## Result and interpretation

Reusing clipped inputs and certifying narrow integer dot products substantially
reduced the previous scalar residual cost:

| H256 head | Old median four-position time | New median | Median paired time reduction |
| --- | ---: | ---: | ---: |
| Direct control | 655 ms | 638 ms | Not claimed; arithmetic unchanged |
| Residual 4 | 1,362 ms | 698 ms | 49.56% |
| Residual 8 | 2,148 ms | 773 ms | 64.35% |

On the new binary, ratios of median elapsed times versus direct are 1.094x
and 1.212x respectively. These are **end-to-end fixed-depth search costs**, not
isolated matrix speedups or Elo measurements. All 24 warmup/measured runs
searched identical trees: 321,287 main nodes and 76,248 quiescence nodes, with
unchanged moves, scores and completed depth. Direct timing differences are
shared-host noise/code-layout effects, not an additional V1 optimisation.

Three measured trials per binary/head, one warmup each, alternating old/new
order and whole order. Depth 10, four standard positions, one thread, 128 MiB
hash, 5,000 ms per-position ceiling, NNUE weight 100%, incremental accumulators,
ProbCut/singular extensions off. This is an exploratory shared-host pilot;
idle-host and cross-platform replication remain. The diagnostic took 33.26
seconds excluding Python startup.

The existing shared source weights were exported with zero-output R4/R8
branches to preserve tree identity. Neither implementation skips zero-output
neurons: all residual dots and divisions execute. Actual nonzero trained heads
were checked separately for numerical parity, not timed in this protocol.
The direct re-export exactly matches the existing network SHA-256. Raw output,
hashes, order and per-position times are in the
[cost record](../../research/experiments/2026-10-15-nnue-residual-simd-cost.json).
Do not multiply these gains by earlier separately measured speedups.
The [validation summary](../../research/experiments/2026-10-15-nnue-residual-simd-validation.json)
records the implementation commit, test outcomes and resource limits.

## Implementation and exactness

V1 inference, V2 wire format, model topology, perspective order and final
rounding are unchanged. For V2, clip the two accumulators **once** into a
contiguous `2H` scratch buffer and reuse it across all residual rows.
For each row, loading computes in int64:

```text
B_j = abs(residual_bias[j]) + A * sum_i(abs(residual_weight[j,i]))
```

If `B_j <= INT32_MAX`, every product and any subset/partial reduction fit
signed int32: PMADDWD pairs, SIMD lanes, horizontal sums, bias and scalar-tail
additions, regardless of signs or accumulation order. Widen before absolute
values, including signed-minimum weights/biases.

When `A <= 32767`, saturating int32-to-int16 packing followed by clipping is
exact even for INT32_MIN/MAX accumulators. Activations are nonnegative, so
PMADDWD's exceptional `-32768 * -32768` pair cannot occur. Certified rows use
baseline SSE2 when `2H >= 8`; remaining elements use exact scalar tails.
Other certified rows use scalar32; uncertified rows use scalar64. Wide-scale
networks clip once into int32 scratch and retain the same row certificates.
Mixed rows select their own safe paths. Inability to narrow a residual row
does not reject a previously supported network.

Scratch is fixed-capacity (4 KiB narrow / 8 KiB wide, with compiler-dependent
stack slot reuse), without heap allocation or zero-fill. Only populated lanes
are read. The original repeated-clipping int64 loop remains independent under
`FORKLIFT_NNUE_FORCE_REFERENCE`. Both branches still combine before **one**
final signed division. Clear/reload resets certificates and kernel diagnostics.

## Validation

- Release build: at most two workers; all 14 CTests pass, 19.19 seconds.
- Normal, forced-scalar and forced-reference residual targets each pass 64
  numeric fixtures and 19,296 independent int64-oracle comparisons: 57,888
  total, plus malformed files, rounding, perspective, special-move/incremental
  and reload checks.
- Exact INT32_MAX / one-above certificate boundaries; mixed safe/unsafe rows;
  signed extrema; A32767, 32768 and INT32_MAX; odd widths; H1023/H1024 scratch
  boundaries in both element widths. Overflow traps have visible, unsaturated
  outputs rather than hiding errors behind large final divisors.
- Focused MSVC Windows ASAN target passes all 64 fixtures, 16.63 seconds, no
  findings. Existing runtime DLL used through process-local PATH; no install.
  Not full cross-platform/UBSAN evidence; `/DEBUG` would improve reports.
- Export/training/resume integration rerun: both tests pass, 4.535 seconds;
  12 export configurations, 56 exact Python/C++ comparisons, preserved frozen
  transformer and initialization provenance. Those labels are artificial.
- Actual tiny trained exports: 72 old/new C++ comparisons agree exactly with
  Python integer reference across three models and 12 FENs, including eight
  recorded validation positions. All three transformer payloads remain
  byte-identical to the original V1 network. Verification took 4.875 seconds.

The first new normal test failed because its dispatch assertion checked a
private SIMD macro after the header deliberately undefined it. The test now
uses compiler architecture/force-scalar flags, as existing output tests do.
This was a test expectation bug, not an arithmetic mismatch. All final checks
above ran after the repair.

## Tiny real-label learning pilot

`scripts/nnue/residual_head_pilot.py` reads only 2,048 training and 512 validation
records using seeded random seeks into existing labelled shards. No corpus
scan, full-source hash, new labels or data-loader processes. Source digest
declarations are labelled as declarations, not newly verified hashes; exact
selection indices/hashes are retained. Sampled game IDs are cross-split
disjoint, which is not a new full-corpus independence audit.

All arms start from the same H256/shared source. Freeze the 10,649,856
transformer parameters; same data order/seed, WDL probability-MSE objective,
learning rate 1e-4, batch size 256, eight AdamW updates (one epoch). Target scale
600, WDL scale 400, A1024, O32, W256, result weight zero. CPU/inter-op pools
one, CUDA disabled, workers zero. The whole pilot took 9.294 seconds including
imports.

| Arm | Prefit held-out WDL loss | Postfit loss | Difference from trained direct |
| --- | ---: | ---: | ---: |
| Direct | 0.0102956197 | 0.0102741234 | 0 |
| Residual 4 | Same | 0.0102739991 | -0.0000001243 |
| Residual 8 | Same | 0.0102740740 | -0.0000000494 |

Differences are negligible, not established architecture/generalisation gains.
Sign accuracy stayed unchanged; centipawn MAE slightly worsened even though
the optimised WDL objective improved. All four R4 and seven of eight R8 output
coefficients quantise nonzero. This establishes learning/export, not a useful
trained candidate. Validation reuses the existing validation set, not a new
untouched test set. No model was promoted.

Full [pilot evidence](../../research/experiments/2026-10-15-nnue-residual-head-pilot.json)
and [post-pilot verification](../../research/experiments/2026-10-15-nnue-residual-head-parity.json)
retain hashes/provenance. The original report is preserved; verification was
added later without rerunning training.

## Reproduction and next gate

Run only when modest CPU use is authorised. Preserve an old tools executable
before rebuilding; forcing the old direct branch to int64 would confound a
claimed residual-only gain.

```powershell
# Actual small training; refuses an existing output directory.
.venv-nnue/Scripts/python.exe scripts/nnue/residual_head_pilot.py `
  --training-data E:/ChessData/halfkp-5m-sf18-20k/train.nnuebin `
  --validation-data E:/ChessData/halfkp-5m-sf18-20k/validation.nnuebin `
  --initialize-from E:/Dev/Forklift-Research/models/nnue-v2-factorization-20260912/shared/model.pt `
  --output-directory artifacts/new-bounded-head-pilot

# Cost only; report destination must not exist.
.venv-nnue/Scripts/python.exe scripts/nnue/residual_cost_pilot.py `
  --baseline-tools artifacts/nnue-residual-optimisation-20261015/baseline-tools.exe `
  --candidate-tools build-singular/Release/chess-engine-tools.exe `
  --initialize-from E:/Dev/Forklift-Research/models/nnue-v2-factorization-20260912/shared/model.pt `
  --output artifacts/new-residual-cost.json
```

The cost tool expects the measured x86 SSE2 kernels, validates runtime options
and rejects partial-depth/tree mismatches. Other hardware needs an explicitly
adapted protocol, not silent fallback. Both tools bound work; neither schedules
itself or runs full training.

Next, when the machine is available: predeclare a larger matched direct/R4
study, independent seed replication, richer held-out and quantisation/error
diagnostics. Only a convincing quality/cost trade-off earns equal-time strength
testing. R4 is the cheaper first study arm, not a selected champion. This is
established fixed-point/SIMD engineering, not a new chess algorithm, Stockfish
parity or scholarly novelty claim.
