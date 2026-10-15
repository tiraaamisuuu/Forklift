# Matched frozen-transformer head study

Protocol frozen before cache creation and substantive training on October 15,
2026, after the user explicitly made the PC available for testing. This follows
the [residual inference optimisation](2026-10-15-nnue-residual-simd.md).
No result is asserted by this initial protocol.

## Question and predeclared budget

Does a four/eight-neuron residual head improve teacher agreement enough to
justify extra evaluation cost? Compare identically fine-tuned direct/R4/R8
heads on a fixed H256/shared transformer, not a newly trained head against a
stale direct control. Cache float32 clipped activations once to avoid repeatedly
recomputing the unchanged sparse feature transformer.

Use 250,000 training positions and 32,000 validation / 32,000 diagnostic-test
positions, two training seeds, 20 epochs per arm, AdamW learning rate 1e-3,
batch 2,048, weight decay 1e-6, no scheduler, WDL probability-MSE. All arms
start with the same direct function and see the same batches for each seed.
Checkpoint selection includes epoch zero, so a damaging fine-tune can retain
the original model. Every arm completes the same update budget or the study
is marked incomplete. Measure actual wall/GPU costs rather than claiming equal
epochs imply equal compute.

Verify fresh source hashes and game-ID separation. Current validation/test
partitioning is by game; selected effective inputs are also audited across
partitions, including colour-flipped feature equivalence. The source network
was historically selected using the original validation corpus: the new
subdivision is not a globally untouched test set. Preserve that limitation.

An architecture earns a short equal-time strength screen only if its mean
relative validation loss improves at least 1% versus the equally trained direct
controls, both seeds improve, the game-cluster diagnostic-test confidence
interval supports the direction, actual integer exports retain the improvement,
and correctness/cost gates pass. Otherwise preserve the negative result and
do not launch hundreds of games merely because a new model exists.

The conditional strength screen is 120 paired-colour games per training seed,
240 total, at `3+0.03`, one thread, 128 MiB hash, four concurrent games. It is
exploratory, not promotion. Do not extend it opportunistically or translate an
offline loss improvement into an invented Elo figure.

See the [frozen machine-readable protocol](../../research/experiments/2026-10-15-nnue-head-study-plan.json)
for exact source identities, seeds, selection rules and execution caps.

## Independent implementation work

MSVC disassembly of the existing residual evaluator shows an 8,256-byte stack
frame and a stack probe on every residual call, including ordinary H256/A1024
models that need only 1 KiB of narrow activation storage. Compact noinline
scratch helpers are being evaluated separately. Any measured speed improvement
must use identical weights and search trees; it is not a training-quality gain.

## Execution and outcomes

Completed on October 15. Cache builder `93de1f8`; study runner `c0fed7c`.
The original protocol remains unchanged. No residual model is promoted and
the conditional 240-game screen was **not launched**.

### Data and cache checks

Fresh full-shard hashes matched the predeclared identities. A full game-ID
audit found no overlap between 1,812,069 training-source and 224,934
holdout-source game IDs. The selected cache has 233,138 training games,
27,806 validation games and 27,804 diagnostic-test games. Selected effective
inputs are unique/disjoint, including colour-mirrored equivalents; 11 training,
two validation and two test duplicates were skipped before filling exact counts.
This is not a full-source effective-input audit.

Cache creation took 29.125 seconds. An independent 3.304-second check verified
all 27 array hashes and all 314,000 source-row board/turn/game records, then
recomputed 16 evenly spaced rows per split against the original CPU float
model: zero activation and direct-score differences from the GPU cache.
See [cache manifest](../../research/experiments/2026-10-15-nnue-head-cache.json)
and [verification](../../research/experiments/2026-10-15-nnue-head-cache-verification.json).

### Matched learning result: insufficient evidence to deploy

All six arms completed 20 epochs / 2,460 updates each, five million training
position presentations per arm. Transformer tensors remain bit-identical,
initial predictions agree exactly, and every exported model passed its four
Python/C++ FEN checks. Twenty epochs is the update budget, not the chosen
checkpoint: validation selection retained epoch zero for both direct controls.

| Seed | Head | Selected epoch | Validation Brier | Float test Brier | Integer test Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| 20261015 | Direct | 0 | 0.0097638812 | 0.0096863301 | 0.0096871652 |
| 20261015 | R4 | 12 | 0.0097259307 | 0.0097002044 | 0.0097033025 |
| 20261015 | R8 | 8 | 0.0097010193 | 0.0096800544 | 0.0096759278 |
| 20261016 | Direct | 0 | 0.0097638812 | 0.0096863301 | 0.0096871652 |
| 20261016 | R4 | 19 | 0.0096672143 | 0.0096546313 | 0.0096517146 |
| 20261016 | R8 | 10 | 0.0096894248 | 0.0096909557 | 0.0096897630 |

Mean relative validation reductions were 0.689% for R4 and 0.703% for R8,
both below the frozen 1% screen. Architecture selection was sealed as **none**
before test labels were opened. On diagnostic test, seed-averaged paired
float-loss differences (residual minus direct) and whole-game-bootstrap 95%
intervals were:

- R4: -0.000008912, interval [-0.000052785, +0.000037311].
- R8: -0.000000825, interval [-0.000050589, +0.000047073].

Both include zero. Full-test exact integer evaluation (192,000 predictions,
not rounded cached float activations) also gives intervals spanning zero.
Integer-versus-float score RMSE ranges from 0.797 to 1.200 cp across arms;
maximum absolute discrepancy is 7.760 cp. This does not establish equivalence
on arbitrary positions, but there is no integer-export mismatch in the checks.

The study took 60.394 seconds, including exporting, C++ probes, full-test integer
evaluation and 2,000-resample game-cluster intervals. Peak allocated GPU tensors
were about 681 MiB; this excludes desktop/driver memory. TF32 was disabled and
deterministic Torch algorithms enabled. Per-arm CUDA event spans include CPU
dispatch gaps and must not be called active GPU compute time. Full histories,
selection hash, network/weight/per-sample hashes, memory and timing are in the
[study result](../../research/experiments/2026-10-15-nnue-head-study-results.json).
The larger arrays/models remain outside Git in the recorded research directory.

Interpretation: the extra heads can learn, but this frozen-transformer setup
does not demonstrate a reliable benefit worth its inference cost. Do not
lower the predeclared threshold, select the lucky seed, promote an offline
winner, or start matches despite the failed screen. The cached training and
integer-evaluation infrastructure is reusable; stronger representation/data or
joint-transformer training are separate future hypotheses, not results here.

### Compact scratch experiment: rejected

Prototype `f0b74c0` reduced the measured MSVC H256 narrow stack allocation from
8,256 to 1,096 bytes and removed its stack probe. Correctness passed all 14
Release CTests, then expanded normal/scalar/reference fixtures (76 numeric
cases and 19,584 oracle comparisons each, 58,752 total) and focused Windows
ASAN (31.24 seconds, no finding).

Five interleaved paired timings did not show an end-to-end win: candidate/old
median paired ratios were 1.0072 for R4 and 1.0296 for R8. Direct control was
0.9812, demonstrating remaining timing variability. All 36 warmup/measured
trees matched. The small sample does not establish a precise slowdown, but
does not justify adopting the added specialisations. Original arithmetic and
scratch layout were restored in `f632217`, retaining the new boundary tests;
all three expanded residual targets passed again (15.88 seconds).
The [raw negative result](../../research/experiments/2026-10-15-nnue-compact-scratch-cost.json)
and prototype remain in Git history. Structural improvements are not assumed
to be speed improvements.
