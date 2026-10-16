# Conservative legal generation and end-to-end NNUE adaptation

Date: 2026-10-16. Host: Ryzen 9 5900X, RTX 3070, Windows/MSVC.
These are two separate interventions: an exact systems optimisation and a
controlled learning experiment. Neither is an architectural novelty claim or
evidence of Stockfish parity. Faster fixed-depth search is not a measured Elo
gain; better offline Brier loss is not a measured playing-strength gain.

## Why these interventions

The previous frozen-transformer head experiment failed its preregistered
quality screen. Repeating that experiment with more output neurons or more
epochs is not the immediate priority. The selected shared H256 source also
already shows validation deterioration under prolonged original-rate training.

Two concrete hypotheses remain worth testing:

1. `genLegalMoves` currently makes/undoes every pseudo move, although search
   makes selected legal moves again. Conservative pin-based acceptance can
   eliminate redundant first-pass work without changing the search tree.
2. A small learning rate on the whole feature transformer may permit useful
   adaptation that a frozen-transformer head cannot achieve. A matched direct
   control is necessary to attribute any gain to the residual head.

The protected source is raw weights `ef3ca127...a52643`, deployed network
`ff25cf44...762c7c`. The full hashes are in the committed protocols. No default
network is replaced by launching this study.

## Legal-generation implementation and correctness

`1b394fa` introduces a mask of potentially pinned friendly pieces. Outside
check, non-king moves from unpinned origins can bypass the original make/undo
filter, except castling, en passant and king-target captures. All uncertain
cases retain the old filter. Missing/inconsistent king caches and invalid
sides also fall back. Pseudo generation, move order and metadata are unchanged.

An independent test oracle retains the original filtering logic. The completed
native AVX2+IPO Release validation compares 6,079 positions and 160,411 ordered
moves, including legal playouts, artificial boards, both-colour special moves,
eight pin directions and malformed-state cases. Six known perfts and all 15
CTests pass (22.20 seconds for the final suite). A focused MSVC AddressSanitizer
run passes in 3.08 seconds with no findings. This is substantial regression
coverage, not exhaustive proof over every board state.

Review found an additional malformed-FEN interaction after the initial suite:
an occupied EP target is accepted by the legacy loader, and legacy EP undo can
erase that occupant, changing pins mid-list. The partial first timing run was
stopped, preserved as
[`2026-10-16-legalgen-aborted-cost.json`](../../research/experiments/2026-10-16-legalgen-aborted-cost.json),
and excluded from the final speed estimate. `21f9ffb` adds a whole-generator
reference fallback for occupied EP targets and three explicit regressions.
This preserves old malformed-input behaviour; it does not repair that separate
loader/undo issue. The full correctness suite and ASAN were rerun afterward.

The final performance protocol is
[`2026-10-16-legalgen-protocol.json`](../../research/experiments/2026-10-16-legalgen-protocol.json).
It uses same-source/configuration uninstrumented reference/candidate builds,
seven interleaved pairs separately for NNUE and classical evaluation, four
depth-12 positions, and 16 additional seeded full-FEN UHO positions at depth 10.
All moves, scores, depths and printed tree counters must match exactly. One
instrumented run per evaluator/build attributes work; its timer overhead is
not used as the speed estimate. No project builds/training/tests run during
the final timing window. The parser has 13 independent stdlib fixtures.

The final [raw measurement](../../research/experiments/2026-10-16-legalgen-cost.json)
completed in 91.02 seconds. All 100 benchmark invocations passed the exact-tree
checks, including the 16 extra positions under both evaluators. The seven NNUE
pairs were all faster: median candidate/reference search time 0.883416,
or 11.66% less time / 13.20% higher identical-tree throughput. Classical also
won all seven pairs: ratio 0.946117, 5.39% less time / 5.70% higher throughput.
The preregistered primary gate (>=2% less NNUE time and >=6/7 faster pairs) passes.

Instrumented NNUE counters show 8,151,725 pseudo candidates in both versions:
full make/undo probes fell to 1,132,814, with 7,018,911 fast acceptances (86.10%).
Instrumented legal-generation time fell from 402.29 to 196.50 ms, but these
timers include overhead and are attribution only. The speed estimate above
uses the separate uninstrumented runs. It is local, one-thread, fixed-depth
evidence, not an equal-time playing-strength estimate or hardware-wide claim.

## Corrected feature-coverage interpretation

The previous 37,890/40,960 raw-slot count does not imply 3,070 useful missing
features. Necessary occupancy rules exclude 2,656 allocated HalfKP slots:
640 anchor-king overlaps + 2,048 back-rank pawn slots - 32 overlaps. The
remaining denominator is 38,304, without a claim of complete legal reachability.
New diagnostics retain legacy counts, add excluded activations and seen-only
rarity, and explicitly distinguish the denominators. Twenty dataset fixtures
pass. Historical JSON evidence is not overwritten.

## Whole-network training design

The frozen
[`joint protocol`](../../research/experiments/2026-10-16-nnue-joint-protocol.json)
uses all 5,000,000 training positions, two complete epochs per arm, and direct
versus R8 heads across seeds 202610161 and 202610162. AdamW transformer/head
learning rates are 1e-5/1e-4, with weight decay 1e-6. Batch size is 2048,
teacher-WDL scale 400 and target scale 600. Both seed-matched arms consume the
same exact row permutations, whose canonical int64 hashes are preserved.

The feature transformer is trainable, and validation re-encodes inputs through
the current model. Cached activations are never used in the forward pass.
Hash auditing may read cache file bytes, including activation files; that is
not training from those activations. Epoch zero is evaluated, saved and eligible
for selection, so degrading adaptation cannot displace the source checkpoint.

Selection uses the previous cache's fixed 32k validation membership. A new 32k
diagnostic excludes games and effective inputs in that cache's validation and
test partitions; its labels are accessed only after all four selections are
sealed. The source was originally selected on the parent validation shard, so
this is **not** a globally untouched final test. Game-level separation is not
a certificate of no repeated board inputs across all training data.

Every selected model gets its own exact quantized transformer calculation,
integer output evaluation and four C++ export probes. Paired uncertainty uses
1,000 whole-game bootstrap resamples of seed-averaged loss differences. It
measures sampled-game uncertainty conditional on these two training seeds,
not broad training-seed uncertainty. No cherry-picking the better seed.

Direct adaptation requires at least 1% relative validation improvement in
each seed, an aggregate float diagnostic 95% interval wholly below zero,
improved integer diagnostic direction and exact C++ probes. R8 additionally
must beat its matched trained direct control by 1% in each seed and pass the
same diagnostics; direct adaptation itself must pass. Inference cost and a
separately preregistered strength screen remain required before promotion.

Eleven tiny CPU fixtures cover actual transformer updates, selectable epoch
zero, matched initialization/order, sealed labels, per-model quantization,
source preservation and fail-closed incomplete runs. A tiny CUDA deterministic
forward/backward smoke passed on the RTX 3070; it was not corpus training.

The protocol launcher pins dataset/model/cache hashes and sizes, records the
exact command, Git/dependency hashes and Python/NumPy/Torch/GPU environment
before starting the child. The study has a 2,700-second cooperative budget;
an external child watchdog permits 30 seconds for cleanup before stopping only
that child process tree. Wrapper preflight is recorded separately. There is no
automatic extension, resume, default-model replacement or strength match.
Six stdlib wrapper fixtures pass, with all real child launches/GPU probes mocked.

Full training outcome: pending at the time this protocol record was created.

## Follow-on decision

If the adaptation screen fails, preserve the negative result and stop that
branch rather than lower its gate. Prioritise better teacher/data error coverage
and controlled representation experiments. A deeper-label experiment must
recover full original positions from PGN/FEN: the existing compact record omits
castling rights, EP, clocks and history, so reconstructing it is not a clean
same-position teacher-budget comparison. Any RIFT claim needs real, measured
teacher compute and matched random/rarity controls, not only a toy selector.
