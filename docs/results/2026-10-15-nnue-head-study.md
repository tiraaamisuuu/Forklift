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

Pending. No architecture, default model or release is promoted by this plan.
