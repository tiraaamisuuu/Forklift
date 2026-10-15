# Research and engineering contribution ledger

This ledger separates newly implemented project designs from established
techniques and measured outcomes. Names are not novelty evidence. Neither
Stockfish parity nor a state-of-the-art result is claimed.

## New project designs and prototypes

| Contribution | What exists | Evidence and limits |
| --- | --- | --- |
| [Proof-Budgeted Singular Search (PBSS)](designs/2026-10-13-proof-budgeted-singular-search.md) | A new Forklift admission scheduler allocating ordinary search work to singular verification; exact work ledger, UCI option and bounded arithmetic. | Implemented and correctness-tested. First grid includes fewer-node and worse-cost outcomes, changed moves and a nonbinding setting. No strength gain or literature novelty established. |
| [RIFT-v1](designs/2026-10-13-rift.md) | The prior RIFT hypothesis is now runnable: rarity/instability/disagreement-based teacher allocation, planned-node budget, transitive game/holdout isolation and lineage. | 44 standard-library tests plus a deliberately signal-aligned toy demo. No real deep-label execution, trained-model improvement or Elo result. Active/cost-sensitive learning overlaps are documented. |

PBSS was designed during the October 13 work. RIFT's working hypothesis
already existed in `RESEARCH_QUESTIONS.md`; its selector/audits are the new
implementation. A paper must review prior art beyond the scoped scans and
compare appropriate baselines across independent data/training seeds.

## Established techniques: implementation and measurement status

- [Optional residual-head NNUE](../docs/results/2026-10-15-nnue-residual-head.md):
  independently implemented small nonlinear head, V2 fixed-point format,
  Python/C++ inference paths, strict warm starts and frozen-transformer controls.
  CPU correctness/export checks now pass. The initial scalar R4/R8 cost pilot
  took 2.09x/3.17x direct time. Subsequent
  [clip-once/certified SIMD](../docs/results/2026-10-15-nnue-residual-simd.md)
  reduced old residual elapsed time by 49.56%/64.35% on identical trees;
  costs are now about 1.09x/1.21x direct in a small shared-host sample.
  An eight-update CPU training pilot found negligible quality differences
  versus an equally trained direct control. No production candidate,
  model-quality/strength gain or scholarly novelty claim; retain promotion gates.
  A subsequent [two-seed GPU head study](../docs/results/2026-10-15-nnue-head-study.md)
  completed matched 20-epoch arms on an audited 250k/32k/32k activation cache.
  Its sub-1% validation gains and inconclusive game-cluster test intervals failed
  the predeclared screen, so no matches or promotion followed. This is useful
  negative research evidence and reusable tooling, not a claimed strength gain.
  The other entries below have their own, separate measured evidence.

- [Certified NNUE output kernel](../docs/results/2026-10-13-nnue-certified-output.md):
  load-time absolute-bound proof, exact SSE2/32-bit output, portable 64-bit
  fallback. Ten identical-tree pairs showed median paired throughput +23.25%
  on four positions under concurrent confirmation load. This is a systems
  implementation/measurement contribution, not a new chess algorithm.
- [Fused accumulator deltas](../docs/results/2026-10-13-nnue-fused-updates.md):
  an earlier independent ten-pair sample showed about +20.5% paired throughput.
  Do not multiply separately measured gains into a claimed combined result
  without a direct comparison under one protocol.
- [Singular extensions](../docs/results/2026-10-13-singular-extensions.md):
  established selective deepening, implemented opt-in with excluded-root TT
  isolation, legal TT identity, path limits and NNUE parity. No equal-time
  strength promotion yet.

## Accurate CV wording

Example implementation statement:

> Designed and implemented experimental compute-budgeted chess search and
> leakage-aware NNUE teacher sampling in C++17/Python, with exact vectorized
> inference, cross-platform sanitizers and reproducible experiment manifests.

Example measured statement:

> Measured a 23% additional NNUE search-throughput gain across ten paired,
> identical-tree four-position benchmarks on a Ryzen 9 5900X; preserved raw
> evidence and shared-host limitations.

Do not describe PBSS/RIFT as novel published algorithms, the toy payoff as a
training result, faster search as a measured Elo gain, or Forklift as stronger
than Stockfish. Contributions are collaborative, AI-assisted development; any
paper must follow its venue's authorship/disclosure policies.

## Next substantive experiments

1. Collect audited cheap signals from a real training-only pool and execute
   random/rarity/RIFT label plans under recorded **actual** teacher compute.
2. Evaluate PBSS on tactical/endgame holdouts, then compare useful candidates
   at equal time. Do not tune on the permanent test set or partial scores.
3. Replicate certified inference timing on idle and other hardware; use
   equal-time matches only to assess strength, not to validate bit-exact speed.
