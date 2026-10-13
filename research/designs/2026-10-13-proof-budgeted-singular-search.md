# New Forklift design: Proof-Budgeted Singular Search (PBSS)

Status: implemented experimental project design; not promoted, no novelty or
playing-strength claim. This name labels a reproducible research hypothesis,
not an assertion that no previous engine has used a related policy.

## Research question

Can a rolling admission quota for singular-verification work preserve useful
tactical extensions while leaving more computation for the ordinary search?
Unrestricted singular search is the control. Singular extensions themselves
are established, not our invention: see Anantharaman, Campbell and Hsu,
[1988](https://journals.sagepub.com/doi/10.3233/ICG-1988-11402) and
[1990](https://www.sciencedirect.com/science/article/pii/0004370290900739).

The broader idea of allocating computation between control and execution also
has prior art: Horvitz and Breese's
[metareasoning resource partition](https://erichorvitz.com/metapart.htm).
This preliminary scan establishes overlap, not originality. A publication
requires a deeper literature/code review and a credible empirical result.

## Designed mechanism

Maintain a per-move, per-worker ledger:

- `T`: all visited main and quiescence nodes.
- `V`: nodes visited inside singular verification, charged once per proof.
- `O = max(T - V, 0)`: ordinary work, including the extended main search.
- `p`: configured soft proof quota as a percentage of ordinary work.
- `Q = 256 + floor(p * O / 100)`: bootstrap plus earned quota.

An otherwise eligible and legal TT candidate is admitted only when `V < Q`.
An admitted proof runs to its normal completion or existing time/abort stop.
Its main and quiescence work are then added to `V`, including work from a
proof interrupted by the overall search deadline. No incomplete proof earns
an extension. Unrestricted mode (`p=0`) preserves the prior policy.

There is deliberately **no mid-proof local cancellation**: that would require
an additional incomplete-search state threaded through all cutoffs to avoid
publishing invalid TT bounds. The prototype instead stops admitting proofs
while its ledger is in debt and resumes after ordinary work earns capacity.

This is a **soft admission target**, not a hard percentage/time bound. A
single expensive admitted proof can overshoot. Node types have unequal costs,
so node share is not wall-time share. The unchanged four-ply path limit still
applies. Each root worker has its own ledger and 256-node bootstrap; parallel
aggregate allowances therefore depend on worker count and scheduling.

The implementation exposes `Singular Proof Budget` (UCI, 0–100, default 0)
and `--singular-budget PERCENT` with `--singular` in the tools benchmark.
It logs proof nodes and denied admissions. Neither the feature nor its budget
is enabled in the default champion.

## Safety and validity

- Proof work never funds itself, preventing an accounting feedback loop.
- Quota arithmetic divides before multiplying and saturates at the maximum
  unsigned counter rather than wrapping.
- No ordinary search is skipped solely because a proof was denied.
- Excluded-root cache isolation, canonical legal TT identity, non-mate/depth
  guards, NNUE parity, deadline handling and path limits remain unchanged.
- Unit checks include exact quota exhaustion, injected debt, live charging,
  large counters and full state restoration.
- Automated tests also require activation before in-flight cancellation and
  actual multiworker verification/extensions; pre-entry abort alone is not
  enough evidence for recursive unwinding.

## Diagnostic protocol fixed before the first PBSS grid

- Same committed executable and frozen shared NNUE for all modes.
- Existing four fixed positions, depth 12, one thread, 128 MiB TT, 60,000 ms
  ceiling, NNUE weight 100, ProbCut off.
- Singular policy unrestricted (`0`) versus quota percentages `2`, `5`, `10`.
- One diagnostic run per mode in ascending percentage order; all reach depth
  12. This is a descriptive pilot, not a powered timing estimate.
- Retain complete stdout, moves, scores, both node counts, proof ledger,
  attempts/extensions/skips, wall time and binary/network checksums.
- Record all modes, including unfavorable/inconclusive outcomes. Do not select
  the best pilot as a proven champion or claim equal-time strength from depth.
- Repeat unrestricted mode against the prior singular implementation to check
  that adding a ledger does not change its deterministic single-thread tree.

Timing remains exploratory while the independent model confirmation uses the
same host. A later equal-time, paired on/off study must freeze its own contract
and assess tactical retention, technical failures and strength. Tactical and
endgame holdouts plus independent opponents are needed before generalizing.

## What could become a research contribution?

A contribution would be a measured relationship between proof-work allocation,
tactical retention and equal-resource strength, backed by transparent negative
results and ablations. A named scheduler or a lower node count alone is not a
publication-level contribution. For a CV, accurately describe designing and
implementing an experimental compute-budget scheduler and reproducible audit
pipeline; do not claim a new state-of-the-art chess algorithm.

## First descriptive pilot: mixed/negative evidence preserved

The predeclared grid completed at depth 12 on the same certified-output
executable and model. The full
[raw/parsed record](../experiments/2026-10-13-proof-budgeted-singular-pilot.json)
is the source of these figures:

| Quota | Main nodes | Proof nodes | Extensions | Denied proofs | Main-tree change |
| --- | ---: | ---: | ---: | ---: | ---: |
| unrestricted | 1,302,166 | 47,167 | 68 | 0 | control |
| 2% | 1,096,251 | 23,233 | 38 | 215 | -15.81% |
| 5% | 1,320,972 | 45,268 | 62 | 4 | +1.44% |
| 10% | 1,302,166 | 47,167 | 68 | 0 | identical |

The 2% setting changes two root moves, including the endgame. We do not know
whether that is improved efficiency or lost tactical information. The 5%
setting makes the tree slightly larger despite skipping four proofs; selective
search is not monotonic in this quota. The 10% setting does not bind here and
reproduces unrestricted moves/scores/node counts.

An important limitation of the original hypothesis is now measured: raw
verification accounts for only **2.94% of all nodes** in the unrestricted
pilot. Extension depth and changes to ordering/TT history can therefore
dominate its extra tree cost. Reducing proof work alone need not recover most
of that cost. These observations motivate tactical holdouts and a separate
extension-cost study, not selecting a quota from four positions.

No setting is promoted, no Elo gain is claimed, and all outcomes are retained.
Timing samples remain exploratory under concurrent match load.
