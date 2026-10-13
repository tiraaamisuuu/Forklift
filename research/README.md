# Forklift Research Notebook

This directory preserves compact, reviewable research metadata. Large raw
datasets, models, PGNs and generated experiment directories belong in the
external archive at `E:\Dev\Forklift-Research`, not in Git.

Engine development now keeps a paper-oriented experiment notebook alongside
the strength work. Each substantial experiment records its hypothesis, fixed
contract, artifact hashes, outcome, and limitations, including failed and
interrupted runs. Preparing this evidence does not establish novelty.

## Current research signal

The five-million-position corpus reached 92.50% input coverage, but its original
centipawn candidate scored only 32.5% against classical evaluation over 400
games. A later WDL candidate reached 84.12% validation score-sign accuracy but
only 43.8% in a noisy 40-game screen. Together, these results suggest a useful
question: which offline dataset and model metrics actually predict downstream
playing strength?

The working paper direction is:

> How should a compact NNUE evaluator allocate a fixed data-generation and
> teacher-search budget to maximize playing strength on consumer hardware?

This is only a direction. Novelty must be established by a proper literature
review before any publication claim.

The first absolute-strength campaign produced a local 50% crossing near 2321.5
over 1,200 games at `10+0.1`. The slower campaign completed 4,200 games at
`30+0.3`: Forklift scored 50.4% against Stockfish 2500 and 43.3% against 2550,
bracketing the interpolated crossing at approximately 2503. Six
Stockfish-side timeout results favoured Forklift at 2500; excluding them leaves
an exactly level `269-56-269` and a conservative sensitivity estimate of 2500.
The time-control difference is itself a research signal and must not be
collapsed into a universal Elo claim.

## Repository contents

- [`experiments/`](experiments/) stores compact experiment definitions committed
  before new runs, including the October 12 long-control recovery contract.
  The [fused NNUE update record](experiments/2026-10-13-nnue-fused-update.json)
  also retains all twenty fixed-depth timing outputs, executable hashes,
  exact-tree parity, and shared-host limitations for the October 13 optimization.
  The [singular-extension prototype record](experiments/2026-10-13-singular-extensions.json)
  preserves the subsequent search policy, on/off activation diagnostics,
  real-network incremental/rebuild parity and CI evidence. It is an
  implementation result, not a playing-strength result.
- [`RESEARCH_QUESTIONS.md`](RESEARCH_QUESTIONS.md) records hypotheses and
  possible new methods without treating them as established inventions.
- [`DATA_POLICY.md`](DATA_POLICY.md) defines what is collected and where it
  lives.
- [`data/matches.csv`](data/matches.csv) is the compact match-result registry.
- [`data/stockfish_calibration.csv`](data/stockfish_calibration.csv) records the
  rung-level absolute-strength calibration.
- [`data/nnue_runs.csv`](data/nnue_runs.csv) records comparable NNUE data,
  validation and inference measurements.
- [`champion.json`](champion.json) freezes the exact promoted engine revision,
  binary checksum, calibration evidence, and default SPRT contract.
- `scripts/generate_research_figures.py` regenerates the checked-in SVG figures.

## Regenerate the figures

```powershell
py -3 scripts\generate_research_figures.py
```

The generated figures are derived summaries. The dated files under
`docs/results/` remain the human-readable evidence, while external manifests,
PGNs and binaries remain the source artifacts.
