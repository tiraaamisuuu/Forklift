# Forklift

Forklift is a high-performance C++17 chess engine combining modern alpha-beta
search with classical and custom trainable NNUE evaluation. The project includes
a local chess interface, UCI support, CUDA/PyTorch training, Stockfish teacher
labelling, incremental NNUE inference, and reproducible strength testing.

`C++` · `UCI` · `NNUE` · `CUDA / PyTorch` · `SFML`

<!-- HERO IMAGE
Recommended: a clean screenshot of the web interface during an interesting
middlegame, with the board, evaluation, PV and engine identity visible.
Suggested size: 1600 × 900 (16:9), WebP or PNG.
Save as: docs/assets/engine-room-hero.webp
Enable with: ![Forklift interface](docs/assets/engine-room-hero.webp)
-->

## Highlights

- Iterative-deepening alpha-beta/PVS with transposition tables, quiescence,
  null-move pruning, LMR, aspiration windows, SEE and history-based ordering
- Correct legal move generation with castling, en passant, promotion,
  repetition and rule-draw handling
- Standalone UCI engine plus a responsive local web interface for PvP, PvC,
  CvC, external engines, live analysis and game export
- Minimalist SFML desktop app with smooth piece motion, original move sounds,
  live search telemetry and manual or position-aware automatic time limits
- Optional versioned HalfKP NNUE with quantized C++ inference and incremental
  per-ply accumulators
- Resumable Stockfish labelling from streamed Lichess `.pgn.zst` archives and
  CUDA training with exact Python/C++ export verification
- Perft, deterministic core tests, CI, fixed-position benchmarks, paired
  opening matches, Stockfish calibration, and a frozen-champion SPRT gate

<!-- ENGINE ARCHITECTURE VISUAL
Recommended: restrained monochrome SVG showing UCI/web input → board state →
iterative search → classical or NNUE evaluation → best move/PV, with TT and
time management as supporting components.
Suggested size: 1400 × 760 (approximately 16:9).
Save as: docs/assets/engine-architecture.svg
Enable with: ![Engine architecture](docs/assets/engine-architecture.svg)
-->

## Performance

All figures below are documented development measurements. The Stockfish
ladder is a qualified local-pool calibration rather than a universal or human
Elo claim. Full hardware, commands, commits and uncertainty are preserved in
[`docs/results/`](docs/results/).

| Measurement | Result | Interpretation |
|---|---:|---|
| Stockfish 18 calibration, 4,200 paired `30+0.3` games | approximately 2503 | Bracketed by 50.4% at 2500 and 43.3% at 2550; timeout-excluded sensitivity is exactly 2500 |
| Stockfish 18 calibration, 1,200 paired `10+0.1` games | approximately 2321.5 | Interpolated 50% crossing in this hardware-, opening- and time-control-specific limited-strength pool |
| v1 vs `v0.4.0`, 400 paired `10+0.1` games | `299–18–83` | 85.1%, +303.0 +/- 37.1 Elo, 100% LOS; release strength gate passed |
| Depth-10 search work | 270,343 → 186,618 nodes | Qsearch SEE pruning plus continuation history; changed tree |
| Incremental null move | 5,108 → 4,951 ms | 3.1% median reduction with identical tree |
| Incremental NNUE accumulator | 13,457 → 46,010 median NPS | 3.42× over full accumulator rebuild on the same smoke network |
| Six-thread root search | 1.51×–1.68× benchmark throughput | No proven playing-strength gain; one thread remains default |

The retained qsearch SEE change scored `38–28–34` in its 100-game diagnostic.
Continuation history scored `37–31–32` and remains provisional pending a
slower, longer match. Pawn correction history and a selection-scan move picker
were rejected after neutral or negative measurements.

![Forklift v1 release match results](docs/assets/release-strength.svg)

### Calibrated strength and informal game review

The slower confirmation campaign and its extensions completed 4,200 games at
`30+0.3`. Forklift scored 50.4% (`274-57-269`) against Stockfish 2500 and 43.3%
(`236-48-316`) against 2550, placing the interpolated 50% crossing at
approximately **2503** in this pool. The 2550 rung had zero technical
failures. Six Stockfish-side timeout results at 2500 favoured Forklift;
excluding them gives an exactly level `269-56-269` and a conservative
sensitivity estimate of **2500.0**. The defensible shorthand is therefore
approximately **2503 local-pool Elo** under this exact contract.

![Forklift slow-control Stockfish calibration curve](docs/assets/stockfish-calibration-slow.svg)

Forklift completed a 1,200-game Stockfish 18 limited-strength ladder at
`10+0.1`. It scored 56.5% against the 2200 rung and 45.8% against 2400, placing
the interpolated 50% crossing at approximately **2321.5** in this exact local
pool. All six rungs completed with zero crashes, time forfeits, illegal moves,
or disconnects. This is the project's first reproducible strength estimate,
but it is not an official FIDE or Chess.com rating and should travel with its
hardware, opening suite, time control, sample size, and uncertainty.

![Forklift Stockfish calibration curve](docs/assets/stockfish-calibration.svg)

One manually relayed game against Chess.com's Magnus Carlsen bot received a
Chess.com game review of **90.6% accuracy**, a **2750 single-game performance
rating**, and **zero misses or blunders** for Forklift. The bot side received
96.4% and 3000. This is encouraging anecdotal evidence from one game—not a
repeatable benchmark or proof that Forklift is 2750 Elo. The measured results
that can currently be reproduced are the approximately 2321.5 fast-control
crossing, the bracketed 2503 slower-control crossing, and the +303.0 +/- 37.1
relative Elo release match above.

<!-- PERFORMANCE VISUAL
Recommended later, once another release-grade match exists: a simple line or
step chart of measured candidate score across tagged engine revisions. Include
sample size and uncertainty; do not mix NPS and Elo on one axis.
Suggested size: 1400 × 800 (7:4).
Save as: docs/assets/strength-development.svg
Enable with: ![Measured engine development](docs/assets/strength-development.svg)
-->

## NNUE

The new shared-feature candidate won two clean 400-game screens: approximately
+83 Elo versus the original NNUE and +104 versus classical at 30+0.3. These are
local relative estimates, not an absolute rating. Independent-seed and longer
time-control confirmation precede any default change.
[Experiment and results](docs/results/2026-09-12-nnue-feature-sharing.md).

The custom `HalfKP-v1` pipeline turns licensed game archives into deterministic,
game-disjoint datasets, labels sampled positions with Stockfish, trains on
CUDA, quantizes the network, and verifies exact C++ predictions before match
testing. It supports target-sized generation, cross-machine shard merging,
feature-coverage audits, teacher-budget comparisons and sliced validation
errors.

The first production experiment retained five million Stockfish-18-labelled
training positions and 616,632 game-disjoint validation positions from the
July 2026 Lichess CC0 archive. It covered 92.5% of the HalfKP inputs, passed
quantization and exact C++ verification, and completed a 400-game match without
technical failures. The fixed 256-wide network still lost decisively to the
classical evaluator and was rejected. A WDL objective and configurable hybrid
evaluation improved the diagnostic results but did not establish a strength
gain, so classical evaluation remains the default.

![NNUE v1 offline and playing metrics](docs/assets/nnue-research-baseline.svg)

The mismatch between strong-looking offline percentages and weak match play is
now preserved as a research baseline. The backburner
[research notebook](research/README.md) records machine-readable results,
candidate hypotheses and a reproducible figure generator without treating an
eventual paper as a current development priority.

<!-- NNUE PIPELINE VISUAL
Recommended: horizontal SVG of Lichess/self-play → sampling → Stockfish teacher
→ compact dataset → CUDA/PyTorch → quantization → C++ verification → paired
match → promote/reject.
Suggested size: 1800 × 700 (18:7).
Save as: docs/assets/nnue-pipeline.svg
Enable with: ![NNUE training and promotion pipeline](docs/assets/nnue-pipeline.svg)
-->

## Quick start

For Windows, download
[`Forklift-1.1.0-Windows-AMD64-GUI.zip`](https://github.com/tiraaamisuuu/Forklift/releases/download/v1.1.0/Forklift-1.1.0-Windows-AMD64-GUI.zip),
extract it, and run `bin/Forklift.exe`. The archive includes the required SFML
and OpenAL runtime files.

Launch the local web interface:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_web_gui.ps1
```

On macOS/Linux, use `scripts/run_web_gui.sh`. The launcher creates its isolated
Python environment, builds the headless engine when necessary, and opens the
local interface.

Build the Forklift desktop app with SFML 2.6 available:

```powershell
cmake -S . -B build-gui -DCHESS_BUILD_GUI=ON -DBUILD_TESTING=ON `
  -DSFML_DIR=C:\path\to\SFML\lib\cmake\SFML
cmake --build build-gui --config Release --parallel
.\build-gui\Release\Forklift.exe
```

The desktop app includes smooth piece motion, original sound cues, live search
telemetry, and manual or position-aware automatic time controls.

Build and test the headless engine directly:

```sh
cmake -S . -B build -DCHESS_BUILD_GUI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

The principal executables are `chess-engine-uci`, `chess-engine-tools`, and
`chess-core-tests`.

## Documentation

- **[Technical Wiki](wiki/Home.md)** — architecture, search, evaluation, NNUE,
  building, testing and development workflows
- [NNUE training and promotion](docs/NNUE.md) — exact production commands
- [Benchmarking and strength testing](docs/BENCHMARKING.md) — paired matches,
  calibration and SPRT
- [Development status](docs/DEVELOPMENT.md) — current configuration and release
  gates
- [Ultimate roadmap](docs/ULTIMATE_ROADMAP.md) — the unbounded path from the
  current engine to a continuously improving research platform
- [Research notebook](research/README.md) — compact evidence, research questions
  and reproducible figures for possible future academic work
- [Raw experimental results](docs/results/) — permanent reproducibility record

The Wiki source is versioned in this repository and mirrored to the project's
GitHub Wiki, so documentation changes remain reviewable with the code.

## Roadmap

The calibrated post-v1 build is frozen in `research/champion.json`, and the
candidate-versus-champion SPRT gate is now automated. The immediate sequence is
to confirm the retained search gains at slower time controls, profile and
improve move ordering, build a stronger NNUE v2, then prove multi-thread
strength.

Beyond that, the project expands into distributed SPRT testing and tuning,
large-scale self-play, tablebases, SIMD/architecture-specific optimization,
and a separate policy/value plus MCTS research engine. The champion will never
be replaced merely because a new technique is fashionable: every candidate
must be correct, reproducible and stronger at a named resource limit.

The complete, deliberately open-ended plan is in the
**[Ultimate Roadmap](docs/ULTIMATE_ROADMAP.md)**.

The project follows one rule throughout: **implement → test → measure → keep or
reject → document**.
