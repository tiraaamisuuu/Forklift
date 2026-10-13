# Forklift: Ultimate Roadmap

This is the deliberately unfinished roadmap for taking Forklift as far as its
developers, hardware and ideas can carry it. It is not a promise that every
item will be implemented, and it is not a claim that the engine will reach a
particular rating. It is a sequence of increasingly ambitious, measurable
research programmes.

The north star is simple: **make the strongest correct, reproducible and usable
Forklift possible at clearly stated compute and time limits**.

## Permanent rules

Every strength change follows the same loop:

```text
idea -> invariant tests -> fixed-work benchmark -> short paired test
     -> statistically powered confirmation -> promote or reject -> archive
```

- Compare candidates with the current champion using identical openings,
  colors, hardware, threads, hash and time controls.
- Separate speed, node-count and playing-strength claims. None substitutes for
  the others.
- Preserve binaries, commits, manifests, PGNs, logs, checksums and decisions.
- Use SPRT or confidence intervals rather than stopping when a result looks
  attractive.
- Keep correctness ahead of Elo. A fast engine that occasionally plays an
  illegal move is a broken engine.
- Keep the best proven configuration as the default; experimental systems stay
  opt-in until they earn promotion.
- Report absolute Elo only against a documented, calibrated opponent pool.

## Horizon 0: establish the truth

Before optimizing against a guessed rating, establish a durable baseline.

- Run a calibrated Stockfish ladder at short and tournament-like time controls.
- Add a second independent engine pool to reduce opponent-specific bias.
- Publish hardware, operating system, compiler, opening suite, adjudication,
  concurrency, hash, threads and confidence intervals with every rating.
- Calibrate the desktop presets separately because Eco and Performance+ spend
  very different resources.
- Add tactical, quiet-position, endgame and time-management suites alongside
  match play.
- Build a regression dashboard covering correctness, nodes, NPS, time losses,
  crashes and relative Elo.
- Feed compact result summaries into the versioned research notebook while
  keeping large raw evidence in `E:\Dev\Forklift-Research`.
- Preserve the measured 2321.5 `10+0.1` crossing and bracketed 2503
  `30+0.3` crossing as separate local-pool anchors; never collapse them into a
  universal rating claim.

**Exit gate: achieved.** The project now has rerunnable rating evidence at two
time controls and a frozen post-v1 champion binary. Future rating work should
add independent opponents and desktop-preset calibration rather than repeat
the same Stockfish ladder without a specific hypothesis.

## Horizon 1: maximize the single-thread classical engine

The fastest path to a stronger hybrid engine is a stronger search and a strong
independent evaluator.

### Search

- Replace full move sorting with a genuinely staged and profiled move picker.
- Confirm and tune continuation history and quiescence SEE pruning.
- Singular extensions now have a bounded, opt-in implementation (2026-10-13),
  with excluded-root TT isolation and cross-platform correctness checks.
  Equal-time strength evidence remains outstanding; see the
  [implementation report](results/2026-10-13-singular-extensions.md).
- Add dedicated invariants before testing ProbCut, singular extensions,
  multi-cut ideas or more aggressive pruning.
- Explore context-aware LMR, history pruning, capture history, correction
  histories, improving flags and better extension/reduction interactions.
- Refine aspiration windows, mate-distance pruning, null-move verification,
  quiescence checks and transposition-table replacement.
- Improve principal-variation stability and make stop handling effectively
  instantaneous without corrupting completed iterations.
- Mine tactical failures and search explosions into permanent regression tests.

### Classical evaluation and time management

- Tune piece values, piece-square terms, mobility, king safety, pawn structure,
  passed pawns, space, threats and endgame scaling as a joint parameter system.
- Use automated tuning rather than accumulating hand-adjusted constants.
- Add phase- and volatility-aware time allocation, panic time, ponder support
  and robust behaviour under increments, delays and low-clock conditions.
- Calibrate every desktop automatic-time profile against both strength and
  latency goals.

**Exit gate:** each retained feature passes correctness checks and a properly
powered match; the combined candidate then beats the frozen v1 champion.

## Horizon 2: make the hot path hardware-efficient

- Profile before changing representations; optimize measured bottlenecks only.
- Evaluate bitboards and magic, PEXT or hybrid attack generation against the
  current representation.
- Make board state, move lists, histories and transposition entries cache-aware.
- Reduce unpredictable branches, allocations, copies and false sharing.
- Add SIMD kernels for NNUE and suitable board operations, with scalar fallbacks.
- Produce tuned x86-64, AVX2, AVX-512, ARM64 and Apple Silicon builds where the
  gain justifies separate artifacts.
- Measure link-time optimization, profile-guided optimization and compiler
  variants on real searches rather than microbenchmarks alone.
- Track nodes per joule and thermal throttling as well as nodes per second.

**Exit gate:** reproducible speed gains with an identical search tree, followed
by equal-time matches proving that the speed translates into strength.

## Horizon 3: production NNUE

The five-million-position HalfKP-v1 experiment proved the pipeline, not the
network. The next effort changes the information available to the model and the
quality of its data before simply multiplying training time.

- Oversample the measured unseen and low-frequency HalfKP features.
- Build stricter filters for duplicate, trivial, corrupted and highly biased
  positions while preserving game-disjoint validation.
- Compare teacher score, WDL/result and blended objectives with controlled
  ablations.
- Prototype two-perspective accumulators, king/input buckets, HalfKAv2-like
  features, threats and multiple material/PSQT subnetworks.
- Tune network width and depth against inference cost, cache pressure and Elo.
- Add quantization-aware training and optimized incremental accumulator refresh.
- Expand validation slices for tactical positions, endgames, king exposure,
  unusual material and rare features.
- Run automatic checkpoint gauntlets so weak networks stop consuming match
  resources early.
- Promote a network only after it beats the classical champion on more than one
  time control and remains stable across supported CPUs.

**Exit gate:** the first NNUE-enabled champion with a statistically supported
gain over the best classical build.

## Horizon 4: scalable parallel search

- Fully profile duplicate work, shared-TT contention and worker utilization in
  the existing root-parallel search.
- Run equal-resource 1/2/4/6/8/12+ thread scaling matches, not throughput-only
  benchmarks.
- Test diversified Lazy SMP, improved root splitting and carefully bounded
  shared search information.
- Partition or align hot shared structures to reduce contention and false
  sharing.
- Add NUMA-aware thread and memory placement on multi-socket hosts.
- Make hash, thread and affinity controls first-class UCI options.
- Explore multi-process and multi-node analysis only after one-machine scaling
  is sound.

**Exit gate:** more threads produce repeatable equal-wall-time Elo gains without
breaking determinism-sensitive tests or interactive stop behaviour.

## Horizon 5: build the data flywheel

- Generate large, diverse self-play corpora using paired openings and controlled
  randomization.
- Mix strong external-teacher labels, Forklift search labels and game outcomes
  only through measured ablations.
- Mine rare structures, tactical disagreements, horizon effects, endgames and
  positions where champion and candidate strongly disagree.
- Deduplicate across generations and maintain permanent holdout sets.
- Track dataset lineage from source game through sample, label, training run,
  quantized network and match decision.
- Continuously compare fresh-only, replay-buffer and curriculum strategies.
- Re-label strategically when a new champion makes the old teacher stale.

**Exit gate:** a reproducible cycle in which new data can create a candidate,
test it and either promote or reject it without manual bookkeeping.

## Horizon 6: distributed experimentation

Build a Fishtest-style service for Forklift:

- A coordinator queues tests, verifies builds and assigns paired games to
  trusted workers.
- Workers report PGNs, crashes, timing data, hashes and reproducibility metadata.
- SPRT stops decisive tests early; fixed-game confidence intervals remain
  available for release gates.
- SPSA or another suitable optimizer tunes coupled parameters against match
  outcomes.
- Resource quotas prevent speculative tests from starving confirmations.
- A public experiment ledger shows the hypothesis, patch, result and decision,
  including failures.
- Reproducible containers and signed/checksummed artifacts make remote results
  auditable.

**Exit gate:** contributors can donate CPU time and the platform can safely turn
it into trustworthy engineering evidence.

## Horizon 7: complete the practical engine

- Add Syzygy probing and exact tablebase-aware search and adjudication.
- Complete pondering, MultiPV, searchmoves, analysis mode and robust UCI option
  handling.
- Support Chess960 without weakening standard-chess correctness.
- Investigate opening-book learning while keeping book and engine strength
  separately measurable.
- Add import/export, analysis annotations, clocks, game history and accessible
  keyboard operation to the desktop client.
- Ship signed, reproducible installers and automatic update checks with a clear
  offline path.
- Maintain fuzzing, sanitizers, perft, protocol torture tests and long-running
  stability matches on every supported platform.

**Exit gate:** Forklift is dependable as a tournament engine, analysis engine,
desktop opponent and research platform.

## Horizon 8: a separate neural-search research line

AlphaZero- and Leela-style work should begin as a sibling experiment, not a
rewrite of the proven alpha-beta champion.

- Implement a policy-and-value network plus batched GPU inference.
- Add Monte Carlo Tree Search with correct virtual loss, batching and tree reuse.
- Generate self-play games, train from policy/value targets and repeatedly pit
  new networks against the current neural champion.
- Scale self-play workers and training independently behind versioned formats.
- Compare pure MCTS, alpha-beta and hybrid systems at equal hardware and time.
- Experiment with a policy network for alpha-beta move ordering and a value
  network for selective evaluation before considering full replacement.
- Research distillation between the alpha-beta and neural-search branches.

**Exit gate:** the neural branch wins a fair match or contributes a measured
component gain. Otherwise it remains valuable research without displacing the
main engine.

## Horizon 9: frontier-scale hardware and research

- Distribute self-play and testing across university, community and rented
  compute with strict cost and provenance tracking.
- Support efficient GPU training and inference backends without coupling the
  core engine to one vendor.
- Study larger networks, mixture-of-experts ideas, uncertainty-aware evaluation,
  learned time allocation and opponent-independent opening discovery.
- Explore distributed analysis, persistent analysis caches and collaborative
  opening-tree research.
- Optimize for laptops, desktops, servers and low-power ARM devices as distinct
  resource classes.
- Publish datasets, networks, tooling and negative results where licensing and
  storage permit.

**Exit gate:** each expensive programme demonstrates that its extra compute
buys useful strength, knowledge or accessibility.

## Horizon 10: assurance and longevity

- Add property-based and coverage-guided fuzz testing for move generation,
  FEN/UCI parsing, make/unmake and serialization.
- Differential-test rules and perft against independent trusted engines.
- Formally specify the most failure-prone state transitions where practical.
- Harden network, archive and model loaders against malformed inputs.
- Keep file formats versioned and migration-tested so old experiments remain
  usable.
- Automate reproducible releases, dependency review, provenance and long-term
  artifact checks.
- Document architecture well enough that a future maintainer can reproduce the
  current champion from source and evidence alone.

## The infinite loop

There is intentionally no final box named “done.” After the last listed horizon,
Forklift returns to measurement: find the largest remaining weakness, design the
smallest experiment that can test it, and promote only what survives. New
algorithms, hardware and research can be inserted into the same loop without
lowering its evidence standard.

The practical order from today is:

1. preserve the completed absolute-strength calibration as the baseline;
2. confirm the retained single-thread search work;
3. automate champion/candidate promotion;
4. build and test NNUE v2;
5. prove parallel-search strength;
6. distribute testing and tuning; and
7. begin the policy/value research branch when the data and compute platform can
   support it.

## External design references

- [Stockfish Fishtest](https://github.com/official-stockfish/fishtest) for
  distributed paired testing and statistical experiment management
- [Stockfish NNUE technical documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
  and [trainer](https://github.com/official-stockfish/nnue-pytorch) for sparse
  incremental networks, feature sets, quantization and checkpoint matches
- [Leela Chess Zero architecture](https://lczero.org/dev/lc0/architecture/),
  [project overview](https://lczero.org/dev/overview/) and
  [training runs](https://lczero.org/play/networks/) for a separated
  self-play/policy/value research system
- [AlphaZero paper](https://arxiv.org/abs/1712.01815) for the original general
  self-play reinforcement-learning programme across chess, shogi and Go
