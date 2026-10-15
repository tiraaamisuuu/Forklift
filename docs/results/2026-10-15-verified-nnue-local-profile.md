# Making the verified NNUE usable locally

Date: 2026-10-15. Web implementation `0ed395f`; build screen `88cb568`.
This closes a practical access gap, not a global/release promotion.

## What changed

The previously confirmed shared V1 model lived outside the repository and
was not selected by ordinary web startup. `--nnue` merely appended a selectable
experimental profile. UCI still defaults to classical; the native SFML GUI
does not yet wire NNUE into its search evaluator. Release packages contain no
approved network, and the revision-only champion registry was not repurposed.

The web server now accepts `--verified-nnue PATH` and
`--default-profile verified-shared-nnue`. The verified profile requires the
exact SHA-256 `ff25cf447dc6137591ecc2522d23f14e2f9bee0a641dda36ec2cd9c67a762c7c`,
rechecks it before activation, and requires a successful UCI load probe.
It configures weight 100%. Reserved profile IDs cannot be supplied by an
arbitrary local comparison manifest. Missing/wrong models or unknown defaults
fail clearly. Ordinary startup, experimental `--nnue`, and classical opt-out
remain available. Checksums do not eliminate a malicious concurrent file-swap
race; this is a local trusted-asset workflow, not a sandbox boundary.

The model's 800/1,200-game evidence and 61-check verifier result are preserved
in the earlier [validation record](../../research/experiments/2026-10-15-nnue-v2-validation.json).
The verified badge identifies that model; it does not pretend every supplied
development executable has newly passed its playing-strength confirmation.

The approved 20 MiB network was copied to the local ignored path
`networks/verified-shared-v1/model.nnue` and its digest rechecked. Source
weights/networks were not overwritten. No weights, release or global
promotion were published. A new checkout still needs the approved artifact.

## Local performance choice

Four otherwise identical MSVC Release builds were screened on the Ryzen 9
5900X: portable, native, IPO, native+IPO. One warmup per binary and three
interleaved paired trials per candidate, depth 12, four standard positions,
one thread, 128 MiB hash, NNUE 100%, incremental, ProbCut/singular off:

| Build | Median paired candidate/portable time |
| --- | ---: |
| Native only | 0.996902 |
| IPO only | 0.982405 |
| Native+IPO | 0.927536 |

All 22 runs have identical moves, scores, completed depth, main nodes and
quiescence nodes. The three native+IPO candidate/portable totals were
1,575/1,640, 1,602/1,769 and 1,664/1,794 ms. This is about 7.25% less
paired elapsed time in an exploratory best-of-three build screen, not
independent confirmation, Elo or portable-release performance evidence.
Do not multiply this with earlier speedups measured against other baselines.
The handwritten NNUE kernel remains SSE2; `/arch:AVX2` permits the compiler
to optimise surrounding code and `/GL` + `/LTCG` enable link-time optimisation.

The local native+IPO executable requires an AVX2-capable processor and must
not silently replace portable binaries on other machines. Historical frozen
build directories were preserved. Raw flags, executable/network hashes,
commands and outputs are in the
[build record](../../research/experiments/2026-10-15-nnue-native-ipo-cost.json).

## Checks and use

The native+IPO configuration passes all 14 CTests (24.07 seconds), including
58,752 normal/scalar/reference residual oracle comparisons. The measured
tools/UCI hashes stayed unchanged when testing was enabled. Actual source
network static scores match the stored references: 43, 38, -5 and -67 cp.

Twelve web profile fixtures pass. Both real-engine HTTP smoke modes pass:
ordinary classical startup and explicit verified-NNUE startup, legal moves,
analysis, PvP/PvC/CvC, import/export, and classical opt-out. The fake Stockfish
installer in the existing smoke test uses a local fixture archive, not a
download or a real Stockfish strength test. Fourteen new cache/study CPU
fixture tests also pass together (3.548 seconds).

On this Windows workspace:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_verified_nnue_web.ps1 -Native
```

Omit `-Native` for the local portable build. `-CheckOnly` verifies prerequisites
and prints the selection without starting a server; both preflight modes and
PowerShell syntax passed. The full HTTP smoke used the Python server CLI.
The launcher delegates environment/bootstrap handling to the existing web
launcher. No server is left running by validation.

See [setup and build commands](../NNUE.md#verified-shared-model-in-the-web-ui)
and the [machine-readable integration record](../../research/experiments/2026-10-15-nnue-local-integration-validation.json).
The separate [six-arm head study](2026-10-15-nnue-head-study.md) failed its
predeclared quality screen; none of its new models replaces this approved V1.
