# Optional NNUE residual head: implementation, not a strength result

Date: 2026-10-15. Initially a source-only milestone; a subsequent user-authorized
ten-minute CPU validation window produced the results below. No production V2
network, strength improvement or Stockfish parity is claimed. Tiny artificial
training fixtures are correctness checks, not model-quality experiments.

Subsequent [clip-once/certified residual SIMD work](2026-10-15-nnue-residual-simd.md)
substantially reduces the scalar cost below and records a tiny real-label
head-learning pilot. It does not establish a strength gain or promote V2.

Implementation commits: `b17437c` (runtime and deferred C++ fixtures), `40ed977`
(training/export, warm starts and deferred Python fixtures). These commits are
local; no push, CI trigger or automatic schedule was started.

## Motivation and prior art

The current model clips two HalfKP accumulators, concatenates them in
side-to-move order, and applies one linear output. This already has nonlinear
feature interactions inside each clipped accumulator; it does not have a
learned nonlinear layer after their concatenation.

The new optional branch can learn nonlinear combinations across those signals.
It deliberately keeps the existing direct output as a residual/skip path:

```text
x = concatenate(clip(own_accumulator), clip(opponent_accumulator))
prediction = direct(x) + residual_output(clip(residual_hidden(x)))
```

The residual output has no bias and starts with zero weights. Thus an enabled
branch initially adds zero. Its hidden weights are nonzero so that it can begin
learning after output coefficients move away from zero. Initialization uses an
isolated CPU RNG context so enabling the branch does not perturb baseline
parameters or subsequent data-order RNG. Those initially deferred properties
passed the later CPU Python validation described below.

This follows established small dense-layer, clipped-activation, residual and
fixed-point techniques. The [Stockfish NNUE guide](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
explains the accuracy/cost trade-off, accumulator perspective ordering and
quantization scaling. It also warns that extra layers can add little accuracy,
hurt quantization and cost runtime. That warning is part of our hypothesis, not
something a larger model automatically overcomes. No Stockfish implementation
was copied and no algorithmic novelty claim is made.

## Scope and compatibility

- Existing `HalfKpV1(hidden, feature_factorization)` calls and direct models
  remain the default. `--residual-hidden 0` exports the existing V1 format.
- `--residual-hidden R`, with `1 <= R <= 32`, requests the experimental branch;
  first proposed configurations are 4 and 8, not a maximal network.
- Python training/export and C++ loading/inference now implement the new V2
  format. Existing V1 magic, version, payload and direct arithmetic are retained.
- The feature set, accumulator refresh/update logic and network inputs are
  unchanged. Existing datasets remain usable; a new trained V2 export is needed.
- Existing champion artifacts, packaged executables and default evaluator are
  untouched. No model or release has been promoted by this implementation.

## Exact wire contract

All fields are little-endian. C++ retains the existing native little-endian
reader assumptions used on supported x86/ARM hosts; big-endian portability is
not newly established.

| Field | V1 | V2 |
| --- | --- | --- |
| Magic | `TNNUE1\0\0` | `TNNUE2\0\0` |
| Version | u32 = 1 | u32 = 2 |
| Common header | feature count u32, H u32, A i32, O i32, output bias i32 | Same |
| Extra header | None | R u32, residual weight scale W i32 |
| Common payload | hidden bias i32[H], transformer i16[40960,H], direct output i16[2H] | Same |
| Extra payload | None | residual bias i32[R], residual matrix i16[R,2H], residual output i16[R] |

The common header including magic is 32 bytes; V2's header is 40 bytes. Matrix
rows are contiguous. The loader checks magic/version pairing, H in 1..1024,
R in 1..32, positive signed-int32 scales, complete payload and V2 trailing bytes.
Clear or load failure resets all format/residual state. V1's old trailing-byte
acceptance is deliberately unchanged rather than silently changing its parser.

## Integer arithmetic

Let clipped accumulator lanes `x_i` be integers in `[0,A]`, and use normalized
float activations in `[0,1]` during training. Quantize:

```text
residual_matrix_q = round(residual_matrix_float * W)
residual_bias_q   = round(residual_bias_float * A * W)
residual_output_q = round(residual_output_float * targetScale * O)
```

The independent reference computes each residual neuron in int64:

```text
z_j = residual_bias_q[j] + sum_i(x_i * residual_matrix_q[j,i])
r_j = clamp(truncate_toward_zero(z_j / W), 0, A)
numerator = direct_bias_q + direct_dot(x) + sum_j(r_j * residual_output_q[j])
score_cp = clamp(truncate_toward_zero(numerator / (A * O)), -32000, 32000)
```

The branches must be added **before** the final division. Dividing them
separately changes rounding, particularly when their signs differ. No floats
or dynamic residual allocations are used during C++ evaluation. The direct
branch retains its existing certified SSE2/32-bit/64-bit dispatch. The first
residual implementation used portable int64. The later linked optimisation
certifies exact SSE2/32-bit residual rows, retaining int64 fallback/reference;
the kernel name describes the selected paths.

With H<=1024, R<=32, signed-int32 positive scales and int16 coefficients, both
the widest residual dot and the total output numerator are below 2^58 in
magnitude; `A*O` is below 2^62. These conservative algebraic bounds fit int64.
V2 also requires, for every transformer lane:

```text
abs(hidden_bias_q) + 64 * max_feature(abs(transformer_weight_q)) <= INT32_MAX
```

Widen before absolute values so INT16_MIN/INT32_MIN are handled. At most 64
square features can be active, and actual two-king boards have fewer; this
bounds refresh and ordinary valid incremental partial sums. V1's parser
acceptance remains unchanged. Python rejects non-finite/out-of-range
quantization before opening the destination network for writing; finite-value
and signed-int32 scale checks also protect direct exports.

## Fair warm starts and bounded experiments

`--initialize-from MODEL.pt` loads an exported tensor state dictionary using
`weights_only=True`, not a pickle-enabled optimizer checkpoint. It hashes the
exact bytes decoded, validates keys/shapes/dtypes/finite values before changing
the model, and either loads a complete matching head or allows a complete new
zero-output residual head on matching base weights. Partial/incompatible heads
are rejected, not silently dropped. It starts a new optimizer.

The raw `.pt` has no target-scale metadata. Preserving its float function is
not proof of preserving centipawn values: experiment commands must retain the
source's documented target and export scales. The current baseline uses target
scale 600, hidden scale 1024 and output scale 32 in the factorization runner.

`--freeze-transformer` freezes embeddings, shared factorization and hidden bias;
only the direct/residual heads are optimized. It requires initialization or a
resume and is a cheap pilot option, not proof that freezing is best. Freeze,
topology and residual-scale settings are checked on resume; legacy checkpoints
default to direct/unfrozen. Initialization provenance survives checkpoint resume.

Before writes, source weights are protected against output, metrics JSON/CSV,
manifest, progress, temporary and periodic checkpoint paths, including existing
hard links. No output path may overwrite the warm-start source. This does not
guard against another process changing filesystem links during a training run.

## Cost, rejection criteria and next execution gate

At H256, the residual hidden matrix alone adds 2,048 MACs/evaluation for R4 or
4,096 for R8; its output adds R more. These are arithmetic counts, not measured
latency. Additional serialized bytes are 4,128 for R4 and 8,248 for R8 including
the extra header, versus roughly 20 MiB of existing transformer weights.
Small file overhead does not imply small runtime overhead.

When the user explicitly makes the PC available:

1. Run V1 regressions and new C++ normal/scalar residual suites, sanitizers and
   Python training/export/initialization/path fixtures. Any failure blocks use.
   Rebuild a separate validation executable first: the frozen confirmation
   executable predates V2 and must not be overwritten or used to verify V2.
2. Verify actual exported Python/C++ scores for real boards, not only separate
   hand-built fixtures. Establish integer/full-rebuild/incremental agreement.
3. Freeze a pilot's data/split identities, common source model/scales, seeds,
   training-update budget and checkpoint rule. Compare identically fine-tuned
   direct controls with R4/R8; frozen-transformer and full fine-tuning are
   separate experiments. Do not compare a fine-tuned head with a stale control.
4. Measure quantization error and end-to-end CPU cost. Reject a candidate whose
   offline advantage disappears at equal time; do not promote on lower loss.
5. Only selected candidates receive predeclared paired strength confirmation.
   Broader opponent/time-control tests and multiple training seeds are needed
   before generalizing. No arbitrary Elo prediction or match extension is used.

The separate [v3 data-selection study](../../research/designs/2026-10-15-nnue-v3.md)
still matters: better information at unchanged inference cost remains valuable.
Implementing this bounded architecture branch does not establish that it should
take precedence over better data, nor does it implement the RIFT collector.

## Deferred validation inventory

- Two CTest targets, normal and forced-scalar. Written numeric fixtures specify
  10,560 independent-reference comparisons per target, plus malformed formats,
  clipping/rounding, reload, STM ordering and special-move/incremental checks.
- Thirteen residual Python tests cover gradients, baseline initialization/RNG,
  exact file layout, independent integer reference, ranges and legacy resume.
- Warm-start/freeze and source/dataset path fixtures cover source preservation
  and configuration safety. Python parsing and Git whitespace checks passed.

The inventory above was deferred at implementation time. The following bounded
session supersedes that deferral for the specifically executed checks only.

## Authorized validation window: October 15, 11:15–11:25 UTC

- Release rebuilt successfully, at most two build workers; all 13 CTests passed
  in 8.64 seconds, including normal/scalar V1 and residual fixtures.
- All 70 Python checks passed with CUDA disabled and CPU pools capped at one:
  paths 16, residual 13, initialization 7, training 17 and dataset 17.
- New export integration: 12 configurations gave 48 exact Python/C++ scores;
  tiny CPU training/export/resume added eight, for 56 exact comparisons. Both
  integration tests passed. Frozen transformer weights and initialization
  provenance survived resume. Training labels were artificial test fixtures.
- Focused MSVC Windows ASAN residual test passed (2.16 seconds), no finding.
  The first attempt could not start because its runtime DLL was absent from
  PATH; using the existing installed DLL directory fixed the process-local
  environment. This is not a full sanitizer/cross-platform result.
- The current Release tools matched the existing V1 model's four stored Python
  reference scores: 43, 38, -5 and -67 cp.
- The existing shared V1 confirmation passed all 61 evidence-verifier checks,
  including hashes, exclusion/recovery lineage and runtime network loading.
  Its recorded 800/1,200-game gains are evidence for that model only. The default
  evaluator and release were not changed.

No production dataset was relabelled and no production network was trained.
GPU training, new strength matches and cross-platform gates remain outstanding.
See the [machine-readable validation record](../../research/experiments/2026-10-15-nnue-v2-validation.json).

### Cost pilot: correct, but currently too expensive

The existing trained H256/shared model was re-exported directly and with
zero-output residual branches R4/R8; direct bytes matched the recorded SHA.
Depth 10, one thread, 128 MiB hash, weight 100, ProbCut/singular off, one warmup
and three interleaved measured trials per mode gave:

| Mode | Median four-position time | Median paired elapsed ratio |
| --- | ---: | ---: |
| Direct | 681 ms | 1.00x |
| R4 | 1,425 ms | 2.09x |
| R8 | 2,157 ms | 3.17x |

All 12 runs had identical moves/scores and 321,287 main / 76,248 quiescence
nodes. This small shared-host diagnostic measures overhead, not strength.
The scalar residual repeatedly clamps lanes and performs int64 dot products;
clamp/pack reuse and certified SIMD are next implementation targets. Skipping
zero-output neurons would make this synthetic benchmark look better without
measuring trained nonzero-head cost, so it is not the proposed solution.
Keep V2 experimental; optimize before production training. Raw timing values
and hashes are retained in the validation record.
