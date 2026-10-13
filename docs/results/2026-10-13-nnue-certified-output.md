# Certified NNUE output reduction — 2026-10-13

## Engineering contribution

The existing HalfKP output loop uses a dependent 64-bit sum for every hidden
lane in both perspectives. The new loader certifies, once per network,

```text
B = abs(outputBias) + hiddenScale * sum(abs(outputWeights))
B <= INT32_MAX
```

Because every clipped activation is between zero and `hiddenScale`, this
conservative absolute bound covers every product and every partial reduction,
not merely the final result. A certified network can use exact 32-bit sums.
When `hiddenScale <= 32767`, baseline SSE2 also permits signed-16-bit
activation packing, clipping, paired products and four-lane reduction. Odd
hidden widths use a scalar tail. Uncertified networks retain 64-bit output;
non-SSE2/ARM/ARM64EC targets use the portable certified-32 or general-64 path.

The method does not change network format, weights, clipping, rounding or
scores. AVX2/native architecture flags are not required. Packing before
clipping is exact only under the scale guard; the certificate rules out
overflow in pair sums, vector lanes and horizontal partials. Absolute values
of extreme biases/weights are widened before computation.

This is established integer/SIMD optimization plus an explicit safety
certificate in this engine, **not** a claim of a novel chess algorithm.
See Microsoft's [x64 architecture defaults](https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64),
[intrinsics reference](https://learn.microsoft.com/en-us/cpp/intrinsics/x86-intrinsics-list),
and [ARM64EC predefined macros](https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros).

## Correctness and portability

`nnue_output_tests.cpp` independently computes scalar 64-bit reference scores
for 31 fixtures and 63,488 accumulator/perspective comparisons per build.
Both normal and forced-portable builds run in CTest. Coverage includes widths
1–1,024 and odd tails, scales 32,767/32,768/INT32_MAX, signed extreme biases
and activations, int16-minimum weights, certificate boundary/+1 dispatch,
invalid accumulator/parent rebuilding, clear, reload and failed load.

Independent review found the arithmetic sound and identified ARM64EC's
shared `_M_X64` macro. The guards now explicitly exclude ARM64EC rather than
selecting untested software SSE emulation. A Windows ASAN run of the kernel
fixtures passed. Full cross-platform/UBSAN validation remains part of CI.

The test-only `FORKLIFT_NNUE_FORCE_REFERENCE` configuration and
`chess-engine-tools-nnue-reference` target expose the original general 64-bit
calculation inside the **same current search implementation**, avoiding a
timing comparison confounded by unrelated search changes. The production
kernel choice appears in the benchmark header as `nnueOutput`.

## Measurements and frozen protocol

Exploratory five-round microbenchmarks observed roughly 405 ns versus 54 ns
for public accumulator inference on the shared 256-wide model. This is
isolated evaluation, **not** engine throughput or Elo. A benchmark harness is
kept at `tests/nnue_kernel_bench.cpp`; no microbenchmark result is used alone
to make a playing-strength claim.

The end-to-end protocol is fixed before its paired sample:

- Current normal and forced-reference tools, identical Release MSVC build,
  native architecture off, IPO off, same search/evaluation settings.
- Shared network SHA-256
  `ff25cf447dc6137591ecc2522d23f14e2f9bee0a641dda36ec2cd9c67a762c7c`.
- Four existing positions, depth 12, one thread, 128 MiB TT, 60,000 ms ceiling;
  NNUE weight 100, ProbCut off, singular extensions off.
- One warm-up per executable, then ten pairs, alternating reference/candidate
  order. Require identical moves/scores/main/quiescence counts at every position
  and full prescribed depth in every trial.
- Preserve raw stdout and hashes; report median total time and paired
  throughput ratio. Do not turn one favorable run into a speed claim.
- Record concurrent confirmation load as a limitation and repeat on an idle
  host before publication. No frozen confirmation binary is replaced.

## End-to-end result

The ten pairs ran from 09:43:43 to 09:44:30 UTC on source
`abaffa97aa8c5cc11c9fd16484864246d0609754`. All twenty runs reached depth 12
and matched moves, scores, main nodes and quiescence nodes at all four
positions (969,129 main nodes / 218,408 quiescence nodes per run).

Median total reported time fell from **2,112.5 ms to 1,719.5 ms** (**18.60%
lower**). The median paired throughput ratio was **1.2325** (**23.25% higher
throughput**), with all ten pairs favoring the candidate. Different median
aggregations and reciprocal time/throughput metrics must not be conflated.

This is an additional identical-tree NNUE search gain over the previously
integrated fused accumulator updates, not an Elo result or a universal CPU
speed claim. All trials, hashes, paired ratios and limitations are preserved
in [the experiment record](../../research/experiments/2026-10-13-nnue-certified-output.json).

[CI run 38378258649](https://github.com/tiraaamisuuu/Forklift/actions/runs/38378258649)
passed all six jobs for the measured source. The sanitizer script now runs
the entire CTest suite, including the new output/scalar, proof-budget and
in-flight/parallel tests, rather than only the original core executable.
The timing remains preliminary under six-game concurrent confirmation load.
