# Shared NNUE confirmation: interruption audit and recovery

## Observation on 2026-10-12

The September 27 series is preserved at
`E:\Dev\Forklift-Research\matches\nnue-low-clock-confirmation-20260927`.
No coordinator, match runner, or Cute Chess process for this series remains
alive. The long-stage log was last written on September 27; its `series.json`
still records `running`. The abrupt termination cause is unknown. There is no
evidence here establishing an engine defect, shutdown, or user action as the
cause.

The completed short stage contains 800 games at `30+0.3`, scoring 415 wins,
204 losses, and 181 draws (63.1875%; Cute Chess +93.9 +/- 21.7 Elo). Its four
technical-failure counters are zero. The executable and shared-network hashes
still match, and a fresh disposable UCI process explicitly loaded the network.

The unfinished `90+0.9` stage contains 387 logged finishes: 207 wins, 74 losses,
106 draws (67.1835%). No technical termination is recorded in those finish
lines. The PGN contains 389 completed result headers and four `*` results.
This discrepancy is consistent with incomplete log finalization but does not
establish its cause. Neither count constitutes the required 1,200-game sample.
The partial score is descriptive only; it is excluded from promotion evidence.

## Immutable input checksums

| Artifact | SHA-256 |
| --- | --- |
| Original `series.json` | `92657068bbcf9feea5200193eb80c635627e0e2c93d5efaa05d23b74b68f141d` |
| Long-stage `match.log` | `e88282fdca81ed015fde8b2c76b7776198cb34a090903374e6a68104f8cb8686` |
| Long-stage `games.pgn` | `c56227f132e81d710e4b70a01e2acb361d210b25b11c2b6c363effb0726d6ab1` |
| Frozen UCI executable | `b97bfa1199e5c3ff13b3d53cb713a3d21ff2f47cfae42c9035e3581ec050c2ff` |
| Shared network | `ff25cf447dc6137591ecc2522d23f14e2f9bee0a641dda36ec2cd9c67a762c7c` |

## Reliability repair

New series write a separate coordinator heartbeat every five seconds. Readers
require a heartbeat no older than 60 seconds and a live local process before
reporting a new run as active. A different host, malformed timestamp, or future
timestamp is unverified. Historical runs fall back to their active match-runner
PID. The verifier fails a dead series rather than reporting it indefinitely
pending; the dashboard derives its displayed state without rewriting evidence.
Terminal completed results remain valid after their processes exit.

Tests cover a fresh live coordinator, expired heartbeat with a reused live PID,
dead process, another host, malformed/future timestamp, legacy interruption,
terminal results, promotion refusal, and preservation of archived metadata.

## Recovery plan

Retain the complete 800-game result and run a new, full 1,200-game long stage
with a different predeclared seed. Preserve the original partial run unchanged.
Record the reused stage as reused evidence with source-artifact hashes. Require
the same frozen executable, network, paired opening pool, one thread, 128 MiB
hash per engine, concurrency six, `90+0.9`, zero technical terminations, and a
positive reported Elo lower bound. Do not aggregate the interrupted sample or
extend a losing/inconclusive sample until a desired boundary appears.

This is an engineering confirmation of one selected network. It does not isolate
the causal contribution of feature factorization, estimate absolute Elo, or
establish a novel learning method. A paper will need independent training seeds,
controlled ablations, and opening/opponent generalization beyond this gate.

### Committed recovery contract

The new contract is `shared-nnue-long-recovery-20261012`, seed `202610120`.
Its compact definition is in
`research/experiments/2026-10-12-shared-nnue-long-recovery.json`. The completed
short stage is referenced unchanged and revalidated against its source checksum;
it is not copied into a new PGN or counted as a fresh replication. The coordinator
refuses a live source, an invalid short result, a changed engine, or an existing
terminal long-stage result. The verifier recognizes only versioned contracts.

Launch after committing from a clean checkout:

```powershell
.\.venv-nnue\Scripts\python.exe -u scripts\nnue\run_strength_screen.py `
  --training-dir E:\Dev\Forklift-Research\models\nnue-v2-factorization-20260912 `
  --engine E:\Dev\Chess-Engine\build-nnue-recovery-confirmation\Release\chess-engine-uci.exe `
  --run-dir E:\Dev\Forklift-Research\matches\nnue-long-recovery-20261012 `
  --recover-long-from E:\Dev\Forklift-Research\matches\nnue-low-clock-confirmation-20260927
```

The new series records runner revision/dirty status, training-manifest hash,
source-series hash, reused-result hash, excluded log/PGN hashes, UTC start time,
and the exact match command. The existing match manifest adds host, opening,
Cute Chess, executable, UCI-option, and disposable activation-probe provenance.

### Validation and launch

The separate recovery workflow smoke completed four games at `2+0.02`, scoring
zero wins, two losses, and two draws, with zero technical terminations. Its
promotion verification correctly failed the fixed 1,200-game requirement and
positive lower-bound requirement. This result is recorded for completeness;
it is a workflow check and is excluded from strength evidence.

Smoke evidence:
`E:\Dev\Forklift-Research\matches\nnue-long-recovery-smoke-20261012`.

The full recovery started on October 12 at approximately 11:58 UTC from clean
runner commit `a71e93feee700e0b03f7a58f355f8947fae90d6f`, retaining the original
engine binary from `8cfb637`. It ran six concurrent games in
`E:\Dev\Forklift-Research\matches\nnue-long-recovery-20261012`.
The live verifier reported zero failed checks, two pending completion checks,
valid source lineage, and successful NNUE activation. The actual match manifest
confirms 1,200 games, `90+0.9`, seed `202610120`, equal 1-thread/128-MiB
resources, and successful configuration probes on both sides. No full result is
available at launch.

The frozen binary's core/UCI CTest gates passed before launch. All Windows,
macOS, Ubuntu, sanitizer, web, and desktop-package CI jobs passed for the
recovery runner commit. No strength or timing measurements from later builds
should be attributed to this frozen executable.

### Benchmark configuration audit

The audit also found that `runSearchBenchmark` and `chess-engine-tools` defaulted
to ProbCut on, while `SearchContext` and UCI play defaulted to off. Both benchmark
defaults now match production; `--probcut` explicitly selects the experimental
path and `--no-probcut` remains compatible. Dedicated CTest checks protect the
tools default, opt-in flag, and desktop benchmark default. This repairs
measurement configuration; it does not alter the frozen confirmation binary
or claim a playing-strength improvement.

### Explicit user pause and restart

The user requested a pause on October 12 at 19:26 UTC. The verified match
process tree was stopped, preserving the original files. The pause record
contains 676 games: 321 wins, 153 losses, 202 draws (62.4260%). The log and PGN
agree on 676 completed games, and no technical termination appears in the log.
The six in-flight games did not become completed result records. This is a
user-requested interruption; the coordinator's exit-1 `failed` metadata reflects
the stopped subprocess, not an inferred engine failure.

On the user's request to continue, the separately committed contract
`shared-nnue-long-resume-20261012` declares a new 1,200-game sample, seed
`202610121`. The exact definition and all four paused-artifact checksums are in
`research/experiments/2026-10-12-shared-nnue-long-resume.json`. Neither prior
partial long sample is aggregated into this gate. The previously completed
800-game short result remains referenced as reused evidence.

The prior partial score was visible before this restart. This history must
remain disclosed in any paper; the restart is not presented as the sole
attempt or as confirmation on an unseen opening distribution.

The runner's `--exclude-paused-series` flag records the explicit pause, checks
provenance, and selects the versioned new seed. The verifier refuses changed
pause artifacts, a missing pause record, or any existing terminal long result.
Status readers derive `paused` from the control record while retaining the
original failed metadata unchanged. Heartbeat writes also now retry temporary
Windows sharing violations; recovered telemetry contention does not turn a
completed clean match into a failure.

Launch from a clean committed checkout into a new directory:

```powershell
.\.venv-nnue\Scripts\python.exe -u scripts\nnue\run_strength_screen.py `
  --training-dir E:\Dev\Forklift-Research\models\nnue-v2-factorization-20260912 `
  --engine E:\Dev\Chess-Engine\build-nnue-recovery-confirmation\Release\chess-engine-uci.exe `
  --run-dir E:\Dev\Forklift-Research\matches\nnue-long-resume-20261012 `
  --recover-long-from E:\Dev\Forklift-Research\matches\nnue-low-clock-confirmation-20260927 `
  --exclude-paused-series E:\Dev\Forklift-Research\matches\nnue-long-recovery-20261012
```
