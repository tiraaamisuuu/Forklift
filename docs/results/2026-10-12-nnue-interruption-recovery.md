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
| Original `series.json` | `92657068bbbcf9feea5200193eb80c635627e0e2c93d5efaa05d23b74b68f141d` |
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
