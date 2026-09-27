# Low-clock recovery after the repaired NNUE confirmation

## Gate evidence

The repaired shared-feature NNUE faced classical evaluation on the same
stack-safe executable for 800 paired games at `30+0.3`, one thread and 128 MiB
hash per side, concurrency six, seed 202609140. The candidate scored
`365-229-206` (58.5%), reported as +59.6 +/- 20.9 Elo with 100% LOS. There were
no crashes, disconnects, or illegal moves, so the stack repair held under the
full workload.

The technical gate nevertheless failed because four games ended on time. Two
forfeits belonged to NNUE and two to classical evaluation. This rules out NNUE
inference as the common cause. The predeclared `90+0.9` stage correctly did not
start. Preserved evidence is under:

```text
E:\Dev\Forklift-Research\matches\nnue-stack-overnight-20260914
```

## Diagnosis

Reconstructing the four clocks from PGN elapsed-time annotations showed the
same failure mode on both evaluators. Once the remaining clock fell below one
second, repeated moves often consumed more than the 300 ms increment. The
clock therefore oscillated downward until the rounded elapsed totals crossed
zero. PGN times are rounded and do not prove the exact transport latency, but
they do establish cumulative clock depletion rather than an evaluator-specific
crash or illegal move.

The existing 25 ms reserve protected a single move but did not force recovery.
At low remaining time, normal complexity scaling and soft-limit extensions
could continue allocating approximately an increment or more, leaving no way
to rebuild a safety cushion under six-process Windows scheduling load.

## Repair

`pickClockTimeBudget` now enters a recovery regime when positive-increment time
falls below roughly two increments plus the configured reserve. In that regime:

- the hard search limit is capped by both one third of safe remaining time and
  one half of the increment;
- the soft limit is capped at half of that hard limit;
- ordinary allocation resumes once the clock rebuilds above the threshold;
- zero-increment controls retain their previous sudden-death policy;
- negative and extreme UCI increment values are clamped before arithmetic.

This intentionally trades search depth for survival only in an emergency clock
state. It does not change fixed-depth search, evaluation, or move ordering.

## Verification

- Optimized MSVC Release core and UCI tests passed.
- The six-position perft suite passed through depth 4.
- The full 79-test Python suite passed in `.venv-nnue`.
- Windows AddressSanitizer core and UCI tests passed with the MSVC runtime on
  `PATH`.
- Unit tests cover one-increment and sub-two-increment recovery, zero/negative
  increments, and an extreme integer increment without overflow.
- The expanded raw UCI regression passed 90/90 sequential probes over clocks
  from 0 to 625 ms, one and two search threads.
- Six simultaneous regression processes passed 324/324 additional probes with
  no overrun. Final tested engine SHA-256:
  `46dd22252731764f9cfcfd367e7a0d5fa232906937295f569c84c6ad988b57f1`.

Final raw evidence is under:

```text
E:\Dev\Forklift-Research\diagnostics\low-clock-recovery-final-20260927.json
E:\Dev\Forklift-Research\diagnostics\low-clock-recovery-final-parallel-20260927
```

## Real-search smoke

Before the arithmetic-only increment hardening, the repaired budget also ran a
100-game paired shared-NNUE-versus-classical smoke at `5+0.05`, one thread,
128 MiB hash, concurrency six, seed 202609270. Both sides used the same binary
and differed only by evaluation backend. The NNUE scored `44-35-21` (54.5%),
reported as +31.4 +/- 61.3 Elo, with zero time forfeits, crashes, illegal moves,
or disconnects. This small fast match is technical and directional evidence,
not a promotion-strength estimate.

Evidence is preserved under:

```text
E:\Dev\Forklift-Research\matches\nnue-low-clock-recovery-smoke-20260927
```

## Decision

Retain the recovery budget as a correctness fix. The shared NNUE remains
disabled by default: promotion still requires a fresh, clean `30+0.3`
confirmation followed by the predeclared longer-control stage. The failed
800-game result remains failed evidence despite its positive playing score.
