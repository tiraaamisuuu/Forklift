# Clock handling after the staged-picker confirmation

## Completed gate

The stack-safe candidate `e90bfdbd476f5dce3fe744b5a5489dd582c3cffc`
faced champion `071b4b624629ead2326b8ff03cace89b0520aa1f` at `30+0.3`,
one thread, 256 MiB hash, concurrency six, seed 6003. Evidence is preserved at
`E:\Dev\Forklift-Research\matches\champion-gates\staged-move-picker-stack-safe-20260910-resume-001`.

- 3,391 scored games: candidate W-D-L **1064-1407-920**, score 52.1%.
- Relative Elo **+14.8 +/- 8.9**, SPRT H1 accepted (LLR 2.95).
- Zero crashes, disconnects and illegal moves; **one candidate time forfeit**.
- The five games still running at the SPRT boundary ended without a result.
- Gate decision: **technical_failure**. The champion registry is unchanged.

Game 798 ended at Black's 58th move, after `Nc8+`, with Black ahead according
to both engines' preceding evaluations. The final FEN was
`2N5/4kp2/1p4p1/4P3/p4P2/4P3/2K4p/4r3 b - - 3 58`.
The PGN records rounded thinking times but no raw UCI clock transcript, so it
does not establish the exact remaining clock or distinguish an engine overrun
from an operating-system scheduling delay. A replay of the position responded
normally with a positive clock. Do not claim that the original incident has
been deterministically reproduced.

## Confirmed defects and corrections

1. A raw UCI clock of zero selected the missing-clock fallback, allowing
   1000/1500 ms of search. The immutable candidate reproduced **1501.1 ms**
   with one thread and **1500.7 ms** with two. Zero now gets the existing
   emergency budget; only an absent clock selects the fallback.
2. Both search modes calculated soft-budget extensions from the previously
   extended budget. Repeated iterations could compound this toward the hard
   limit. Extensions now use the original allocation every time.
3. The search clock started inside the worker after UCI preparation and thread
   scheduling. It now starts when the `go` command is handled, and the first
   stop check examines the deadline before searching nodes.
4. Deadline polling remains amortized for normal searches, but checks every
   node for budgets at most 10 ms and every 32 checks up to 100 ms.

These address demonstrable time-handling problems; their playing-strength
effect still requires an independent match. The configured 25 ms transport
reserve remains in place. No software deadline can guarantee timely delivery
while the OS suspends its process.

## Verification

- Clean optimized MSVC Release build: core and UCI tests passed.
- Windows AddressSanitizer: core and UCI tests passed with the MSVC runtime
  directory on PATH. The first invocation without that runtime failed to load
  (`0xc0000135`), before running engine code.
- Raw UCI clock regression: 42/42 cases passed across 0, 1, 20, 50, 100,
  300 and 325 ms clocks, one/two threads, three repetitions.
- Deterministic tests cover an already expired worker deadline, legal fallback,
  board preservation and bounded soft-budget extensions in both search modes.
- Depth-12 diagnostic: all four best moves, scores, principal and quiescence
  node counts unchanged. Principal-node total **703,600** on both revisions.
  Optimized wall times were 2,121 ms versus 2,116 ms in single sequential samples;
  these are a sanity check, not a measured speed gain.
- 68 Python tooling tests passed. The dashboard now distinguishes a strength
  boundary from a gate blocked by technical failures.

Raw clock samples and executable checksums are under
`E:\Dev\Forklift-Research\diagnostics\clock-20260911`.

```powershell
py -3 tests/uci_clock_regression.py build-clock-safe/Release/chess-engine-uci.exe `
  --output E:/Dev/Forklift-Research/diagnostics/clock-20260911/after-release.json
```

## Fast-clock diagnostic

Committed candidate `8b9bd9a7a7e11368631ee3d0129000774573a448` played the
previous candidate `e90bfdbd476f5dce3fe744b5a5489dd582c3cffc` for 100 paired
UHO games at `2+0.02`, one thread, 256 MiB hash, concurrency six, seed 6101.
Both binaries were built from committed sources using the same MSVC Release
configuration.

- W-D-L: **53-27-20**, score **66.5%**.
- Relative Elo: **+119.1 +/- 61.0** in this fast-clock diagnostic only.
- Zero time forfeits, crashes, illegal moves or disconnects on either side.
- Full match: `E:\Dev\Forklift-Research\matches\clock-stress-20260911`.
- All six GitHub CI jobs passed for the candidate, including the new Windows
  raw UCI clock regression and Windows desktop packaging.

This is preliminary evidence that the revised allocation helps under time
pressure. It does not establish a +119 Elo gain at longer controls or a new
absolute rating. The predeclared next step is a new immutable `30+0.3`
champion gate with seed 6102, maximum 10,000 games and unchanged resources,
at `E:\Dev\Forklift-Research\matches\champion-gates\staged-picker-clock-safe-20260911`.
Promotion still requires a clean passing gate.
