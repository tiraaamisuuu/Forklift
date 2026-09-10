# Engine and NNUE Match Testing

`scripts/compare_engines.py` builds two committed Git revisions in isolated
directories and runs a paired Cute Chess match. The repository stays on the
current branch throughout the process.

## First-time setup

On macOS or Linux, build the pinned Cute Chess CLI dependency:

```sh
scripts/install_cutechess.sh
```

The source installer needs Qt 6.8 or newer. On Windows PowerShell, the following
downloads and verifies the pinned official binary package instead:

```powershell
scripts\install_cutechess.ps1
py -3 scripts\compare_engines.py --quick
```

The comparison command automatically downloads and verifies the CC0
`UHO_4060_v4` opening suite from the official Stockfish books repository. The
bundled 24-position suite is used only by `--quick` installation checks.

## Compare the v1.0 release with the current mainline

```sh
scripts/compare_engines.py \
  --baseline v1.0.0 \
  --candidate main \
  --games 400 \
  --tc 10+0.1 \
  --threads 1 \
  --hash 256
```

Every opening is played with reversed colours. PGN games and the complete
match log are written under `artifacts/elo/`.

Each match directory also contains:

- `manifest.json` with engine paths, Git commits or external selectors,
  executable SHA-256 values, UCI identity/options, configured option values,
  opening and Cute Chess checksums, and match settings
- `result.json` with score, relative Elo and uncertainty when finite,
  termination reasons, crashes, illegal moves, disconnects, time forfeits, and
  process completion status

The reported Elo is a **relative difference**. For example, `+100` means the
candidate performed about 100 Elo above the selected baseline under those test
conditions; it does not assign either engine an absolute human-style rating.

Both refs must be committed because the builder uses `git archive`. Old refs
are handled automatically: before v1, UCI mode lived in the SFML `gui` binary;
new refs use the independent `chess-engine-uci` executable. Set `SFML_PREFIX` or
pass `--sfml-prefix` if SFML 2.6 is not in the normal CMake search path.

After the two binaries are built, they are also registered as local web GUI
profiles. Open **New game → CvC** to watch those revisions play each other, or
select either one as the opponent in PvC. This is useful for visual inspection;
use the paired Cute Chess workflow for meaningful strength measurements.

## Check the installation

```sh
scripts/compare_engines.py --quick
```

This runs four paired games at a short time control. It proves the binaries,
protocol, openings, and tournament runner work; four games say effectively
nothing about Elo.

## Compare external UCI engines

Either side can be an existing executable instead of a Git revision. Repeated
`--candidate-option` and `--baseline-option` values use `NAME=VALUE` syntax and
are validated against the options advertised by the engine's live UCI
handshake. Executable arguments can likewise be repeated with
`--candidate-arg` or `--baseline-arg`; external engines receive no implicit
arguments.

For example, calibrate the current revision against a pinned Stockfish binary:

```powershell
python scripts\compare_engines.py `
  --candidate HEAD `
  --baseline-exe C:\Engines\stockfish-windows-x86-64.exe `
  --baseline-name "Stockfish 18 @ 1800" `
  --baseline-version 18 `
  --baseline-option UCI_LimitStrength=true `
  --baseline-option UCI_Elo=1800 `
  --games 100 --tc 10+0.1 --threads 1 --hash 256
```

The runner discovers and records the engine's supported `UCI_Elo` bounds at
runtime. A limited-strength value is a calibration anchor in this local test
pool, not a universal human rating.

To compare resource configurations of the same executable, use the explicit
per-engine overrides. The manifest records both sides separately:

```powershell
python scripts\compare_engines.py `
  --baseline-exe .\build-pc-max\Release\chess-engine-uci.exe `
  --candidate-exe .\build-pc-max\Release\chess-engine-uci.exe `
  --baseline-name "Single thread" --candidate-name "Root split 6 threads" `
  --baseline-threads 1 --candidate-threads 6 `
  --baseline-hash 256 --candidate-hash 256 `
  --games 100 --tc 10+0.1 --concurrency 2
```

This gives both engines equal wall-clock time but intentionally gives the
candidate more CPU resources. Describe it as a scaling experiment, not a
same-resource engine comparison.

## Run a calibrated Stockfish ladder

`scripts/calibrate_rating.py` reuses the paired runner across several
limited-strength rungs:

```powershell
python scripts\calibrate_rating.py `
  --stockfish-exe C:\Engines\stockfish-windows-x86-64.exe `
  --stockfish-version 18 `
  --engine-ref HEAD `
  --rungs 1600,1900,2200,2500 `
  --games 100 --tc 10+0.1 `
  --threads 1 --hash 256 --concurrency 4 `
  --run-dir artifacts\rating\windows-10s
```

The ladder:

- resolves a Git candidate to one immutable commit before the first rung
- validates every rung against the live `UCI_Elo` minimum and maximum
- applies `UCI_LimitStrength=true` at every rung
- uses the same deterministic opening seed and resource settings throughout
- preserves per-rung PGN, logs, manifests, results, and driver output
- writes an aggregate `summary.json` and qualified `report.md`
- reports no point estimate when the observed scores do not bracket 50%

Repeat the exact command with the same `--run-dir` to resume. Completed rungs
are skipped. An incomplete rung attempt is retained unchanged and retried in a
new numbered attempt directory, so evidence is not silently overwritten.
Individual games inside an interrupted attempt are not spliced into the retry.

On Windows, the production pilot and its live dashboard can be started together:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_rating_calibration.ps1
```

The default contract is 200 paired games at each of 2200, 2400, 2600, 2800,
3000, and 3190, using `10+0.1`, one thread per engine, 256 MiB hash, and six
simultaneous games on the 12-core development host. Results are stored under
`E:\Dev\Forklift-Research\matches`; the dashboard is served on port 8766 and
keeps the computer awake without requiring the displays to remain on. Running
the launcher again resumes the same ladder rather than overwriting evidence.

When a ladder is extended in separate run directories, pass each earlier
directory with `-HistoryRunDir` to display one deduplicated campaign. The last
directory is the active run and wins if the same rung appears more than once:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_rating_calibration.ps1 `
  -RunDir 'E:\Dev\Forklift-Research\matches\current-extension' `
  -HistoryRunDir @(
    'E:\Dev\Forklift-Research\matches\original-ladder',
    'E:\Dev\Forklift-Research\matches\earlier-extension'
  )
```

To pin a live engine-vs-engine confirmation above the ladder, pass its run
directory with `-MatchRunDir` (or use `--match-run-dir` when starting
`calibration_dashboard.py` directly). The dashboard then shows its current
games, W/D/L, score, live Elo estimate, SPRT state, elapsed time, maximum ETA,
and match contract while the completed calibration remains below it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_rating_calibration.ps1 `
  -RunDir 'E:\Dev\Forklift-Research\matches\current-extension' `
  -HistoryRunDir @('E:\Dev\Forklift-Research\matches\original-ladder') `
  -MatchRunDir 'E:\Dev\Forklift-Research\matches\retained-search-confirmation'
```

Use `--quick` only to verify the ladder workflow; it selects four games per
rung at `2+0.02`. For a rating claim, use substantially larger samples and
repeat the ladder at a second time control. Even then, describe the result as a
rating anchor in the named local pool.

## Measure multicore scaling

`scripts/benchmark_threads.py` runs the fixed-position benchmark sequentially
at several thread counts and preserves the executable checksum, raw logs,
manifest, JSON, CSV, and Markdown report:

```powershell
python scripts\benchmark_threads.py `
  --engine .\build-pc-max\Release\chess-engine-tools.exe `
  --threads 1,2,4,6,12 `
  --repetitions 3 --time-ms 1000 --depth 64 --hash 256
```

The Ryzen 9 5900X diagnostic on 2026-07-24 found the best median node
throughput at six threads in a short 500 ms-per-position run: 1.677x the
one-thread result. Twelve threads fell to 1.565x. This identifies a scaling
bottleneck and a useful experimental setting; it does not demonstrate an Elo
gain. Keep one thread as the default until equal-time paired games prove a
multi-thread configuration stronger.

## Serious acceptance testing

The supported path is the frozen-champion gate. Inspect its exact revision and
default contract:

```powershell
py -3 scripts\champion_gate.py show
```

Run a committed candidate against that champion. The run directory must be new
or empty and should live in the external evidence archive:

```powershell
py -3 scripts\champion_gate.py run `
  --candidate <candidate-commit> `
  --run-dir E:\Dev\Forklift-Research\matches\champion-gates\candidate-001
```

The live dashboard can expose pause/resume controls for a displayed champion
gate:

```powershell
py -3 scripts\calibration_dashboard.py `
  --run-dir E:\Dev\Forklift-Research\matches\absolute-calibration-extension-2550-20260829 `
  --match-run-dir E:\Dev\Forklift-Research\matches\champion-gates\candidate-001 `
  --enable-match-controls `
  --port 8766
```

Pause terminates the gate process tree and preserves its partial PGN, logs and
manifest. Resume deliberately does not append to that interrupted SPRT sample:
it creates a new `-resume-NNN` sibling directory, increments the opening seed
and starts the same immutable champion/candidate contract there. This keeps the
buttons convenient without presenting stitched evidence as one experiment.
The controls accept same-origin requests only and remain disabled unless the
explicit flag is present.

The default contract permits up to 10,000 paired-opening games at `10+0.1`,
one thread and 256 MiB hash per engine, concurrency six, and an SPRT interval
of 0 to +5 Elo. A valid early SPRT boundary is classified as `promote` or
`reject`; reaching the maximum without a boundary is `inconclusive`. Any crash,
illegal move, disconnect, or time forfeit blocks promotion as a technical
failure. `--quick` runs a four-game workflow smoke and can only produce
`smoke_pass`, never strength evidence.

Every gate preserves its immutable manifest, underlying match manifest, PGN,
logs, strict JSON result, and a Markdown report. An interrupted directory is
never overwritten or silently resumed; preserve it and use a new directory.

Only after reviewing a `promote` result, update the versioned registry:

```powershell
py -3 scripts\champion_gate.py apply `
  --run-dir E:\Dev\Forklift-Research\matches\champion-gates\candidate-001
```

The apply step verifies that the tested baseline is still the registered
champion, records the new binary checksum and evidence hash, and leaves a normal
Git change for review. It does not commit or push automatically.

The lower-level runner remains available for bespoke experiments. For a
candidate expected to be at least five Elo stronger:

```sh
scripts/compare_engines.py \
  --baseline v0.4.0 \
  --candidate main \
  --games 10000 \
  --tc 10+0.1 \
  --sprt --elo0 0 --elo1 5
```

Keep the machine on AC power, close heavy background work, disable sleep, and
use identical thread/hash values. Concurrency changes throughput, but each
engine must still receive the same resources. Do not interpret a few dozen
games as a strength conclusion. On Windows the runner requests that the system
remain awake for the process lifetime; it does not prevent the displays from
turning off.

## Compare NNUE networks

An NNUE file is a model; the executable loading it is still the engine. To
isolate the effect of two networks, use the same engine ref on both sides:

```sh
scripts/compare_engines.py \
  --baseline main \
  --candidate main \
  --baseline-eval-file networks/network-a.nnue \
  --candidate-eval-file networks/network-b.nnue \
  --games 1000 --tc 10+0.1
```

To compare NNUE against classical evaluation, omit the eval-file option on the
classical side. A network should not become the default until it passes loader
tests and a properly powered paired match.

To screen a classical/NNUE blend without changing the network or executable,
set the candidate's UCI weight explicitly. For example, this compares a 25%
NNUE blend against pure classical evaluation:

```powershell
python scripts\compare_engines.py `
  --baseline main --candidate main `
  --candidate-eval-file D:\ChessNetworks\candidate.nnue `
  --candidate-option "NNUE Weight=25" `
  --games 100 --tc 10+0.1 --threads 1 --hash 256
```

Screen several predeclared weights with identical openings and settings. Do not
select a weight from a tiny match and then report that same match as independent
confirmation; rerun the selected candidate with a fresh seed and larger sample.
