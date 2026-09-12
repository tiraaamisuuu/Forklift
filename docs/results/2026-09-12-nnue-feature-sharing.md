# NNUE v2: shared piece-square learning

## Question and predeclared comparison

Can sharing piece-square parameters across king squares improve generalization
without increasing the engine's inference work? This is an implementation of
known [feature factorization](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html),
not a claim of novelty or of higher Elo.

Candidate: existing 256-wide HalfKP network plus a zero-initialized 640-row shared
piece-square embedding. Control: original embedding only. The shared term folds
into the original matrix before quantization, so both export the same format.
Initialization and random-number state are matched by tests. No Stockfish source
is copied into this implementation.

Both runs reuse the five-million-position Stockfish-18/20k teacher dataset and
616,632-position held-out validation split. The runner checks source-manifest
hashes before starting. No new labels or evaluation games are generated here.

Fixed settings: seed 11, hidden 256, batch 2048, CUDA, two loader workers,
two CPU threads, WDL probability loss, result weight 0, WDL scale 400, target
scale 600, AdamW learning rate 0.001 (weight decay 0.000001), cosine decay to 0.00001, maximum 16 epochs,
early stopping after five epochs without validation improvement. Select each
model's minimum-validation-loss checkpoint. Candidate runs first, control second.
One seed is an exploratory screen, not a robust estimate across training seeds.

## Evidence and monitoring

Run directory: `E:\Dev\Forklift-Research\models\nnue-v2-factorization-20260912`.
Each variant preserves logs, epoch metrics, checkpoints, network, quantization
checks, validation slices and provenance. `experiment.json` records exact commands
and source revision. The dashboard's `--training-run-dir` reads this directory.
It reports a per-epoch ETA, not an unsupported estimate for the entire pipeline.

Launch from a clean committed checkout:

```powershell
.venv-nnue\Scripts\python.exe scripts/nnue/run_factorization_experiment.py --dataset-dir E:\ChessData\halfkp-5m-sf18-20k --run-dir E:\Dev\Forklift-Research\models\nnue-v2-factorization-20260912 --cpp-tools build-clock-safe\Release\chess-engine-tools.exe
```

Compare held-out loss, MAE/RMSE, king-square and phase slices, and quantization
error. The runner requires exact C++ integer-evaluation agreement before accepting
either export. An offline improvement is only a reason to run a controlled match;
it does not change the default evaluator or establish a playing-strength gain.
The previously paused search match remains paused. No automatic promotion.

If interrupted, checkpoints are retained; the trainer supports `--resume` with
matching configuration. The experiment runner refuses a nonempty run directory
unless `--resume-failed` is explicitly supplied. Recovery validates settings/data,
archives the failed manifest, appends logs and resumes the last saved epoch.

## Recovery

The initial run completed epoch 1, then failed in epoch 2 with Windows error 5
replacing `model.progress.json`. This was a telemetry write failure, not a CUDA
or training-loss failure. JSON replacement now retries transient permission errors;
progress-only write errors are non-fatal, while checkpoint/evidence failures still
propagate. Recovery repeats unfinished epoch 2 from the saved epoch-1 checkpoint.
The recovery revision is recorded separately from the original experiment commit.

## Offline result and playing-strength screen

Both exports passed exact C++ checks. Shared features reduced held-out loss from
0.010551882 to 0.009687731 (8.19%); MAE fell from 200.292 to 193.781 cp. Shared
features peaked at epoch 3 and stopped at epoch 8; control stopped at epoch 9.
This is one seed, with a checkpoint recovery, not a playing-strength conclusion.

The next predeclared screen uses `scripts/nnue/run_strength_screen.py`: 400 games
shared versus control, followed by 400 shared versus classical; 30+0.3, eight
concurrent games, one thread and 128 MiB hash per engine, paired random UHO
openings, seeds 20260912 and 20260913. Both sides use the same clock-safe Release
UCI executable, with its checksum and configured options recorded. Neural sides
use pure NNUE (default weight 100); the classical side explicitly disables NNUE.
Fixed-length screens, not SPRT or automatic promotion. Technical failures block
the next stage for investigation. GPU training is finished; matches use CPU.

Evidence: `E:\Dev\Forklift-Research\matches\nnue-v2-strength-20260912`.
The dashboard follows `series.json` through both matches and retains each result.
The separate four-game-per-stage `nnue-v2-screen-smoke-20260912` checks only the
workflow and is excluded from strength evidence.
