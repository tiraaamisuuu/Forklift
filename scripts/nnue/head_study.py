#!/usr/bin/env python3
"""Matched, frozen-transformer direct/residual-head study on cached activations.

This is a bounded research experiment, not a production trainer or an Elo test.
All arms use the same pretrained transformer, examples, permutations, optimizer,
update budget and validation-only checkpoint rule. Test labels are opened only
after every arm's checkpoint selection has been written to disk.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

# Required by deterministic CUDA linear algebra; set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from initialization import freeze_feature_transformer, load_initial_weights
from nnue_dataset import active_features_from_packed, sha256_file, write_json_atomic
from train import HalfKpV1, QuantizedNetwork, export_network, quantize_network, verify_cpp_export


@dataclass(frozen=True)
class StudyConfig:
    cache: Path
    source_weights: Path
    output: Path
    seeds: tuple[int, ...] = (1729, 2718)
    residual_widths: tuple[int, ...] = (0, 4, 8)
    epochs: int = 20
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    device: str = "cpu"
    cpu_threads: int = 2
    max_seconds: float = 900.0
    hidden_scale: int = 1024
    output_scale: int = 32
    residual_weight_scale: int = 256
    target_scale: float = 600.0
    wdl_scale: float = 400.0
    cpp_tools: Path | None = None
    bootstrap_resamples: int = 2000


class CachedHead(nn.Module):
    """Only the trainable output modules; no embedding copy lives on the GPU."""

    def __init__(self, model: HalfKpV1):
        super().__init__()
        self.output = copy.deepcopy(model.output)
        self.residual_hidden_layer = copy.deepcopy(model.residual_hidden_layer)
        self.residual_output = copy.deepcopy(model.residual_output)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        value = self.output(activation)
        if self.residual_hidden_layer is not None:
            value = value + self.residual_output(
                torch.clamp(self.residual_hidden_layer(activation), 0.0, 1.0),
            )
        return value.squeeze(1)


def transformer_digest(model: HalfKpV1) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key in ("hidden_bias", "feature_weights.weight", "piece_square_weights.weight"):
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stitch_head(model: HalfKpV1, head: CachedHead, expected_digest: str) -> None:
    if transformer_digest(model) != expected_digest:
        raise ValueError("transformer changed before stitching cached head")
    values = {key: value.detach().cpu() for key, value in head.state_dict().items()}
    expected = {key for key in model.state_dict()
                if key.startswith(("output.", "residual_hidden_layer.", "residual_output."))}
    if set(values) != expected:
        raise ValueError("cached head keys do not match full model")
    model.load_state_dict(values, strict=False)
    if transformer_digest(model) != expected_digest:
        raise ValueError("transformer changed while stitching cached head")


def brier_losses(prediction: torch.Tensor, target: torch.Tensor,
                 target_scale: float, wdl_scale: float) -> torch.Tensor:
    return (torch.sigmoid(prediction * (target_scale / wdl_scale)) - target).square()


def epoch_permutation(count: int, seed: int, epoch: int) -> np.ndarray:
    # Independent of model initialization, arm width and global CPU/CUDA RNG.
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count)


def evaluate_head(head: CachedHead, activation: torch.Tensor, target: torch.Tensor,
                  batch_size: int, target_scale: float, wdl_scale: float,
                  *, predictions: bool = False) -> tuple[float, np.ndarray | None]:
    head.eval()
    losses = 0.0
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(activation), batch_size):
            prediction = head(activation[start:start + batch_size])
            loss = brier_losses(prediction, target[start:start + batch_size],
                                target_scale, wdl_scale)
            losses += float(loss.double().sum().item())
            if predictions:
                outputs.append(prediction.cpu().numpy().copy() * target_scale)
    return losses / len(activation), np.concatenate(outputs) if predictions else None


def paired_cluster_ci(differences: np.ndarray, game_ids: np.ndarray,
                      resamples: int, seed: int,
                      *, deadline_check: Callable[[], None] | None = None) -> dict[str, Any]:
    """Position-weighted paired mean, resampling whole games, not positions."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or len(differences) != len(game_ids) or not len(differences):
        raise ValueError("paired losses require equal, nonempty one-dimensional arrays")
    if not np.isfinite(differences).all() or resamples < 1:
        raise ValueError("paired losses must be finite and resamples positive")
    _, inverse = np.unique(game_ids, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=differences)
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 64):
        if deadline_check:
            deadline_check()
        draws = rng.integers(0, len(counts), size=(min(64, resamples - start), len(counts)))
        replicates[start:start + len(draws)] = sums[draws].sum(1) / counts[draws].sum(1)
    lower, upper = np.quantile(replicates, (0.025, 0.975))
    return {
        "meanResidualMinusDirect": float(differences.mean()),
        "interval95": [float(lower), float(upper)], "positions": len(differences),
        "games": len(counts), "bootstrapResamples": resamples,
        "method": "paired position-weighted percentile bootstrap, whole-game clusters",
        "interpretation": "negative favors residual; exploratory, not Elo or cross-seed certainty",
    }


def validation_screen(arms: list[dict[str, Any]], seeds: tuple[int, ...],
                      widths: tuple[int, ...]) -> dict[str, Any]:
    """Pre-registered architecture choice, sealed before opening test labels."""
    indexed = {(arm["seed"], arm["residualHidden"]): arm for arm in arms}
    if len(indexed) != len(arms) or set(indexed) != {(seed, width) for seed in seeds for width in widths}:
        raise ValueError("validation screen requires exactly one complete arm per seed/width")
    comparisons = []
    for width in widths:
        if width == 0:
            continue
        per_seed = []
        for seed in seeds:
            direct = indexed[(seed, 0)]["validationBrier"]
            residual = indexed[(seed, width)]["validationBrier"]
            if (not math.isfinite(direct) or not math.isfinite(residual)
                    or direct < 0 or residual < 0):
                raise ValueError("validation losses must be finite and nonnegative")
            relative = 100 * (direct - residual) / direct if direct > 0 else None
            per_seed.append({"seed": seed, "directBrier": direct, "residualBrier": residual,
                             "relativeReductionPercent": relative, "improved": residual < direct})
        mean_relative = (sum(row["relativeReductionPercent"] for row in per_seed) / len(per_seed)
                         if all(row["relativeReductionPercent"] is not None for row in per_seed) else None)
        both_improved = all(row["improved"] for row in per_seed)
        comparisons.append({"residualHidden": width, "perSeed": per_seed,
                            "meanRelativeReductionPercent": mean_relative,
                            "everySeedImproved": both_improved,
                            "eligible": both_improved and mean_relative is not None and mean_relative >= 1.0})
    eligible = [row for row in comparisons if row["eligible"]]
    winner = min(eligible, key=lambda row: (-row["meanRelativeReductionPercent"], row["residualHidden"])) if eligible else None
    return {"minimumMeanRelativeReductionPercent": 1.0, "everySeedMustImprove": True,
            "selectionRule": "largest mean relative validation reduction; narrower width breaks ties",
            "comparisons": comparisons,
            "selectedResidualHidden": winner["residualHidden"] if winner else None,
            "selectionUsedTestLabels": False}


def validate_config(config: StudyConfig) -> None:
    if len(config.seeds) < 2 or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("use at least two distinct seeds")
    if any(seed < 0 or seed >= 2**32 for seed in config.seeds):
        raise ValueError("seeds must be unsigned 32-bit integers")
    if (not config.residual_widths or config.residual_widths[0] != 0
            or len(set(config.residual_widths)) != len(config.residual_widths)
            or any(not 0 <= width <= 32 for width in config.residual_widths)):
        raise ValueError("residual widths must be unique, start with 0 and lie in [0,32]")
    for name in ("epochs", "batch_size", "cpu_threads", "bootstrap_resamples"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("learning_rate", "max_seconds", "target_scale", "wdl_scale"):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    for name in ("hidden_scale", "output_scale", "residual_weight_scale"):
        if not 1 <= getattr(config, name) <= np.iinfo(np.int32).max:
            raise ValueError(f"{name} must be positive int32")
    if config.output.exists():
        raise ValueError(f"output must not already exist: {config.output}")
    # Prevent accidental nesting that could mingle source/cache files and results.
    output = config.output.resolve()
    for source in (config.source_weights.resolve(), config.cache.resolve()):
        if source == output or output in source.parents:
            raise ValueError("output cannot contain the source weights or cache manifest")
    if config.cpp_tools is not None and not config.cpp_tools.is_file():
        raise ValueError("C++ tools executable does not exist")


def _json_config(config: StudyConfig) -> dict[str, Any]:
    return {key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in asdict(config).items()}


def _validate_cache(manifest: dict[str, Any], config: StudyConfig, source_hash: str) -> tuple[int, bool]:
    if manifest.get("schemaVersion") != 1:
        raise ValueError("unsupported activation cache schema")
    if manifest.get("sourceWeights", {}).get("sha256") != source_hash:
        raise ValueError("activation cache belongs to different source weights")
    options = manifest["configuration"]
    for key, expected in (("targetScale", config.target_scale), ("wdlScale", config.wdl_scale),
                          ("resultWeight", 0.0)):
        if options.get(key) != expected:
            raise ValueError(f"activation cache has incompatible {key}")
    hidden = options["hidden"]
    if isinstance(hidden, bool) or not isinstance(hidden, int) or not 1 <= hidden <= 1024:
        raise ValueError("invalid cached transformer width")
    shared = options["featureFactorization"]
    if not isinstance(shared, bool):
        raise ValueError("invalid cached factorization flag")
    return hidden, shared


def _validate_arrays(arrays: dict[str, np.ndarray], hidden: int) -> None:
    x = arrays["activations"]
    y = arrays["target_wdl"]
    if x.ndim != 2 or x.shape[1] != 2 * hidden or len(x) < 1 or x.dtype != np.float32:
        raise ValueError("invalid activation shape/dtype")
    if y.shape != (len(x),) or y.dtype != np.float32:
        raise ValueError("invalid WDL target shape/dtype")
    # Chunked checks bound temporary memory even for the full training cache.
    for start in range(0, len(x), 8192):
        values = x[start:start + 8192]
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("cached activations must be finite and clipped to [0,1]")
    if not np.isfinite(y).all() or np.any(y < 0) or np.any(y > 1):
        raise ValueError("cached WDL targets must be finite probabilities")


def _to_device(arrays: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # tensor() copies read-only mmaps, avoiding writable-storage aliases/warnings.
    return (torch.tensor(np.asarray(arrays["activations"]), device=device),
            torch.tensor(np.asarray(arrays["target_wdl"]), device=device))


def _head_state(head: CachedHead) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}


def _check_time(started: float, config: StudyConfig) -> None:
    if time.monotonic() - started > config.max_seconds:
        raise TimeoutError("study time budget exceeded; incomplete arms are not comparable")


def quantized_activations(quantized: QuantizedNetwork, test: dict[str, np.ndarray],
                          batch_size: int = 256, *,
                          deadline_check: Callable[[], None] | None = None) -> np.ndarray:
    """Actual integer transformer, not a rounding approximation of float cache.

    Reconstruct each perspective's feature list once, sum quantized weights in
    int64 and clip exactly as the deployment reference. Chunking bounds gather
    temporaries; empty king-only feature bags are supported without reduceat.
    """
    count = len(test["boards"])
    hidden = len(quantized.hidden_bias)
    result = np.empty((count, 2 * hidden), dtype=np.int64)
    for start in range(0, count, batch_size):
        if deadline_check:
            deadline_check()
        stop = min(count, start + batch_size)
        bags: list[list[int]] = []
        for index in range(start, stop):
            packed = bytes(test["boards"][index])
            turn = bool(test["turns"][index])
            bags.extend((active_features_from_packed(packed, turn),
                         active_features_from_packed(packed, not turn)))
        # Include a zero sentinel for empty bags to keep every reduction valid.
        nonempty = [bag if bag else [0] for bag in bags]
        lengths = np.fromiter(map(len, nonempty), dtype=np.int64)
        offsets = np.r_[0, lengths.cumsum()[:-1]]
        flat = np.fromiter((feature for bag in nonempty for feature in bag), dtype=np.int64)
        gathered = quantized.input_weights[flat].astype(np.int64)
        for index, bag in enumerate(bags):
            if not bag:
                gathered[offsets[index]].fill(0)
        sums = np.add.reduceat(gathered, offsets, axis=0)
        sums += quantized.hidden_bias.astype(np.int64)
        np.clip(sums, 0, quantized.hidden_scale, out=sums)
        result[start:stop] = sums.reshape(stop - start, 2 * hidden)
    return result


def integer_divide_toward_zero(values: np.ndarray, divisor: int) -> np.ndarray:
    # All supported NNUE intermediates are strictly inside int64 endpoints.
    return np.where(values >= 0, values // divisor, -((-values) // divisor))


def quantized_head_predictions(quantized: QuantizedNetwork, activation: np.ndarray,
                               batch_size: int = 8192, *,
                               deadline_check: Callable[[], None] | None = None) -> np.ndarray:
    result = np.empty(len(activation), dtype=np.int32)
    for start in range(0, len(activation), batch_size):
        if deadline_check:
            deadline_check()
        x = activation[start:start + batch_size]
        output = x @ quantized.output_weights.astype(np.int64) + quantized.output_bias
        if quantized.residual_width:
            numerator = x @ quantized.residual_weights.astype(np.int64).T
            numerator += quantized.residual_bias.astype(np.int64)
            residual = integer_divide_toward_zero(numerator, quantized.residual_weight_scale)
            np.clip(residual, 0, quantized.hidden_scale, out=residual)
            output += residual @ quantized.residual_output_weights.astype(np.int64)
        score = integer_divide_toward_zero(output, quantized.hidden_scale * quantized.output_scale)
        result[start:start + len(x)] = np.clip(score, -32000, 32000)
    return result


def run_study(config: StudyConfig) -> dict[str, Any]:
    validate_config(config)
    # Import lazily: unit tests for head/math do not depend on cache creation.
    from head_cache import load_cache

    started = time.monotonic()
    manifest_path = config.cache.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    source_hash = sha256_file(config.source_weights)
    hidden, shared = _validate_cache(manifest, config, source_hash)
    train = load_cache(manifest_path, "train")
    validation = load_cache(manifest_path, "validation")
    _validate_arrays(train, hidden)
    _validate_arrays(validation, hidden)
    if set(np.unique(train["game_ids"])) & set(np.unique(validation["game_ids"])):
        raise ValueError("training and validation game IDs overlap")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    train_x, train_y = _to_device(train, device)
    validation_x, validation_y = _to_device(validation, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    config.output.mkdir(parents=True, exist_ok=False)
    progress = {"state": "running", "startedAt": datetime.now(timezone.utc).isoformat(),
                "configuration": _json_config(config), "armsCompleted": 0,
                "testOpened": False}
    write_json_atomic(config.output / "progress.json", progress)
    report: dict[str, Any] = {
        "schemaVersion": 1, "configuration": _json_config(config),
        "cache": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "sourceWeights": {"path": str(config.source_weights.resolve()), "sha256": source_hash},
        "device": {"requested": str(device), "torchVersion": str(torch.__version__),
                   "cudaVersion": torch.version.cuda, "deterministicAlgorithms": True,
                   "tf32": False, "cpuThreads": config.cpu_threads,
                   "gpuName": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "gpuCapability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None},
        "protocol": {"objective": "teacher WDL Brier score", "resultWeight": 0,
                     "checkpointSelection": "minimum validation Brier over epoch 0..epochs; earliest tie",
                     "optimizer": "AdamW, constant learning rate", "frozenTransformer": True,
                     "equalUpdatesPerArm": math.ceil(len(train_x) / config.batch_size) * config.epochs,
                     "testPolicy": "open test arrays only after all selections are written",
                     "testCaveat": "diagnostic subdivision of the existing validation corpus, already used to select pretrained source; not globally untouched",
                     "strengthEvidence": False, "novelArchitectureClaim": False},
        "arms": [],
    }
    initial_prediction: torch.Tensor | None = None
    common_transformer_digest: str | None = None
    selected: list[tuple[int, int, Path]] = []
    deadline = lambda: _check_time(started, config)
    try:
        for seed in config.seeds:
            for width in config.residual_widths:
                _check_time(started, config)
                cuda_start = cuda_end = None
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    cuda_start, cuda_end = (torch.cuda.Event(enable_timing=True),
                                            torch.cuda.Event(enable_timing=True))
                    cuda_start.record(torch.cuda.current_stream(device))
                arm_started = time.monotonic()
                torch.manual_seed(seed)
                model = HalfKpV1(hidden, feature_factorization=shared, residual_hidden=width)
                initialization = load_initial_weights(model, config.source_weights)
                if initialization["sha256"] != source_hash:
                    raise ValueError("source weights changed during study")
                freeze_feature_transformer(model)
                digest = transformer_digest(model)
                if common_transformer_digest is None:
                    common_transformer_digest = digest
                if digest != common_transformer_digest:
                    raise ValueError("arms do not share bit-identical frozen transformer")
                head = CachedHead(model).to(device)
                with torch.no_grad():
                    prediction = head(validation_x[:min(1024, len(validation_x))]).detach()
                if initial_prediction is None:
                    initial_prediction = prediction.clone()
                if not torch.equal(prediction, initial_prediction):
                    raise ValueError("arms have different initial predictions")
                optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate,
                                               weight_decay=config.weight_decay)
                initial_loss, _ = evaluate_head(head, validation_x, validation_y,
                                                config.batch_size, config.target_scale, config.wdl_scale)
                best_loss, best_epoch, best_state = initial_loss, 0, _head_state(head)
                history: list[dict[str, Any]] = [{"epoch": 0, "validationBrier": initial_loss,
                                                 "updates": 0}]
                updates = 0
                arm_dir = config.output / f"seed-{seed}" / ("direct" if width == 0 else f"residual-{width}")
                arm_dir.mkdir(parents=True, exist_ok=False)
                for epoch in range(1, config.epochs + 1):
                    epoch_started = time.monotonic()
                    order = torch.tensor(epoch_permutation(len(train_x), seed, epoch), device=device)
                    head.train()
                    training_loss = torch.zeros((), device=device)
                    for offset in range(0, len(order), config.batch_size):
                        _check_time(started, config)
                        indices = order[offset:offset + config.batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        prediction = head(train_x.index_select(0, indices))
                        losses = brier_losses(prediction, train_y.index_select(0, indices),
                                               config.target_scale, config.wdl_scale)
                        loss = losses.mean()
                        loss.backward()
                        optimizer.step()
                        training_loss += losses.detach().sum()
                        updates += 1
                    val_loss, _ = evaluate_head(head, validation_x, validation_y, config.batch_size,
                                                config.target_scale, config.wdl_scale)
                    train_loss = float(training_loss.item()) / len(train_x)
                    if not math.isfinite(val_loss) or not math.isfinite(train_loss):
                        raise ValueError("non-finite training/validation loss")
                    if val_loss < best_loss:
                        best_loss, best_epoch, best_state = val_loss, epoch, _head_state(head)
                    history.append({"epoch": epoch, "trainBrier": train_loss,
                                    "validationBrier": val_loss, "updates": updates,
                                    "elapsedSeconds": time.monotonic() - epoch_started})
                    write_json_atomic(arm_dir / "history.json", history)
                _check_time(started, config)
                head.load_state_dict(best_state)
                stitch_head(model, head, digest)
                raw = arm_dir / "model.pt"
                torch.save(model.state_dict(), raw)
                network = arm_dir / "model.nnue"
                quantized = export_network(model, network, config.hidden_scale, config.output_scale,
                                            config.target_scale, config.residual_weight_scale)
                cpp = verify_cpp_export(config.cpp_tools, network, quantized) if config.cpp_tools else None
                gpu_metrics = None
                if device.type == "cuda":
                    cuda_end.record(torch.cuda.current_stream(device))
                    torch.cuda.synchronize(device)
                    gpu_metrics = {
                        "peakAllocatedBytes": torch.cuda.max_memory_allocated(device),
                        "peakReservedBytes": torch.cuda.max_memory_reserved(device),
                        "elapsedCudaStreamSpanMs": cuda_start.elapsed_time(cuda_end),
                        "timingNote": "event span includes CPU dispatch gaps; not active GPU compute time",
                        "memoryNote": "peaks include resident shared training/validation tensors",
                    }
                _check_time(started, config)
                arm = {"seed": seed, "residualHidden": width, "selectedEpoch": best_epoch,
                       "validationBrier": best_loss, "initialValidationBrier": initial_loss,
                       "updates": updates, "initialPredictionsExact": True,
                       "transformerSha256": digest, "transformerBitIdentical": True,
                       "initialization": initialization, "history": history,
                       "rawWeights": {"path": str(raw.resolve()), "sha256": sha256_file(raw)},
                       "network": {"path": str(network.resolve()), "sha256": sha256_file(network)},
                       "cppVerification": cpp, "gpuMetrics": gpu_metrics,
                       "elapsedSeconds": time.monotonic() - arm_started}
                report["arms"].append(arm)
                selected.append((seed, width, raw))
                write_json_atomic(arm_dir / "manifest.json", arm)
                progress["armsCompleted"] += 1
                write_json_atomic(config.output / "progress.json", progress)
                print(json.dumps({"seed": seed, "residualHidden": width,
                                  "selectedEpoch": best_epoch, "validationBrier": best_loss,
                                  "updates": updates}), flush=True)
                del head, model, optimizer
        report["transformerSha256"] = common_transformer_digest
        report["validationScreen"] = validation_screen(report["arms"], config.seeds,
                                                         config.residual_widths)
        selection_path = config.output / "selection.json"
        write_json_atomic(selection_path, report)
        report["frozenSelectionSha256"] = sha256_file(selection_path)
        _check_time(started, config)
        # This is deliberately the first access to test data or test labels.
        test = load_cache(manifest_path, "test")
        _validate_arrays(test, hidden)
        if (set(np.unique(test["game_ids"])) & set(np.unique(train["game_ids"]))
                or set(np.unique(test["game_ids"])) & set(np.unique(validation["game_ids"]))):
            raise ValueError("held-out test game IDs overlap training/validation")
        progress["testOpened"] = True
        write_json_atomic(config.output / "progress.json", progress)
        test_x, test_y = _to_device(test, device)
        losses_by_arm: dict[tuple[int, int], np.ndarray] = {}
        integer_losses_by_arm: dict[tuple[int, int], np.ndarray] = {}
        integer_activation: np.ndarray | None = None
        for arm, (seed, width, raw) in zip(report["arms"], selected):
            _check_time(started, config)
            torch.manual_seed(seed)
            model = HalfKpV1(hidden, feature_factorization=shared, residual_hidden=width)
            load_initial_weights(model, raw)
            head = CachedHead(model).to(device)
            brier, prediction_cp = evaluate_head(head, test_x, test_y, config.batch_size,
                                                  config.target_scale, config.wdl_scale, predictions=True)
            assert prediction_cp is not None
            probabilities = 1 / (1 + np.exp(-np.clip(prediction_cp.astype(np.float64) / config.wdl_scale,
                                                     -700, 700)))
            losses = (probabilities - np.asarray(test["target_wdl"], dtype=np.float64))**2
            losses_by_arm[(seed, width)] = losses
            quantized = quantize_network(model, config.hidden_scale, config.output_scale,
                                          config.target_scale, config.residual_weight_scale)
            if transformer_digest(model) != common_transformer_digest:
                raise ValueError("selected model transformer no longer matches cache")
            if integer_activation is None:
                integer_activation = quantized_activations(quantized, test, deadline_check=deadline)
            integer_cp = quantized_head_predictions(quantized, integer_activation, deadline_check=deadline)
            integer_probabilities = 1 / (1 + np.exp(-integer_cp.astype(np.float64) / config.wdl_scale))
            integer_losses = (integer_probabilities - test["target_wdl"])**2
            integer_losses_by_arm[(seed, width)] = integer_losses
            per_sample = raw.parent / "test-samples.npz"
            np.savez_compressed(per_sample, prediction_cp=prediction_cp, brier_loss=losses,
                                quantized_cp=integer_cp, quantized_brier_loss=integer_losses,
                                game_ids=test["game_ids"], source_indices=test["source_indices"],
                                identity=test["identity"])
            arm["test"] = {"samples": len(test_x), "floatBrier": brier,
                           "floatTeacherRmseCp": float(np.sqrt(np.mean(
                               (prediction_cp - test["target_cp"])**2))),
                           "perSample": {"path": str(per_sample.resolve()), "sha256": sha256_file(per_sample)},
                           "quantization": {
                               "samples": len(integer_cp), "brierScore": float(integer_losses.mean()),
                               "teacherRmseCp": float(np.sqrt(np.mean(
                                   (integer_cp.astype(np.float64) - test["target_cp"])**2))),
                               "rmseCpVsFloat": float(np.sqrt(np.mean((integer_cp - prediction_cp)**2))),
                               "maxAbsErrorCpVsFloat": float(np.max(np.abs(integer_cp - prediction_cp))),
                               "note": "full test set, exact quantized transformer/head Python integer reference"}}
            write_json_atomic(raw.parent / "manifest.json", arm)
            del head, model
        report["pairedComparisons"] = [
            {"seed": seed, "residualHidden": width,
             "float": paired_cluster_ci(losses_by_arm[(seed, width)] - losses_by_arm[(seed, 0)],
                                         test["game_ids"], config.bootstrap_resamples, seed + width,
                                         deadline_check=deadline),
             "quantized": paired_cluster_ci(integer_losses_by_arm[(seed, width)] - integer_losses_by_arm[(seed, 0)],
                                             test["game_ids"], config.bootstrap_resamples, seed + width,
                                             deadline_check=deadline)}
            for seed in config.seeds for width in config.residual_widths if width
        ]
        report["primaryPairedComparisons"] = [
            {"residualHidden": width, "seeds": list(config.seeds),
             "aggregation": "average paired per-position differences over seeds, then bootstrap games",
             "float": paired_cluster_ci(np.mean([
                 losses_by_arm[(seed, width)] - losses_by_arm[(seed, 0)] for seed in config.seeds
             ], axis=0), test["game_ids"], config.bootstrap_resamples, 20261015 + width,
                 deadline_check=deadline),
             "quantized": paired_cluster_ci(np.mean([
                 integer_losses_by_arm[(seed, width)] - integer_losses_by_arm[(seed, 0)]
                 for seed in config.seeds
             ], axis=0), test["game_ids"], config.bootstrap_resamples, 20261015 + width,
                 deadline_check=deadline)}
            for width in config.residual_widths if width
        ]
        winner = report["validationScreen"]["selectedResidualHidden"]
        if winner is None:
            report["offlineTestGate"] = {"passed": False,
                                          "reason": "no architecture passed pre-registered validation screen"}
        else:
            comparison = next(row for row in report["primaryPairedComparisons"]
                              if row["residualHidden"] == winner)
            cpp_exact = all(arm["cppVerification"] is not None
                            and arm["cppVerification"].get("exactMatch", False)
                            for arm in report["arms"] if arm["residualHidden"] in (0, winner))
            float_ci = comparison["float"]["interval95"][1] < 0
            integer_direction = comparison["quantized"]["meanResidualMinusDirect"] < 0
            report["offlineTestGate"] = {
                "selectedResidualHidden": winner, "floatPairedCiUpperBelowZero": float_ci,
                "aggregateIntegerDirectionImproved": integer_direction, "cppExportsExact": cpp_exact,
                "passed": float_ci and integer_direction and cpp_exact,
                "note": "offline screening only; speed/strength match gates remain separate"}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            report["device"]["studyPeakAllocatedBytes"] = max(
                torch.cuda.max_memory_allocated(device),
                *(arm["gpuMetrics"]["peakAllocatedBytes"] for arm in report["arms"]))
            report["device"]["studyPeakReservedBytes"] = max(
                torch.cuda.max_memory_reserved(device),
                *(arm["gpuMetrics"]["peakReservedBytes"] for arm in report["arms"]))
        _check_time(started, config)
        report["completedAt"] = datetime.now(timezone.utc).isoformat()
        report["elapsedSeconds"] = time.monotonic() - started
        report["state"] = "complete"
        write_json_atomic(config.output / "study.json", report)
        progress.update(state="complete", elapsedSeconds=report["elapsedSeconds"])
        write_json_atomic(config.output / "progress.json", progress)
        return report
    except BaseException as error:
        progress.update(state="incomplete", error=str(error) or type(error).__name__,
                        elapsedSeconds=time.monotonic() - started)
        write_json_atomic(config.output / "progress.json", progress)
        raise


def parse_args() -> StudyConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path, help="head_cache manifest.json")
    parser.add_argument("--source-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="fresh study directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 2718])
    parser.add_argument("--residual-widths", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--cpp-tools", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    values = vars(parser.parse_args())
    values["seeds"] = tuple(values["seeds"])
    values["residual_widths"] = tuple(values["residual_widths"])
    return StudyConfig(**values)


if __name__ == "__main__":
    try:
        run_study(parse_args())
    except (Exception, KeyboardInterrupt) as error:
        print(f"head study failed: {error}", file=sys.stderr)
        raise SystemExit(1)
