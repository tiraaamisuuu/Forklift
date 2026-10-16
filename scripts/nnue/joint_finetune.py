#!/usr/bin/env python3
"""Matched whole-network NNUE fine tuning, with sealed diagnostic evaluation.

Unlike head_study.py this runner never uses cached activations: every forward
pass includes the current model's trainable feature transformer. The existing
cache supplies only fixed validation membership and labels, and metadata used
to exclude previously examined diagnostic games/inputs. This is an offline
screen, not an Elo claim or a globally untouched final test.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import chess
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from head_cache import _source, effective_input_identity
from head_study import (
    brier_losses, epoch_permutation, paired_cluster_ci, quantized_activations,
    quantized_head_predictions, transformer_digest,
)
from initialization import load_initial_weights
from nnue_dataset import active_features, active_features_from_packed, sha256_file, write_json_atomic
from train import (CPP_VERIFICATION_FENS, HalfKpV1, PositionsDataset, batches, collate,
                   export_network, quantized_predict)


@dataclass(frozen=True)
class JointConfig:
    train: Path
    validation: Path
    cache: Path
    source_weights: Path
    output: Path
    seeds: tuple[int, ...] = (202610161, 202610162)
    residual_widths: tuple[int, ...] = (0, 8)
    epochs: int = 2
    batch_size: int = 2048
    transformer_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-6
    device: str = "cuda"
    cpu_threads: int = 4
    workers: int = 2
    max_seconds: float = 2700.0
    hidden: int = 256
    feature_factorization: bool = True
    diagnostic_count: int = 32000
    diagnostic_seed: int = 20261016
    hidden_scale: int = 1024
    output_scale: int = 32
    residual_weight_scale: int = 256
    target_scale: float = 600.0
    wdl_scale: float = 400.0
    cpp_tools: Path | None = None
    bootstrap_resamples: int = 1000
    expected_source_sha256: str | None = None


def _json_config(config: JointConfig) -> dict[str, Any]:
    return {key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in asdict(config).items()}


def validate_config(config: JointConfig) -> None:
    if len(config.seeds) < 2 or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("at least two distinct seeds are required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
           for seed in (*config.seeds, config.diagnostic_seed)):
        raise ValueError("seeds must be unsigned 32-bit integers")
    if (not config.residual_widths or config.residual_widths[0] != 0
            or len(set(config.residual_widths)) != len(config.residual_widths)
            or any(isinstance(width, bool) or not isinstance(width, int)
                   or not 0 <= width <= 32 for width in config.residual_widths)):
        raise ValueError("residual widths must be unique, start at zero and lie in [0,32]")
    for name in ("epochs", "batch_size", "cpu_threads", "hidden", "diagnostic_count",
                 "bootstrap_resamples", "hidden_scale", "output_scale", "residual_weight_scale"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.hidden > 1024 or any(getattr(config, name) > np.iinfo(np.int32).max
                                   for name in ("hidden_scale", "output_scale", "residual_weight_scale")):
        raise ValueError("unsupported hidden width or export scale")
    if isinstance(config.workers, bool) or not isinstance(config.workers, int) or config.workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    for name in ("transformer_lr", "head_lr", "max_seconds", "target_scale", "wdl_scale"):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight decay must be finite and nonnegative")
    if not isinstance(config.feature_factorization, bool):
        raise ValueError("feature factorization must be boolean")
    output = config.output.resolve()
    if output.exists():
        raise ValueError(f"output must not already exist: {output}")
    inputs = (config.train, config.validation, config.cache, config.source_weights)
    for path in inputs:
        source = path.resolve(strict=True)
        if output == source or output in source.parents:
            raise ValueError("output must not contain an input")
    if config.train.samefile(config.validation):
        raise ValueError("training and validation source paths must differ")
    if config.cpp_tools is not None and not config.cpp_tools.is_file():
        raise ValueError("C++ tools executable does not exist")
    if config.expected_source_sha256 is not None and (
            len(config.expected_source_sha256) != 64
            or any(value not in "0123456789abcdef" for value in config.expected_source_sha256)):
        raise ValueError("expected source SHA256 must be lowercase hexadecimal")


def load_metadata_cache(cache: Path, split: str, fields: Iterable[str]) -> dict[str, np.ndarray]:
    """Hash-check only explicitly requested arrays; never open activations here."""
    manifest = json.loads(cache.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1 or manifest.get("status") != "complete":
        raise ValueError("unsupported or incomplete metadata cache")
    entry = manifest["splits"][split]
    arrays = {}
    for field in fields:
        if field == "activations":
            raise ValueError("cached activations are prohibited in joint training")
        descriptor = entry["files"][field]
        path = (cache.resolve().parent / descriptor["path"]).resolve(strict=True)
        if cache.resolve().parent not in path.parents:
            raise ValueError("cache metadata path escapes cache directory")
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"cache metadata hash mismatch: {path}")
        value = np.load(path, allow_pickle=False)
        if (list(value.shape) != descriptor["shape"] or str(value.dtype) != descriptor["dtype"]
                or len(value) != entry["count"]):
            raise ValueError(f"cache metadata shape/dtype/count mismatch: {path}")
        arrays[field] = value
    return arrays


def sample_diagnostic(records: np.ndarray, excluded_games: Iterable[int], count: int,
                      seed: int, excluded_identities: Iterable[bytes] = (), *,
                      deadline_check: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    """Choose immutable metadata without ever reading score or result fields.

    Excluding effective HalfKP identities also catches color-flipped vertically
    mirrored representations, not just byte-identical boards.
    """
    if count < 1 or count > len(records):
        raise ValueError("diagnostic count exceeds source rows or is not positive")
    excluded = set(map(int, excluded_games))
    seen = {bytes(identity) for identity in excluded_identities}
    indices, boards, turns, games, identities = [], [], [], [], []
    for iteration, index in enumerate(np.random.default_rng(seed).permutation(len(records))):
        if iteration % 512 == 0 and deadline_check:
            deadline_check()
        game = int(records["game"][index])
        if game in excluded:
            continue
        packed = bytes(records["board"][index])
        turn = bool(records["turn"][index])
        identity = effective_input_identity(active_features_from_packed(packed, turn),
                                            active_features_from_packed(packed, not turn))
        if identity in seen:
            continue
        seen.add(identity)
        indices.append(int(index)); boards.append(np.frombuffer(packed, dtype=np.uint8))
        turns.append(int(turn)); games.append(game)
        identities.append(np.frombuffer(identity, dtype=np.uint8))
        if len(indices) == count:
            break
    if len(indices) != count:
        raise ValueError("not enough fresh, distinct diagnostic inputs outside previously examined games")
    return {"source_indices": np.asarray(indices, dtype=np.int64),
            "boards": np.asarray(boards, dtype=np.uint8),
            "turns": np.asarray(turns, dtype=np.uint8),
            "game_ids": np.asarray(games, dtype=np.uint32),
            "identity": np.asarray(identities, dtype=np.uint8)}


class MetadataPositions(Dataset):
    """Feature bags are reusable input encodings, never model activations."""

    def __init__(self, metadata: dict[str, np.ndarray]):
        self.metadata = metadata
        self.features = []
        count = len(metadata["boards"])
        for name in ("turns", "target_cp", "target_wdl"):
            if len(metadata[name]) != count:
                raise ValueError("metadata arrays have different lengths")
        if count < 1 or not np.isfinite(metadata["target_wdl"]).all():
            raise ValueError("metadata targets must be nonempty and finite")
        if not np.isfinite(metadata["target_cp"]).all():
            raise ValueError("metadata centipawn targets must be finite")
        if np.any(metadata["target_wdl"] < 0) or np.any(metadata["target_wdl"] > 1):
            raise ValueError("metadata WDL targets must be probabilities")
        for packed, turn in zip(metadata["boards"], metadata["turns"]):
            self.features.append((active_features_from_packed(bytes(packed), bool(turn)),
                                  active_features_from_packed(bytes(packed), not bool(turn))))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple:
        first, second = self.features[index]
        cp = float(self.metadata["target_cp"][index])
        return first, second, cp, float(self.metadata["target_wdl"][index]), cp


class MatchedEpochSampler(Sampler[int]):
    def __init__(self, count: int, seed: int):
        self.count, self.seed, self.epoch = count, seed, 1
        self.last_permutation_sha256: str | None = None

    def __iter__(self):
        order = epoch_permutation(self.count, self.seed, self.epoch)
        # Hash precisely the array this iterator yields, not a separately
        # regenerated approximation. Explicit endianness makes the provenance
        # portable across machines and NumPy platform integer defaults.
        self.last_permutation_sha256 = hashlib.sha256(
            order.astype("<i8", copy=False).tobytes(order="C")).hexdigest()
        return iter(order.tolist())

    def __len__(self):
        return self.count


def build_optimizer(model: HalfKpV1, config: JointConfig) -> torch.optim.AdamW:
    transformer, head = [], []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(True)
        (transformer if name in ("hidden_bias", "feature_weights.weight", "piece_square_weights.weight")
         else head).append(parameter)
    if not transformer or not head:
        raise ValueError("joint optimizer requires transformer and head parameters")
    return torch.optim.AdamW([
        {"params": transformer, "lr": config.transformer_lr, "name": "transformer"},
        {"params": head, "lr": config.head_lr, "name": "head"},
    ], weight_decay=config.weight_decay)


def select_checkpoint(best_loss: float, candidate_loss: float) -> bool:
    if not math.isfinite(candidate_loss):
        raise ValueError("validation loss is non-finite")
    return candidate_loss < best_loss  # Epoch zero and earliest ties remain selectable.


def evaluate_model(model: HalfKpV1, loader: DataLoader, config: JointConfig, *,
                   deadline_check: Callable[[], None] | None = None) -> tuple[float, np.ndarray]:
    model.eval()
    outputs, total = [], 0.0
    with torch.no_grad():
        for batch in batches(loader, torch.device(config.device)):
            if deadline_check:
                deadline_check()
            prediction = model(*batch[:4])
            loss = brier_losses(prediction, batch[5], config.target_scale, config.wdl_scale)
            total += float(loss.double().sum().item())
            outputs.append(prediction.cpu().numpy().copy() * config.target_scale)
    values = np.concatenate(outputs)
    return total / len(values), values


def _cpu_state(model: HalfKpV1) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _losses(prediction_cp: np.ndarray, target: np.ndarray, scale: float) -> np.ndarray:
    probability = 1 / (1 + np.exp(-np.clip(prediction_cp.astype(np.float64) / scale, -700, 700)))
    return (probability - np.asarray(target, dtype=np.float64)) ** 2


def _checked_sha256(path: Path, deadline_check: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            deadline_check()
            digest.update(chunk)
    deadline_check()
    return digest.hexdigest()


def _verify_cpp_bounded(tools: Path, network: Path, quantized: Any,
                        seconds_remaining: Callable[[], float]) -> dict[str, Any]:
    """Same reference FENs/math as train.verify_cpp_export, bounded subprocesses."""
    positions = []
    for fen in CPP_VERIFICATION_FENS:
        remaining = seconds_remaining()
        if remaining <= 0:
            raise TimeoutError("no time remains for C++ export verification")
        board = chess.Board(fen)
        expected = quantized_predict(quantized, active_features(board, board.turn),
                                     active_features(board, not board.turn))
        result = subprocess.run([str(tools.resolve()), "--eval", "--nnue", str(network.resolve()),
                                 "--fen", fen], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False, timeout=min(30.0, remaining))
        match = re.search(r"\bevaluation_cp=(-?\d+)\s+backend=nnue\b", result.stdout)
        if result.returncode or match is None:
            raise RuntimeError(f"C++ NNUE verification failed: {result.stderr or result.stdout}")
        actual = int(match.group(1))
        if actual != expected:
            raise RuntimeError(f"C++ NNUE mismatch: Python={expected}, C++={actual}, fen={fen}")
        positions.append({"fen": fen, "pythonCp": expected, "cppCp": actual})
    return {"executable": str(tools.resolve()), "executableSha256": sha256_file(tools),
            "positions": positions, "exactMatch": True}


def _shutdown_loader(loader: DataLoader | None) -> None:
    # DataLoader owns worker processes. Shut down precisely those processes on
    # success/error instead of leaving persistent workers alive until GC.
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def run_joint_study(config: JointConfig) -> dict[str, Any]:
    validate_config(config)
    started = time.monotonic()
    output = config.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    progress = {"state": "running", "startedAt": datetime.now(timezone.utc).isoformat(),
                "configuration": _json_config(config), "armsCompleted": 0,
                "diagnosticLabelsOpened": False, "stage": "source-audit"}
    write_json_atomic(output / "progress.json", progress)
    records = validation_records = dataset = training_loader = None
    report: dict[str, Any] = {"schemaVersion": 1, "configuration": _json_config(config), "arms": []}
    last_progress = 0.0

    def deadline() -> None:
        if time.monotonic() - started > config.max_seconds:
            raise TimeoutError("joint study hard cap exceeded; incomplete arms are not comparable")

    def update(force: bool = False, **values: Any) -> None:
        nonlocal last_progress
        deadline()
        progress.update(values)
        if force or time.monotonic() - last_progress >= 2:
            progress["elapsedSeconds"] = time.monotonic() - started
            write_json_atomic(output / "progress.json", progress)
            last_progress = time.monotonic()

    try:
        input_paths = (config.train, config.validation, config.cache, config.source_weights) + (
            (config.cpp_tools,) if config.cpp_tools is not None else ())
        input_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in input_paths}
        source_hash = _checked_sha256(config.source_weights, deadline)
        cpp_hash = _checked_sha256(config.cpp_tools, deadline) if config.cpp_tools else None
        if config.expected_source_sha256 and source_hash != config.expected_source_sha256:
            raise ValueError("source weights do not match pre-registered SHA256")
        records, train_source = _source(config.train)
        validation_records, validation_source = _source(config.validation)
        deadline()
        train_games = np.unique(records["game"])
        parent_games = np.unique(validation_records["game"])
        if len(np.intersect1d(train_games, parent_games, assume_unique=True)):
            raise ValueError("full training and parent validation game IDs overlap")
        cache_bytes = config.cache.read_bytes()
        manifest = json.loads(cache_bytes)
        cache_paths = {}
        for split in manifest["splits"].values():
            for descriptor in split["files"].values():
                path = (config.cache.resolve().parent / descriptor["path"]).resolve(strict=True)
                if config.cache.resolve().parent not in path.parents:
                    raise ValueError("cache array path escapes cache directory")
                if _checked_sha256(path, deadline) != descriptor["sha256"]:
                    raise ValueError(f"cache array hash mismatch: {path}")
                cache_paths[path] = descriptor["sha256"]
        options = manifest.get("configuration", {})
        if manifest.get("sourceWeights", {}).get("sha256") != source_hash:
            raise ValueError("validation metadata belongs to different pretrained source")
        if manifest.get("sources", {}).get("validation", {}).get("sha256") != validation_source["sha256"]:
            raise ValueError("validation metadata belongs to different parent validation source")
        for key, value in (("hidden", config.hidden), ("featureFactorization", config.feature_factorization),
                           ("targetScale", config.target_scale), ("wdlScale", config.wdl_scale),
                           ("resultWeight", 0.0)):
            if options.get(key) != value:
                raise ValueError(f"validation metadata has incompatible {key}")
        validation = load_metadata_cache(config.cache, "validation", (
            "boards", "turns", "target_cp", "target_wdl", "game_ids", "source_indices", "identity"))
        rows = validation["source_indices"]
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(validation_records)):
            raise ValueError("validation source indices are out of bounds")
        for cache_field, source_field in (("boards", "board"), ("turns", "turn"), ("game_ids", "game")):
            if not np.array_equal(validation[cache_field], validation_records[source_field][rows]):
                raise ValueError(f"validation metadata disagrees with parent source: {cache_field}")
        old_test = load_metadata_cache(config.cache, "test", ("game_ids", "identity"))
        if set(map(int, validation["game_ids"])) & set(map(int, train_games)):
            raise ValueError("cached validation game IDs overlap training")
        excluded_games = np.union1d(validation["game_ids"], old_test["game_ids"])
        excluded_inputs = [bytes(row) for row in np.concatenate((validation["identity"], old_test["identity"]))]
        diagnostic = sample_diagnostic(validation_records, excluded_games, config.diagnostic_count,
                                       config.diagnostic_seed, excluded_inputs, deadline_check=deadline)
        descriptor = output / "diagnostic-descriptor.npz"
        np.savez_compressed(descriptor, **diagnostic)
        report.update({"sourceWeights": {"path": str(config.source_weights.resolve()), "sha256": source_hash},
                       "sources": {"train": train_source, "parentValidation": validation_source},
                       "metadataCache": {"path": str(config.cache.resolve()),
                                         "sha256": hashlib.sha256(cache_bytes).hexdigest()},
                       "audit": {"fullSourceGameIdsDisjoint": True, "trainingGames": len(train_games),
                                 "parentValidationGames": len(parent_games),
                                 "diagnosticGamesExcludeOldValidationAndTest": True,
                                 "diagnosticInputsUniqueAndExcludeOldValidationAndTest": True,
                                 "inputAuditScope": "selected diagnostic versus old validation/test; not full training-source input audit",
                                 "descriptor": {"path": str(descriptor), "sha256": sha256_file(descriptor)}},
                       "protocol": {"checkpointSelection": "minimum validation Brier over epochs 0..epochs; earliest tie",
                                    "optimizer": "AdamW, constant differential learning rates",
                                    "frozenTransformer": False, "cachedActivationsUsed": False,
                                    "resultWeight": 0.0, "objective": "teacher WDL Brier",
                                    "testPolicy": "fresh diagnostic labels opened only after all-arm selection seal",
                                    "testCaveat": "source model was selected on parent validation; diagnostic is not globally untouched",
                                    "deadlinePolicy": "cooperative checks between batches, hash chunks, evaluation chunks and stages; C++ subprocess timeout; no auto-extension",
                                    "deadlineCaveat": "an OS file read or individual CUDA call cannot be pre-empted by this Python process; use an external watchdog for an absolute wall cap",
                                    "strengthEvidence": False, "novelArchitectureClaim": False}})
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        report["device"] = {"requested": str(device), "torchVersion": str(torch.__version__),
                            "cudaVersion": torch.version.cuda, "cpuThreads": config.cpu_threads,
                            "loaderWorkers": config.workers, "deterministicAlgorithms": True, "tf32": False,
                            "gpuName": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        dataset = PositionsDataset([config.train], result_weight=0.0, wdl_scale=config.wdl_scale)
        if len(dataset) != train_source["count"]:
            raise ValueError("training source count changed")
        validation_loader = DataLoader(MetadataPositions(validation), batch_size=config.batch_size,
                                       shuffle=False, collate_fn=collate, num_workers=0)
        initial_cp = None
        source_digest = None
        selected_paths = []
        matched_permutations: dict[tuple[int, int], str] = {}
        for seed in config.seeds:
            for width in config.residual_widths:
                update(True, stage="initializing-arm", seed=seed, residualHidden=width, epoch=0,
                       positionsInEpoch=0, positionsPerEpoch=len(dataset))
                arm_started = time.monotonic()
                torch.manual_seed(seed)
                model = HalfKpV1(config.hidden, feature_factorization=config.feature_factorization,
                                 residual_hidden=width)
                initialization = load_initial_weights(model, config.source_weights)
                if initialization["sha256"] != source_hash:
                    raise ValueError("source weights changed during study")
                digest = transformer_digest(model)
                if source_digest is None:
                    source_digest = digest
                if digest != source_digest:
                    raise ValueError("initial transformers differ between arms")
                model.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    cuda_start.record()
                initial_loss, cp = evaluate_model(model, validation_loader, config, deadline_check=deadline)
                if initial_cp is None:
                    initial_cp = cp.copy()
                if not np.array_equal(initial_cp, cp):
                    raise ValueError("all arms must begin with exactly identical full-validation predictions")
                best_loss, best_epoch, best_state = initial_loss, 0, _cpu_state(model)
                if not math.isfinite(initial_loss):
                    raise ValueError("initial validation Brier is non-finite")
                arm_dir = output / f"seed-{seed}" / ("direct" if width == 0 else f"residual-{width}")
                arm_dir.mkdir(parents=True, exist_ok=False)
                torch.save(best_state, arm_dir / "epoch-0.pt")
                optimizer = build_optimizer(model, config)
                sampler = MatchedEpochSampler(len(dataset), seed)
                training_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler,
                                             collate_fn=collate, num_workers=config.workers,
                                             persistent_workers=config.workers > 0,
                                             pin_memory=device.type == "cuda",
                                             generator=torch.Generator().manual_seed(seed))
                updates, positions = 0, 0
                history = [{"epoch": 0, "validationBrier": initial_loss, "updates": 0,
                            "trainingPositions": 0, "selected": True}]
                write_json_atomic(arm_dir / "history.json", history)
                for epoch in range(1, config.epochs + 1):
                    sampler.epoch = epoch
                    epoch_started = time.monotonic()
                    model.train()
                    train_total, epoch_positions = 0.0, 0
                    for batch in batches(training_loader, device):
                        deadline()
                        optimizer.zero_grad(set_to_none=True)
                        prediction = model(*batch[:4])
                        losses = brier_losses(prediction, batch[5], config.target_scale, config.wdl_scale)
                        loss = losses.mean()
                        if not torch.isfinite(loss):
                            raise ValueError("non-finite training loss")
                        loss.backward()
                        optimizer.step()
                        train_total += float(losses.detach().double().sum().item())
                        batch_count = len(prediction)
                        epoch_positions += batch_count; positions += batch_count; updates += 1
                        update(stage="training", epoch=epoch, positionsInEpoch=epoch_positions,
                               trainingPositions=positions, updates=updates)
                    if epoch_positions != len(dataset):
                        raise ValueError("epoch did not process the complete training source")
                    permutation_hash = sampler.last_permutation_sha256
                    if permutation_hash is None:
                        raise ValueError("training sampler did not record the epoch permutation")
                    permutation_key = (seed, epoch)
                    expected_permutation = matched_permutations.setdefault(permutation_key, permutation_hash)
                    if expected_permutation != permutation_hash:
                        raise ValueError("matched arms used different epoch permutations")
                    update(True, stage="validation", epoch=epoch)
                    val_loss, _ = evaluate_model(model, validation_loader, config, deadline_check=deadline)
                    improved = select_checkpoint(best_loss, val_loss)
                    if improved:
                        best_loss, best_epoch, best_state = val_loss, epoch, _cpu_state(model)
                    history.append({"epoch": epoch, "trainBrier": train_total / epoch_positions,
                                    "validationBrier": val_loss, "updates": updates,
                                    "trainingPositions": positions, "selected": improved,
                                    "permutationSha256": permutation_hash,
                                    "permutationEncoding": "C-contiguous little-endian int64 source row indices",
                                    "elapsedSeconds": time.monotonic() - epoch_started})
                    write_json_atomic(arm_dir / "history.json", history)
                    print(json.dumps({"seed": seed, "residualHidden": width, **history[-1]}), flush=True)
                _shutdown_loader(training_loader); training_loader = None
                model.load_state_dict(best_state)
                model.cpu()
                final_digest = transformer_digest(model)
                raw = arm_dir / "model.pt"
                torch.save(model.state_dict(), raw)
                network = arm_dir / "model.nnue"
                quantized = export_network(model, network, config.hidden_scale, config.output_scale,
                                           config.target_scale, config.residual_weight_scale)
                deadline()
                cpp = (_verify_cpp_bounded(config.cpp_tools, network, quantized,
                                          lambda: config.max_seconds - (time.monotonic() - started))
                       if config.cpp_tools else None)
                if cpp is not None and cpp["executableSha256"] != cpp_hash:
                    raise ValueError("C++ verification executable changed during study")
                gpu = None
                if device.type == "cuda":
                    cuda_end.record(); torch.cuda.synchronize(device)
                    gpu = {"peakAllocatedBytes": torch.cuda.max_memory_allocated(device),
                           "peakReservedBytes": torch.cuda.max_memory_reserved(device),
                           "elapsedCudaStreamSpanMs": cuda_start.elapsed_time(cuda_end),
                           "timingNote": "CUDA event span includes CPU dispatch/export gaps; not active GPU compute time"}
                arm = {"seed": seed, "residualHidden": width, "selectedEpoch": best_epoch,
                       "validationBrier": best_loss, "initialValidationBrier": initial_loss,
                       "initialPredictionsExact": True, "initialTransformerSha256": digest,
                       "selectedTransformerSha256": final_digest,
                       "transformerChanged": digest != final_digest, "initialization": initialization,
                       "updates": updates, "trainingPositions": positions, "history": history,
                       "rawWeights": {"path": str(raw), "sha256": sha256_file(raw)},
                       "network": {"path": str(network), "sha256": sha256_file(network)},
                       "cppVerification": cpp, "gpuMetrics": gpu,
                       "elapsedSeconds": time.monotonic() - arm_started}
                report["arms"].append(arm); selected_paths.append(raw)
                write_json_atomic(arm_dir / "manifest.json", arm)
                write_json_atomic(output / "partial-study.json", report)
                update(True, armsCompleted=len(report["arms"]), stage="arm-complete")
                del model, optimizer, best_state, quantized
        indexed = {(arm["seed"], arm["residualHidden"]): arm for arm in report["arms"]}
        baseline_loss = report["arms"][0]["initialValidationBrier"]
        direct_reductions = [100 * (baseline_loss - indexed[(seed, 0)]["validationBrier"]) / baseline_loss
                             if baseline_loss > 0 else 0.0 for seed in config.seeds]
        residual_screens = []
        for width in config.residual_widths:
            if not width:
                continue
            reductions = [100 * (indexed[(seed, 0)]["validationBrier"] - indexed[(seed, width)]["validationBrier"])
                          / indexed[(seed, 0)]["validationBrier"]
                          if indexed[(seed, 0)]["validationBrier"] > 0 else 0.0 for seed in config.seeds]
            residual_screens.append({"residualHidden": width, "perSeedRelativeReductionPercent": reductions,
                                     "eligible": all(value >= 1.0 for value in reductions)})
        report["validationScreen"] = {"directPerSeedReductionVsSourcePercent": direct_reductions,
                                      "directEligible": all(value >= 1.0 for value in direct_reductions),
                                      "residualComparisons": residual_screens,
                                      "rule": "at least 1% relative reduction in every seed; test labels not used"}
        # The immutable selection seal is the boundary before any fresh test label.
        selection_path = output / "selection.json"
        write_json_atomic(selection_path, report)
        report["frozenSelectionSha256"] = sha256_file(selection_path)
        update(True, stage="diagnostic", diagnosticLabelsOpened=True)
        indices = diagnostic["source_indices"]
        cp = np.clip(validation_records["score"][indices].astype(np.float32), -3000, 3000)
        diagnostic["target_cp"] = cp
        diagnostic["target_wdl"] = (1 / (1 + np.exp(-cp.astype(np.float64) / config.wdl_scale))).astype(np.float32)
        diagnostic_loader = DataLoader(MetadataPositions(diagnostic), batch_size=config.batch_size,
                                       shuffle=False, collate_fn=collate, num_workers=0)
        float_losses, integer_losses = {}, {}
        evaluation_entries = [(None, 0, config.source_weights)] + [
            (arm["seed"], arm["residualHidden"], raw) for arm, raw in zip(report["arms"], selected_paths)]
        for seed, width, raw in evaluation_entries:
            deadline()
            model = HalfKpV1(config.hidden, feature_factorization=config.feature_factorization,
                             residual_hidden=width)
            load_initial_weights(model, raw)
            model.to(device)
            brier, prediction = evaluate_model(model, diagnostic_loader, config, deadline_check=deadline)
            model.cpu()
            # Crucially recomputed for EACH selected transformer's quantized weights.
            from train import quantize_network
            quantized = quantize_network(model, config.hidden_scale, config.output_scale,
                                         config.target_scale, config.residual_weight_scale)
            integer_x = quantized_activations(quantized, diagnostic, deadline_check=deadline)
            integer_cp = quantized_head_predictions(quantized, integer_x, deadline_check=deadline)
            key = (seed, width)
            float_losses[key] = _losses(prediction, diagnostic["target_wdl"], config.wdl_scale)
            integer_losses[key] = _losses(integer_cp, diagnostic["target_wdl"], config.wdl_scale)
            destination = output / "source-diagnostic.npz" if seed is None else raw.parent / "diagnostic-samples.npz"
            np.savez_compressed(destination, prediction_cp=prediction, brier_loss=float_losses[key],
                                quantized_cp=integer_cp, quantized_brier_loss=integer_losses[key],
                                game_ids=diagnostic["game_ids"], source_indices=indices,
                                identity=diagnostic["identity"])
            summary = {"samples": len(indices), "floatBrier": brier,
                       "quantizedBrier": float(integer_losses[key].mean()),
                       "floatTeacherRmseCp": float(np.sqrt(np.mean(
                           (prediction.astype(np.float64) - diagnostic["target_cp"]) ** 2))),
                       "quantizedTeacherRmseCp": float(np.sqrt(np.mean(
                           (integer_cp.astype(np.float64) - diagnostic["target_cp"]) ** 2))),
                       "quantizedRmseCpVsFloat": float(np.sqrt(np.mean(
                           (integer_cp.astype(np.float64) - prediction) ** 2))),
                       "quantizedMaxAbsErrorCpVsFloat": float(np.max(np.abs(integer_cp - prediction))),
                       "perSample": {"path": str(destination), "sha256": sha256_file(destination)},
                       "quantizedTransformerRecomputed": True,
                       "selectedTransformerSha256": transformer_digest(model)}
            if seed is None:
                report["sourceDiagnostic"] = summary
            else:
                indexed[key]["diagnostic"] = summary
                write_json_atomic(raw.parent / "manifest.json", indexed[key])
            del model, quantized, integer_x
            update(True, diagnosticModelsCompleted=len(float_losses))
        comparisons = []
        for width in config.residual_widths:
            for seed in config.seeds:
                reference = (None, 0) if width == 0 else (seed, 0)
                comparisons.append({"seed": seed, "residualHidden": width,
                                    "reference": "pretrained-source" if width == 0 else "matched-trained-direct",
                                    "float": paired_cluster_ci(float_losses[(seed, width)] - float_losses[reference],
                                        diagnostic["game_ids"], config.bootstrap_resamples, seed + width,
                                        deadline_check=deadline),
                                    "quantized": paired_cluster_ci(integer_losses[(seed, width)] - integer_losses[reference],
                                        diagnostic["game_ids"], config.bootstrap_resamples, seed + width,
                                        deadline_check=deadline)})
        primary = []
        for width in config.residual_widths:
            reference_keys = [(None, 0) if width == 0 else (seed, 0) for seed in config.seeds]
            entry = {"residualHidden": width, "seeds": list(config.seeds),
                     "reference": "pretrained-source" if width == 0 else "matched-trained-direct",
                     "aggregation": "mean paired per-position difference over seeds, then bootstrap whole games"}
            for name, losses in (("float", float_losses), ("quantized", integer_losses)):
                differences = np.mean([losses[(seed, width)] - losses[reference]
                                       for seed, reference in zip(config.seeds, reference_keys)], axis=0)
                entry[name] = paired_cluster_ci(differences, diagnostic["game_ids"],
                                                config.bootstrap_resamples, config.diagnostic_seed + width,
                                                deadline_check=deadline)
            validation_pass = (report["validationScreen"]["directEligible"] if width == 0 else
                               next(row["eligible"] for row in residual_screens if row["residualHidden"] == width))
            cpp_exact = all(arm["cppVerification"] and arm["cppVerification"].get("exactMatch")
                            for arm in report["arms"] if arm["residualHidden"] in (0, width))
            entry["offlineGate"] = {"validationPassed": validation_pass,
                                    "floatCiUpperBelowZero": entry["float"]["interval95"][1] < 0,
                                    "integerDirectionImproved": entry["quantized"]["meanResidualMinusDirect"] < 0,
                                    "cppExportsExact": bool(cpp_exact)}
            if width:
                entry["offlineGate"]["directAdaptationGatePassed"] = primary[0]["offlineGate"]["passed"]
            entry["offlineGate"]["passed"] = all(entry["offlineGate"].values())
            primary.append(entry)
        report.update(pairedComparisons=comparisons, primaryPairedComparisons=primary)
        for path, expected_stat in input_stats.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != expected_stat:
                raise ValueError(f"input changed during study: {path}")
        expected_hashes = {config.train: train_source["sha256"],
                           config.validation: validation_source["sha256"],
                           config.source_weights: source_hash,
                           config.cache: hashlib.sha256(cache_bytes).hexdigest(), **cache_paths}
        if config.cpp_tools is not None:
            expected_hashes[config.cpp_tools] = cpp_hash
        for path, expected_hash in expected_hashes.items():
            if _checked_sha256(path, deadline) != expected_hash:
                raise ValueError(f"input hash changed during study: {path}")
        report["audit"]["sourceAndAllCacheFileHashesRecheckedAtCompletion"] = True
        deadline()
        report.update(state="complete", completedAt=datetime.now(timezone.utc).isoformat(),
                      elapsedSeconds=time.monotonic() - started)
        write_json_atomic(output / "study.json", report)
        update(True, state="complete", stage="complete")
        return report
    except BaseException as error:
        progress.update(state="incomplete", error=str(error) or type(error).__name__,
                        elapsedSeconds=time.monotonic() - started,
                        comparable=False, autoExtended=False)
        write_json_atomic(output / "progress.json", progress)
        report.update(state="incomplete", error=progress["error"], comparable=False)
        write_json_atomic(output / "partial-study.json", report)
        raise
    finally:
        _shutdown_loader(training_loader)
        if dataset is not None:
            dataset.close()
        for source in (records, validation_records):
            if isinstance(source, np.memmap):
                source._mmap.close()


run_study = run_joint_study


def parse_args() -> JointConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "cache", "source_weights", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[202610161, 202610162])
    parser.add_argument("--residual-widths", type=int, nargs="+", default=[0, 8])
    for name, default in (("epochs", 2), ("batch_size", 2048), ("cpu_threads", 4),
                          ("workers", 2), ("hidden", 256), ("diagnostic_count", 32000),
                          ("diagnostic_seed", 20261016), ("hidden_scale", 1024),
                          ("output_scale", 32), ("residual_weight_scale", 256),
                          ("bootstrap_resamples", 1000)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in (("transformer_lr", 1e-5), ("head_lr", 1e-4), ("weight_decay", 1e-6),
                          ("max_seconds", 2700), ("target_scale", 600), ("wdl_scale", 400)):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-factorization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpp-tools", type=Path)
    parser.add_argument("--expected-source-sha256")
    values = vars(parser.parse_args())
    values["seeds"] = tuple(values["seeds"]); values["residual_widths"] = tuple(values["residual_widths"])
    return JointConfig(**values)


if __name__ == "__main__":
    try:
        run_joint_study(parse_args())
    except (Exception, KeyboardInterrupt) as error:
        print(f"joint fine-tuning study failed: {error}", file=sys.stderr)
        raise SystemExit(1)
