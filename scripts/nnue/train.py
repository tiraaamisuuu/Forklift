#!/usr/bin/env python3
"""Train, validate, checkpoint, and export the engine's HalfKP-v1 NNUE."""

from __future__ import annotations

import argparse
from array import array
import bisect
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import time
from typing import BinaryIO, Iterator

import chess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from nnue_dataset import (
    SHARD_HEADER,
    SHARD_RECORD,
    active_features,
    active_features_from_packed,
    decode_record,
    pack_board,
    read_shard_header,
    sha256_file,
    unpack_piece_codes,
    write_json_atomic,
)
from dataset_diagnostics import labelled_bucket, position_distribution


FEATURE_COUNT = 64 * 10 * 64
FORMAT_VERSION = 1
MAGIC = b"TNNUE1\0\0"
CHECKPOINT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    parser.add_argument("--validation-data", type=Path, nargs="+",
                        help="Game-disjoint validation shard(s)")
    parser.add_argument("--allow-position-split", action="store_true",
                        help="Permit a random position split for pipeline smoke only")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--feature-factorization", action="store_true",
                        help="Share piece-square learning across king squares; fold weights at export")
    parser.add_argument("--cpu-threads", type=int, default=2,
                        help="PyTorch CPU threads (data loader workers are separate)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--result-weight", type=float, default=0.15)
    parser.add_argument(
        "--loss", choices=("centipawn-huber", "wdl"), default="centipawn-huber",
        help="Training objective; WDL optimizes calibrated win probability",
    )
    parser.add_argument(
        "--wdl-scale", type=float, default=400.0,
        help="Centipawn logistic scale used by the WDL objective",
    )
    parser.add_argument("--target-scale", type=float, default=600.0,
                        help="Centipawns per normalized model output unit")
    parser.add_argument("--huber-beta-cp", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", "--checkpoint", dest="resume", type=Path,
                        help="Resume a versioned optimizer/scheduler checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--verify-samples", type=int, default=256)
    parser.add_argument("--hidden-scale", type=int, default=1024)
    parser.add_argument("--output-scale", type=int, default=64)
    parser.add_argument("--metrics", type=Path, help="Defaults beside --output")
    parser.add_argument("--manifest", type=Path, help="Defaults beside --output")
    parser.add_argument("--cpp-tools", type=Path,
                        help="Optional chess-engine-tools executable for exact export verification")
    return parser.parse_args()


@dataclass
class DataSource:
    path: Path
    format: str
    start: int
    count: int
    offsets: array | None = None


@dataclass(frozen=True)
class TrainingSample:
    first: list[int]
    second: list[int]
    target: float
    wdl_target: float
    board_bytes: bytes
    turn: chess.Color
    teacher_score_cp: int


def score_to_win_probability(score_cp: float, scale_cp: float) -> float:
    if scale_cp <= 0:
        raise ValueError("WDL scale must be positive")
    value = score_cp / scale_cp
    if value >= 0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def training_loss(
    prediction: torch.Tensor,
    target_cp: torch.Tensor,
    target_wdl: torch.Tensor,
    loss_mode: str,
    target_scale: float,
    huber_beta_cp: float,
    wdl_scale: float,
) -> torch.Tensor:
    if loss_mode == "centipawn-huber":
        return torch.nn.functional.smooth_l1_loss(
            prediction, target_cp / target_scale,
            beta=huber_beta_cp / target_scale,
        )
    if loss_mode == "wdl":
        predicted_wdl = torch.sigmoid(prediction * target_scale / wdl_scale)
        return torch.nn.functional.mse_loss(predicted_wdl, target_wdl)
    raise ValueError(f"unsupported loss mode: {loss_mode}")


class PositionsDataset(Dataset):
    """Random-access mixed JSONL/compact position dataset."""

    def __init__(self, paths: list[Path], result_weight: float, wdl_scale: float = 400.0):
        self.paths = [path.resolve() for path in paths]
        self.sources: list[DataSource] = []
        self.starts: list[int] = []
        self._handles: dict[int, BinaryIO] = {}
        self.result_weight = result_weight
        self.wdl_scale = wdl_scale
        total = 0

        for path in self.paths:
            if not path.is_file():
                raise ValueError(f"dataset does not exist: {path}")
            if path.suffix.casefold() == ".nnuebin":
                with path.open("rb") as source:
                    header = read_shard_header(source)
                expected_size = SHARD_HEADER.size + header.count * header.record_size
                if path.stat().st_size != expected_size:
                    raise ValueError(f"compact shard size/count mismatch: {path}")
                descriptor = DataSource(path, "compact", total, header.count)
            else:
                offsets = array("Q")
                with path.open("rb") as source:
                    while True:
                        offset = source.tell()
                        line = source.readline()
                        if not line:
                            break
                        if line.strip():
                            offsets.append(offset)
                descriptor = DataSource(path, "jsonl", total, len(offsets), offsets)

            self.starts.append(total)
            self.sources.append(descriptor)
            total += descriptor.count
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        if hasattr(self, "_handles"):
            self._handles.clear()

    def __enter__(self) -> PositionsDataset:
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _source_for(self, index: int) -> tuple[int, DataSource, int]:
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        source_index = bisect.bisect_right(self.starts, index) - 1
        source = self.sources[source_index]
        return source_index, source, index - source.start

    def _handle(self, source_index: int, source: DataSource) -> BinaryIO:
        handle = self._handles.get(source_index)
        if handle is None:
            handle = source.path.open("rb")
            self._handles[source_index] = handle
        return handle

    def sample(self, index: int) -> TrainingSample:
        source_index, source, local_index = self._source_for(index)
        handle = self._handle(source_index, source)
        if source.format == "compact":
            handle.seek(SHARD_HEADER.size + local_index * SHARD_RECORD.size)
            record = decode_record(handle.read(SHARD_RECORD.size))
            first = active_features_from_packed(record.board_bytes, record.turn)
            second = active_features_from_packed(record.board_bytes, not record.turn)
            score_cp = record.score_cp
            result = float(record.result)
            board_bytes = record.board_bytes
            turn = record.turn
        else:
            assert source.offsets is not None
            handle.seek(source.offsets[local_index])
            record_json = json.loads(handle.readline())
            board = chess.Board(record_json["fen"])
            first = active_features(board, board.turn)
            second = active_features(board, not board.turn)
            score_cp = int(record_json["score_cp"])
            result = float(record_json.get("result", 0.0))
            board_bytes = pack_board(board)
            turn = board.turn

        score = max(-3000.0, min(3000.0, float(score_cp)))
        result_target = result * 1000.0
        target = score * (1.0 - self.result_weight) + result_target * self.result_weight
        teacher_probability = score_to_win_probability(score, self.wdl_scale)
        result_probability = (result + 1.0) * 0.5
        wdl_target = (
            teacher_probability * (1.0 - self.result_weight)
            + result_probability * self.result_weight
        )
        return TrainingSample(
            first, second, target, wdl_target, board_bytes, turn, score_cp,
        )

    def __getitem__(
        self, index: int,
    ) -> tuple[list[int], list[int], float, float, float]:
        sample = self.sample(index)
        return (
            sample.first, sample.second, sample.target,
            sample.wdl_target, float(sample.teacher_score_cp),
        )


def collate(samples: list[tuple]) -> tuple[torch.Tensor, ...]:
    first_flat: list[int] = []
    second_flat: list[int] = []
    first_offsets: list[int] = []
    second_offsets: list[int] = []
    targets: list[float] = []
    wdl_targets: list[float] = []
    teacher_scores: list[float] = []
    for sample in samples:
        if len(sample) == 3:
            first, second, target = sample
            wdl_target = 0.5
            teacher_score = target
        elif len(sample) == 5:
            first, second, target, wdl_target, teacher_score = sample
        else:
            raise ValueError("training samples must contain 3 or 5 values")
        first_offsets.append(len(first_flat))
        second_offsets.append(len(second_flat))
        first_flat.extend(first)
        second_flat.extend(second)
        targets.append(target)
        wdl_targets.append(wdl_target)
        teacher_scores.append(teacher_score)
    return (
        torch.tensor(first_flat, dtype=torch.long),
        torch.tensor(first_offsets, dtype=torch.long),
        torch.tensor(second_flat, dtype=torch.long),
        torch.tensor(second_offsets, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(wdl_targets, dtype=torch.float32),
        torch.tensor(teacher_scores, dtype=torch.float32),
    )


class HalfKpV1(nn.Module):
    def __init__(self, hidden: int, feature_factorization: bool = False):
        super().__init__()
        self.hidden = hidden
        self.feature_weights = nn.EmbeddingBag(FEATURE_COUNT, hidden, mode="sum")
        self.hidden_bias = nn.Parameter(torch.zeros(hidden))
        self.output = nn.Linear(hidden * 2, 1)
        nn.init.normal_(self.feature_weights.weight, mean=0.0, std=0.005)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.05)
        self.feature_factorization = feature_factorization
        self.piece_square_weights = None
        if feature_factorization:
            # Preserve baseline initialization and shuffle RNG; the shared part starts at zero.
            with torch.random.fork_rng(devices=[]):
                self.piece_square_weights = nn.EmbeddingBag(640, hidden, mode="sum")
                nn.init.zeros_(self.piece_square_weights.weight)

    def effective_input_weights(self) -> torch.Tensor:
        weights = self.feature_weights.weight
        if self.piece_square_weights is not None:
            weights = weights + self.piece_square_weights.weight.repeat(64, 1)
        return weights

    def accumulator(self, features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        value = self.feature_weights(features, offsets) + self.hidden_bias
        if self.piece_square_weights is not None:
            value = value + self.piece_square_weights(features % 640, offsets)
        return torch.clamp(value, 0.0, 1.0)

    def forward(self, first: torch.Tensor, first_offsets: torch.Tensor,
                second: torch.Tensor, second_offsets: torch.Tensor) -> torch.Tensor:
        first_activation = self.accumulator(first, first_offsets)
        second_activation = self.accumulator(second, second_offsets)
        return self.output(torch.cat((first_activation, second_activation), dim=1)).squeeze(1)


def batches(loader: DataLoader, device: torch.device) -> Iterator[tuple[torch.Tensor, ...]]:
    for batch in loader:
        yield tuple(value.to(device, non_blocking=True) for value in batch)


@dataclass
class QuantizedNetwork:
    hidden_scale: int
    output_scale: int
    hidden_bias: np.ndarray
    input_weights: np.ndarray
    output_weights: np.ndarray
    output_bias: int


def quantize_network(model: HalfKpV1, hidden_scale: int = 1024,
                     output_scale: int = 64, target_scale: float = 1.0) -> QuantizedNetwork:
    if hidden_scale <= 0 or output_scale <= 0 or target_scale <= 0:
        raise ValueError("quantization scales must be positive")
    state = model.cpu().eval()
    # Coalesce virtual features before rounding, preserving the deployed HalfKP-v1 layout.
    input_weights = np.rint(state.effective_input_weights().detach().numpy() * hidden_scale)
    if np.any(input_weights < -32768) or np.any(input_weights > 32767):
        raise ValueError("input weights exceed int16 at the requested hidden scale")
    input_weights = input_weights.astype("<i2")
    hidden_bias = np.rint(state.hidden_bias.detach().numpy() * hidden_scale)
    if (np.any(hidden_bias < np.iinfo(np.int32).min) or
            np.any(hidden_bias > np.iinfo(np.int32).max)):
        raise ValueError("hidden bias exceeds int32 at the requested hidden scale")
    hidden_bias = hidden_bias.astype("<i4")
    output_weights = np.rint(
        state.output.weight.detach().numpy().reshape(-1) * target_scale * output_scale,
    )
    if np.any(output_weights < -32768) or np.any(output_weights > 32767):
        raise ValueError("output weights exceed int16 at the requested output/target scale")
    output_weights = output_weights.astype("<i2")
    output_bias = int(round(
        float(state.output.bias.detach()[0]) * target_scale * hidden_scale * output_scale,
    ))
    if not np.iinfo(np.int32).min <= output_bias <= np.iinfo(np.int32).max:
        raise ValueError("quantized output bias exceeds the NNUE file format")
    return QuantizedNetwork(
        hidden_scale, output_scale, hidden_bias, input_weights, output_weights, output_bias,
    )


def export_network(model: HalfKpV1, destination: Path, hidden_scale: int = 1024,
                   output_scale: int = 64, target_scale: float = 1.0) -> QuantizedNetwork:
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantized = quantize_network(model, hidden_scale, output_scale, target_scale)
    with destination.open("wb") as output:
        output.write(struct.pack(
            "<8sIIIiii", MAGIC, FORMAT_VERSION, FEATURE_COUNT, model.hidden,
            hidden_scale, output_scale, quantized.output_bias,
        ))
        output.write(quantized.hidden_bias.tobytes(order="C"))
        output.write(quantized.input_weights.tobytes(order="C"))
        output.write(quantized.output_weights.tobytes(order="C"))
    return quantized


def truncating_division(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


def quantized_predict(quantized: QuantizedNetwork, first: list[int],
                      second: list[int]) -> int:
    first_accumulator = quantized.hidden_bias.astype(np.int64)
    second_accumulator = quantized.hidden_bias.astype(np.int64)
    if first:
        first_accumulator = first_accumulator + np.sum(
            quantized.input_weights[first].astype(np.int64), axis=0,
        )
    if second:
        second_accumulator = second_accumulator + np.sum(
            quantized.input_weights[second].astype(np.int64), axis=0,
        )
    first_accumulator = np.clip(first_accumulator, 0, quantized.hidden_scale)
    second_accumulator = np.clip(second_accumulator, 0, quantized.hidden_scale)
    hidden = quantized.hidden_bias.size
    output = quantized.output_bias
    output += int(np.dot(first_accumulator, quantized.output_weights[:hidden].astype(np.int64)))
    output += int(np.dot(second_accumulator, quantized.output_weights[hidden:].astype(np.int64)))
    divisor = quantized.hidden_scale * quantized.output_scale
    return max(-32_000, min(32_000, truncating_division(output, divisor)))


def verify_quantization(model: HalfKpV1, dataset: Dataset, sample_count: int,
                        seed: int, hidden_scale: int, output_scale: int,
                        target_scale: float = 1.0) -> dict[str, float | int]:
    if len(dataset) == 0 or sample_count <= 0:
        return {"samples": 0, "rmseCp": 0.0, "meanErrorCp": 0.0, "maxAbsErrorCp": 0.0}
    state = model.cpu().eval()
    quantized = quantize_network(state, hidden_scale, output_scale, target_scale)
    random_generator = random.Random(seed)
    indices = list(range(len(dataset)))
    random_generator.shuffle(indices)
    indices = indices[:min(sample_count, len(indices))]
    errors: list[float] = []
    with torch.no_grad():
        for index in indices:
            first, second, *_unused = dataset[index]
            batch = collate([(first, second, 0.0)])
            prediction = float(state(*batch[:4]).item()) * target_scale
            quantized_prediction = quantized_predict(quantized, first, second)
            errors.append(float(quantized_prediction) - prediction)
    return {
        "samples": len(errors),
        "rmseCp": round(math.sqrt(sum(error * error for error in errors) / len(errors)), 6),
        "meanErrorCp": round(sum(errors) / len(errors), 6),
        "maxAbsErrorCp": round(max(abs(error) for error in errors), 6),
    }


CPP_VERIFICATION_FENS = (
    chess.STARTING_FEN,
    "r2q1rk1/pp2bppp/2np1n2/2p1p1B1/2P1P3/2NP1N2/PP2QPPP/R4RK1 w - - 0 10",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 b - - 0 9",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
)


def verify_cpp_export(tools: Path, network: Path,
                      quantized: QuantizedNetwork) -> dict[str, object]:
    if not tools.is_file():
        raise ValueError(f"C++ tools executable does not exist: {tools}")
    positions: list[dict[str, object]] = []
    for fen in CPP_VERIFICATION_FENS:
        board = chess.Board(fen)
        expected = quantized_predict(
            quantized,
            active_features(board, board.turn),
            active_features(board, not board.turn),
        )
        result = subprocess.run(
            [str(tools.resolve()), "--eval", "--nnue", str(network.resolve()), "--fen", fen],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        match = re.search(r"\bevaluation_cp=(-?\d+)\s+backend=nnue\b", result.stdout)
        if result.returncode != 0 or match is None:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"C++ NNUE verification failed to run: {detail}")
        actual = int(match.group(1))
        positions.append({"fen": fen, "pythonCp": expected, "cppCp": actual})
        if actual != expected:
            raise RuntimeError(
                f"C++ NNUE mismatch for {fen}: Python={expected} cp C++={actual} cp"
            )
    return {
        "executable": str(tools.resolve()),
        "executableSha256": sha256_file(tools),
        "positions": positions,
        "exactMatch": True,
    }


def evaluate(
    model: HalfKpV1,
    loader: DataLoader,
    device: torch.device,
    target_scale: float,
    loss_mode: str = "centipawn-huber",
    huber_beta_cp: float = 100.0,
    wdl_scale: float = 400.0,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_brier = 0.0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_teacher_squared_error = 0.0
    correct_signs = 0
    samples = 0
    with torch.no_grad():
        for (first, first_offsets, second, second_offsets, target,
             target_wdl, teacher_score) in batches(loader, device):
            normalized_prediction = model(first, first_offsets, second, second_offsets)
            batch_loss = training_loss(
                normalized_prediction, target, target_wdl, loss_mode,
                target_scale, huber_beta_cp, wdl_scale,
            )
            prediction = normalized_prediction * target_scale
            error = prediction - target
            bounded_teacher = torch.clamp(teacher_score, -3000.0, 3000.0)
            teacher_error = prediction - bounded_teacher
            predicted_wdl = torch.sigmoid(prediction / wdl_scale)
            total_loss += batch_loss.item() * target.numel()
            total_brier += torch.sum((predicted_wdl - target_wdl) ** 2).item()
            total_squared_error += torch.sum(error ** 2).item()
            total_absolute_error += torch.sum(torch.abs(error)).item()
            total_teacher_squared_error += torch.sum(teacher_error ** 2).item()
            correct_signs += int(torch.sum((prediction >= 0) == (target >= 0)).item())
            samples += target.numel()
    denominator = max(1, samples)
    return {
        "samples": samples,
        "loss": total_loss / denominator,
        "brierScore": total_brier / denominator,
        "rmseCp": math.sqrt(total_squared_error / denominator),
        "maeCp": total_absolute_error / denominator,
        "teacherRmseCp": math.sqrt(total_teacher_squared_error / denominator),
        "signAccuracy": correct_signs / denominator,
    }


@dataclass
class ErrorAccumulator:
    samples: int = 0
    signed_error: float = 0.0
    absolute_error: float = 0.0
    squared_error: float = 0.0
    correct_signs: int = 0

    def add(self, prediction: float, target: float) -> None:
        error = prediction - target
        self.samples += 1
        self.signed_error += error
        self.absolute_error += abs(error)
        self.squared_error += error * error
        self.correct_signs += int((prediction >= 0) == (target >= 0))

    def report(self) -> dict[str, float | int]:
        denominator = max(1, self.samples)
        return {
            "samples": self.samples,
            "meanErrorCp": self.signed_error / denominator,
            "maeCp": self.absolute_error / denominator,
            "rmseCp": math.sqrt(self.squared_error / denominator),
            "signAccuracy": self.correct_signs / denominator,
        }


def diagnostic_sample(dataset: Dataset, index: int) -> TrainingSample:
    if isinstance(dataset, PositionsDataset):
        return dataset.sample(index)
    if isinstance(dataset, Subset):
        return diagnostic_sample(dataset.dataset, int(dataset.indices[index]))
    raise TypeError("validation diagnostics require PositionsDataset or Subset")


def validation_error_diagnostics(model: HalfKpV1, dataset: Dataset,
                                 device: torch.device, target_scale: float,
                                 batch_size: int) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("diagnostic batch size must be positive")
    overall = ErrorAccumulator()
    groups: dict[str, dict[str, ErrorAccumulator]] = {
        "sideToMoveKingSquare": {},
        "phase": {},
        "materialImbalance": {},
        "teacherEvalMagnitude": {},
    }

    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            samples = [diagnostic_sample(dataset, index)
                       for index in range(start, min(len(dataset), start + batch_size))]
            batch = collate([(sample.first, sample.second, sample.target) for sample in samples])
            predictions = model(*[
                value.to(device, non_blocking=True) for value in batch[:4]
            ]) * target_scale
            values = predictions.detach().cpu().tolist()
            for sample, prediction in zip(samples, values):
                target = sample.target
                overall.add(prediction, target)
                codes = unpack_piece_codes(sample.board_bytes)
                king_code = chess.KING + (0 if sample.turn == chess.WHITE else 6)
                king_square = chess.square_name(codes.index(king_code))
                phase, material = position_distribution(sample.board_bytes)
                magnitude = labelled_bucket(
                    sample.teacher_score_cp,
                    ((99, "0-99"), (299, "100-299"), (699, "300-699")),
                    "700+",
                )
                for group_name, label in (
                    ("sideToMoveKingSquare", king_square),
                    ("phase", phase),
                    ("materialImbalance", material),
                    ("teacherEvalMagnitude", magnitude),
                ):
                    groups[group_name].setdefault(label, ErrorAccumulator()).add(
                        prediction, target,
                    )

    return {
        "target": "teacher-result blend configured for training",
        "overall": overall.report(),
        **{
            group_name: {
                label: accumulator.report()
                for label, accumulator in sorted(accumulators.items())
            }
            for group_name, accumulators in groups.items()
        },
    }


def atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def save_metrics(metrics: list[dict[str, object]], json_path: Path) -> tuple[Path, Path]:
    csv_path = json_path.with_suffix(".csv")
    write_json_atomic(json_path, metrics)
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        fieldnames = list(dict.fromkeys(
            key for metric in metrics for key in metric
        )) if metrics else ["epoch"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    temporary.replace(csv_path)
    return json_path, csv_path


def git_state(root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def find_dataset_manifest(dataset_path: Path, dataset_sha256: str) -> Path | None:
    candidates = [
        dataset_path.with_suffix(dataset_path.suffix + ".manifest.json"),
        dataset_path.parent / "dataset.manifest.json",
        *sorted(dataset_path.parent.glob("*.manifest.json")),
    ]
    checked: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in checked or not candidate.is_file():
            continue
        checked.add(resolved_candidate)
        try:
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
            outputs = manifest.get("outputs", [])
            if not isinstance(outputs, list):
                continue
            for output in outputs:
                if not isinstance(output, dict):
                    continue
                output_path = Path(str(output.get("path", "")))
                if (output_path.resolve() == dataset_path.resolve() and
                        output.get("sha256") == dataset_sha256):
                    return resolved_candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def checkpoint_payload(model: HalfKpV1, optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler.LRScheduler | None,
                       epoch: int, best_objective: float, epochs_without_improvement: int,
                       metrics: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    best_rmse = min(
        (float(metric["validationRmseCp"]) for metric in metrics),
        default=math.inf,
    )
    return {
        "checkpointVersion": CHECKPOINT_VERSION,
        "epoch": epoch,
        "hidden": model.hidden,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "bestValidationRmseCp": best_rmse,
        "bestValidationObjective": best_objective,
        "bestValidationObjectiveName": (
            "validationLoss" if args.loss == "wdl" else "validationRmseCp"
        ),
        "epochsWithoutImprovement": epochs_without_improvement,
        "metrics": metrics,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "training": {
            "featureFactorization": args.feature_factorization,
            "resultWeight": args.result_weight,
            "loss": args.loss,
            "wdlScale": args.wdl_scale,
            "seed": args.seed,
            "scheduler": args.scheduler,
            "epochsRequested": args.epochs,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "minimumLearningRate": args.minimum_learning_rate,
            "targetScale": args.target_scale,
            "huberBetaCp": args.huber_beta_cp,
        },
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state.get("cuda")
    if torch.cuda.is_available() and cuda_states is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def validate_resume_configuration(payload: dict[str, object], args: argparse.Namespace) -> None:
    saved = payload.get("training")
    if not isinstance(saved, dict):
        raise SystemExit("resume checkpoint is missing training configuration")
    exact_settings = {
        "featureFactorization": args.feature_factorization,
        "resultWeight": args.result_weight,
        "loss": args.loss,
        "wdlScale": args.wdl_scale,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "minimumLearningRate": args.minimum_learning_rate,
        "targetScale": args.target_scale,
        "huberBetaCp": args.huber_beta_cp,
    }
    for key, requested in exact_settings.items():
        saved_value = saved.get(
            key,
            False if key == "featureFactorization" else
            "centipawn-huber" if key == "loss" else (400.0 if key == "wdlScale" else None),
        )
        if saved_value != requested:
            raise SystemExit(
                f"resume checkpoint {key}={saved_value!r} does not match requested {requested!r}"
            )
    if args.scheduler == "cosine" and saved.get("epochsRequested", args.epochs) != args.epochs:
        raise SystemExit("cosine-scheduler resume must use the original --epochs value")


def validate_args(args: argparse.Namespace) -> None:
    if args.cpu_threads < 1:
        raise SystemExit("--cpu-threads must be positive")
    if not 1 <= args.hidden <= 1024:
        raise SystemExit("--hidden must be between 1 and 1024 (the C++ loader limit)")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("--epochs and --batch-size must be positive; --workers cannot be negative")
    if args.learning_rate <= 0 or args.minimum_learning_rate < 0:
        raise SystemExit("learning rates must be non-negative and the initial rate must be positive")
    if not 0.0 <= args.result_weight <= 1.0:
        raise SystemExit("--result-weight must be in [0, 1]")
    if args.target_scale <= 0 or args.huber_beta_cp <= 0 or args.wdl_scale <= 0:
        raise SystemExit("--target-scale, --huber-beta-cp, and --wdl-scale must be positive")
    if args.validation_data is None and not args.allow_position_split:
        raise SystemExit("--validation-data is required unless --allow-position-split is explicit")
    if not args.validation_data and not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in (0, 1)")
    if args.checkpoint_every < 1 or args.early_stopping_patience < 0 or args.verify_samples < 0:
        raise SystemExit("checkpoint cadence and verification counts must be valid")
    if args.hidden_scale <= 0 or args.output_scale <= 0:
        raise SystemExit("quantization scales must be positive")
    if args.cpp_tools and not args.cpp_tools.is_file():
        raise SystemExit(f"--cpp-tools does not exist: {args.cpp_tools}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but the installed PyTorch build cannot use it")


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    torch.set_num_threads(args.cpu_threads)
    started_at = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics or args.output.with_suffix(".metrics.json")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    checkpoint_path = args.output.with_suffix(".checkpoint.pt")
    best_checkpoint_path = args.output.with_suffix(".best.checkpoint.pt")
    periodic_directory = args.output.with_suffix(".checkpoints")
    progress_path = args.output.with_suffix(".progress.json")
    progress = {
        "schemaVersion": 1, "state": "preparing", "pid": os.getpid(),
        "startedAt": started_at.isoformat(), "epoch": 0, "epochs": args.epochs,
        "featureFactorization": args.feature_factorization, "device": args.device,
    }

    def update_progress(state: str, **values: object) -> None:
        progress.update(values)
        progress.update(state=state, updatedAt=datetime.now(timezone.utc).isoformat())
        write_json_atomic(progress_path, progress)

    update_progress("preparing")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    dataset = PositionsDataset(args.data, args.result_weight, args.wdl_scale)
    if len(dataset) < 100:
        raise SystemExit("dataset must contain at least 100 positions")
    split_method = "game-disjoint-shards"
    if args.validation_data:
        training: Dataset = dataset
        validation: Dataset = PositionsDataset(
            args.validation_data, args.result_weight, args.wdl_scale,
        )
        if len(validation) == 0:
            raise SystemExit("validation dataset is empty")
    else:
        split_method = "random-position-smoke-only"
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        validation_size = max(1, int(len(indices) * args.validation_fraction))
        validation = Subset(dataset, indices[:validation_size])
        training = Subset(dataset, indices[validation_size:])

    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    training_loader = DataLoader(training, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)

    model = HalfKpV1(args.hidden, args.feature_factorization).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.minimum_learning_rate,
        )
    start_epoch = 1
    best_objective = math.inf
    epochs_without_improvement = 0
    metrics: list[dict[str, object]] = []
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("checkpointVersion") != CHECKPOINT_VERSION:
            raise SystemExit("resume file is not a supported versioned training checkpoint")
        if int(payload.get("hidden", -1)) != args.hidden:
            raise SystemExit("resume checkpoint hidden size does not match --hidden")
        validate_resume_configuration(payload, args)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler and payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_objective = float(payload.get(
            "bestValidationObjective",
            payload.get("bestValidationRmseCp", math.inf),
        ))
        epochs_without_improvement = int(payload.get("epochsWithoutImprovement", 0))
        metrics = list(payload.get("metrics", []))
        rng_state = payload.get("rng")
        if not isinstance(rng_state, dict):
            raise SystemExit("resume checkpoint is missing random-number generator state")
        restore_rng_state(rng_state)
        print(
            f"resumed={args.resume} next_epoch={start_epoch} "
            f"best_objective={best_objective:.6f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        total_loss = 0.0
        samples = 0
        last_progress = 0.0
        update_progress("training", epoch=epoch, positions=0, trainingPositions=len(training),
                        validationPositions=len(validation))
        for (first, first_offsets, second, second_offsets, target,
             target_wdl, _teacher_score) in batches(training_loader, device):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(first, first_offsets, second, second_offsets)
            loss = training_loss(
                prediction, target, target_wdl, args.loss,
                args.target_scale, args.huber_beta_cp, args.wdl_scale,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.numel()
            samples += target.numel()
            now = time.perf_counter()
            if now - last_progress >= 2.0:
                update_progress("training", positions=samples,
                                positionsPerSecond=round(samples / max(now - epoch_started, 1e-9), 1),
                                trainingLoss=total_loss / max(1, samples),
                                gpuMemoryBytes=torch.cuda.memory_allocated(device) if device.type == "cuda" else 0)
                last_progress = now
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - epoch_started
        update_progress("validating", positions=samples)
        validation_metrics = evaluate(
            model, validation_loader, device, args.target_scale,
            args.loss, args.huber_beta_cp, args.wdl_scale,
        )
        rmse = float(validation_metrics["rmseCp"])
        validation_loss = float(validation_metrics["loss"])
        objective = validation_loss if args.loss == "wdl" else rmse
        learning_rate = float(optimizer.param_groups[0]["lr"])
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        metric: dict[str, object] = {
            "epoch": epoch,
            "lossMode": args.loss,
            "trainingLoss": round(total_loss / max(1, samples), 8),
            "trainingLossNormalized": round(total_loss / max(1, samples), 6),
            "validationLoss": round(validation_loss, 8),
            "validationBrierScore": round(float(validation_metrics["brierScore"]), 8),
            "validationRmseCp": round(rmse, 6),
            "validationMaeCp": round(float(validation_metrics["maeCp"]), 6),
            "validationTeacherRmseCp": round(
                float(validation_metrics["teacherRmseCp"]), 6,
            ),
            "validationSignAccuracy": round(float(validation_metrics["signAccuracy"]), 6),
            "learningRate": learning_rate,
            "durationSeconds": round(duration, 6),
            "positionsPerSecond": round(samples / max(duration, 1e-9), 3),
            "gpuPeakMemoryBytes": peak_memory,
        }
        metrics.append(metric)

        improved = objective < best_objective
        if improved:
            best_objective = objective
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler:
            scheduler.step()

        payload = checkpoint_payload(
            model, optimizer, scheduler, epoch, best_objective,
            epochs_without_improvement, metrics, args,
        )
        atomic_torch_save(payload, checkpoint_path)
        if improved:
            atomic_torch_save(payload, best_checkpoint_path)
        if epoch % args.checkpoint_every == 0:
            atomic_torch_save(payload, periodic_directory / f"epoch-{epoch:04d}.pt")
        save_metrics(metrics, metrics_path)
        update_progress("checkpointed", latestMetric=metric, bestObjective=best_objective)
        print(
            f"epoch={epoch} train_loss_normalized={metric['trainingLossNormalized']:.4f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_rmse_cp={rmse:.2f} "
            f"validation_mae_cp={metric['validationMaeCp']:.2f} "
            f"validation_sign_accuracy={metric['validationSignAccuracy']:.3f} "
            f"positions_per_second={metric['positionsPerSecond']:.0f} "
            f"gpu_peak_mb={peak_memory / (1024 * 1024):.1f}",
            flush=True,
        )

        if (args.early_stopping_patience > 0 and
                epochs_without_improvement >= args.early_stopping_patience):
            print(f"early_stopping epoch={epoch} patience={args.early_stopping_patience}")
            break

    if best_checkpoint_path.is_file():
        best_payload = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model"])

    update_progress("diagnosing")
    best_validation_metrics = evaluate(
        model, validation_loader, device, args.target_scale,
        args.loss, args.huber_beta_cp, args.wdl_scale,
    )
    best_validation_slices = validation_error_diagnostics(
        model, validation, device, args.target_scale, args.batch_size,
    )

    update_progress("exporting")
    quantization = verify_quantization(
        model, validation, args.verify_samples, args.seed,
        args.hidden_scale, args.output_scale, args.target_scale,
    )
    quantized = export_network(
        model, args.output, args.hidden_scale, args.output_scale, args.target_scale,
    )
    cpp_verification = (
        verify_cpp_export(args.cpp_tools, args.output, quantized) if args.cpp_tools else None
    )
    model_path = args.output.with_suffix(".pt")
    atomic_torch_save(model.state_dict(), model_path)
    metrics_json, metrics_csv = save_metrics(metrics, metrics_path)

    completed_at = datetime.now(timezone.utc)
    data_inputs: list[dict[str, object]] = []
    position_counts = {source.path: source.count for source in dataset.sources}
    if isinstance(validation, PositionsDataset):
        position_counts.update({source.path: source.count for source in validation.sources})
    for split, paths in (("training", args.data), ("validation", args.validation_data or [])):
        for path in paths:
            resolved = path.resolve()
            dataset_sha256 = sha256_file(resolved)
            dataset_manifest = find_dataset_manifest(resolved, dataset_sha256)
            data_inputs.append({
                "split": split,
                "path": str(resolved),
                "positions": position_counts.get(resolved),
                "sizeBytes": resolved.stat().st_size,
                "sha256": dataset_sha256,
                "manifest": ({
                    "path": str(dataset_manifest),
                    "sha256": sha256_file(dataset_manifest),
                } if dataset_manifest else None),
            })

    device_manifest: dict[str, object] = {
        "requested": args.device,
        "torch": torch.__version__,
        "cudaRuntime": torch.version.cuda,
        "cudaAvailable": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_manifest.update({
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "memoryBytes": properties.total_memory,
        })

    manifest = {
        "schemaVersion": 1,
        "networkFormat": "HalfKP-v1",
        "startedAt": started_at.isoformat(),
        "completedAt": completed_at.isoformat(),
        "durationSeconds": round((completed_at - started_at).total_seconds(), 3),
        "trainer": {**git_state(root), "script": str(Path(__file__).resolve())},
        "device": device_manifest,
        "data": {
            "splitMethod": split_method,
            "trainingPositions": len(training),
            "validationPositions": len(validation),
            "inputs": data_inputs,
        },
        "configuration": {
            "featureFactorization": args.feature_factorization,
            "cpuThreads": args.cpu_threads,
            "hidden": args.hidden,
            "epochsRequested": args.epochs,
            "epochsCompleted": metrics[-1]["epoch"] if metrics else 0,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "minimumLearningRate": args.minimum_learning_rate,
            "scheduler": args.scheduler,
            "resultWeight": args.result_weight,
            "loss": args.loss,
            "wdlScale": args.wdl_scale,
            "targetScale": args.target_scale,
            "huberBetaCp": args.huber_beta_cp,
            "seed": args.seed,
            "hiddenScale": args.hidden_scale,
            "outputScale": args.output_scale,
            "resumedFrom": str(args.resume.resolve()) if args.resume else None,
        },
        "bestValidationRmseCp": best_validation_metrics["rmseCp"],
        "bestValidationObjective": {
            "name": "validationLoss" if args.loss == "wdl" else "validationRmseCp",
            "value": best_objective,
        },
        "bestValidation": best_validation_metrics,
        "bestValidationSlices": best_validation_slices,
        "quantization": quantization,
        "cppVerification": cpp_verification,
        "outputs": {
            "network": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
            "model": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)},
            "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
            "bestCheckpoint": ({
                "path": str(best_checkpoint_path.resolve()),
                "sha256": sha256_file(best_checkpoint_path),
            } if best_checkpoint_path.is_file() else None),
            "metricsJson": {"path": str(metrics_json.resolve()), "sha256": sha256_file(metrics_json)},
            "metricsCsv": {"path": str(metrics_csv.resolve()), "sha256": sha256_file(metrics_csv)},
        },
    }
    write_json_atomic(manifest_path, manifest)
    update_progress("complete", network=str(args.output.resolve()),
                    bestValidation=best_validation_metrics, cppVerification=cpp_verification)
    print(
        f"exported={args.output} hidden={args.hidden} train_positions={len(training)} "
        f"validation_positions={len(validation)} "
        f"best_rmse_cp={float(best_validation_metrics['rmseCp']):.2f} "
        f"best_objective={best_objective:.6f} "
        f"quantization_rmse_cp={quantization['rmseCp']:.3f} manifest={manifest_path}"
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return train(args)
    except (Exception, KeyboardInterrupt) as error:
        progress_path = args.output.with_suffix(".progress.json")
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            progress = {}
        progress.update(state="failed", error=str(error) or type(error).__name__,
                        updatedAt=datetime.now(timezone.utc).isoformat())
        write_json_atomic(progress_path, progress)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
