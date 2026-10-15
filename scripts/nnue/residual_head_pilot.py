#!/usr/bin/env python3
"""Bounded CPU-only head-learning diagnostic; not a strength experiment.

Read only a deterministic random subset of already-labelled compact shards.
Keep the feature transformer frozen and give direct/R4/R8 the same eight
updates. Source shard hashes are copied from provenance, never recomputed by
scanning a large corpus. Outputs must be placed in a new run directory.
"""

from __future__ import annotations

import os

# Set before importing NumPy/PyTorch: this pilot must never acquire a GPU or
# silently use several CPU pools while the shared machine is in use.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import struct
import subprocess
import time

_STARTED = time.perf_counter()

import torch
from torch.utils.data import DataLoader
import chess
import numpy as np

from initialization import freeze_feature_transformer, load_initial_weights
from nnue_dataset import (
    SHARD_HEADER, SHARD_RECORD, active_features_from_packed, decode_record,
    active_features, read_shard_header, sha256_file, unpack_piece_codes, write_json_atomic,
)
from train import (
    CPP_VERIFICATION_FENS, HalfKpV1, QuantizedNetwork, collate, evaluate,
    export_network, quantized_predict, score_to_win_probability, training_loss,
)


TRAINING_ROWS = 2048
VALIDATION_ROWS = 512
BATCH_SIZE = 256
SEED = 20261015


def check_deadline(maximum_seconds: float) -> None:
    if time.perf_counter() - _STARTED > maximum_seconds:
        raise TimeoutError("bounded pilot exceeded its elapsed-time budget")


def sample_shard(path: Path, count: int, seed: int) -> tuple[list[tuple], dict, set[int]]:
    """Read only the header and `count` fixed-width records, with no full scan."""
    path = path.resolve(strict=True)
    samples: list[tuple] = []
    game_ids: set[int] = set()
    with path.open("rb") as source:
        header = read_shard_header(source)
        if path.stat().st_size != SHARD_HEADER.size + header.count * SHARD_RECORD.size:
            raise ValueError(f"source shard size/count mismatch: {path}")
        if header.count < count:
            raise ValueError(f"source shard contains fewer than {count} records: {path}")
        # random.sample(range(...)) does not materialise/shuffle the full corpus.
        indices = sorted(random.Random(seed).sample(range(header.count), count))
        for index in indices:
            source.seek(SHARD_HEADER.size + index * SHARD_RECORD.size)
            record = decode_record(source.read(SHARD_RECORD.size))
            score = max(-3000.0, min(3000.0, float(record.score_cp)))
            samples.append((
                active_features_from_packed(record.board_bytes, record.turn),
                active_features_from_packed(record.board_bytes, not record.turn),
                score, score_to_win_probability(score, 400.0), float(record.score_cp),
            ))
            game_ids.add(record.game_id)

    # Metadata-only provenance lookup. The declaration is explicitly not a
    # freshly verified full-file digest: doing so would defeat bounded reads.
    manifest_path = path.parent / "dataset.manifest.json"
    declared_sha256 = None
    manifest_sha256 = None
    if manifest_path.is_file():
        if manifest_path.stat().st_size > 1024 * 1024:
            raise ValueError("source manifest unexpectedly exceeds one MiB")
        payload = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(payload).hexdigest()
        manifest = json.loads(payload)
        for output in manifest.get("outputs", []):
            if Path(output.get("path", "")).resolve() == path:
                declared_sha256 = output.get("sha256")
                break
    encoded_indices = json.dumps(indices, separators=(",", ":")).encode("ascii")
    return samples, {
        "path": str(path), "sizeBytes": path.stat().st_size,
        "modifiedTimeNs": path.stat().st_mtime_ns,
        "availableRows": header.count, "sampledRows": count,
        "selectionSeed": seed, "indices": indices,
        "indicesSha256": hashlib.sha256(encoded_indices).hexdigest(),
        "indicesEncoding": "UTF-8 compact JSON array, sorted ascending",
        "declaredSourceSha256": declared_sha256,
        "sourceDigestVerifiedThisRun": False,
        "manifestPath": str(manifest_path) if manifest_path.is_file() else None,
        "manifestSha256": manifest_sha256,
        "recordBytesRead": count * SHARD_RECORD.size,
        "distinctSampledGameIds": len(game_ids),
    }, game_ids


def read_exported_network(path: Path) -> tuple[QuantizedNetwork, bytes]:
    """Parse the exact V1/V2 layout without loading float model/optimizer state."""
    payload = path.read_bytes()
    magic, version, features, hidden, a_scale, o_scale, bias = struct.unpack_from("<8sIIIiii", payload)
    if ((magic, version) not in ((b"TNNUE1\0\0", 1), (b"TNNUE2\0\0", 2))
            or features != 40960 or not 1 <= hidden <= 1024
            or a_scale <= 0 or o_scale <= 0):
        raise ValueError("invalid exported NNUE header")
    offset = 32
    width, weight_scale = 0, 256
    if version == 2:
        width, weight_scale = struct.unpack_from("<Ii", payload, offset)
        offset += 8
        if not 1 <= width <= 32 or weight_scale <= 0:
            raise ValueError("invalid exported residual header")
    transformer_start = offset

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal offset
        result = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
        offset += np.dtype(dtype).itemsize * count
        return result

    hidden_bias = take("<i4", hidden)
    input_weights = take("<i2", features * hidden).reshape(features, hidden)
    transformer_bytes = payload[transformer_start:offset]
    output_weights = take("<i2", hidden * 2)
    residual_bias = take("<i4", width) if width else None
    residual_weights = take("<i2", width * hidden * 2).reshape(width, hidden * 2) if width else None
    residual_output_weights = take("<i2", width) if width else None
    if offset != len(payload):
        raise ValueError("exported NNUE payload size does not match its dimensions")
    return QuantizedNetwork(
        a_scale, o_scale, hidden_bias, input_weights, output_weights, bias,
        width, weight_scale, residual_bias, residual_weights, residual_output_weights,
    ), transformer_bytes


def verify_existing_run(args: argparse.Namespace) -> int:
    """Compare saved exports with independent int64 math and two C++ binaries."""
    verification_started = time.perf_counter()
    if not args.baseline_tools or not args.cpp_tools or not args.base_network:
        raise ValueError("verification requires --baseline-tools, --cpp-tools and --base-network")
    directory = args.output_directory.resolve(strict=True)
    verification_path = directory / "verification.json"
    if verification_path.exists():
        raise ValueError("verification report already exists; preserve the original evidence")
    report = json.loads((directory / "pilot.json").read_text(encoding="utf-8"))
    baseline_tools, tools = args.baseline_tools.resolve(strict=True), args.cpp_tools.resolve(strict=True)
    base_path = args.base_network.resolve(strict=True)
    _, base_transformer = read_exported_network(base_path)
    fens = list(CPP_VERIFICATION_FENS)
    validation = report["data"]["validation"]
    # Eight recorded sampled rows spread through the validation shard. Compact
    # records omit rights/clocks; HalfKP uses only placement and side to move.
    with Path(validation["path"]).open("rb") as source:
        read_shard_header(source)
        for index in validation["indices"][::64][:8]:
            source.seek(SHARD_HEADER.size + index * SHARD_RECORD.size)
            record = decode_record(source.read(SHARD_RECORD.size))
            board = chess.Board(None)
            for square, code in enumerate(unpack_piece_codes(record.board_bytes)):
                if code:
                    board.set_piece_at(square, chess.Piece(code if code <= 6 else code - 6, code <= 6))
            board.turn = record.turn
            fens.append(board.fen())
    arms = []
    for arm in report["arms"]:
        network_path = Path(arm["network"]["path"])
        if sha256_file(network_path) != arm["network"]["sha256"]:
            raise ValueError("saved pilot network no longer matches original evidence")
        quantized, transformer = read_exported_network(network_path)
        if transformer != base_transformer:
            raise AssertionError("frozen transformer bytes changed from original source network")
        positions = []
        for fen in fens:
            if time.perf_counter() - verification_started > 10:
                raise TimeoutError("export-only parity checks exceeded ten seconds")
            board = chess.Board(fen)
            expected = quantized_predict(quantized, active_features(board, board.turn),
                                         active_features(board, not board.turn))
            row = {"fen": fen, "pythonInt64Cp": expected}
            for key, executable in (("oldCppCp", baseline_tools), ("newCppCp", tools)):
                run = subprocess.run(
                    [str(executable), "--eval", "--nnue", str(network_path), "--fen", fen],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=2, check=False,
                )
                match = re.search(r"\bevaluation_cp=(-?\d+)\s+backend=nnue\b", run.stdout)
                if run.returncode or match is None:
                    raise RuntimeError(f"C++ parity check failed: {run.stderr or run.stdout}")
                row[key] = int(match.group(1))
                if row[key] != expected:
                    raise AssertionError(f"integer parity mismatch: {row}")
            positions.append(row)
        arms.append({"arm": arm["arm"], "transformerBytesMatchOriginal": True,
                     "positions": positions})
    result = {
        "schemaVersion": 1, "kind": "post-pilot-export-only-verification",
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "elapsedSeconds": time.perf_counter() - verification_started,
        "trainingRepeated": False, "pythonReference": "train.quantized_predict independent int64",
        "scriptSha256": sha256_file(Path(__file__)),
        "originalPilotScriptSha256": report["script"]["sha256"],
        "baseNetwork": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "oldTools": {"path": str(baseline_tools), "sha256": sha256_file(baseline_tools)},
        "newTools": {"path": str(tools), "sha256": sha256_file(tools)},
        "positionsPerArm": len(fens), "arms": arms, "exactMatch": True,
        "cppComparisons": len(fens) * len(arms) * 2,
        "fenCaveat": "Sampled compact records reconstruct placement/turn only; "
                     "castling/en-passant/clocks are unavailable and unnecessary for this HalfKP evaluation.",
    }
    write_json_atomic(verification_path, result)
    print(f"export-only parity passed: {result['cppComparisons']} C++ comparisons; "
          f"three frozen transformers unchanged; elapsed={result['elapsedSeconds']:.3f}s", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--maximum-seconds", type=float, default=180.0)
    parser.add_argument("--verify-only-existing-run", action="store_true")
    parser.add_argument("--baseline-tools", type=Path)
    parser.add_argument("--cpp-tools", type=Path)
    parser.add_argument("--base-network", type=Path)
    args = parser.parse_args()
    if args.verify_only_existing_run:
        return verify_existing_run(args)
    if not args.training_data or not args.validation_data or not args.initialize_from:
        raise ValueError("training mode requires --training-data, --validation-data and --initialize-from")
    if not 0 < args.maximum_seconds <= 180:
        raise ValueError("maximum seconds must be in (0, 180]")
    training_path = args.training_data.resolve(strict=True)
    validation_path = args.validation_data.resolve(strict=True)
    initial_path = args.initialize_from.resolve(strict=True)
    if training_path == validation_path or training_path.samefile(validation_path):
        raise ValueError("training and validation sources must be distinct files")
    output_directory = args.output_directory.resolve()
    # Refuse reuse so this one-shot diagnostic cannot overwrite previous evidence.
    output_directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    started_at = datetime.now(timezone.utc).isoformat()
    check_deadline(args.maximum_seconds)
    training, training_identity, training_games = sample_shard(training_path, TRAINING_ROWS, SEED)
    validation, validation_identity, validation_games = sample_shard(validation_path, VALIDATION_ROWS, SEED + 1)
    overlap = sorted(training_games & validation_games)
    if overlap:
        raise ValueError(f"sampled cross-split game IDs overlap: {overlap}")
    validation_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=0, collate_fn=collate)
    results: list[dict] = []
    baseline_prefit = None
    for residual_width in (0, 4, 8):
        check_deadline(args.maximum_seconds)
        arm_started = time.perf_counter()
        torch.manual_seed(SEED)
        model = HalfKpV1(256, feature_factorization=True, residual_hidden=residual_width)
        initialization = load_initial_weights(model, initial_path)
        frozen_count = freeze_feature_transformer(model)
        model.to(device)
        prefit = evaluate(model, validation_loader, device, 600.0, "wdl", 100.0, 400.0)
        if baseline_prefit is None:
            baseline_prefit = prefit
        elif prefit != baseline_prefit:
            raise AssertionError("zero-output residual arm changed prefit float predictions")
        training_loader = DataLoader(
            training, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
            collate_fn=collate, generator=torch.Generator().manual_seed(SEED),
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-4, weight_decay=1e-6,
        )
        model.train()
        updates = 0
        summed_loss = 0.0
        for first, first_offsets, second, second_offsets, target, wdl_target, _ in training_loader:
            check_deadline(args.maximum_seconds)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(first, first_offsets, second, second_offsets)
            loss = training_loss(prediction, target, wdl_target, "wdl", 600.0, 100.0, 400.0)
            loss.backward()
            optimizer.step()
            summed_loss += loss.item() * target.numel()
            updates += 1
        if updates != 8:
            raise AssertionError("pilot update budget changed")
        postfit = evaluate(model, validation_loader, device, 600.0, "wdl", 100.0, 400.0)
        check_deadline(args.maximum_seconds)
        name = "direct" if residual_width == 0 else f"residual-{residual_width}"
        network_path = output_directory / f"{name}.nnue"
        quantized = export_network(model, network_path, 1024, 32, 600.0, 256)
        nonzero_output = (int((quantized.residual_output_weights != 0).sum())
                          if quantized.residual_output_weights is not None else 0)
        result = {
            "arm": name, "residualWidth": residual_width, "initialization": initialization,
            "frozenTransformerParameters": frozen_count, "updates": updates,
            "trainingPositionsSeen": TRAINING_ROWS, "trainingLoss": summed_loss / TRAINING_ROWS,
            "validationPrefit": prefit, "validationPostfit": postfit,
            "validationLossChangeFromPrefit": postfit["loss"] - prefit["loss"],
            "quantizedNonzeroResidualOutputWeights": nonzero_output,
            "network": {"path": str(network_path), "sha256": sha256_file(network_path)},
            "elapsedSeconds": time.perf_counter() - arm_started,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, quantized

    direct_loss = results[0]["validationPostfit"]["loss"]
    for result in results:
        result["validationLossChangeFromDirectControl"] = result["validationPostfit"]["loss"] - direct_loss
    report = {
        "schemaVersion": 1, "experiment": "bounded-frozen-transformer-head-pilot",
        "startedAt": started_at, "completedAt": datetime.now(timezone.utc).isoformat(),
        "elapsedSeconds": time.perf_counter() - _STARTED,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "configuration": {
            "device": "cpu", "cpuThreads": 1, "interopThreads": 1, "workers": 0,
            "cudaVisibleDevices": os.environ["CUDA_VISIBLE_DEVICES"],
            "epochs": 1, "trainingRows": TRAINING_ROWS, "validationRows": VALIDATION_ROWS,
            "batchSize": BATCH_SIZE, "learningRate": 1e-4, "weightDecay": 1e-6,
            "scheduler": "none", "seed": SEED, "loss": "wdl", "resultWeight": 0.0,
            "targetScale": 600, "wdlScale": 400, "hidden": 256,
            "featureFactorization": True, "hiddenScale": 1024, "outputScale": 32,
            "residualWeightScale": 256, "maximumSeconds": args.maximum_seconds,
        },
        "data": {"training": training_identity, "validation": validation_identity,
                 "sampledGameIdIntersection": overlap,
                 "splitCaveat": "Sampled game IDs only; not a full source-content/split audit. "
                                "Validation rows reuse an existing validation split, not a new independent test set."},
        "arms": results,
        "interpretation": "Exploratory tiny one-seed one-epoch learning/export diagnostic only. "
                          "No novelty, generalization, Elo, playing-strength, or promotion claim. "
                          "Compare residuals with the equally fine-tuned direct control, not only prefit.",
        "cppVerification": "not-run; awaiting rebuilt tools",
    }
    write_json_atomic(output_directory / "pilot.json", report)
    print(f"completed pilot in {report['elapsedSeconds']:.3f}s; report={output_directory / 'pilot.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
