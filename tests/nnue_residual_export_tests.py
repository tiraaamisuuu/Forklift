#!/usr/bin/env python3
"""CPU-only synthetic Python/C++ export and warm-start/resume integration tests.

All weights, labels and datasets are disposable fixtures, not strength evidence.
Run with --tools /absolute/path/to/chess-engine-tools.exe.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

# Set these before importing NumPy/PyTorch, including when run standalone.
os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

import chess
import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

from nnue_dataset import BinaryShardWriter  # noqa: E402
from train import HalfKpV1, export_network, main, verify_cpp_export  # noqa: E402

TOOLS: Path | None = None


class NnueResidualExportTests(unittest.TestCase):
    def test_nonzero_export_matrix_matches_cpp_exactly(self) -> None:
        self.assertIsNotNone(TOOLS)
        # Exercise scalar tails, SIMD widths, direct and residual formats, shared
        # feature folding, and non-power-of-two residual quantization division.
        cases = (
            (1, 0, 7, False), (1, 1, 7, True),
            (4, 3, 64, False), (4, 4, 256, True),
            (7, 8, 512, True), (7, 1, 64, False),
            (8, 0, 256, True), (8, 3, 7, True),
            (17, 4, 512, False), (17, 8, 64, True),
            (8, 4, 256, False), (17, 1, 7, True),
        )
        comparisons = 0
        with tempfile.TemporaryDirectory(prefix="forklift-export-test-") as directory:
            for index, (hidden, residual, scale, shared) in enumerate(cases):
                with self.subTest(hidden=hidden, residual=residual,
                                  scale=scale, shared=shared):
                    torch.manual_seed(1200 + index)
                    model = HalfKpV1(hidden, feature_factorization=shared,
                                     residual_hidden=residual)
                    with torch.no_grad():
                        model.hidden_bias.fill_(0.2)
                        model.output.bias.fill_(0.15)
                        if shared:
                            model.piece_square_weights.weight.uniform_(-0.003, 0.003)
                        if residual:
                            model.residual_hidden_layer.weight.uniform_(-0.2, 0.2)
                            model.residual_hidden_layer.bias.fill_(0.25)
                            model.residual_output.weight.uniform_(0.02, 0.08)
                    network = Path(directory) / f"case-{index}.nnue"
                    quantized = export_network(model, network, target_scale=600.0,
                                               residual_weight_scale=scale)
                    if residual:
                        self.assertTrue(np.any(quantized.residual_output_weights != 0))
                        self.assertTrue(np.any(quantized.residual_bias != 0))
                    report = verify_cpp_export(TOOLS, network, quantized)
                    self.assertTrue(report["exactMatch"])
                    self.assertEqual(len(report["positions"]), 4)
                    self.assertTrue(any(row["pythonCp"] != 0 for row in report["positions"]))
                    comparisons += len(report["positions"])
        print(f"export_matrix_cases={len(cases)} exact_cpp_comparisons={comparisons}")

    def test_tiny_frozen_training_export_and_resume(self) -> None:
        self.assertIsNotNone(TOOLS)
        with tempfile.TemporaryDirectory(prefix="forklift-train-test-") as directory:
            fixture = Path(directory)
            torch.manual_seed(141)
            base = HalfKpV1(4, feature_factorization=True)
            initial = fixture / "base.pt"
            torch.save(base.state_dict(), initial)
            train_data, validation_data = fixture / "train.nnuebin", fixture / "validation.nnuebin"
            for path, count, first_move, offset in (
                (train_data, 100, "e2e4", 0),
                (validation_data, 20, "d2d4", 1000),
            ):
                with BinaryShardWriter(path) as writer:
                    for index in range(count):
                        board = chess.Board()
                        board.push_uci(first_move)
                        replies = list(board.legal_moves)
                        board.push(replies[index % len(replies)])
                        # Deliberately artificial labels: tests pipeline mechanics only.
                        writer.write(board, (index % 9 - 4) * 25, 0.0,
                                     game_id=offset + index, ply=2)
            output = fixture / "model.nnue"
            common = [
                "train.py", "--data", str(train_data),
                "--validation-data", str(validation_data), "--output", str(output),
                "--hidden", "4", "--feature-factorization", "--residual-hidden", "4",
                "--freeze-transformer", "--cpu-threads", "1", "--workers", "0",
                "--device", "cpu", "--batch-size", "20", "--scheduler", "none",
                "--verify-samples", "20", "--cpp-tools", str(TOOLS),
            ]
            reports = []
            for extra in (
                ["--initialize-from", str(initial), "--epochs", "1"],
                ["--resume", str(output.with_suffix(".checkpoint.pt")), "--epochs", "2"],
            ):
                with mock.patch.object(sys, "argv", common + extra):
                    self.assertEqual(main(), 0)
                report = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
                reports.append(report)
                self.assertTrue(report["cppVerification"]["exactMatch"])
                self.assertEqual(report["networkFormat"], "HalfKP-v2-residual")
                self.assertEqual(report["architecture"]["formatVersion"], 2)
                self.assertTrue(report["configuration"]["freezeTransformer"])
                self.assertEqual(report["configuration"]["residualHidden"], 4)
                self.assertFalse(report["device"]["cudaAvailable"])
                self.assertEqual(report["device"]["requested"], "cpu")
                self.assertEqual(report["data"]["trainingPositions"], 100)
                self.assertEqual(report["data"]["validationPositions"], 20)
                final = torch.load(output.with_suffix(".pt"), weights_only=True)
                for key in ("feature_weights.weight", "piece_square_weights.weight", "hidden_bias"):
                    self.assertTrue(torch.equal(final[key], base.state_dict()[key]), key)
            self.assertEqual(reports[0]["initialization"], reports[1]["initialization"])
            self.assertEqual(reports[0]["initialization"]["source"], str(initial.resolve()))
            self.assertTrue(reports[0]["initialization"]["newZeroOutputResidualHead"])
            self.assertEqual(reports[0]["configuration"]["epochsCompleted"], 1)
            self.assertEqual(reports[1]["configuration"]["epochsCompleted"], 2)
            checkpoint = torch.load(output.with_suffix(".checkpoint.pt"), weights_only=False)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["initialization"], reports[0]["initialization"])
            self.assertTrue(checkpoint["training"]["freezeTransformer"])
            self.assertEqual(checkpoint["training"]["residualHidden"], 4)
            print("training_epochs=2 frozen_transformer_exact=true cpp_comparisons=8 provenance_retained=true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools", required=True, type=Path)
    arguments, remainder = parser.parse_known_args()
    TOOLS = arguments.tools.resolve()
    if not TOOLS.is_file():
        parser.error("--tools executable does not exist")
    started = time.monotonic()
    try:
        unittest.main(argv=[sys.argv[0], *remainder])
    finally:
        print(f"elapsed_seconds={time.monotonic()-started:.3f}")
