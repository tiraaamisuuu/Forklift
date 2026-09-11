#!/usr/bin/env python3

from __future__ import annotations

import json
import io
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import chess
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

from nnue_dataset import BinaryShardWriter  # noqa: E402
from train import (  # noqa: E402
    FEATURE_COUNT,
    FORMAT_VERSION,
    MAGIC,
    HalfKpV1,
    PositionsDataset,
    active_features,
    collate,
    evaluate,
    export_network,
    find_dataset_manifest,
    main,
    quantize_network,
    quantized_predict,
    restore_rng_state,
    score_to_win_probability,
    training_loss,
    truncating_division,
    validation_error_diagnostics,
    verify_quantization,
)


class NnueTrainingTests(unittest.TestCase):
    def test_factorization_preserves_initialization_and_folds_exactly(self) -> None:
        torch.manual_seed(41)
        plain = HalfKpV1(8)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(41)
        shared = HalfKpV1(8, feature_factorization=True)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(plain.feature_weights.weight, shared.effective_input_weights()))
        self.assertTrue(torch.equal(plain.output.weight, shared.output.weight))
        with torch.no_grad():
            shared.piece_square_weights.weight.uniform_(-0.01, 0.01)
            plain.feature_weights.weight.copy_(shared.effective_input_weights())
        for fen in (chess.STARTING_FEN, "8/8/4k3/3p4/4P3/3K4/8/8 w - - 0 1"):
            board = chess.Board(fen)
            batch = collate([(active_features(board, board.turn), active_features(board, not board.turn), 0.0)])
            with torch.no_grad():
                torch.testing.assert_close(shared(*batch[:4]), plain(*batch[:4]), atol=1e-6, rtol=1e-6)
        a, b = quantize_network(shared), quantize_network(plain)
        np.testing.assert_array_equal(a.input_weights, b.input_weights)

    def test_shared_training_updates_unseen_king_bucket(self) -> None:
        model = HalfKpV1(2, feature_factorization=True)
        with torch.no_grad():
            model.feature_weights.weight.fill_(0.1)
        before = model.effective_input_weights().detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Training one king bucket must also improve the shared component of unseen buckets.
        model.accumulator(torch.tensor([5]), torch.tensor([0])).sum().backward()
        optimizer.step()
        after = model.effective_input_weights().detach()
        self.assertFalse(torch.equal(before[640 + 5], after[640 + 5]))
        self.assertTrue(torch.equal(before[640 + 6], after[640 + 6]))

    def test_compact_and_jsonl_datasets_match(self) -> None:
        board = chess.Board()
        board.push_uci("e2e4")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compact = root / "fixture.nnuebin"
            jsonl = root / "fixture.jsonl"
            with BinaryShardWriter(compact) as writer:
                writer.write(board, 125, -1.0, game_id=7, ply=9)
            jsonl.write_text(json.dumps({
                "fen": board.fen(en_passant="fen"),
                "score_cp": 125,
                "result": -1.0,
                "game_id": 7,
                "ply": 9,
            }) + "\n", encoding="utf-8")

            with PositionsDataset([compact], result_weight=0.2) as compact_dataset:
                compact_sample = compact_dataset[0]
            with PositionsDataset([jsonl], result_weight=0.2) as jsonl_dataset:
                jsonl_sample = jsonl_dataset[0]
            self.assertEqual(compact_sample, jsonl_sample)

    def test_quantized_prediction_tracks_float_model(self) -> None:
        torch.manual_seed(17)
        model = HalfKpV1(hidden=8)
        board = chess.Board()
        with tempfile.TemporaryDirectory() as directory:
            compact = Path(directory) / "fixture.nnuebin"
            with BinaryShardWriter(compact) as writer:
                for game_id, move in enumerate(("e2e4", "d2d4", "g1f3"), start=1):
                    position = board.copy()
                    position.push_uci(move)
                    writer.write(position, game_id * 10, 0.0, game_id, 1)
            with PositionsDataset([compact], result_weight=0.0) as dataset:
                report = verify_quantization(model, dataset, 3, 17, 127, 64)
                self.assertEqual(report["samples"], 3)
                self.assertLess(report["maxAbsErrorCp"], 1.0)

                first, second, *_rest = dataset[0]
                quantized = quantize_network(model, 127, 64)
                prediction = quantized_predict(quantized, first, second)
                self.assertIsInstance(prediction, int)

    def test_scaled_model_exports_centipawn_outputs(self) -> None:
        torch.manual_seed(29)
        model = HalfKpV1(hidden=8)
        board = chess.Board()
        first = active_features(board, board.turn)
        second = active_features(board, not board.turn)
        quantized = quantize_network(model, 1024, 64, target_scale=600.0)
        batch = collate([(first, second, 0.0)])
        with torch.no_grad():
            expected_cp = float(model(*batch[:4]).item()) * 600.0
        actual_cp = quantized_predict(quantized, first, second)
        self.assertLess(abs(actual_cp - expected_cp), 2.0)

    def test_validation_reports_rmse_mae_and_sign_accuracy(self) -> None:
        model = HalfKpV1(hidden=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        board = chess.Board()
        first = active_features(board, board.turn)
        second = active_features(board, not board.turn)
        loader = [collate([
            (first, second, -100.0),
            (first, second, 100.0),
        ])]
        report = evaluate(model, loader, torch.device("cpu"), target_scale=1.0)
        self.assertEqual(report["samples"], 2)
        self.assertAlmostEqual(report["rmseCp"], 100.0)
        self.assertAlmostEqual(report["maeCp"], 100.0)
        self.assertAlmostEqual(report["signAccuracy"], 0.5)

    def test_wdl_probability_is_stable_and_symmetric(self) -> None:
        self.assertEqual(score_to_win_probability(0.0, 400.0), 0.5)
        positive = score_to_win_probability(800.0, 400.0)
        negative = score_to_win_probability(-800.0, 400.0)
        self.assertAlmostEqual(positive + negative, 1.0)
        self.assertGreater(positive, 0.8)
        self.assertLess(negative, 0.2)
        self.assertGreater(score_to_win_probability(32_000.0, 400.0), 0.999)
        self.assertLess(score_to_win_probability(-32_000.0, 400.0), 0.001)

    def test_wdl_target_blends_teacher_probability_and_game_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            compact = Path(directory) / "fixture.nnuebin"
            with BinaryShardWriter(compact) as writer:
                writer.write(chess.Board(), 0, 1.0, game_id=1, ply=8)
            with PositionsDataset(
                [compact], result_weight=0.2, wdl_scale=400.0,
            ) as dataset:
                sample = dataset.sample(0)
            self.assertAlmostEqual(sample.target, 200.0)
            self.assertAlmostEqual(sample.wdl_target, 0.6)

    def test_wdl_loss_rewards_calibrated_probability(self) -> None:
        target_cp = torch.tensor([0.0])
        target_wdl = torch.tensor([0.75])
        zero_prediction = torch.tensor([0.0])
        calibrated_prediction = torch.tensor([
            400.0 * np.log(3.0) / 600.0,
        ])
        zero_loss = training_loss(
            zero_prediction, target_cp, target_wdl, "wdl",
            target_scale=600.0, huber_beta_cp=100.0, wdl_scale=400.0,
        )
        calibrated_loss = training_loss(
            calibrated_prediction, target_cp, target_wdl, "wdl",
            target_scale=600.0, huber_beta_cp=100.0, wdl_scale=400.0,
        )
        self.assertAlmostEqual(float(zero_loss), 0.0625)
        self.assertLess(float(calibrated_loss), 1e-12)

    def test_wdl_training_exports_reproducible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "train.nnuebin"
            validation = root / "validation.nnuebin"
            random_board = chess.Board()
            with BinaryShardWriter(training) as writer:
                for index in range(100):
                    if random_board.is_game_over():
                        random_board.reset()
                    move = list(random_board.legal_moves)[index % random_board.legal_moves.count()]
                    random_board.push(move)
                    writer.write(
                        random_board, (index % 21 - 10) * 25,
                        float((index % 3) - 1), game_id=index + 1, ply=index + 8,
                    )
            with BinaryShardWriter(validation) as writer:
                for index in range(20):
                    board = chess.Board()
                    board.push(list(board.legal_moves)[index % board.legal_moves.count()])
                    writer.write(
                        board, (index - 10) * 20, 0.0,
                        game_id=1000 + index, ply=index + 8,
                    )

            output = root / "wdl-smoke.nnue"
            arguments = [
                "train.py", "--data", str(training),
                "--validation-data", str(validation), "--output", str(output),
                "--hidden", "4", "--epochs", "2", "--batch-size", "16",
                "--workers", "0", "--device", "cpu", "--loss", "wdl",
                "--result-weight", "0.15", "--wdl-scale", "400",
                "--target-scale", "600", "--verify-samples", "4", "--feature-factorization",
            ]
            with mock.patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)

            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertTrue(output.is_file())
            self.assertTrue(manifest["configuration"]["featureFactorization"])
            self.assertEqual(json.loads(output.with_suffix(".progress.json").read_text())["state"], "complete")
            self.assertEqual(manifest["configuration"]["loss"], "wdl")
            self.assertEqual(manifest["configuration"]["wdlScale"], 400.0)
            self.assertEqual(
                manifest["bestValidationObjective"]["name"], "validationLoss",
            )
            self.assertEqual(manifest["cppVerification"], None)

    def test_validation_error_slices_cover_position_groups(self) -> None:
        model = HalfKpV1(hidden=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        with tempfile.TemporaryDirectory() as directory:
            compact = Path(directory) / "validation.nnuebin"
            with BinaryShardWriter(compact) as writer:
                writer.write(chess.Board(), 50, 0.0, game_id=1, ply=8)
                writer.write(
                    chess.Board("8/8/8/8/8/4k3/4p3/4K3 w - - 0 1"),
                    -250, -1.0, game_id=2, ply=60,
                )
            with PositionsDataset([compact], result_weight=0.0) as dataset:
                report = validation_error_diagnostics(
                    model, dataset, torch.device("cpu"), target_scale=1.0,
                    batch_size=2,
                )
            self.assertEqual(report["overall"]["samples"], 2)
            self.assertEqual(set(report["phase"]), {"endgame", "opening"})
            self.assertEqual(set(report["teacherEvalMagnitude"]), {"0-99", "100-299"})
            self.assertEqual(set(report["sideToMoveKingSquare"]), {"e1"})
            self.assertAlmostEqual(report["overall"]["maeCp"], 150.0)

    def test_quantization_rejects_saturation(self) -> None:
        model = HalfKpV1(hidden=2)
        with torch.no_grad():
            model.feature_weights.weight[0, 0] = 40.0
        with self.assertRaisesRegex(ValueError, "input weights exceed int16"):
            quantize_network(model, hidden_scale=1024, output_scale=64)

    def test_export_header_and_size(self) -> None:
        model = HalfKpV1(hidden=4)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.nnue"
            export_network(model, destination)
            with destination.open("rb") as source:
                magic, version, features, hidden, hidden_scale, output_scale, _bias = struct.unpack(
                    "<8sIIIiii", source.read(struct.calcsize("<8sIIIiii")),
                )
            self.assertEqual(magic, MAGIC)
            self.assertEqual(version, FORMAT_VERSION)
            self.assertEqual(features, FEATURE_COUNT)
            self.assertEqual(hidden, 4)
            self.assertEqual(hidden_scale, 1024)
            self.assertEqual(output_scale, 64)
            expected = struct.calcsize("<8sIIIiii") + 4 * 4 + FEATURE_COUNT * 4 * 2 + 8 * 2
            self.assertEqual(destination.stat().st_size, expected)

    def test_integer_division_matches_cpp_truncation(self) -> None:
        self.assertEqual(truncating_division(7, 3), 2)
        self.assertEqual(truncating_division(-7, 3), -2)

    def test_rng_state_restores_all_cpu_generators(self) -> None:
        import random

        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": None,
        }
        expected = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
        random.random()
        np.random.random()
        torch.rand(1)
        restore_rng_state(state)
        actual = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
        self.assertEqual(actual, expected)

    def test_dataset_manifest_discovery_validates_output_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "train.nnuebin"
            dataset.write_bytes(b"fixture")
            import hashlib
            checksum = hashlib.sha256(b"fixture").hexdigest()
            manifest = root / "dataset.manifest.json"
            manifest.write_text(json.dumps({
                "outputs": [{"path": str(dataset.resolve()), "sha256": checksum}],
            }), encoding="utf-8")
            self.assertEqual(find_dataset_manifest(dataset, checksum), manifest.resolve())
            self.assertIsNone(find_dataset_manifest(dataset, "0" * 64))


if __name__ == "__main__":
    unittest.main()
