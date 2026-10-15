#!/usr/bin/env python3
"""Tiny CPU fixtures for the matched head study, never chess-strength evidence."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1")

import chess
import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

from head_study import (CachedHead, StudyConfig, epoch_permutation, integer_divide_toward_zero,
                        paired_cluster_ci, quantized_activations, quantized_head_predictions,
                        run_study, stitch_head, transformer_digest, validate_config, validation_screen)
from initialization import freeze_feature_transformer, load_initial_weights
from nnue_dataset import active_features, pack_board, sha256_file
from train import HalfKpV1, collate, quantize_network, quantized_predict


def fixture_arrays(model: HalfKpV1, count: int, offset: int) -> dict[str, np.ndarray]:
    samples = []
    boards = []
    turns = []
    for index in range(count):
        board = chess.Board()
        board.push_uci(("e2e4", "d2d4", "g1f3", "c2c4")[index % 4])
        board.push(list(board.legal_moves)[index % board.legal_moves.count()])
        boards.append(np.frombuffer(pack_board(board), dtype=np.uint8))
        turns.append(int(board.turn))
        samples.append((active_features(board, board.turn), active_features(board, not board.turn), 0))
    batch = collate(samples)
    with torch.no_grad():
        x = torch.cat((model.accumulator(batch[0], batch[1]),
                       model.accumulator(batch[2], batch[3])), dim=1).numpy()
    cp = ((np.arange(count) % 7) * 40 - 80).astype(np.float32)
    return {"activations": x, "target_cp": cp,
            "target_wdl": (1 / (1 + np.exp(-cp / 400))).astype(np.float32),
            "teacher_cp": cp.astype(np.int16), "game_ids": (offset + np.arange(count) // 2).astype(np.uint32),
            "source_indices": np.arange(count, dtype=np.int64), "boards": np.stack(boards),
            "turns": np.array(turns, dtype=np.uint8), "identity": (offset + np.arange(count)).astype(np.uint64)}


class HeadStudyTests(unittest.TestCase):
    def test_initial_heads_identical_and_transformer_stitch_unchanged(self) -> None:
        torch.manual_seed(31)
        with tempfile.TemporaryDirectory(prefix="forklift-head-init-") as directory:
            base = HalfKpV1(4, feature_factorization=True)
            source = Path(directory) / "base.pt"
            torch.save(base.state_dict(), source)
            x = torch.rand(20, 8)
            expected = CachedHead(base)(x)
            for width in (0, 1, 4, 8):
                torch.manual_seed(91)
                model = HalfKpV1(4, feature_factorization=True, residual_hidden=width)
                load_initial_weights(model, source)
                freeze_feature_transformer(model)
                digest = transformer_digest(model)
                head = CachedHead(model)
                self.assertTrue(torch.equal(head(x), expected))
                with torch.no_grad():
                    head.output.bias.add_(0.05)
                    if width:
                        head.residual_output.weight.fill_(0.02)
                stitch_head(model, head, digest)
                self.assertEqual(transformer_digest(model), digest)
                self.assertTrue(torch.equal(CachedHead(model)(x), head(x)))
                with torch.no_grad():
                    model.hidden_bias.add_(1)
                with self.assertRaisesRegex(ValueError, "transformer changed"):
                    stitch_head(model, head, digest)

    def test_permutations_do_not_depend_on_global_rng_or_arm(self) -> None:
        expected = epoch_permutation(100, 1729, 2)
        torch.manual_seed(5)
        np.random.seed(17)
        HalfKpV1(3, residual_hidden=8)
        self.assertTrue(np.array_equal(expected, epoch_permutation(100, 1729, 2)))
        self.assertFalse(np.array_equal(expected, epoch_permutation(100, 1729, 3)))
        self.assertEqual(sorted(expected), list(range(100)))

    def test_integer_batched_reference_including_empty_bags_and_odd_tails(self) -> None:
        torch.manual_seed(113)
        boards = [chess.Board(), chess.Board("8/8/8/8/8/2k5/8/K7 w - - 0 1")]
        board = chess.Board()
        for move in ("e2e4", "c7c5", "g1f3"):
            board.push_uci(move)
            boards.append(board.copy())
        arrays = {"boards": np.stack([np.frombuffer(pack_board(board), dtype=np.uint8) for board in boards]),
                  "turns": np.array([int(board.turn) for board in boards])}
        for hidden in (1, 7, 8, 17):
            for width in (0, 3, 8):
                model = HalfKpV1(hidden, feature_factorization=True, residual_hidden=width)
                with torch.no_grad():
                    model.hidden_bias.uniform_(-0.1, 0.4)
                    if width:
                        model.residual_hidden_layer.bias.uniform_(-0.3, 0.4)
                        model.residual_output.weight.uniform_(-0.05, 0.05)
                quantized = quantize_network(model, target_scale=600, output_scale=32,
                                              residual_weight_scale=7)
                activation = quantized_activations(quantized, arrays, batch_size=2)
                actual = quantized_head_predictions(quantized, activation, batch_size=2)
                expected = [quantized_predict(quantized, active_features(board, board.turn),
                                                active_features(board, not board.turn)) for board in boards]
                self.assertEqual(actual.tolist(), expected, (hidden, width))
        values = np.array([-19, -14, -1, 0, 1, 14, 19], dtype=np.int64)
        self.assertEqual(integer_divide_toward_zero(values, 7).tolist(), [-2, -2, 0, 0, 0, 2, 2])

    def test_game_cluster_paired_interval(self) -> None:
        report = paired_cluster_ci(np.full(20, -0.01), np.repeat(np.arange(5), 4), 100, 3)
        self.assertAlmostEqual(report["meanResidualMinusDirect"], -0.01)
        self.assertEqual(report["games"], 5)
        for endpoint in report["interval95"]:
            self.assertAlmostEqual(endpoint, -0.01)
        with self.assertRaises(ValueError):
            paired_cluster_ci(np.array([np.nan]), np.array([1]), 10, 1)

    def test_config_rejects_unmatched_or_overwriting_study(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-head-config-") as directory:
            root = Path(directory)
            config = StudyConfig(root / "cache.json", root / "base.pt", root / "out")
            validate_config(config)
            for altered in (replace(config, seeds=(1,)), replace(config, seeds=(1, 1)),
                            replace(config, residual_widths=(4, 0)), replace(config, epochs=0),
                            replace(config, learning_rate=float("nan")), replace(config, output=root)):
                with self.assertRaises(ValueError):
                    validate_config(altered)

    def test_validation_only_architecture_selection_threshold_and_tie(self) -> None:
        arms = [{"seed": seed, "residualHidden": width, "validationBrier": loss}
                for seed in (11, 13) for width, loss in ((0, 0.1), (4, 0.098), (8, 0.098))]
        result = validation_screen(arms, (11, 13), (0, 4, 8))
        self.assertEqual(result["selectedResidualHidden"], 4)
        self.assertTrue(all(row["eligible"] for row in result["comparisons"]))
        arms[1]["validationBrier"] = 0.101
        self.assertEqual(validation_screen(arms, (11, 13), (0, 4, 8))["selectedResidualHidden"], 8)
        for arm in arms:
            if arm["residualHidden"]:
                arm["validationBrier"] = 0.0995
        self.assertIsNone(validation_screen(arms, (11, 13), (0, 4, 8))["selectedResidualHidden"])

    def test_tiny_complete_study_never_opens_test_until_all_selections(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-head-run-") as directory:
            root = Path(directory)
            source = root / "base.pt"
            torch.manual_seed(41)
            model = HalfKpV1(4, feature_factorization=True)
            with torch.no_grad():
                model.hidden_bias.fill_(0.2)
            torch.save(model.state_dict(), source)
            cached = {name: fixture_arrays(model, count, offset)
                      for name, count, offset in (("train", 32, 0), ("validation", 12, 100), ("test", 12, 200))}
            cache = root / "cache.json"
            cache.write_text(json.dumps({"schemaVersion": 1, "sourceWeights": {"sha256": sha256_file(source)},
                                        "configuration": {"hidden": 4, "featureFactorization": True,
                                                          "targetScale": 600, "wdlScale": 400, "resultWeight": 0}}),
                             encoding="utf-8")
            output = root / "study"
            loads = []

            def loader(_path: Path, split: str) -> dict[str, np.ndarray]:
                loads.append(split)
                if split == "test":
                    self.assertTrue((output / "selection.json").is_file())
                    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
                    self.assertEqual(len(selection["arms"]), 6)
                    self.assertTrue(all("test" not in arm for arm in selection["arms"]))
                    self.assertIn("validationScreen", selection)
                    self.assertFalse(selection["validationScreen"]["selectionUsedTestLabels"])
                return cached[split]

            config = StudyConfig(cache, source, output, seeds=(11, 13), residual_widths=(0, 2, 4),
                                  epochs=2, batch_size=8, device="cpu", cpu_threads=1,
                                  max_seconds=60, bootstrap_resamples=30)
            with mock.patch("head_cache.load_cache", side_effect=loader):
                report = run_study(config)
            self.assertEqual(loads, ["train", "validation", "test"])
            self.assertEqual(report["state"], "complete")
            self.assertEqual(len(report["arms"]), 6)
            self.assertEqual(len(report["pairedComparisons"]), 4)
            self.assertEqual(report["protocol"]["equalUpdatesPerArm"], 8)
            for arm in report["arms"]:
                self.assertEqual(arm["updates"], 8)
                self.assertIn(arm["selectedEpoch"], (0, 1, 2))
                state = torch.load(arm["rawWeights"]["path"], weights_only=True)
                for key in ("feature_weights.weight", "piece_square_weights.weight", "hidden_bias"):
                    self.assertTrue(torch.equal(state[key], model.state_dict()[key]))
                with np.load(arm["test"]["perSample"]["path"]) as values:
                    self.assertEqual(len(values["quantized_cp"]), 12)
                self.assertEqual(arm["test"]["quantization"]["samples"], 12)
            self.assertEqual(sha256_file(output / "selection.json"), report["frozenSelectionSha256"])
            with self.assertRaisesRegex(ValueError, "already exist"):
                run_study(config)
            loads.clear()
            failed = replace(config, output=root / "time-limited", max_seconds=1e-8)
            with mock.patch("head_cache.load_cache", side_effect=loader):
                with self.assertRaises(TimeoutError):
                    run_study(failed)
            self.assertEqual(loads, ["train", "validation"])
            progress = json.loads((failed.output / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["state"], "incomplete")
            self.assertFalse(progress["testOpened"])


if __name__ == "__main__":
    unittest.main()
