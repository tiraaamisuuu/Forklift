#!/usr/bin/env python3
"""Small CPU fixtures for frozen-transformer sampling, identity and provenance."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

import chess
import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

from head_cache import (  # noqa: E402
    _close_arrays, build_cache, effective_input_identity, load_cache, validation_partition,
)
from nnue_dataset import BinaryShardWriter, active_features_from_packed  # noqa: E402
from train import HalfKpV1, collate  # noqa: E402


class HeadCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="forklift-head-cache-test-")
        self.directory = Path(self.temporary.name)
        self.train = self.directory / "train.nnuebin"
        self.validation = self.directory / "validation.nnuebin"
        self.weights = self.directory / "source.pt"
        torch.manual_seed(991)
        self.model = HalfKpV1(4, feature_factorization=True)
        with torch.no_grad():
            self.model.hidden_bias.fill_(0.3)
            self.model.piece_square_weights.weight.uniform_(-0.01, 0.01)
        torch.save(self.model.state_dict(), self.weights)
        with BinaryShardWriter(self.train) as writer:
            for index in range(20):
                writer.write(self.board(index), (index - 10) * 400, 0.0,
                             game_id=index // 2, ply=2)
        with BinaryShardWriter(self.validation) as writer:
            for index in range(80):
                writer.write(self.board(index + 30), (index - 40) * 100, 0.0,
                             game_id=100 + index // 2, ply=2)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def board(index: int) -> chess.Board:
        board = chess.Board(None)
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(8 + index % 32, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(40 + index // 32, chess.Piece(chess.KNIGHT, chess.BLACK))
        board.turn = index % 2 == 0
        return board

    def build(self, name: str = "cache", **extra: object) -> dict:
        return build_cache(self.train, self.validation, self.weights,
                           self.directory / name, train_count=12,
                           validation_count=8, test_count=8, hidden=4,
                           seed=123, batch_size=3, cpu_threads=1,
                           max_seconds=30.0, **extra)

    def test_cache_is_exact_float_forward_and_has_disjoint_membership(self) -> None:
        manifest = self.build()
        self.assertTrue(manifest["audit"]["fullSourceGameIdsDisjoint"])
        self.assertTrue(manifest["audit"]["selectedEffectiveInputsDisjoint"])
        self.assertEqual(manifest["configuration"]["resultWeight"], 0.0)
        seen_games: set[int] = set()
        seen_inputs: set[bytes] = set()
        for split in ("train", "validation", "test"):
            arrays = load_cache(self.directory / "cache" / "manifest.json", split)
            try:
                self.assertEqual(arrays["activations"].dtype, np.float32)
                self.assertFalse(arrays["activations"].flags.writeable)
                games = set(arrays["game_ids"].tolist())
                self.assertFalse(seen_games & games)
                seen_games |= games
                identities = {row.tobytes() for row in arrays["identity"]}
                self.assertEqual(len(identities), len(arrays["identity"]))
                self.assertFalse(seen_inputs & identities)
                seen_inputs |= identities
                samples = []
                for packed, turn in zip(arrays["boards"], arrays["turns"]):
                    first = active_features_from_packed(packed.tobytes(), bool(turn))
                    second = active_features_from_packed(packed.tobytes(), not bool(turn))
                    samples.append((first, second, 0.0))
                batch = collate(samples)
                with torch.inference_mode():
                    expected = torch.cat((self.model.accumulator(batch[0], batch[1]),
                                          self.model.accumulator(batch[2], batch[3])), dim=1)
                np.testing.assert_array_equal(arrays["activations"], expected.numpy())
                cp = np.clip(arrays["teacher_cp"], -3000, 3000).astype(np.float32)
                np.testing.assert_array_equal(arrays["target_cp"], cp)
                np.testing.assert_allclose(arrays["target_wdl"], 1.0 / (1.0 + np.exp(-cp / 400.0)),
                                           rtol=1e-6, atol=1e-7)
                if split != "train":
                    self.assertTrue(all(validation_partition(game, 123) == split for game in games))
            finally:
                _close_arrays(arrays)

    def test_repeated_builds_select_and_compute_identically(self) -> None:
        first = self.build("first")
        second = self.build("second")
        for split in ("train", "validation", "test"):
            for name in first["splits"][split]["files"]:
                self.assertEqual(first["splits"][split]["files"][name]["sha256"],
                                 second["splits"][split]["files"][name]["sha256"])

    def test_existing_output_and_shared_source_game_ids_rejected(self) -> None:
        self.build()
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.build()
        with BinaryShardWriter(self.validation) as writer:
            writer.write(self.board(50), 0, 0.0, game_id=1, ply=2)
        with self.assertRaisesRegex(ValueError, "game IDs overlap"):
            self.build("overlap")
        self.assertFalse((self.directory / "overlap").exists())

    def test_selected_cross_split_effective_duplicates_are_filtered(self) -> None:
        # A mirrored color-swapped position is identical HalfKP input although
        # both raw board bytes and the side to move differ.
        with BinaryShardWriter(self.validation) as writer:
            for index in range(20):
                writer.write(self.board(index).mirror(), 0, 0.0, game_id=100 + index, ply=2)
            for index in range(80):
                writer.write(self.board(50 + index), 0, 0.0, game_id=200 + index, ply=2)
        manifest = self.build()
        self.assertGreater(sum(manifest["splits"][split]["excludedDuplicateInputs"]
                               for split in ("validation", "test")), 0)
        self.assertTrue(manifest["audit"]["selectedEffectiveInputsDisjoint"])

    def test_input_identity_encodes_perspectives_and_boundaries(self) -> None:
        self.assertNotEqual(effective_input_identity([1], [2]), effective_input_identity([2], [1]))
        self.assertNotEqual(effective_input_identity([1, 2], []), effective_input_identity([1], [2]))
        from nnue_dataset import active_features
        board = self.board(12)
        mirrored = board.mirror()
        self.assertEqual(
            effective_input_identity(active_features(board, board.turn), active_features(board, not board.turn)),
            effective_input_identity(active_features(mirrored, mirrored.turn),
                                     active_features(mirrored, not mirrored.turn)),
        )

    def test_tamper_missing_manifest_and_unsafe_paths_rejected(self) -> None:
        self.build()
        manifest_path = self.directory / "cache" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        array_path = self.directory / "cache" / "train" / "cp.npy"
        with array_path.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"X")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_cache(manifest_path, "train")
        payload["splits"]["test"]["files"]["activations"]["path"] = "../source.pt"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes cache"):
            load_cache(manifest_path, "test")
        manifest_path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_cache(manifest_path, "train")

    def test_exhaustion_and_time_budget_do_not_publish_manifest(self) -> None:
        with BinaryShardWriter(self.train) as writer:
            for index in range(20):
                writer.write(self.board(1), 0, 0.0, game_id=index, ply=2)
        with self.assertRaisesRegex(ValueError, "not enough distinct"):
            self.build("too-few")
        self.assertFalse((self.directory / "too-few" / "manifest.json").exists())
        with self.assertRaises(TimeoutError):
            build_cache(self.train, self.validation, self.weights, self.directory / "timeout",
                        hidden=4, train_count=1, validation_count=1, test_count=1,
                        max_seconds=1e-20)
        self.assertFalse((self.directory / "timeout" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
