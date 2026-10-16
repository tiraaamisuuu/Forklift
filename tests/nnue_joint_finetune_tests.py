#!/usr/bin/env python3
"""Tiny CPU fixtures for matched whole-network adaptation, not chess strength."""

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

import joint_finetune as joint
from head_cache import build_cache, effective_input_identity
from head_study import epoch_permutation, transformer_digest
from initialization import load_initial_weights
from nnue_dataset import (BinaryShardWriter, active_features, pack_board, sha256_file)
from train import HalfKpV1, collate, quantize_network, quantized_predict


def fixture_board(index: int) -> chess.Board:
    """Distinct compact inputs; no teacher or engine workload is involved."""
    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(8 + index % 32, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(40 + index // 32, chess.Piece(chess.KNIGHT, chess.BLACK))
    board.turn = index % 2 == 0
    return board


def input_identity(board: chess.Board) -> bytes:
    return effective_input_identity(active_features(board, board.turn),
                                    active_features(board, not board.turn))


def tiny_inputs(root: Path) -> tuple[Path, Path, Path, Path, HalfKpV1]:
    train = root / "train.nnuebin"
    validation = root / "validation.nnuebin"
    source = root / "source.pt"
    torch.manual_seed(191)
    model = HalfKpV1(4, feature_factorization=True)
    with torch.no_grad():
        model.feature_weights.weight.uniform_(-0.01, 0.01)
        model.piece_square_weights.weight.uniform_(-0.01, 0.01)
        model.hidden_bias.fill_(0.3)
        model.output.weight.fill_(0.1)
        model.output.bias.fill_(0.05)
    torch.save(model.state_dict(), source)
    with BinaryShardWriter(train) as writer:
        for index in range(24):
            writer.write(fixture_board(index), (index % 5 - 2) * 100, 0,
                         game_id=index // 2, ply=2)
    with BinaryShardWriter(validation) as writer:
        for index in range(128):
            writer.write(fixture_board(index + 40), (index % 7 - 3) * 80, 0,
                         game_id=1000 + index // 2, ply=2)
    cache = root / "cache"
    build_cache(train, validation, source, cache, train_count=12,
                validation_count=8, test_count=8, hidden=4, batch_size=8,
                seed=1729, device="cpu", cpu_threads=1, max_seconds=30)
    return train, validation, source, cache / "manifest.json", model


class JointFineTuneTests(unittest.TestCase):
    def test_checkpoint_ties_and_degradation_preserve_epoch_zero(self) -> None:
        self.assertFalse(joint.select_checkpoint(0.01, 0.01))
        self.assertFalse(joint.select_checkpoint(0.01, 0.02))
        self.assertTrue(joint.select_checkpoint(0.01, 0.009))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            joint.select_checkpoint(0.01, float("nan"))

    def test_epoch_sampler_is_matched_and_independent_of_model_rng(self) -> None:
        sampler = joint.MatchedEpochSampler(25, 1729)
        expected = list(sampler)
        torch.manual_seed(9)
        HalfKpV1(4, residual_hidden=8)
        self.assertEqual(list(sampler), expected)
        self.assertEqual(expected, epoch_permutation(25, 1729, 1).tolist())
        sampler.epoch = 2
        self.assertEqual(list(sampler), epoch_permutation(25, 1729, 2).tolist())
        self.assertNotEqual(list(sampler), expected)
        self.assertEqual(sorted(list(sampler)), list(range(25)))

    def test_optimizer_has_differential_rates_and_unfrozen_transformer(self) -> None:
        root = Path("unused-fixture")
        config = joint.JointConfig(root / "train", root / "validation", root / "cache",
                                   root / "source", root / "output", transformer_lr=1e-5,
                                   head_lr=1e-4, device="cpu", cpu_threads=1, workers=0,
                                   hidden=4)
        model = HalfKpV1(4, feature_factorization=True, residual_hidden=8)
        optimizer = joint.build_optimizer(model, config)
        transformer = {id(value) for name, value in model.named_parameters()
                       if name in ("hidden_bias", "feature_weights.weight",
                                   "piece_square_weights.weight")}
        head = {id(value) for name, value in model.named_parameters()
                if name not in ("hidden_bias", "feature_weights.weight",
                                "piece_square_weights.weight")}
        seen: set[int] = set()
        for group in optimizer.param_groups:
            identities = {id(value) for value in group["params"]}
            self.assertFalse(seen & identities)
            seen |= identities
            self.assertTrue(all(value.requires_grad for value in group["params"]))
            if identities & transformer:
                self.assertTrue(identities <= transformer)
                self.assertEqual(group["lr"], config.transformer_lr)
            else:
                self.assertTrue(identities <= head)
                self.assertEqual(group["lr"], config.head_lr)
        self.assertEqual(seen, transformer | head)

    def test_same_initial_function_and_order_despite_new_residual_parameters(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-joint-initial-") as directory:
            root = Path(directory)
            torch.manual_seed(99)
            source = HalfKpV1(4, feature_factorization=True)
            path = root / "base.pt"
            torch.save(source.state_dict(), path)
            board = fixture_board(19)
            batch = collate([(active_features(board, board.turn),
                              active_features(board, not board.turn), 0)])
            with torch.no_grad():
                expected = source(*batch[:4])
            order = epoch_permutation(24, 11, 1)
            for seed in (11, 13):
                for width in (0, 8):
                    torch.manual_seed(seed)
                    model = HalfKpV1(4, feature_factorization=True, residual_hidden=width)
                    load_initial_weights(model, path)
                    with torch.no_grad():
                        self.assertTrue(torch.equal(model(*batch[:4]), expected))
                    self.assertEqual(transformer_digest(model), transformer_digest(source))
                    self.assertTrue(all(value.requires_grad for value in model.parameters()))
                    self.assertTrue(np.array_equal(epoch_permutation(24, 11, 1), order))

    def test_training_can_change_transformer_without_changing_source_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-joint-unfrozen-") as directory:
            root = Path(directory)
            torch.manual_seed(5)
            model = HalfKpV1(4, feature_factorization=True, residual_hidden=8)
            with torch.no_grad():
                model.hidden_bias.fill_(0.3)
                model.feature_weights.weight.zero_()
                model.output.weight.fill_(0.1)
            source = root / "source.pt"
            torch.save(model.state_dict(), source)
            source_hash = sha256_file(source)
            original = transformer_digest(model)
            config = joint.JointConfig(root / "train", root / "validation", root / "cache",
                                       source, root / "output", transformer_lr=1e-3,
                                       head_lr=2e-3, device="cpu", cpu_threads=1,
                                       workers=0, hidden=4)
            optimizer = joint.build_optimizer(model, config)
            board = fixture_board(8)
            batch = collate([(active_features(board, board.turn),
                              active_features(board, not board.turn), 0)])
            optimizer.zero_grad(set_to_none=True)
            model(*batch[:4]).square().mean().backward()
            optimizer.step()
            self.assertNotEqual(transformer_digest(model), original)
            self.assertEqual(sha256_file(source), source_hash)

    def test_diagnostic_sampler_uses_no_labels_and_excludes_games_and_equivalent_inputs(self) -> None:
        # Deliberately omit score/result fields: merely selecting diagnostic rows
        # must not inspect labels before the checkpoint selection is sealed.
        records = np.zeros(20, dtype=[("board", "u1", (32,)),
                                     ("turn", "u1"), ("game", "<u4")])
        positions = [fixture_board(index + 50) for index in range(20)]
        positions[1] = positions[0].mirror()
        positions[3] = positions[2].copy()
        for index, board in enumerate(positions):
            records["board"][index] = np.frombuffer(pack_board(board), dtype=np.uint8)
            records["turn"][index] = int(board.turn)
            records["game"][index] = 100 + index
        excluded = (input_identity(positions[0]),)
        selected_rows = joint.sample_diagnostic(records, {104, 105}, 8, 123,
                                                excluded_identities=excluded)
        indices, identities = selected_rows["source_indices"], selected_rows["identity"]
        self.assertEqual(len(indices), 8)
        self.assertEqual(len(identities), 8)
        self.assertFalse({0, 1, 4, 5} & set(map(int, indices)))
        selected = [input_identity(positions[int(index)]) for index in indices]
        self.assertEqual(len(set(selected)), 8)
        self.assertFalse(set(selected) & set(excluded))
        np.testing.assert_array_equal(identities,
                                      np.stack([np.frombuffer(value, dtype=np.uint8) for value in selected]))
        repeat = joint.sample_diagnostic(records, {104, 105}, 8, 123,
                                         excluded_identities=excluded)
        np.testing.assert_array_equal(indices, repeat["source_indices"])
        np.testing.assert_array_equal(identities, repeat["identity"])
        with self.assertRaisesRegex(ValueError, "distinct|enough|exhaust"):
            joint.sample_diagnostic(records, {104, 105}, 19, 123,
                                    excluded_identities=excluded)

    def test_metadata_loader_forbids_activations_and_hash_checks_requested_arrays(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-joint-metadata-") as directory:
            root = Path(directory)
            _train, _validation, _source, cache, _model = tiny_inputs(root)
            arrays = joint.load_metadata_cache(cache, "test", ("game_ids", "identity"))
            self.assertEqual(set(arrays), {"game_ids", "identity"})
            with self.assertRaisesRegex(ValueError, "activations.*prohibited"):
                joint.load_metadata_cache(cache, "validation", ("activations",))
            manifest = json.loads(cache.read_text(encoding="utf-8"))
            target = cache.parent / manifest["splits"]["test"]["files"]["game_ids"]["path"]
            with target.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"X")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                joint.load_metadata_cache(cache, "test", ("game_ids",))

    def test_tiny_whole_network_study_seals_selection_and_recomputes_each_integer_transformer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-joint-run-") as directory:
            root = Path(directory)
            train, validation, source, cache, initial = tiny_inputs(root)
            source_hash = sha256_file(source)
            config = joint.JointConfig(train, validation, cache, source, root / "study",
                                       seeds=(11, 13), epochs=1, batch_size=8,
                                       transformer_lr=1e-3, head_lr=2e-3, device="cpu",
                                       cpu_threads=1, workers=0, max_seconds=60, hidden=4,
                                       diagnostic_count=8, bootstrap_resamples=20)
            metadata_calls = []
            quantized_calls = []
            label_reads = []
            evaluations = 0
            actual_source = joint._source
            actual_metadata = joint.load_metadata_cache
            actual_evaluate = joint.evaluate_model
            actual_integer = joint.quantized_activations
            case = self

            class LabelGuardedRecords(np.memmap):
                def __getitem__(self, key):
                    if isinstance(key, str) and key in ("score", "result"):
                        seal = config.output / "selection.json"
                        case.assertTrue(seal.is_file(), "diagnostic labels read before selection seal")
                        frozen = json.loads(seal.read_text(encoding="utf-8"))
                        case.assertEqual(len(frozen["arms"]), 4)
                        case.assertTrue(all("diagnostic" not in arm for arm in frozen["arms"]))
                        label_reads.append(key)
                    return super().__getitem__(key)

            def guarded_source(path):
                records, descriptor = actual_source(path)
                if Path(path) == validation:
                    records = records.view(LabelGuardedRecords)
                return records, descriptor

            def guarded_metadata(path, split, fields):
                fields = tuple(fields)
                metadata_calls.append((split, fields))
                case.assertNotIn("activations", fields)
                if split == "test":
                    case.assertEqual(set(fields), {"game_ids", "identity"})
                return actual_metadata(path, split, fields)

            def controlled_evaluate(model, loader, options, **kwargs):
                nonlocal evaluations
                value, predictions = actual_evaluate(model, loader, options, **kwargs)
                # Force epoch one to be selected for every arm. Numeric predictions
                # and all parameter updates remain real, tiny CPU computations.
                if evaluations < 8:
                    value = 0.2 if evaluations % 2 == 0 else 0.1
                evaluations += 1
                return value, predictions

            def observed_integer(quantized, arrays, *args, **kwargs):
                quantized_calls.append((quantized.input_weights.copy(), quantized.hidden_bias.copy()))
                return actual_integer(quantized, arrays, *args, **kwargs)

            with mock.patch.object(joint, "_source", side_effect=guarded_source), \
                    mock.patch.object(joint, "load_metadata_cache", side_effect=guarded_metadata), \
                    mock.patch.object(joint, "evaluate_model", side_effect=controlled_evaluate), \
                    mock.patch.object(joint, "quantized_activations", side_effect=observed_integer):
                report = joint.run_joint_study(config)
            self.assertEqual(report["state"], "complete")
            self.assertEqual(len(report["arms"]), 4)
            self.assertEqual(len(quantized_calls), 5)  # Source plus EACH selected arm.
            self.assertEqual(label_reads, ["score"])
            self.assertEqual([split for split, _fields in metadata_calls], ["validation", "test"])
            self.assertFalse(report["protocol"]["cachedActivationsUsed"])
            self.assertFalse(report["protocol"]["frozenTransformer"])
            self.assertEqual(sha256_file(source), source_hash)
            self.assertEqual(sha256_file(config.output / "selection.json"),
                             report["frozenSelectionSha256"])
            source_quantized = quantize_network(initial, target_scale=600, output_scale=32)
            np.testing.assert_array_equal(quantized_calls[0][0], source_quantized.input_weights)
            descriptor = np.load(report["audit"]["descriptor"]["path"])
            try:
                boards = descriptor["boards"].copy()
                turns = descriptor["turns"].copy()
                self.assertNotIn("target_cp", descriptor.files)
                self.assertNotIn("target_wdl", descriptor.files)
            finally:
                descriptor.close()
            for index, arm in enumerate(report["arms"], 1):
                self.assertEqual(arm["selectedEpoch"], 1)
                self.assertEqual(arm["updates"], 3)
                self.assertEqual(arm["trainingPositions"], 24)
                self.assertTrue(arm["transformerChanged"])
                model = HalfKpV1(4, feature_factorization=True,
                                 residual_hidden=arm["residualHidden"])
                load_initial_weights(model, Path(arm["rawWeights"]["path"]))
                quantized = quantize_network(model, target_scale=600, output_scale=32)
                np.testing.assert_array_equal(quantized_calls[index][0], quantized.input_weights)
                np.testing.assert_array_equal(quantized_calls[index][1], quantized.hidden_bias)
                with np.load(arm["diagnostic"]["perSample"]["path"]) as samples:
                    expected = []
                    from nnue_dataset import active_features_from_packed
                    for packed, turn in zip(boards, turns):
                        first = active_features_from_packed(bytes(packed), bool(turn))
                        second = active_features_from_packed(bytes(packed), not bool(turn))
                        expected.append(quantized_predict(quantized, first, second))
                    self.assertEqual(samples["quantized_cp"].tolist(), expected)
            with self.assertRaisesRegex(ValueError, "already exist"):
                joint.run_joint_study(config)

    def test_hard_cap_marks_incomplete_and_does_not_publish_completion_or_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-joint-cap-") as directory:
            root = Path(directory)
            train, validation, source, cache, _model = tiny_inputs(root)
            source_hash = sha256_file(source)
            config = joint.JointConfig(train, validation, cache, source, root / "capped",
                                       seeds=(11, 13), epochs=1, batch_size=8,
                                       device="cpu", cpu_threads=1, workers=0, hidden=4,
                                       diagnostic_count=8, max_seconds=1e-20)
            with self.assertRaises(TimeoutError):
                joint.run_joint_study(config)
            progress = json.loads((config.output / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["state"], "incomplete")
            self.assertFalse(progress["diagnosticLabelsOpened"])
            self.assertFalse(progress["comparable"])
            self.assertFalse(progress["autoExtended"])
            self.assertFalse((config.output / "selection.json").exists())
            self.assertFalse((config.output / "study.json").exists())
            self.assertEqual(sha256_file(source), source_hash)
            with self.assertRaisesRegex(ValueError, "already exist"):
                joint.run_joint_study(config)

    def test_epoch_zero_can_remain_the_selected_full_model_after_real_updates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-joint-zero-") as directory:
            root = Path(directory)
            train, validation, source, cache, initial = tiny_inputs(root)
            config = joint.JointConfig(train, validation, cache, source, root / "zero-best",
                                       seeds=(11, 13), epochs=1, batch_size=8,
                                       transformer_lr=1e-3, head_lr=2e-3, device="cpu",
                                       cpu_threads=1, workers=0, max_seconds=60, hidden=4,
                                       diagnostic_count=8, bootstrap_resamples=10)
            actual_evaluate = joint.evaluate_model
            evaluations = 0

            def degraded_validation(model, loader, options, **kwargs):
                nonlocal evaluations
                loss, predictions = actual_evaluate(model, loader, options, **kwargs)
                if evaluations < 8:
                    loss = 0.1 if evaluations % 2 == 0 else 0.2
                evaluations += 1
                return loss, predictions

            with mock.patch.object(joint, "evaluate_model", side_effect=degraded_validation):
                report = joint.run_joint_study(config)
            self.assertEqual(report["state"], "complete")
            for arm in report["arms"]:
                self.assertEqual(arm["selectedEpoch"], 0)
                self.assertEqual(arm["updates"], 3)
                self.assertEqual(arm["trainingPositions"], 24)
                self.assertEqual(arm["history"][1]["validationBrier"], 0.2)
                self.assertFalse(arm["transformerChanged"])
                selected = torch.load(arm["rawWeights"]["path"], weights_only=True)
                initial_file = Path(arm["rawWeights"]["path"]).parent / "epoch-0.pt"
                epoch_zero = torch.load(initial_file, weights_only=True)
                self.assertEqual(set(selected), set(epoch_zero))
                for name, value in epoch_zero.items():
                    self.assertTrue(torch.equal(selected[name], value), name)
                for name, value in initial.state_dict().items():
                    self.assertTrue(torch.equal(selected[name], value), name)
                if arm["residualHidden"]:
                    self.assertEqual(torch.count_nonzero(selected["residual_output.weight"]).item(), 0)

    def test_invalid_controls_and_source_pin_fail_without_completing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-joint-config-") as directory:
            root = Path(directory)
            train, validation, source, cache, _model = tiny_inputs(root)
            config = joint.JointConfig(train, validation, cache, source, root / "output",
                                       hidden=4, workers=0, device="cpu", cpu_threads=1)
            for altered in (replace(config, seeds=(1,)), replace(config, seeds=(1, 1)),
                            replace(config, residual_widths=(8, 0)), replace(config, epochs=0),
                            replace(config, workers=-1), replace(config, transformer_lr=0),
                            replace(config, head_lr=float("nan")), replace(config, diagnostic_count=0),
                            replace(config, train=validation), replace(config, output=root)):
                with self.assertRaises(ValueError):
                    joint.validate_config(altered)
            self.assertFalse(config.output.exists())
            config = replace(config, expected_source_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "pre-registered SHA256"):
                joint.run_joint_study(config)
            self.assertFalse((config.output / "selection.json").exists())
            self.assertFalse((config.output / "study.json").exists())
            progress = json.loads((config.output / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["state"], "incomplete")
            self.assertFalse(progress["diagnosticLabelsOpened"])


if __name__ == "__main__":
    unittest.main()
