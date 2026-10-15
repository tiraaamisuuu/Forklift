#!/usr/bin/env python3
"""Deferred CPU fixtures for strict warm starts and frozen-transformer pilots."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

from initialization import freeze_feature_transformer, load_initial_weights  # noqa: E402
from train import HalfKpV1  # noqa: E402


class InitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "base.pt"

    def test_lifting_direct_head_preserves_base_tensors(self) -> None:
        direct = HalfKpV1(4, True)
        torch.save(direct.state_dict(), self.path)
        candidate = HalfKpV1(4, True, residual_hidden=4)
        report = load_initial_weights(candidate, self.path)
        for name, value in direct.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        self.assertEqual(torch.count_nonzero(candidate.residual_output.weight).item(), 0)
        self.assertTrue(report["newZeroOutputResidualHead"])
        self.assertEqual(len(report["sha256"]), 64)

    def test_same_architecture_loads_all_residual_weights(self) -> None:
        source = HalfKpV1(4, residual_hidden=4)
        with torch.no_grad():
            source.residual_output.weight.fill_(0.25)
        torch.save(source.state_dict(), self.path)
        target = HalfKpV1(4, residual_hidden=4)
        report = load_initial_weights(target, self.path)
        self.assertFalse(report["newZeroOutputResidualHead"])
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[name]), name)

    def test_mismatched_hidden_or_factorization_is_rejected(self) -> None:
        torch.save(HalfKpV1(4, True).state_dict(), self.path)
        for target in (HalfKpV1(8, True), HalfKpV1(4, False)):
            with self.subTest(hidden=target.hidden, factorized=target.feature_factorization):
                with self.assertRaises(ValueError):
                    load_initial_weights(target, self.path)

    def test_partial_residual_head_is_rejected(self) -> None:
        state = HalfKpV1(4, residual_hidden=4).state_dict()
        del state["residual_output.weight"]
        torch.save(state, self.path)
        with self.assertRaisesRegex(ValueError, "missing required"):
            load_initial_weights(HalfKpV1(4, residual_hidden=4), self.path)

    def test_nonfinite_source_rejected_without_model_mutation(self) -> None:
        state = HalfKpV1(4).state_dict()
        state["output.weight"][0, 0] = float("nan")
        torch.save(state, self.path)
        target = HalfKpV1(4)
        before = {name: value.clone() for name, value in target.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_initial_weights(target, self.path)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, target.state_dict()[name]), name)

    def test_full_checkpoint_is_not_accepted_as_initial_weights(self) -> None:
        torch.save({"checkpointVersion": 2, "model": HalfKpV1(4).state_dict()}, self.path)
        with self.assertRaises(ValueError):
            load_initial_weights(HalfKpV1(4), self.path)

    def test_freeze_preserves_trainable_heads(self) -> None:
        model = HalfKpV1(4, True, residual_hidden=4)
        count = freeze_feature_transformer(model)
        self.assertEqual(count, (40960 + 640) * 4 + 4)
        for name, parameter in model.named_parameters():
            should_train = name.startswith(("output.",) + ("residual_hidden_layer.", "residual_output."))
            self.assertEqual(parameter.requires_grad, should_train, name)


if __name__ == "__main__":
    unittest.main()
