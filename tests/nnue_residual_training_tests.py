#!/usr/bin/env python3
"""Deferred CPU-only residual NNUE tests; no engine or training subprocesses."""

from __future__ import annotations

from argparse import Namespace
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

from train import (  # noqa: E402
    FEATURE_COUNT,
    HalfKpV1,
    QuantizedNetwork,
    active_features,
    collate,
    export_network,
    quantize_network,
    quantized_predict,
    validate_resume_configuration,
)


def trunc_div(numerator: int, denominator: int) -> int:
    """Integer-only reference, independent of the production division helper."""
    magnitude = abs(numerator) // denominator
    return -magnitude if numerator < 0 else magnitude


def reference(quantized: QuantizedNetwork, first: list[int], second: list[int]) -> int:
    """Use Python integers/lists rather than NumPy reductions or model.forward."""
    activation: list[int] = []
    hidden = len(quantized.hidden_bias)
    for features in (first, second):
        for unit in range(hidden):
            value = int(quantized.hidden_bias[unit])
            value += sum(int(quantized.input_weights[feature, unit]) for feature in features)
            activation.append(max(0, min(quantized.hidden_scale, value)))
    output = int(quantized.output_bias)
    output += sum(value * int(weight) for value, weight in zip(activation, quantized.output_weights))
    if quantized.residual_width:
        assert quantized.residual_bias is not None
        assert quantized.residual_weights is not None
        assert quantized.residual_output_weights is not None
        for row in range(quantized.residual_width):
            value = int(quantized.residual_bias[row])
            value += sum(activation[unit] * int(quantized.residual_weights[row, unit])
                         for unit in range(2 * hidden))
            value = max(0, min(quantized.hidden_scale,
                               trunc_div(value, quantized.residual_weight_scale)))
            output += value * int(quantized.residual_output_weights[row])
    return max(-32_000, min(32_000, trunc_div(output,
                                           quantized.hidden_scale * quantized.output_scale)))


def zero_model(hidden: int = 1, residual: int = 1) -> HalfKpV1:
    model = HalfKpV1(hidden, residual_hidden=residual)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


class NnueResidualTrainingTests(unittest.TestCase):
    def test_zero_head_preserves_baseline_parameters_rng_and_predictions(self) -> None:
        for factorization in (False, True):
            with self.subTest(factorization=factorization):
                torch.manual_seed(71)
                plain = HalfKpV1(4, feature_factorization=factorization)
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(71)
                residual = HalfKpV1(4, feature_factorization=factorization, residual_hidden=3)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                for name, value in plain.state_dict().items():
                    self.assertTrue(torch.equal(value, residual.state_dict()[name]), name)
                self.assertEqual(plain.residual_hidden, 0)
                self.assertIsNone(plain.residual_hidden_layer)
                self.assertIsNone(plain.residual_output)
                for fen in (chess.STARTING_FEN,
                            "8/8/4k3/3p4/4P3/3K4/8/8 w - - 0 1",
                            "8/8/4k3/3p4/4P3/3K4/8/8 b - - 0 1"):
                    board = chess.Board(fen)
                    batch = collate([(active_features(board, board.turn),
                                      active_features(board, not board.turn), 0.0)])
                    with torch.no_grad():
                        self.assertTrue(torch.equal(plain(*batch[:4]), residual(*batch[:4])))
                    plain_q = quantize_network(plain)
                    residual_q = quantize_network(residual)
                    first, second = active_features(board, board.turn), active_features(board, not board.turn)
                    self.assertEqual(quantized_predict(plain_q, first, second),
                                     quantized_predict(residual_q, first, second))
                    np.testing.assert_array_equal(plain_q.input_weights, residual_q.input_weights)
                    np.testing.assert_array_equal(plain_q.output_weights, residual_q.output_weights)

    def test_nonzero_head_adds_clipped_nonlinear_stm_sensitive_output(self) -> None:
        model = zero_model(1, 2)
        with torch.no_grad():
            model.feature_weights.weight[0, 0] = 0.2
            model.feature_weights.weight[1, 0] = 0.7
            model.residual_hidden_layer.weight.copy_(torch.tensor([[1.0, 1.0], [-1.0, 1.0]]))
            model.residual_hidden_layer.bias.copy_(torch.tensor([-0.5, 0.0]))
            model.residual_output.weight.copy_(torch.tensor([[2.0, -1.0]]))
        forward = collate([([0], [1], 0.0)])
        reverse = collate([([1], [0], 0.0)])
        output = model(*forward[:4])
        self.assertAlmostEqual(float(output.item()), 0.3, places=5)
        self.assertAlmostEqual(float(model(*reverse[:4]).item()), 0.8, places=5)
        output.sum().backward()
        self.assertGreater(float(model.residual_hidden_layer.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.residual_output.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.feature_weights.weight.grad.abs().sum()), 0.0)
        quantized = quantize_network(model, 128, 16, target_scale=32.0, residual_weight_scale=64)
        self.assertEqual(quantized_predict(quantized, [0], [1]), reference(quantized, [0], [1]))
        self.assertLess(abs(quantized_predict(quantized, [0], [1]) - 0.3 * 32.0), 1.0)

    def test_zero_initialized_output_branch_can_receive_learning_signal(self) -> None:
        model = zero_model(1, 2)
        with torch.no_grad():
            model.residual_hidden_layer.bias.fill_(0.5)
        batch = collate([([0], [1], 0.0)])
        loss = (model(*batch[:4]) - 7.0).square().mean()
        loss.backward()
        self.assertGreater(float(model.residual_output.weight.grad.abs().sum()), 0.0)
        # With zero residual-output coefficients, its input layer deliberately
        # receives no signal until the output coefficients first move off zero.
        self.assertEqual(float(model.residual_hidden_layer.weight.grad.abs().sum()), 0.0)

    def test_quantized_reference_and_exact_little_endian_v2_layout(self) -> None:
        torch.manual_seed(97)
        model = HalfKpV1(3, feature_factorization=True, residual_hidden=2)
        with torch.no_grad():
            model.piece_square_weights.weight.fill_(0.001)
            model.residual_output.weight.copy_(torch.tensor([[0.125, -0.25]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "residual.nnue"
            quantized = export_network(model, path, 128, 32, 64.0, residual_weight_scale=64)
            self.assertEqual(quantized.residual_width, 2)
            self.assertEqual(quantized.residual_weight_scale, 64)
            self.assertEqual(quantized.residual_weights.shape, (2, 6))
            expected = struct.pack("<8sIIIiiiIi", b"TNNUE2\0\0", 2, FEATURE_COUNT, 3,
                                   128, 32, quantized.output_bias, 2, 64)
            for values, dtype in ((quantized.hidden_bias, "<i4"),
                                  (quantized.input_weights, "<i2"),
                                  (quantized.output_weights, "<i2"),
                                  (quantized.residual_bias, "<i4"),
                                  (quantized.residual_weights, "<i2"),
                                  (quantized.residual_output_weights, "<i2")):
                expected += np.asarray(values, dtype=dtype).tobytes(order="C")
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(len(expected), 40 + 4 * 3 + 2 * FEATURE_COUNT * 3 +
                             2 * 6 + 4 * 2 + 2 * 2 * 6 + 2 * 2)
            for first, second in (([], []), ([0], [1]), ([640, 1281], [1, 13, 83]),
                                  ([FEATURE_COUNT - 1], [0])):
                self.assertEqual(quantized_predict(quantized, first, second),
                                 reference(quantized, first, second))

    def test_disabled_residual_exports_legacy_layout_without_extra_fields(self) -> None:
        model = HalfKpV1(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.nnue"
            quantized = export_network(model, path, 16, 8)
            expected = struct.pack("<8sIIIiii", b"TNNUE1\0\0", 1, FEATURE_COUNT, 2,
                                   16, 8, quantized.output_bias)
            expected += quantized.hidden_bias.tobytes(order="C")
            expected += quantized.input_weights.tobytes(order="C")
            expected += quantized.output_weights.tobytes(order="C")
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(quantized.residual_width, 0)

    def test_integer_reference_combines_branches_before_final_rounding(self) -> None:
        quantized = QuantizedNetwork(
            hidden_scale=10, output_scale=1, hidden_bias=np.zeros(1, dtype="<i4"),
            input_weights=np.zeros((FEATURE_COUNT, 1), dtype="<i2"),
            output_weights=np.zeros(2, dtype="<i2"), output_bias=0,
            residual_width=1, residual_weight_scale=7,
            residual_bias=np.array([13], dtype="<i4"),
            residual_weights=np.zeros((1, 2), dtype="<i2"),
            residual_output_weights=np.array([6], dtype="<i2"),
        )
        self.assertEqual(quantized_predict(quantized, [], []), 0)  # (13/7)*6 = 6, not 12.
        quantized.output_bias = 6
        self.assertEqual(quantized_predict(quantized, [], []), 1)  # (6 + 6)/10, not 6/10 + 6/10.
        quantized.residual_output_weights[0] = 19
        quantized.output_bias = -28
        self.assertEqual(quantized_predict(quantized, [], []), 0)  # -9/10 truncates to zero.
        quantized.residual_output_weights[0] = 15
        quantized.output_bias = -9
        self.assertEqual(quantized_predict(quantized, [], []), 0)  # -9/10 + 15/10 would be wrong.
        quantized.residual_bias[0] = -1
        quantized.output_bias = 0
        self.assertEqual(quantized_predict(quantized, [], []), 0)
        quantized.residual_bias[0] = np.iinfo(np.int32).max
        quantized.residual_output_weights[0] = 3
        self.assertEqual(quantized_predict(quantized, [], []), 3)

    def test_integer_reference_handles_wide_products_and_output_saturation(self) -> None:
        for scale in (1, 32_767, 32_768, np.iinfo(np.int32).max):
            with self.subTest(scale=scale):
                quantized = QuantizedNetwork(
                    hidden_scale=scale, output_scale=1,
                    hidden_bias=np.array([scale, scale], dtype="<i4"),
                    input_weights=np.zeros((FEATURE_COUNT, 2), dtype="<i2"),
                    output_weights=np.zeros(4, dtype="<i2"), output_bias=0,
                    residual_width=1, residual_weight_scale=1,
                    residual_bias=np.array([0], dtype="<i4"),
                    residual_weights=np.full((1, 4), 32_767, dtype="<i2"),
                    residual_output_weights=np.array([2], dtype="<i2"),
                )
                self.assertEqual(quantized_predict(quantized, [], []), 2)
                self.assertEqual(reference(quantized, [], []), 2)
                for weight, expected in ((32_767, 32_000), (-32_768, -32_000)):
                    quantized.residual_output_weights[0] = weight
                    self.assertEqual(quantized_predict(quantized, [], []), expected)
                    self.assertEqual(quantized_predict(quantized, [], []), reference(quantized, [], []))

    def test_invalid_widths_and_scales_are_rejected(self) -> None:
        for width in (-1, 33, True, 1.5):
            with self.subTest(width=width), self.assertRaises(ValueError):
                HalfKpV1(1, residual_hidden=width)
        model = zero_model()
        for key in ("hidden_scale", "output_scale", "residual_weight_scale"):
            for value in (0, -1, True, 1.5, 2**31):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    quantize_network(model, **{key: value})
        for value in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(target_scale=value), self.assertRaises(ValueError):
                quantize_network(model, target_scale=value)

    def test_nonfinite_parameters_and_quantization_overflows_are_rejected(self) -> None:
        names_and_overflows = (
            ("feature_weights.weight", 32.0),
            ("hidden_bias", float(2**21)),
            ("output.weight", 512.0),
            ("output.bias", 32_768.0),
            ("residual_hidden_layer.weight", 128.0),
            ("residual_hidden_layer.bias", 8192.0),
            ("residual_output.weight", 512.0),
        )
        for name, overflow in names_and_overflows:
            for value in (float("nan"), float("inf"), -float("inf"), overflow):
                with self.subTest(parameter=name, value=value):
                    model = zero_model()
                    with torch.no_grad():
                        dict(model.named_parameters())[name].view(-1)[0] = value
                    with self.assertRaises(ValueError):
                        quantize_network(model)

    def test_legacy_direct_export_also_rejects_nonfinite_parameters(self) -> None:
        for name in ("feature_weights.weight", "hidden_bias", "output.weight", "output.bias"):
            with self.subTest(parameter=name):
                model = zero_model(residual=0)
                with torch.no_grad():
                    dict(model.named_parameters())[name].view(-1)[0] = float("nan")
                with self.assertRaisesRegex(ValueError, "finite"):
                    quantize_network(model)

    def test_transformer_certificate_uses_wide_absolute_values(self) -> None:
        # Float64 permits testing the exact INT32_MAX boundary, not the nearby
        # representable float32 number that would already round out of range.
        model = zero_model().double()
        with torch.no_grad():
            model.feature_weights.weight[0, 0] = -32_768
            model.hidden_bias[0] = np.iinfo(np.int32).max - 64 * 32_768
        quantize_network(model, 1, 1, residual_weight_scale=1)
        with torch.no_grad():
            model.hidden_bias[0] += 1
        with self.assertRaisesRegex(ValueError, "accumulator"):
            quantize_network(model, 1, 1, residual_weight_scale=1)
        with torch.no_grad():
            model.feature_weights.weight.zero_()
            model.hidden_bias[0] = np.iinfo(np.int32).min
        with self.assertRaisesRegex(ValueError, "accumulator"):
            quantize_network(model, 1, 1, residual_weight_scale=1)

    def test_failed_quantization_does_not_overwrite_existing_network(self) -> None:
        model = zero_model()
        with torch.no_grad():
            model.residual_output.weight.fill_(float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owned-fixture.nnue"
            sentinel = b"existing owned fixture, not a model"
            path.write_bytes(sentinel)
            with self.assertRaises(ValueError):
                export_network(model, path)
            self.assertEqual(path.read_bytes(), sentinel)

    def test_resume_legacy_defaults_and_architecture_mismatch(self) -> None:
        settings = {
            "featureFactorization": False, "resultWeight": 0.15,
            "loss": "centipawn-huber", "wdlScale": 400.0, "seed": 1,
            "scheduler": "none", "batchSize": 16, "learningRate": 0.001,
            "minimumLearningRate": 0.00001, "targetScale": 600.0,
            "huberBetaCp": 100.0,
        }
        args = Namespace(feature_factorization=False, residual_hidden=0,
                         residual_weight_scale=256, result_weight=0.15,
                         loss="centipawn-huber", wdl_scale=400.0, seed=1,
                         scheduler="none", batch_size=16, learning_rate=0.001,
                         minimum_learning_rate=0.00001, target_scale=600.0,
                         huber_beta_cp=100.0, epochs=1)
        validate_resume_configuration({"training": settings}, args)
        args.residual_hidden = 2
        with self.assertRaisesRegex(SystemExit, "residualHidden"):
            validate_resume_configuration({"training": settings}, args)
        saved = dict(settings, residualHidden=2, residualWeightScale=256)
        validate_resume_configuration({"training": saved}, args)
        args.residual_weight_scale = 512
        with self.assertRaisesRegex(SystemExit, "residualWeightScale"):
            validate_resume_configuration({"training": saved}, args)


if __name__ == "__main__":
    unittest.main()
