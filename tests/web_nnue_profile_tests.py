#!/usr/bin/env python3
"""Unit tests for web NNUE profile activation evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "web_server", ROOT / "web" / "server.py"
)
assert SPEC and SPEC.loader
web_server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = web_server
SPEC.loader.exec_module(web_server)


class WebNnueProfileTests(unittest.TestCase):
    def test_accepts_exact_network_load_confirmation(self) -> None:
        network = Path("approved.nnue").resolve()
        output = "\n".join((
            "id name Forklift v1.1.0",
            "uciok",
            f"info string NNUE loaded: {network}",
            "readyok",
        ))

        report = web_server.classify_nnue_activation(output, 0, network)

        self.assertTrue(report["successful"])
        self.assertTrue(report["networkLoaded"])

    def test_rejects_silent_classical_fallback(self) -> None:
        report = web_server.classify_nnue_activation(
            "uciok\nreadyok\n", 0, Path("approved.nnue").resolve()
        )

        self.assertFalse(report["successful"])
        self.assertFalse(report["networkLoaded"])

    def test_rejects_reported_load_and_activation_errors(self) -> None:
        output = "\n".join((
            "uciok",
            "info string NNUE load failed: invalid NNUE magic",
            "info string Use NNUE requires a valid EvalFile",
            "readyok",
        ))

        report = web_server.classify_nnue_activation(
            output, 0, Path("approved.nnue").resolve()
        )

        self.assertFalse(report["successful"])
        self.assertEqual(len(report["errors"]), 2)


if __name__ == "__main__":
    unittest.main()
