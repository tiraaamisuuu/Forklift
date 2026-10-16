#!/usr/bin/env python3
"""Stdlib-only legality benchmark parser fixtures; no engines or builds."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_legalgen import parse_run, signature  # noqa: E402


def fixture(evaluator: str = "nnue", diagnostics: str = "") -> str:
    weight, mode, kernel = ("100", "incremental", "sse2") if evaluator == "nnue" else ("0", "n/a", "n/a")
    lines = [
        "Benchmark: depth=12 timeLimit=30000ms threads=1 "
        f"evaluator={evaluator} nnueWeight={weight} nnueMode={mode} nnueOutput={kernel} "
        "probCut=off singular=off singularBudget=0 positions=4",
    ]
    for index, name in enumerate(("Start", "Middlegame 1", "Middlegame 2", "Endgame")):
        lines.append(
            f"{name:<12} best=e2e4 depth=12 score={index - 2} nodes={1000 + index} qnodes={100 + index} "
            "probcut=0/0 singular=0/0 singularPathMax=0 singularProofNodes=0 singularSkipped=0 "
            f"time={100 + index}ms nps=10000"
        )
    lines.append("Benchmark summary: nodes=4006 time=406ms nps=10000")
    if diagnostics:
        lines.append("Legalgen diagnostics: " + diagnostics)
    return "\n".join(lines) + "\n"


DIAGNOSTICS = "calls=10 pseudo=100 probes=20 fast=80 checked=2 legalgenNs=1000 harnessNs=2000"


class LegalgenBenchmarkTests(unittest.TestCase):
    def parse(self, output: str | None = None, evaluator: str = "nnue", diagnostics: bool = False) -> dict:
        return parse_run(fixture(evaluator) if output is None else output,
                         12, 30000, evaluator, 4, diagnostics)

    def test_valid_nnue_fixture_preserves_rows_and_counters(self) -> None:
        run = self.parse()
        self.assertEqual(run["totalMilliseconds"], 406)
        self.assertEqual([row["name"] for row in run["positions"]],
                         ["Start", "Middlegame 1", "Middlegame 2", "Endgame"])
        self.assertEqual(run["positions"][0]["score"], -2)
        self.assertEqual(run["positions"][0]["searchCounters"], {
            "probcut": "0/0", "singular": "0/0", "singularPathMax": "0",
            "singularProofNodes": "0", "singularSkipped": "0",
        })

    def test_valid_classical_fixture(self) -> None:
        self.assertEqual(self.parse(evaluator="classical")["header"]["nnueMode"], "n/a")

    def test_settings_mismatch_fails_closed(self) -> None:
        for before, after in (("depth=12", "depth=11"), ("threads=1", "threads=2"),
                              ("timeLimit=30000ms", "timeLimit=500ms"),
                              ("evaluator=nnue", "evaluator=classical"),
                              ("nnueWeight=100", "nnueWeight=50"),
                              ("nnueMode=incremental", "nnueMode=rebuild"),
                              ("probCut=off", "probCut=on"), ("singular=off", "singular=on"),
                              ("singularBudget=0", "singularBudget=5"),
                              ("positions=4", "positions=3")):
            with self.subTest(setting=before), self.assertRaises(ValueError):
                self.parse(fixture().replace(before, after, 1))

    def test_missing_or_extra_position_fails_closed(self) -> None:
        lines = fixture().splitlines()
        for output in ("\n".join(lines[:1] + lines[2:]),
                       "\n".join(lines[:2] + lines[1:])):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.parse(output)

    def test_incomplete_position_depth_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete"):
            self.parse(fixture().replace("best=e2e4 depth=12", "best=e2e4 depth=11", 1))

    def test_missing_or_insufficient_search_time_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing elapsed"):
            self.parse(fixture().replace("time=100ms", "elapsed=100ms", 1))
        output = fixture()
        for elapsed in range(100, 104):
            output = output.replace(f"time={elapsed}ms", "time=0ms")
        with self.assertRaisesRegex(ValueError, "resolution"):
            self.parse(output)

    def test_timing_and_nps_are_excluded_from_signature(self) -> None:
        original = self.parse()
        different_time = self.parse(fixture().replace("time=100ms", "time=999ms", 1)
                                   .replace("nps=10000", "nps=999", 1))
        self.assertNotEqual(original["totalMilliseconds"], different_time["totalMilliseconds"])
        self.assertEqual(signature(original), signature(different_time))

    def test_all_printed_search_signature_changes_are_detected(self) -> None:
        original = signature(self.parse())
        for before, after in (("Start", "Changed"), ("best=e2e4", "best=d2d4"),
                              ("score=-2", "score=-3"), ("nodes=1000", "nodes=1001"),
                              ("qnodes=100", "qnodes=101"), ("probcut=0/0", "probcut=0/1"),
                              ("singular=0/0", "singular=0/1"),
                              ("singularPathMax=0", "singularPathMax=1"),
                              ("singularProofNodes=0", "singularProofNodes=1"),
                              ("singularSkipped=0", "singularSkipped=1")):
            with self.subTest(field=before):
                changed = self.parse(fixture().replace(before, after, 1))
                self.assertNotEqual(original, signature(changed))

    def test_diagnostic_fixture_balances_and_stays_out_of_signature(self) -> None:
        run = self.parse(fixture(diagnostics=DIAGNOSTICS), diagnostics=True)
        self.assertEqual(run["diagnostics"]["fast"], 80)
        self.assertEqual(signature(run), signature(self.parse()))

    def test_instrumented_binary_is_rejected_as_speed_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "instrumented"):
            self.parse(fixture(diagnostics=DIAGNOSTICS))

    def test_diagnostics_are_required_exactly_once(self) -> None:
        for output in (fixture(), fixture(diagnostics=DIAGNOSTICS) +
                       "Legalgen diagnostics: " + DIAGNOSTICS + "\n"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "missing or duplicate"):
                self.parse(output, diagnostics=True)

    def test_unknown_or_missing_diagnostic_field_fails_closed(self) -> None:
        for counters in (DIAGNOSTICS.replace("calls=", "ignored="),
                         DIAGNOSTICS.replace("calls=10 ", ""),
                         DIAGNOSTICS + " unexpected=1"):
            with self.subTest(counters=counters), self.assertRaisesRegex(ValueError, "fields"):
                self.parse(fixture(diagnostics=counters), diagnostics=True)

    def test_unbalanced_diagnostic_accounting_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "balance"):
            self.parse(fixture(diagnostics=DIAGNOSTICS.replace("fast=80", "fast=79")), diagnostics=True)


if __name__ == "__main__":
    unittest.main()
