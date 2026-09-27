#!/usr/bin/env python3
"""Deterministic tests for match-runner metadata and result parsing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_engines", ROOT / "scripts" / "compare_engines.py"
)
assert SPEC and SPEC.loader
compare_engines = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_engines)


class CompareEnginesTests(unittest.TestCase):
    def test_resolves_per_engine_resource_overrides(self) -> None:
        resources = compare_engines.per_engine_resources(
            threads=1,
            hash_mb=256,
            baseline_threads=None,
            candidate_threads=6,
            baseline_hash_mb=128,
            candidate_hash_mb=None,
        )

        self.assertEqual(resources["baseline"], {"threads": 1, "hashMb": 128})
        self.assertEqual(resources["candidate"], {"threads": 6, "hashMb": 256})

    def test_rejects_invalid_per_engine_resources(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must be positive"):
            compare_engines.per_engine_resources(1, 256, None, 0, None, None)

    def test_parses_uci_option_bounds_and_values(self) -> None:
        output = "\n".join(
            [
                "id name Stockfish 18",
                "option name Threads type spin default 1 min 1 max 1024",
                "option name UCI_LimitStrength type check default false",
                "option name UCI_Elo type spin default 1320 min 1320 max 3190",
                "option name Style type combo default Normal var Solid var Normal var Risky",
                "uciok",
            ]
        )

        options = compare_engines.parse_uci_options(output)

        self.assertEqual(options[0]["name"], "Threads")
        self.assertEqual(options[0]["min"], 1)
        self.assertEqual(options[0]["max"], 1024)
        self.assertEqual(options[2]["default"], "1320")
        self.assertEqual(options[2]["min"], 1320)
        self.assertEqual(options[2]["max"], 3190)
        self.assertEqual(options[3]["vars"], ["Solid", "Normal", "Risky"])

    def test_configures_supported_options_and_allows_overrides(self) -> None:
        inspection = {
            "options": [
                {"name": "Hash", "type": "spin"},
                {"name": "Threads", "type": "spin"},
                {"name": "UCI_LimitStrength", "type": "check"},
                {"name": "UCI_Elo", "type": "spin", "min": 1320, "max": 3190},
            ]
        }

        options = compare_engines.configured_options(
            inspection,
            "Baseline",
            threads=2,
            hash_mb=128,
            custom_values=["Hash=64", "UCI_LimitStrength=true", "UCI_Elo=1800"],
            eval_file=None,
        )

        self.assertEqual(
            options,
            [
                ("Hash", "64"),
                ("Threads", "2"),
                ("UCI_LimitStrength", "true"),
                ("UCI_Elo", "1800"),
            ],
        )

    def test_configuration_probe_requires_explicit_forklift_nnue_load(self) -> None:
        network = Path("approved.nnue").resolve()
        output = "\n".join(
            [
                "id name Forklift v1.1.0",
                "uciok",
                f"info string NNUE loaded: {network}",
                "readyok",
            ]
        )

        report = compare_engines.classify_configuration_probe(
            output, 0, network
        )

        self.assertTrue(report["successful"])
        self.assertTrue(report["nnueConfirmed"])

    def test_configuration_probe_rejects_silent_nnue_fallback(self) -> None:
        report = compare_engines.classify_configuration_probe(
            "uciok\nreadyok\n", 0, Path("approved.nnue").resolve()
        )

        self.assertFalse(report["successful"])
        self.assertFalse(report["nnueConfirmed"])

    def test_configuration_probe_rejects_reported_load_error(self) -> None:
        output = "\n".join(
            [
                "uciok",
                "info string NNUE load failed: checksum mismatch",
                "info string Use NNUE requires a valid EvalFile",
                "readyok",
            ]
        )

        report = compare_engines.classify_configuration_probe(output, 0)

        self.assertFalse(report["successful"])
        self.assertEqual(len(report["errors"]), 2)

    def test_generic_engine_probe_needs_protocol_success_not_custom_message(self) -> None:
        report = compare_engines.classify_configuration_probe(
            "uciok\nreadyok\n", 0
        )

        self.assertTrue(report["successful"])
        self.assertFalse(report["nnueConfirmationRequired"])

    def test_rejects_option_outside_discovered_bounds(self) -> None:
        inspection = {
            "options": [
                {"name": "Hash", "type": "spin"},
                {"name": "Threads", "type": "spin"},
                {"name": "UCI_Elo", "type": "spin", "min": 1320, "max": 3190},
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "below its minimum 1320"):
            compare_engines.configured_options(
                inspection,
                "Baseline",
                threads=1,
                hash_mb=16,
                custom_values=["UCI_Elo=1200"],
                eval_file=None,
            )

    def test_rejects_unadvertised_custom_option(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not advertise"):
            compare_engines.configured_options(
                {
                    "options": [
                        {"name": "Hash", "type": "spin"},
                        {"name": "Threads", "type": "spin"},
                    ]
                },
                "Candidate",
                threads=1,
                hash_mb=16,
                custom_values=["Unknown=1"],
                eval_file=None,
            )

    def test_engine_definition_preserves_external_arguments(self) -> None:
        definition = compare_engines.engine_definition(
            "Calibrated Stockfish",
            Path("stockfish.exe"),
            ["--some-argument"],
            [("UCI_LimitStrength", "true"), ("UCI_Elo", "1800")],
        )

        self.assertIn("arg=--some-argument", definition)
        self.assertIn("option.UCI_LimitStrength=true", definition)
        self.assertIn("option.UCI_Elo=1800", definition)
        self.assertNotIn("arg=--uci", definition)

    def test_parses_machine_readable_result_and_failures(self) -> None:
        log = """\
Finished game 1 (Candidate vs Baseline): 1-0 {White mates}
Score of Candidate vs Baseline: 1 - 0 - 0  [1.000] 1
Finished game 2 (Baseline vs Candidate): 1-0 {Black loses on time}
Score of Candidate vs Baseline: 1 - 1 - 0  [0.500] 2
Finished game 3 (Candidate vs Baseline): 0-1 {White makes an illegal move}
Score of Candidate vs Baseline: 1 - 2 - 0  [0.333] 3
Finished game 4 (Baseline vs Candidate): 1/2-1/2 {Draw by adjudication}
Score of Candidate vs Baseline: 1 - 2 - 1  [0.375] 4
Elo difference: -88.7 +/- 100.1, LOS: 4.1 %, DrawRatio: 25.0 %
"""

        result = compare_engines.parse_match_result(
            log, "Candidate", "Baseline", expected_games=4, exit_code=0
        )

        self.assertTrue(result["completed"])
        self.assertEqual(result["score"]["candidateWins"], 1)
        self.assertEqual(result["score"]["baselineWins"], 2)
        self.assertEqual(result["score"]["draws"], 1)
        self.assertEqual(result["elo"]["difference"], -88.7)
        self.assertEqual(result["failures"]["timeForfeits"], 1)
        self.assertEqual(result["failures"]["illegalMoves"], 1)

    def test_non_finite_elo_is_valid_strict_json_data(self) -> None:
        log = """\
Finished game 1 (Candidate vs Baseline): 1-0 {White mates}
Score of Candidate vs Baseline: 1 - 0 - 0  [1.000] 1
Elo difference: inf +/- nan, LOS: 97.7 %, DrawRatio: 0.0 %
"""

        result = compare_engines.parse_match_result(
            log, "Candidate", "Baseline", expected_games=1, exit_code=0
        )

        self.assertIsNone(result["elo"]["difference"])
        self.assertIsNone(result["elo"]["uncertainty"])
        self.assertEqual(result["elo"]["display"], "inf +/- nan")
        json.dumps(result, allow_nan=False)

    def test_accepts_valid_early_sprt_h1_stop(self) -> None:
        log = """\
Finished game 1 (Candidate vs Champion): 1-0 {White mates}
Finished game 2 (Champion vs Candidate): 0-1 {Black mates}
SPRT: llr 3.02 (102.6%), lbound -2.94444, ubound 2.94444
"""

        result = compare_engines.parse_match_result(
            log,
            "Candidate",
            "Champion",
            expected_games=10000,
            exit_code=0,
            sprt_enabled=True,
        )

        self.assertTrue(result["completed"])
        self.assertEqual(result["finishedGames"], 2)
        self.assertEqual(result["sprt"]["decision"], "accepted_h1")

    def test_records_maximum_game_sprt_as_inconclusive(self) -> None:
        log = """\
Finished game 1 (Candidate vs Champion): 1/2-1/2 {Draw by adjudication}
Finished game 2 (Champion vs Candidate): 1/2-1/2 {Draw by adjudication}
SPRT: llr 0.12 (4.1%), lbound -2.94444, ubound 2.94444
"""

        result = compare_engines.parse_match_result(
            log,
            "Candidate",
            "Champion",
            expected_games=2,
            exit_code=0,
            sprt_enabled=True,
        )

        self.assertTrue(result["completed"])
        self.assertEqual(result["sprt"]["decision"], "inconclusive")

    def test_accepts_cutechess_sprt_suffix(self) -> None:
        log = "SPRT: llr 2.97 (100.7%), lbound -2.94, ubound 2.94 - H1 was accepted\n"

        result = compare_engines.parse_sprt_result(log, enabled=True)

        self.assertEqual(result["decision"], "accepted_h1")


if __name__ == "__main__":
    unittest.main()
