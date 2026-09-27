#!/usr/bin/env python3
"""Deterministic tests for fail-closed NNUE promotion verification."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_promotion", ROOT / "scripts" / "nnue" / "verify_promotion.py"
)
assert SPEC and SPEC.loader
verify_promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_promotion)


class PromotionFixture:
    def __init__(self, root: Path):
        self.root = root
        self.training = root / "training"
        self.series_dir = root / "series"
        self.network = self.training / "shared" / "model.nnue"
        self.engine = root / "engine.bin"
        self.network.parent.mkdir(parents=True)
        self.series_dir.mkdir()
        self.network.write_bytes(b"approved-network")
        self.engine.write_bytes(b"frozen-engine")
        self.network_hash = verify_promotion.file_sha256(self.network)
        self.engine_hash = verify_promotion.file_sha256(self.engine)
        self.write_json(
            self.network.with_suffix(".manifest.json"),
            {
                "cppVerification": {"exactMatch": True},
                "outputs": {"network": {"sha256": self.network_hash}},
            },
        )
        self.write_json(
            self.training / "experiment.json",
            {
                "state": "complete",
                "variants": [
                    {
                        "id": "shared",
                        "state": "complete",
                        "network": {
                            "path": str(self.network),
                            "sha256": self.network_hash,
                        },
                    }
                ],
            },
        )
        self.series = {
            "state": "complete",
            "training": str(self.training),
            "engineSha256": self.engine_hash,
            "stages": [
                self.stage(stage_id, contract)
                for stage_id, contract in verify_promotion.PROMOTION_CONTRACT.items()
            ],
        }
        self.save()

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def stage(self, stage_id: str, contract: dict[str, object]) -> dict[str, object]:
        difference = 60.0 if stage_id == "network" else 35.0
        uncertainty = 20.0
        result = {
            "completed": True,
            "processExitCode": 0,
            "expectedGames": contract["games"],
            "finishedGames": contract["games"],
            "score": {
                "candidateWins": 1,
                "baselineWins": 0,
                "draws": int(contract["games"]) - 1,
                "games": contract["games"],
            },
            "failures": {
                "crashes": 0,
                "disconnects": 0,
                "illegalMoves": 0,
                "timeForfeits": 0,
            },
            "elo": {
                "difference": difference,
                "uncertainty": uncertainty,
                "display": f"{difference} +/- {uncertainty}",
            },
        }
        command = [
            "python",
            "compare_engines.py",
            "--candidate-exe", str(self.engine),
            "--baseline-exe", str(self.engine),
            "--candidate-eval-file", str(self.network),
            "--candidate-option", "NNUE Weight=100",
            "--baseline-option", "Use NNUE=false",
            "--games", str(contract["games"]),
            "--tc", str(contract["timeControl"]),
            "--threads", str(contract["threads"]),
            "--hash", str(contract["hashMb"]),
            "--concurrency", str(contract["concurrency"]),
            "--seed", str(contract["seed"]),
            "--output-dir", str(self.series_dir / stage_id),
        ]
        return {"id": stage_id, "state": "complete", "command": command, "result": result}

    def save(self) -> None:
        self.write_json(self.series_dir / "series.json", self.series)
        for stage in self.series["stages"]:
            if stage.get("state") == "complete" and isinstance(stage.get("result"), dict):
                self.write_json(
                    self.series_dir / str(stage["id"]) / "result.json",
                    stage["result"],
                )

    def verify(self, expected_hash: str | None = None) -> dict[str, object]:
        return verify_promotion.PromotionVerifier(self.series_dir, expected_hash).verify()


class NnuePromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = PromotionFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def failed_codes(self, report: dict[str, object]) -> set[str]:
        return {
            check["code"]
            for check in report["checks"]
            if check["status"] == "fail"
        }

    def test_complete_clean_positive_series_passes(self) -> None:
        report = self.fixture.verify(self.fixture.network_hash)

        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(len(report["stages"]), 2)
        self.assertEqual(report["networkSha256"], self.fixture.network_hash)
        self.assertEqual(report["engineSha256"], self.fixture.engine_hash)

    def test_running_stage_is_pending_and_cannot_pass(self) -> None:
        self.fixture.series["state"] = "running"
        stage = self.fixture.series["stages"][1]
        stage["state"] = "running"
        stage.pop("result")
        self.fixture.save()

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "pending")
        self.assertGreaterEqual(report["summary"]["pending"], 2)

    def test_any_technical_termination_fails_closed(self) -> None:
        result = self.fixture.series["stages"][0]["result"]
        result["failures"]["timeForfeits"] = 1
        self.fixture.save()

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "fail")
        self.assertIn("network.technical_clean", self.failed_codes(report))

    def test_boolean_failure_counter_is_not_accepted_as_zero(self) -> None:
        result = self.fixture.series["stages"][0]["result"]
        result["failures"]["crashes"] = False
        self.fixture.save()

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "fail")
        self.assertIn("network.technical_clean", self.failed_codes(report))

    def test_score_components_must_sum_to_finished_games(self) -> None:
        result = self.fixture.series["stages"][0]["result"]
        result["score"]["draws"] -= 1
        self.fixture.save()

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "fail")
        self.assertIn("network.score_consistent", self.failed_codes(report))

    def test_boolean_process_exit_code_is_not_accepted_as_zero(self) -> None:
        result = self.fixture.series["stages"][0]["result"]
        result["processExitCode"] = False
        self.fixture.save()

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "fail")
        self.assertIn("network.result_complete", self.failed_codes(report))

    def test_non_positive_lower_elo_bound_fails(self) -> None:
        result = self.fixture.series["stages"][1]["result"]
        result["elo"].update(difference=19.9, uncertainty=20.0)
        self.fixture.save()

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "fail")
        self.assertIn("classical.positive_lower_bound", self.failed_codes(report))

    def test_network_checksum_mismatch_fails(self) -> None:
        self.fixture.network.write_bytes(b"tampered-network")

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "fail")
        self.assertIn("network.checksum", self.failed_codes(report))

    def test_mutated_match_contract_fails(self) -> None:
        command = self.fixture.series["stages"][0]["command"]
        command[command.index("--seed") + 1] = "99"
        self.fixture.save()

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "fail")
        self.assertIn("network.contract.seed", self.failed_codes(report))

    def test_embedded_result_must_match_preserved_artifact(self) -> None:
        result = self.fixture.series["stages"][0]["result"]
        result["score"]["candidateWins"] = 2
        self.fixture.write_json(self.fixture.series_dir / "series.json", self.fixture.series)

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "fail")
        self.assertIn("network.result_artifact", self.failed_codes(report))


if __name__ == "__main__":
    unittest.main()
