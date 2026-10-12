#!/usr/bin/env python3
"""Deterministic tests for fail-closed NNUE promotion verification."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))
from promotion_contracts import (
    RECOVERY_CONTRACT_ID, RECOVERY_CONTRACT, RESUME_CONTRACT_ID, RESUME_CONTRACT,
)
SPEC = importlib.util.spec_from_file_location(
    "verify_promotion", ROOT / "scripts" / "nnue" / "verify_promotion.py"
)
assert SPEC and SPEC.loader
verify_promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_promotion)


class PromotionFixture:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.training = self.root / "training"
        self.series_dir = self.root / "series"
        self.network = self.training / "shared" / "model.nnue"
        self.engine = self.root / "engine.bin"
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
            "current": "classical",
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
        successful_probe = lambda engine, network: {
            "successful": True,
            "engine": str(engine),
            "network": str(network),
        }
        return verify_promotion.PromotionVerifier(
            self.series_dir, expected_hash, runtime_probe=successful_probe
        ).verify()


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

    def make_recovery(self):
        original = self.fixture
        original.series["state"] = "running"
        interrupted = original.series["stages"][1]
        interrupted["state"] = "running"
        interrupted.pop("result")
        # The fixture originally created a terminal artifact; an interrupted
        # source has none. Remove only this test-owned temporary artifact.
        (original.series_dir / "classical" / "result.json").unlink()
        original.save()
        source_path = original.series_dir / "series.json"
        recovery_dir = original.root / "recovery"
        recovery_dir.mkdir()
        state = copy.deepcopy(original.series)
        state.update(
            state="complete", contractId=RECOVERY_CONTRACT_ID,
            recovery={
                "sourceSeries": str(source_path),
                "sourceSeriesSha256": verify_promotion.file_sha256(source_path),
                "reusedResultSha256": verify_promotion.file_sha256(original.series_dir / "network" / "result.json"),
            },
        )
        long_stage = original.stage("classical", RECOVERY_CONTRACT["classical"])
        state["stages"][1] = long_stage
        original.write_json(recovery_dir / "series.json", state)
        original.write_json(recovery_dir / "classical" / "result.json", long_stage["result"])
        return recovery_dir, state

    def verify_recovery(self, path):
        return verify_promotion.PromotionVerifier(
            path, runtime_probe=lambda engine, network: {"successful": True}
        ).verify()

    def make_paused_restart(self):
        paused_dir, paused = self.make_recovery()
        paused["state"] = "failed"
        paused["stages"][1]["state"] = "failed"
        paused["stages"][1].pop("result")
        (paused_dir / "classical" / "result.json").unlink()
        control = {"status": "paused", "pausedAt": "2026-10-12T19:26:00+00:00"}
        self.fixture.write_json(paused_dir / "series.json", paused)
        self.fixture.write_json(paused_dir / "classical" / "dashboard-control.json", control)
        (paused_dir / "classical" / "match.log").write_text("preserved partial match\n")
        (paused_dir / "classical" / "games.pgn").write_text("preserved partial games\n")
        restarted = copy.deepcopy(paused)
        restarted["state"] = "complete"
        restarted["contractId"] = RESUME_CONTRACT_ID
        restarted["stages"][1] = self.fixture.stage("classical", RESUME_CONTRACT["classical"])
        restarted["recovery"]["pausedRun"] = {
            "directory": str(paused_dir),
            "seriesSha256": verify_promotion.file_sha256(paused_dir / "series.json"),
            "pauseRecordSha256": verify_promotion.file_sha256(paused_dir / "classical" / "dashboard-control.json"),
            "logSha256": verify_promotion.file_sha256(paused_dir / "classical" / "match.log"),
            "pgnSha256": verify_promotion.file_sha256(paused_dir / "classical" / "games.pgn"),
        }
        new_dir = self.fixture.root / "restart"
        self.fixture.write_json(new_dir / "series.json", restarted)
        self.fixture.write_json(new_dir / "classical" / "result.json", restarted["stages"][1]["result"])
        return new_dir, restarted, paused_dir

    def test_fresh_restart_records_user_pause_and_passes_complete_contract(self):
        path, _, _ = self.make_paused_restart()
        report = self.verify_recovery(path)
        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["contractId"], RESUME_CONTRACT_ID)

    def test_restart_rejects_changed_excluded_pgn(self):
        path, _, paused_dir = self.make_paused_restart()
        (paused_dir / "classical" / "games.pgn").write_text("changed archive\n")
        report = self.verify_recovery(path)
        self.assertEqual(report["decision"], "fail")
        self.assertIn("pause_exclusion.pgnSha256", self.failed_codes(report))

    def test_restart_requires_explicit_pause_record(self):
        path, _, paused_dir = self.make_paused_restart()
        self.fixture.write_json(paused_dir / "classical" / "dashboard-control.json", {"status": "failed"})
        report = self.verify_recovery(path)
        self.assertEqual(report["decision"], "fail")
        self.assertIn("pause_exclusion.user_pause", self.failed_codes(report))

    def test_restart_cannot_replace_terminal_long_result(self):
        path, _, paused_dir = self.make_paused_restart()
        self.fixture.write_json(paused_dir / "classical" / "result.json", {"completed": True})
        report = self.verify_recovery(path)
        self.assertEqual(report["decision"], "fail")
        self.assertIn("pause_exclusion.user_pause", self.failed_codes(report))

    def test_complete_clean_positive_series_passes(self) -> None:
        report = self.fixture.verify(self.fixture.network_hash)

        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(len(report["stages"]), 2)
        self.assertEqual(report["networkSha256"], self.fixture.network_hash)
        self.assertEqual(report["engineSha256"], self.fixture.engine_hash)

    def test_recovery_reuses_only_complete_short_stage_and_fresh_full_long_stage(self):
        path, _ = self.make_recovery()
        report = self.verify_recovery(path)
        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["contractId"], RECOVERY_CONTRACT_ID)
        self.assertFalse((path / "network").exists())

    def test_changed_source_checksum_blocks_recovery(self):
        path, state = self.make_recovery()
        state["recovery"]["sourceSeriesSha256"] = "0" * 64
        self.fixture.write_json(path / "series.json", state)
        report = self.verify_recovery(path)
        self.assertEqual(report["decision"], "fail")
        self.assertIn("recovery.source_checksum", self.failed_codes(report))

    def test_changed_reused_short_result_blocks_recovery(self):
        path, state = self.make_recovery()
        state["stages"][0]["result"]["elo"]["difference"] += 1
        self.fixture.write_json(path / "series.json", state)
        report = self.verify_recovery(path)
        self.assertEqual(report["decision"], "fail")
        self.assertIn("recovery.reused_stage", self.failed_codes(report))

    def test_terminal_long_result_cannot_be_replaced(self):
        path, _ = self.make_recovery()
        self.fixture.write_json(self.fixture.series_dir / "classical" / "result.json", {"completed": True})
        report = self.verify_recovery(path)
        self.assertEqual(report["decision"], "fail")
        self.assertIn("recovery.interrupted_source", self.failed_codes(report))

    def test_original_seed_cannot_be_reused_for_recovery_long_stage(self):
        path, state = self.make_recovery()
        command = state["stages"][1]["command"]
        command[command.index("--seed") + 1] = "202609141"
        self.fixture.write_json(path / "series.json", state)
        report = self.verify_recovery(path)
        self.assertEqual(report["decision"], "fail")
        self.assertIn("classical.contract.seed", self.failed_codes(report))

    def test_running_stage_is_pending_and_cannot_pass(self) -> None:
        self.fixture.series["state"] = "running"
        stage = self.fixture.series["stages"][1]
        stage["state"] = "running"
        stage.pop("result")
        (self.fixture.series_dir / "classical").mkdir(exist_ok=True)
        (self.fixture.series_dir / "classical" / "match.pid").write_text(str(os.getpid()))
        self.fixture.save()

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "pending")
        self.assertGreaterEqual(report["summary"]["pending"], 2)

    def test_dead_runner_is_interrupted_not_pending(self) -> None:
        self.fixture.series["state"] = "running"
        stage = self.fixture.series["stages"][1]
        stage["state"] = "running"
        stage.pop("result")
        self.fixture.save()

        report = self.fixture.verify()

        self.assertEqual(report["decision"], "fail")
        self.assertEqual(report["observedSeriesState"], "interrupted")

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

    def test_runtime_activation_failure_blocks_promotion(self) -> None:
        failed_probe = lambda engine, network: {
            "successful": False,
            "errors": ["info string NNUE load failed: invalid network"],
        }

        report = verify_promotion.PromotionVerifier(
            self.fixture.series_dir,
            runtime_probe=failed_probe,
        ).verify()

        self.assertEqual(report["decision"], "fail")
        self.assertIn("engine.nnue_runtime_probe", self.failed_codes(report))


if __name__ == "__main__":
    unittest.main()
