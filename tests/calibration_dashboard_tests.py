#!/usr/bin/env python3
"""Deterministic tests for the live calibration dashboard aggregation."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibration_dashboard", ROOT / "scripts" / "calibration_dashboard.py"
)
assert SPEC and SPEC.loader
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


class CalibrationDashboardTests(unittest.TestCase):
    def test_allocates_a_new_resume_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "gate"
            source.mkdir()
            (root / "gate-resume-001").mkdir()

            self.assertEqual(
                dashboard.next_resume_directory(source),
                root / "gate-resume-002",
            )
            self.assertEqual(
                dashboard.next_resume_directory(root / "gate-resume-001"),
                root / "gate-resume-002",
            )

    def test_follows_dashboard_resume_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "gate"
            resumed = root / "gate-resume-001"
            source.mkdir()
            resumed.mkdir()
            (source / "dashboard-control.json").write_text(
                json.dumps({"resumedAs": str(resumed)})
            )

            self.assertEqual(dashboard.latest_control_run(source), resumed.resolve())

    def test_rebuilds_champion_gate_with_a_new_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "gate"
            destination = Path(temporary) / "gate-resume-001"
            source.mkdir()
            (source / "gate-manifest.json").write_text(
                json.dumps(
                    {
                        "championRegistry": str(ROOT / "research" / "champion.json"),
                        "candidateCommit": "a" * 40,
                        "contract": {
                            "games": 10000,
                            "timeControl": "30+0.3",
                            "threads": 1,
                            "hashMb": 256,
                            "concurrency": 6,
                            "seed": 6002,
                            "elo0": 0,
                            "elo1": 5,
                        },
                    }
                )
            )

            command, seed = dashboard.champion_resume_command(source, destination)

            self.assertEqual(seed, 6003)
            self.assertIn(str(destination), command)
            self.assertEqual(command[command.index("--candidate") + 1], "a" * 40)
            self.assertEqual(command[command.index("--seed") + 1], "6003")
            self.assertEqual(command[command.index("--tc") + 1], "30+0.3")

    def test_parses_live_score_and_elo(self) -> None:
        result = dashboard.parse_live_result(
            "Score of Forklift vs Stockfish 18 @ 2600: 17 - 11 - 12 [0.575] 40\n"
            "Elo difference: +52.5 +/- 91.0, LOS: 87.2 %, DrawRatio: 30.0 %\n"
        )
        self.assertEqual((result["wins"], result["draws"], result["losses"]), (17, 12, 11))
        self.assertEqual(result["games"], 40)
        self.assertEqual(result["score"], 0.575)
        self.assertEqual(result["relativeElo"], 52.5)

    def test_builds_live_direct_match_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            match_dir = run_dir / "match"
            match_dir.mkdir()
            (match_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "createdAt": datetime.now(timezone.utc).isoformat(),
                        "configuration": {
                            "games": 5000,
                            "timeControl": "10+0.1",
                            "threads": 1,
                            "hashMb": 256,
                            "concurrency": 6,
                            "seed": 5701,
                            "sprt": {"enabled": True, "elo0": 0, "elo1": 5},
                        },
                        "engines": [
                            {
                                "side": "Candidate",
                                "matchName": "Search candidate",
                                "commit": "candidate123",
                            },
                            {
                                "side": "Baseline",
                                "matchName": "Pre-change baseline",
                                "commit": "baseline123",
                            },
                        ],
                    }
                )
            )
            (match_dir / "match.log").write_text(
                "Score of Search candidate vs Pre-change baseline: "
                "3 - 1 - 2  [0.667] 6\n"
                "Elo difference: +120.4 +/- 90.0, LOS: 99.0 %, "
                "DrawRatio: 33.3 %\n"
                "SPRT: llr 0.42 (14.3%), lbound -2.94444, ubound 2.94444\n"
            )

            snapshot = dashboard.build_match_snapshot(run_dir)

            self.assertEqual(snapshot["games"], 6)
            self.assertEqual(
                (snapshot["wins"], snapshot["draws"], snapshot["losses"]),
                (3, 2, 1),
            )
            self.assertEqual(snapshot["scorePercent"], 66.7)
            self.assertEqual(snapshot["relativeElo"], 120.4)
            self.assertFalse(snapshot["relativeEloEstimated"])
            self.assertEqual(snapshot["sprt"]["llr"], 0.42)
            self.assertEqual(snapshot["candidate"]["commit"], "candidate123")

    def test_interpolates_completed_rating_bracket(self) -> None:
        estimate = dashboard.local_pool_estimate(
            [
                {"elo": 2400, "state": "complete", "scorePercent": 60.0},
                {"elo": 2600, "state": "complete", "scorePercent": 40.0},
            ]
        )
        self.assertEqual(estimate["status"], "bracketed")
        self.assertEqual(estimate["estimate"], 2500.0)

    def test_completed_unbracketed_run_recommends_another_rung(self) -> None:
        estimate = dashboard.local_pool_estimate(
            [
                {"elo": 2350, "state": "complete", "scorePercent": 62.7},
                {"elo": 2400, "state": "complete", "scorePercent": 50.2},
            ],
            run_complete=True,
        )
        self.assertEqual(estimate["status"], "above_range")
        self.assertEqual(estimate["display"], "above 2400; add a higher rung")

    def test_timeout_sensitivity_excludes_flagged_game_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "attempt-001"
            match = attempt / "match"
            match.mkdir(parents=True)
            (match / "manifest.json").write_text(
                json.dumps(
                    {
                        "engines": [
                            {"side": "Candidate", "matchName": "Forklift"},
                            {"side": "Baseline", "matchName": "Stockfish"},
                        ]
                    }
                )
            )
            (match / "games.pgn").write_text(
                '[Event "?"]\n[White "Forklift"]\n[Black "Stockfish"]\n'
                '[Result "1-0"]\n[Termination "time forfeit"]\n\n1. e4 1-0\n\n'
                '[Event "?"]\n[White "Stockfish"]\n[Black "Forklift"]\n'
                '[Result "1-0"]\n[Termination "adjudication"]\n\n1. e4 1-0\n'
            )
            sensitivity = dashboard.timeout_sensitivity(
                attempt,
                {"wins": 1, "draws": 0, "losses": 1},
            )
            self.assertIsNotNone(sensitivity)
            assert sensitivity is not None
            self.assertEqual(sensitivity["excluded"], 1)
            self.assertEqual(
                (sensitivity["wins"], sensitivity["draws"], sensitivity["losses"]),
                (0, 0, 1),
            )
            self.assertEqual(sensitivity["scorePercent"], 0.0)

    def test_builds_snapshot_from_complete_and_live_rungs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = {
                "configuration": {
                    "rungs": [2400, 2600],
                    "gamesPerRung": 4,
                    "timeControl": "10+0.1",
                    "threads": 1,
                    "hashMb": 256,
                    "concurrency": 2,
                    "candidate": {"commit": "abc123"},
                }
            }
            (run_dir / "ladder-manifest.json").write_text(json.dumps(manifest))

            complete = run_dir / "rung-2400" / "attempt-001" / "match"
            complete.mkdir(parents=True)
            (complete / "manifest.json").write_text(
                json.dumps({"createdAt": datetime.now(timezone.utc).isoformat()})
            )
            (complete / "result.json").write_text(
                json.dumps(
                    {
                        "completed": True,
                        "score": {
                            "candidateWins": 3,
                            "baselineWins": 0,
                            "draws": 1,
                            "candidateScore": 0.875,
                            "games": 4,
                        },
                        "elo": {"difference": 338.0, "uncertainty": 100.0},
                    }
                )
            )

            live_attempt = run_dir / "rung-2600" / "attempt-001"
            live_match = live_attempt / "match"
            live_match.mkdir(parents=True)
            (live_match / "manifest.json").write_text(
                json.dumps({"createdAt": datetime.now(timezone.utc).isoformat()})
            )
            (live_attempt / "driver.log").write_text(
                "Score of Forklift vs Stockfish: 1 - 0 - 1 [0.750] 2\n"
            )

            snapshot = dashboard.build_snapshot(run_dir)
            self.assertEqual(snapshot["completedGames"], 6)
            self.assertEqual(snapshot["totalGames"], 8)
            self.assertEqual(snapshot["currentRung"], 2600)
            self.assertEqual(snapshot["rungs"][1]["scorePercent"], 75.0)

    def test_combines_completed_history_with_current_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def completed_run(name: str, rung: int, wins: int, draws: int) -> Path:
                run_dir = root / name
                run_dir.mkdir()
                (run_dir / "ladder-manifest.json").write_text(
                    json.dumps(
                        {
                            "configuration": {
                                "rungs": [rung],
                                "gamesPerRung": 600,
                                "candidate": {"commit": "abc123"},
                            }
                        }
                    )
                )
                attempt = run_dir / f"rung-{rung}" / "attempt-001" / "match"
                attempt.mkdir(parents=True)
                (attempt / "manifest.json").write_text(
                    json.dumps({"createdAt": datetime.now(timezone.utc).isoformat()})
                )
                losses = 600 - wins - draws
                (attempt / "result.json").write_text(
                    json.dumps(
                        {
                            "completed": True,
                            "score": {
                                "candidateWins": wins,
                                "baselineWins": losses,
                                "draws": draws,
                                "candidateScore": (wins + 0.5 * draws) / 600,
                                "games": 600,
                            },
                            "elo": {"difference": 0.0, "uncertainty": 26.0},
                        }
                    )
                )
                (run_dir / "summary.json").write_text(json.dumps({"completed": True}))
                return run_dir

            history = completed_run("2500", 2500, 274, 57)
            current = completed_run("2550", 2550, 236, 48)
            snapshot = dashboard.build_snapshot(current, [history])

            self.assertEqual(snapshot["completedGames"], 1200)
            self.assertEqual(snapshot["totalRungs"], 2)
            self.assertEqual(snapshot["state"], "complete")
            self.assertEqual(snapshot["estimate"]["status"], "bracketed")
            self.assertEqual(snapshot["estimate"]["estimate"], 2502.8)

    def test_rejects_history_from_a_different_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "current"
            history = root / "history"
            current.mkdir()
            history.mkdir()
            base = {
                "rungs": [],
                "gamesPerRung": 600,
                "timeControl": "30+0.3",
                "candidate": {"commit": "abc123"},
            }
            (current / "ladder-manifest.json").write_text(
                json.dumps({"configuration": base})
            )
            mismatched = {**base, "timeControl": "10+0.1"}
            (history / "ladder-manifest.json").write_text(
                json.dumps({"configuration": mismatched})
            )

            with self.assertRaisesRegex(RuntimeError, "campaign contract"):
                dashboard.build_snapshot(current, [history])


if __name__ == "__main__":
    unittest.main()
