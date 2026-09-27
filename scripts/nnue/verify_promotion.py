#!/usr/bin/env python3
"""Verify the frozen shared-NNUE promotion evidence without changing engine state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


PROMOTION_CONTRACT = {
    "network": {
        "games": 800,
        "timeControl": "30+0.3",
        "seed": 202609140,
        "threads": 1,
        "hashMb": 128,
        "concurrency": 6,
    },
    "classical": {
        "games": 1200,
        "timeControl": "90+0.9",
        "seed": 202609141,
        "threads": 1,
        "hashMb": 128,
        "concurrency": 6,
    },
}
FAILURE_KEYS = ("crashes", "disconnects", "illegalMoves", "timeForfeits")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unable to read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def option_values(command: list[Any], option: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(command[:-1]):
        if item == option:
            values.append(str(command[index + 1]))
    return values


def is_exact_int(value: Any, expected: int | None = None) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return expected is None or value == expected


class PromotionVerifier:
    def __init__(self, series_dir: Path, expected_network_sha256: str | None = None):
        self.series_dir = series_dir.expanduser().resolve()
        self.expected_network_sha256 = expected_network_sha256
        self.checks: list[dict[str, Any]] = []
        self.stage_reports: list[dict[str, Any]] = []

    def record(
        self, code: str, status: str, message: str, **details: Any
    ) -> None:
        check: dict[str, Any] = {"code": code, "status": status, "message": message}
        if details:
            check["details"] = details
        self.checks.append(check)

    def require(self, condition: bool, code: str, message: str, **details: Any) -> bool:
        self.record(code, "pass" if condition else "fail", message, **details)
        return condition

    def pending(self, code: str, message: str, **details: Any) -> None:
        self.record(code, "pending", message, **details)

    def verify(self) -> dict[str, Any]:
        series_path = self.series_dir / "series.json"
        series = read_object(series_path)
        series_state = series.get("state")
        if series_state == "complete":
            self.record("series.complete", "pass", "promotion series is complete")
        elif series_state == "running":
            self.pending("series.complete", "promotion series is still running")
        else:
            self.record(
                "series.complete", "fail", "promotion series did not complete",
                state=series_state,
            )

        training_value = series.get("training")
        self.require(
            isinstance(training_value, str) and bool(training_value),
            "training.path",
            "series records its training experiment",
        )
        training_dir = (
            Path(training_value).expanduser().resolve()
            if isinstance(training_value, str)
            else Path()
        )
        experiment = read_object(training_dir / "experiment.json")
        self.require(
            experiment.get("state") == "complete",
            "training.complete",
            "training experiment is complete",
        )

        variants = experiment.get("variants")
        variants = variants if isinstance(variants, list) else []
        shared = [
            item
            for item in variants
            if isinstance(item, dict) and item.get("id") == "shared"
        ]
        self.require(
            len(shared) == 1,
            "network.variant",
            "training experiment contains exactly one shared network",
            count=len(shared),
        )
        if len(shared) != 1:
            return self.report(series, None, None)
        self.require(
            shared[0].get("state") == "complete",
            "network.training_complete",
            "shared-network training is complete",
        )

        network = shared[0].get("network")
        network = network if isinstance(network, dict) else {}
        network_value = network.get("path")
        network_path = (
            Path(network_value).expanduser().resolve()
            if isinstance(network_value, str)
            else Path()
        )
        recorded_network_hash = network.get("sha256")
        network_exists = network_path.is_file()
        self.require(
            network_exists,
            "network.exists",
            "shared network file exists",
            path=str(network_path),
        )
        actual_network_hash = file_sha256(network_path) if network_exists else None
        self.require(
            isinstance(recorded_network_hash, str)
            and actual_network_hash == recorded_network_hash,
            "network.checksum",
            "shared network matches its training checksum",
            expected=recorded_network_hash,
            actual=actual_network_hash,
        )
        if self.expected_network_sha256 is not None:
            self.require(
                actual_network_hash == self.expected_network_sha256.lower(),
                "network.expected_checksum",
                "shared network matches the explicitly approved checksum",
                expected=self.expected_network_sha256.lower(),
                actual=actual_network_hash,
            )

        manifest_path = network_path.with_suffix(".manifest.json")
        manifest = read_object(manifest_path)
        cpp_verification = manifest.get("cppVerification")
        cpp_verification = cpp_verification if isinstance(cpp_verification, dict) else {}
        self.require(
            cpp_verification.get("exactMatch") is True,
            "network.cpp_exact",
            "network passed exact Python/C++ export verification",
        )
        manifest_outputs = manifest.get("outputs")
        manifest_outputs = manifest_outputs if isinstance(manifest_outputs, dict) else {}
        manifest_network = manifest_outputs.get("network")
        manifest_network = manifest_network if isinstance(manifest_network, dict) else {}
        self.require(
            manifest_network.get("sha256") == actual_network_hash,
            "network.manifest_checksum",
            "training manifest records the deployed network checksum",
            expected=manifest_network.get("sha256"),
            actual=actual_network_hash,
        )

        stages_value = series.get("stages")
        stages = stages_value if isinstance(stages_value, list) else []
        by_id = {
            stage.get("id"): stage
            for stage in stages
            if isinstance(stage, dict) and isinstance(stage.get("id"), str)
        }
        self.require(
            len(stages) == len(PROMOTION_CONTRACT)
            and set(by_id) == set(PROMOTION_CONTRACT),
            "series.stages",
            "series contains exactly the frozen promotion stages",
            expected=list(PROMOTION_CONTRACT),
            actual=list(by_id),
        )

        engine_path: Path | None = None
        for stage_id, contract in PROMOTION_CONTRACT.items():
            stage = by_id.get(stage_id)
            if not isinstance(stage, dict):
                continue
            stage_engine = self.verify_stage(stage_id, stage, contract, network_path)
            if stage_engine is not None:
                if engine_path is None:
                    engine_path = stage_engine
                else:
                    self.require(
                        stage_engine == engine_path,
                        f"{stage_id}.engine_identity",
                        "both promotion stages use the same frozen executable",
                        expected=str(engine_path),
                        actual=str(stage_engine),
                    )

        recorded_engine_hash = series.get("engineSha256")
        engine_exists = engine_path is not None and engine_path.is_file()
        self.require(
            engine_exists,
            "engine.exists",
            "frozen promotion executable exists",
            path=str(engine_path) if engine_path else None,
        )
        actual_engine_hash = file_sha256(engine_path) if engine_exists and engine_path else None
        self.require(
            isinstance(recorded_engine_hash, str)
            and actual_engine_hash == recorded_engine_hash,
            "engine.checksum",
            "promotion executable matches the series checksum",
            expected=recorded_engine_hash,
            actual=actual_engine_hash,
        )
        return self.report(series, actual_network_hash, actual_engine_hash)

    def verify_stage(
        self,
        stage_id: str,
        stage: dict[str, Any],
        contract: dict[str, Any],
        network_path: Path,
    ) -> Path | None:
        command_value = stage.get("command")
        command = command_value if isinstance(command_value, list) else []

        def exact_option(option: str, expected: Any, label: str) -> bool:
            values = option_values(command, option)
            return self.require(
                values == [str(expected)],
                f"{stage_id}.contract.{label}",
                f"{stage_id} stage preserves {label}",
                expected=str(expected),
                actual=values,
            )

        exact_option("--games", contract["games"], "games")
        exact_option("--tc", contract["timeControl"], "time_control")
        exact_option("--seed", contract["seed"], "seed")
        exact_option("--threads", contract["threads"], "threads")
        exact_option("--hash", contract["hashMb"], "hash")
        exact_option("--concurrency", contract["concurrency"], "concurrency")
        exact_option("--candidate-option", "NNUE Weight=100", "candidate_nnue_weight")
        exact_option("--baseline-option", "Use NNUE=false", "classical_baseline")
        exact_option("--candidate-eval-file", network_path, "network_path")

        candidate_values = option_values(command, "--candidate-exe")
        baseline_values = option_values(command, "--baseline-exe")
        self.require(
            len(candidate_values) == 1 and baseline_values == candidate_values,
            f"{stage_id}.contract.engine_pair",
            f"{stage_id} compares evaluators in the same executable",
            candidate=candidate_values,
            baseline=baseline_values,
        )
        engine_path = (
            Path(candidate_values[0]).expanduser().resolve()
            if len(candidate_values) == 1
            else None
        )

        state = stage.get("state")
        if state == "running":
            self.pending(f"{stage_id}.complete", f"{stage_id} stage is still running")
            self.stage_reports.append({"id": stage_id, "state": "running"})
            return engine_path
        self.require(
            state == "complete",
            f"{stage_id}.complete",
            f"{stage_id} stage is complete",
            state=state,
        )
        result = stage.get("result")
        result = result if isinstance(result, dict) else {}
        expected_games = int(contract["games"])
        score = result.get("score")
        score = score if isinstance(score, dict) else {}
        complete = (
            result.get("completed") is True
            and is_exact_int(result.get("processExitCode"), 0)
            and is_exact_int(result.get("expectedGames"), expected_games)
            and is_exact_int(result.get("finishedGames"), expected_games)
            and is_exact_int(score.get("games"), expected_games)
        )
        self.require(
            complete,
            f"{stage_id}.result_complete",
            f"{stage_id} result contains all predeclared games",
            expectedGames=expected_games,
            finishedGames=result.get("finishedGames"),
            scoreGames=score.get("games"),
            processExitCode=result.get("processExitCode"),
        )
        score_counts = [
            score.get(key) for key in ("candidateWins", "baselineWins", "draws")
        ]
        valid_score = (
            all(is_exact_int(value) and value >= 0 for value in score_counts)
            and sum(score_counts) == expected_games
        )
        self.require(
            valid_score,
            f"{stage_id}.score_consistent",
            f"{stage_id} win/draw counts sum to the game total",
            candidateWins=score.get("candidateWins"),
            baselineWins=score.get("baselineWins"),
            draws=score.get("draws"),
            expectedGames=expected_games,
        )

        failures = result.get("failures")
        failures = failures if isinstance(failures, dict) else {}
        failure_shape = set(failures) == set(FAILURE_KEYS)
        failure_count = sum(
            value for key in FAILURE_KEYS
            if isinstance((value := failures.get(key)), int) and not isinstance(value, bool)
        )
        clean = failure_shape and all(
            isinstance(failures.get(key), int)
            and not isinstance(failures.get(key), bool)
            and failures.get(key) == 0
            for key in FAILURE_KEYS
        )
        self.require(
            clean,
            f"{stage_id}.technical_clean",
            f"{stage_id} has zero technical terminations",
            failures=failures,
        )

        elo = result.get("elo")
        elo = elo if isinstance(elo, dict) else {}
        difference = elo.get("difference")
        uncertainty = elo.get("uncertainty")
        finite = (
            isinstance(difference, (int, float))
            and not isinstance(difference, bool)
            and isinstance(uncertainty, (int, float))
            and not isinstance(uncertainty, bool)
            and math.isfinite(float(difference))
            and math.isfinite(float(uncertainty))
            and float(uncertainty) >= 0
        )
        lower_bound = float(difference) - float(uncertainty) if finite else None
        self.require(
            finite and lower_bound is not None and lower_bound > 0,
            f"{stage_id}.positive_lower_bound",
            f"{stage_id} Elo interval is entirely above zero",
            difference=difference,
            uncertainty=uncertainty,
            lowerBound=lower_bound,
        )

        result_path = self.series_dir / stage_id / "result.json"
        artifact_matches = result_path.is_file() and read_object(result_path) == result
        self.require(
            artifact_matches,
            f"{stage_id}.result_artifact",
            f"{stage_id} embedded result matches its preserved result artifact",
            path=str(result_path),
        )
        self.stage_reports.append(
            {
                "id": stage_id,
                "state": state,
                "games": result.get("finishedGames"),
                "score": score,
                "elo": elo,
                "technicalFailures": failure_count,
                "lowerBoundElo": lower_bound,
            }
        )
        return engine_path

    def report(
        self,
        series: dict[str, Any],
        network_sha256: str | None,
        engine_sha256: str | None,
    ) -> dict[str, Any]:
        failed = sum(check["status"] == "fail" for check in self.checks)
        pending = sum(check["status"] == "pending" for check in self.checks)
        decision = "fail" if failed else ("pending" if pending else "pass")
        return {
            "schemaVersion": 1,
            "decision": decision,
            "summary": {"failed": failed, "pending": pending, "total": len(self.checks)},
            "series": str(self.series_dir / "series.json"),
            "seriesState": series.get("state"),
            "networkSha256": network_sha256,
            "engineSha256": engine_sha256,
            "stages": self.stage_reports,
            "checks": self.checks,
        }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-network-sha256",
        help="optional independently approved lowercase SHA-256",
    )
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    expected_hash = args.expected_network_sha256
    if expected_hash is not None:
        expected_hash = expected_hash.lower()
        if len(expected_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_hash
        ):
            raise RuntimeError("expected network SHA-256 must contain 64 hexadecimal characters")
    report = PromotionVerifier(args.series_dir, expected_hash).verify()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    print(payload, end="")
    return {"pass": 0, "fail": 1, "pending": 2}[report["decision"]]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
