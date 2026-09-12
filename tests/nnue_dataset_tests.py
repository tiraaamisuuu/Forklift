#!/usr/bin/env python3

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import chess


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

from nnue_dataset import (  # noqa: E402
    BinaryShardWriter,
    SHARD_HEADER,
    SHARD_RECORD,
    active_features,
    active_features_from_packed,
    decode_record,
    encode_record,
    iter_compact_records,
    pack_board,
    read_shard_header,
    sha256_file,
    write_json_atomic,
)
from generate_dataset import (  # noqa: E402
    TeacherComparison,
    clear_teacher_hash,
    game_is_validation,
    game_shard,
    open_pgn_text,
    position_is_sampled,
    result_for_side_to_move,
)
from dataset_diagnostics import analyze_records  # noqa: E402
from generate_shards import (  # noqa: E402
    aggregate_teacher_comparisons,
    merge_shards,
    positions_per_shard_for_target,
    record_key,
)
from merge_datasets import (  # noqa: E402
    load_bundle,
    portable_name,
    validate_contracts,
)


class NnueDatasetTests(unittest.TestCase):
    def test_atomic_json_retries_transient_windows_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "progress.json"
            original = Path.replace
            attempts = []
            def replace(source, target):
                attempts.append(target)
                if len(attempts) < 3:
                    raise PermissionError("reader lock")
                return original(source, target)
            with mock.patch.object(Path, "replace", replace), mock.patch("nnue_dataset.time.sleep"):
                write_json_atomic(destination, {"state": "training"})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(destination.read_text())["state"], "training")

    def test_atomic_json_does_not_hide_persistent_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(Path, "replace", side_effect=PermissionError("locked")) as replace, mock.patch("nnue_dataset.time.sleep"):
                with self.assertRaises(PermissionError):
                    write_json_atomic(Path(directory) / "progress.json", {})
            self.assertEqual(replace.call_count, 6)

    def test_packed_features_match_board_features(self) -> None:
        board = chess.Board()
        random_generator = random.Random(20260727)
        for _ in range(80):
            packed = pack_board(board)
            for perspective in (chess.WHITE, chess.BLACK):
                self.assertEqual(
                    active_features(board, perspective),
                    active_features_from_packed(packed, perspective),
                )
            if board.is_game_over():
                board.reset()
            board.push(random_generator.choice(list(board.legal_moves)))

    def test_record_round_trip(self) -> None:
        board = chess.Board("r3k2r/ppp2ppp/2n5/3qp3/8/2N2N2/PPP2PPP/R2Q1RK1 b kq - 7 14")
        payload = encode_record(board, -432, -1.0, game_id=1234, ply=27)
        self.assertEqual(len(payload), SHARD_RECORD.size)
        record = decode_record(payload)
        self.assertEqual(record.score_cp, -432)
        self.assertEqual(record.result, -1)
        self.assertEqual(record.turn, chess.BLACK)
        self.assertEqual(record.game_id, 1234)
        self.assertEqual(record.ply, 27)
        self.assertEqual(
            active_features(board, chess.WHITE),
            active_features_from_packed(record.board_bytes, chess.WHITE),
        )

    def test_writer_finalizes_count_and_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.nnuebin"
            boards = [chess.Board(), chess.Board()]
            boards[1].push_uci("e2e4")
            with BinaryShardWriter(destination) as writer:
                writer.write(boards[0], 12, 0.0, game_id=1, ply=8)
                writer.write(boards[1], -34, 1.0, game_id=2, ply=9)

            with destination.open("rb") as source:
                header = read_shard_header(source)
                self.assertEqual(header.count, 2)
                self.assertEqual(destination.stat().st_size,
                                 SHARD_HEADER.size + 2 * SHARD_RECORD.size)

            records = list(iter_compact_records([destination]))
            self.assertEqual([record.score_cp for record in records], [12, -34])
            self.assertEqual([record.game_id for record in records], [1, 2])

    def test_game_split_is_deterministic_and_whole_game(self) -> None:
        assignments = [game_is_validation("ab" * 32, game, 17, 0.2)
                       for game in range(1, 1001)]
        repeated = [game_is_validation("ab" * 32, game, 17, 0.2)
                    for game in range(1, 1001)]
        self.assertEqual(assignments, repeated)
        self.assertGreater(sum(assignments), 150)
        self.assertLess(sum(assignments), 250)
        self.assertNotEqual(
            assignments,
            [game_is_validation("ab" * 32, game, 18, 0.2)
             for game in range(1, 1001)],
        )

    def test_result_is_oriented_to_side_to_move(self) -> None:
        self.assertEqual(result_for_side_to_move("1-0", chess.WHITE), 1.0)
        self.assertEqual(result_for_side_to_move("1-0", chess.BLACK), -1.0)
        self.assertEqual(result_for_side_to_move("1/2-1/2", chess.BLACK), 0.0)

    def test_teacher_budget_comparison_metrics(self) -> None:
        comparison = TeacherComparison()
        comparison.add(100, 130)
        comparison.add(-50, -90)
        comparison.add(10, -20)
        report = comparison.report(5_000, 20_000)
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["primaryNodes"], 5_000)
        self.assertEqual(report["comparisonNodes"], 20_000)
        self.assertAlmostEqual(report["maeCp"], 100 / 3)
        self.assertAlmostEqual(report["signAgreement"], 2 / 3)

    def test_teacher_comparison_clears_hash_between_budgets(self) -> None:
        class FakeEngine:
            options = {"Clear Hash": object()}

            def __init__(self) -> None:
                self.configurations = []

            def configure(self, options) -> None:
                self.configurations.append(options)

        engine = FakeEngine()
        clear_teacher_hash(engine)  # type: ignore[arg-type]
        clear_teacher_hash(engine)  # type: ignore[arg-type]
        self.assertEqual(
            engine.configurations,
            [{"Clear Hash": None}, {"Clear Hash": None}],
        )

    def test_teacher_comparison_rejects_engine_without_clear_hash(self) -> None:
        class FakeEngine:
            options = {}

        with self.assertRaisesRegex(RuntimeError, "Clear Hash"):
            clear_teacher_hash(FakeEngine())  # type: ignore[arg-type]

    def test_teacher_comparisons_aggregate_across_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, report in enumerate((
                {"samples": 2, "primaryNodes": 5_000, "comparisonNodes": 20_000,
                 "meanDifferenceCp": 10.0, "maeCp": 20.0, "rmseCp": 25.0,
                 "maxAbsDifferenceCp": 30, "signAgreement": 0.5},
                {"samples": 3, "primaryNodes": 5_000, "comparisonNodes": 20_000,
                 "meanDifferenceCp": -5.0, "maeCp": 10.0, "rmseCp": 15.0,
                 "maxAbsDifferenceCp": 40, "signAgreement": 1.0},
            )):
                path = Path(directory) / f"part-{index}.json"
                path.write_text(json.dumps({"teacherComparison": report}), encoding="utf-8")
                paths.append(path)
            aggregate = aggregate_teacher_comparisons(paths)
            assert aggregate is not None
            self.assertEqual(aggregate["samples"], 5)
            self.assertAlmostEqual(aggregate["maeCp"], 14.0)
            self.assertEqual(aggregate["maxAbsDifferenceCp"], 40)
            self.assertAlmostEqual(aggregate["signAgreement"], 0.8)

    def test_sampling_and_game_shards_are_stateless(self) -> None:
        checksum = "cd" * 32
        selected = [position_is_sampled(checksum, 17, ply, 9, 0.25)
                    for ply in range(1, 1001)]
        self.assertEqual(
            selected,
            [position_is_sampled(checksum, 17, ply, 9, 0.25)
             for ply in range(1, 1001)],
        )
        self.assertGreater(sum(selected), 200)
        self.assertLess(sum(selected), 300)
        assignments = [game_shard(checksum, game, 9, 4) for game in range(1, 1001)]
        self.assertTrue(all(0 <= shard < 4 for shard in assignments))
        for shard in range(4):
            self.assertGreater(assignments.count(shard), 200)
            self.assertLess(assignments.count(shard), 300)

    def test_global_merge_deduplicates_parts_and_validation_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = chess.Board()
            e4 = chess.Board()
            e4.push_uci("e2e4")
            d4 = chess.Board()
            d4.push_uci("d2d4")
            c4 = chess.Board()
            c4.push_uci("c2c4")
            first = root / "first.nnuebin"
            second = root / "second.nnuebin"
            validation = root / "validation-part.nnuebin"
            with BinaryShardWriter(first) as writer:
                writer.write(start, 12, 0.0, 1, 8)
                writer.write(e4, 20, 0.0, 1, 9)
            with BinaryShardWriter(second) as writer:
                writer.write(start, 99, 0.0, 2, 8)
                writer.write(d4, 30, 0.0, 2, 9)
            with BinaryShardWriter(validation) as writer:
                writer.write(e4, 25, 0.0, 3, 9)
                writer.write(c4, 40, 0.0, 3, 9)

            seen: set[int] = set()
            training_stats = merge_shards(
                [first, second], root / "train.nnuebin", seen,
            )
            validation_stats = merge_shards(
                [validation], root / "validation.nnuebin", seen,
            )
            self.assertEqual(training_stats, {"read": 4, "written": 3, "duplicatesSkipped": 1})
            self.assertEqual(validation_stats, {"read": 2, "written": 1, "duplicatesSkipped": 1})
            self.assertEqual(len(seen), 4)
            start_record = next(iter(iter_compact_records([first])))
            self.assertNotEqual(
                record_key(start_record.board_bytes, start_record.turn),
                record_key(start_record.board_bytes, not start_record.turn),
            )

    def test_target_position_planning_and_merge_cap(self) -> None:
        self.assertEqual(
            positions_per_shard_for_target(5_000_000, 8, 0.1, 1.10),
            763_889,
        )
        self.assertEqual(positions_per_shard_for_target(0, 8, 0.1, 1.10), 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = root / "part.nnuebin"
            boards = []
            for move in ("e2e4", "d2d4", "c2c4"):
                board = chess.Board()
                board.push_uci(move)
                boards.append(board)
            with BinaryShardWriter(part) as writer:
                for game_id, board in enumerate(boards, start=1):
                    writer.write(board, game_id, 0.0, game_id, 1)

            destination = root / "target.nnuebin"
            stats = merge_shards([part], destination, set(), maximum_records=2)
            self.assertEqual(stats, {"read": 2, "written": 2, "duplicatesSkipped": 0})
            self.assertEqual(len(list(iter_compact_records([destination]))), 2)

    def test_dataset_diagnostics_report_coverage_and_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.nnuebin"
            start = chess.Board()
            endgame = chess.Board("8/8/8/8/8/4k3/4p3/4K3 w - - 0 1")
            with BinaryShardWriter(path) as writer:
                writer.write(start, 25, 0.0, 1, 8)
                writer.write(endgame, -250, -1.0, 2, 60)

            report = analyze_records([path])
            self.assertEqual(report["positions"], 2)
            self.assertGreater(report["featureCoverage"]["seen"], 0)
            self.assertEqual(report["phaseCounts"], {"endgame": 1, "opening": 1})
            self.assertEqual(report["teacherEvalMagnitudeCounts"], {"0-99": 1, "100-299": 1})
            self.assertEqual(report["resultCounts"], {"draw": 1, "loss": 1})

    def test_portable_dataset_bundle_and_multimachine_contract(self) -> None:
        self.assertEqual(portable_name(r"D:\\data\\train.nnuebin"), "train.nnuebin")
        self.assertEqual(portable_name("/data/train.nnuebin"), "train.nnuebin")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles = []
            for index, binary_sha in enumerate(("windows", "linux")):
                bundle_dir = root / str(index)
                bundle_dir.mkdir()
                training = bundle_dir / "train.nnuebin"
                validation = bundle_dir / "validation.nnuebin"
                with BinaryShardWriter(training) as writer:
                    writer.write(chess.Board(), index, 0.0, index + 1, 8)
                board = chess.Board()
                board.push_uci("e2e4")
                with BinaryShardWriter(validation) as writer:
                    writer.write(board, index, 0.0, index + 1, 9)
                manifest = {
                    "schemaVersion": 1,
                    "datasetFormat": "HalfKP-v1-sharded",
                    "teacher": {"nodes": 20_000, "comparisonNodes": None,
                                "sha256": binary_sha},
                    "sampling": {"seed": 7, "validationFractionByGame": 0.1},
                    "provenance": {"name": f"source-{index}"},
                    "outputs": [
                        {"split": "training", "path": r"D:\\copied\\train.nnuebin",
                         "sha256": sha256_file(training)},
                        {"split": "validation", "path": r"D:\\copied\\validation.nnuebin",
                         "sha256": sha256_file(validation)},
                    ],
                }
                path = bundle_dir / "dataset.manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                bundles.append(load_bundle(path))

            contract = validate_contracts(bundles)
            self.assertEqual(contract["nodes"], 20_000)
            self.assertEqual(contract["binarySha256"], ["linux", "windows"])

            bundles[1]["splitContract"] = {
                "seed": 8, "validationFractionByGame": 0.1,
            }
            with self.assertRaisesRegex(ValueError, "split seed"):
                validate_contracts(bundles)

    def test_zstd_pgn_stream(self) -> None:
        import zstandard

        pgn = b"[Event \"fixture\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pgn.zst"
            path.write_bytes(zstandard.ZstdCompressor().compress(pgn))
            with open_pgn_text(path) as source:
                game = chess.pgn.read_game(source)
            self.assertIsNotNone(game)
            assert game is not None
            self.assertEqual(game.headers["Result"], "1-0")
            self.assertEqual([move.uci() for move in game.mainline_moves()], ["e2e4", "e7e5"])


if __name__ == "__main__":
    unittest.main()
