#!/usr/bin/env python3
"""Pure standard-library checks for the exploratory RIFT planner."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

from rift_sampling import (  # noqa: E402
    Candidate, UCI_NODE_MAX, canonical_hash, demo, fen_descriptor, finite_number, group_source_games, main,
    normalized_ranks, read_candidates, select_candidates,
)


def candidate(index: int, **overrides: object) -> Candidate:
    record: dict[str, object] = {
        "candidate_id": f"candidate-{index:03}", "source_sha256": "a" * 64,
        "game_id": index, "ply": 12, "split": "training",
        "features": [10, 100 + index], "position_key": f"position-{index}",
        "leakage_key": f"input-{index}", "screen_nodes": 5, "label_nodes": 20,
        "teacher_scores_cp": [0, index * 10], "evaluator_scores_cp": [0, index * 20],
    }
    record.update(overrides)
    return Candidate.from_mapping(record)


def ids(plan: dict[str, object]) -> list[str]:
    return [record["candidate_id"] for record in plan["selected"]]


class RiftSamplingTests(unittest.TestCase):
    def test_fen_features_match_hand_calculated_halfkp_indices(self) -> None:
        features, _, _ = fen_descriptor("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        self.assertEqual(features, (4 * 640 + 12, 40960 + 4 * 640 + 5 * 64 + 52))

    def test_fen_features_orient_black_and_encode_relative_piece_type(self) -> None:
        features, _, _ = fen_descriptor("4k3/4q3/8/8/8/8/4R3/4K3 b - - 0 1")
        self.assertEqual(features, tuple(sorted((
            4 * 640 + 3 * 64 + 12, 4 * 640 + 9 * 64 + 52,
            40960 + 4 * 640 + 8 * 64 + 52, 40960 + 4 * 640 + 4 * 64 + 12,
        ))))

    def test_fen_dedup_ignores_clocks_but_leakage_ignores_rights_and_ep(self) -> None:
        base = "r3k2r/8/8/8/8/8/8/R3K2R w"
        _, position, neural_input = fen_descriptor(base + " KQkq - 0 1")
        _, repeated, repeated_input = fen_descriptor(base + " qkQK - 12 48")
        _, altered, altered_input = fen_descriptor(base + " - e6 0 1")
        self.assertEqual(position, repeated)
        self.assertEqual(neural_input, repeated_input)
        self.assertNotEqual(position, altered)
        self.assertEqual(neural_input, altered_input)

    def test_fen_invalid_structure_rejected(self) -> None:
        for fen in (
            "8/8/8/8/8/8/8/8 w - - 0 1",  # no kings
            "4k3/8/8/8/8/8/8/4KK2 w - - 0 1",  # two white kings
            "4k3/8/8/8/8/8/8/4K3 x - - 0 1",  # turn
            "4k3/8/8/8/8/8/8/4K3 w KK - 0 1",  # rights
            "4k3/8/8/8/8/8/8/4K3 w - z3 0 1",  # EP
            "4k3/8/8/8/8/8/8/4K3 w - - -1 1",  # clock
            "4k3/8/8/8/8/8/8/4K3 w - - 0 0",  # move number
            "4k3/8/8/8/8/8/8/4K21 w - - 0 1",  # noncanonical rank
        ):
            with self.subTest(fen=fen), self.assertRaises(ValueError):
                fen_descriptor(fen)

    def test_fen_can_supply_all_descriptor_fields(self) -> None:
        record = candidate(1).record()
        for name in ("features", "position_key", "leakage_key"):
            del record[name]
        record["fen"] = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        parsed = Candidate.from_mapping(record)
        self.assertEqual(len(parsed.features), 2)
        self.assertEqual(len(parsed.position_key), 64)
        self.assertEqual(len(parsed.leakage_key), 64)

    def test_fen_and_supplied_descriptor_mismatch_rejected(self) -> None:
        record = candidate(1).record()
        record["fen"] = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        with self.assertRaisesRegex(ValueError, "features disagree"):
            Candidate.from_mapping(record)
        record["features"] = list(fen_descriptor(record["fen"])[0])
        with self.assertRaisesRegex(ValueError, "position_key disagrees"):
            Candidate.from_mapping(record)

    def test_deep_labels_and_unknown_fields_rejected(self) -> None:
        for name in ("score_cp", "result", "deep_teacher_score", "syntheticPayoff"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unexpected"):
                candidate(1, **{name: 1})

    def test_integer_costs_and_indices_are_strict_and_nonnegative(self) -> None:
        for name, value in (("screen_nodes", True), ("screen_nodes", -1),
                            ("label_nodes", 0), ("label_nodes", 2.5),
                            ("game_id", -1), ("ply", True), ("features", [False]),
                            ("features", [81920])):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                candidate(1, **{name: value})

    def test_nonfinite_bool_and_mate_scores_rejected(self) -> None:
        for scores in ([float("nan")], [float("inf")], [True], [32000], [-32000]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                candidate(1, teacher_scores_cp=scores)

    def test_huge_positive_and_negative_scores_are_validation_errors(self) -> None:
        for value in (10 ** 400, -(10 ** 400)):
            with self.subTest(value_sign=value > 0):
                with self.assertRaisesRegex(ValueError, "finite number"):
                    finite_number(value, "score")
                with self.assertRaisesRegex(ValueError, "finite number"):
                    normalized_ranks([value])
                for name in ("teacher_scores_cp", "evaluator_scores_cp"):
                    with self.assertRaisesRegex(ValueError, "finite number"):
                        candidate(1, **{name: [value]})
        with self.assertRaisesRegex(ValueError, "finite number"):
            select_candidates([candidate(1)], 100,
                              weights={"rarity": 10 ** 400, "instability": 1, "disagreement": 1})

    def test_node_cost_and_budget_values_are_bounded_to_uci_uint64(self) -> None:
        for value in (UCI_NODE_MAX + 1, 10 ** 400):
            for name in ("screen_nodes", "label_nodes"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "uint64"):
                    candidate(1, **{name: value})
            for method in ("random", "rarity", "rift"):
                with self.subTest(method=method, value=value), self.assertRaisesRegex(ValueError, "uint64"):
                    select_candidates([], value, method=method)
        invalid = replace(candidate(1), label_nodes=10 ** 400)
        for method in ("random", "rarity", "rift"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "uint64"):
                select_candidates([invalid], UCI_NODE_MAX, method=method)

    def test_uint64_max_is_valid_for_label_screen_and_teacher_budget(self) -> None:
        for method in ("random", "rarity", "rift"):
            with self.subTest(method=method):
                full_label = candidate(1, screen_nodes=0, label_nodes=UCI_NODE_MAX)
                plan = select_candidates([full_label], UCI_NODE_MAX, method=method)
                self.assertEqual(ids(plan), ["candidate-001"])
                self.assertEqual(plan["budget"]["totalNodeAllocations"], UCI_NODE_MAX)
                self.assertEqual(plan["budget"]["unallocatedNodes"], 0)
                full_screen = candidate(1, screen_nodes=UCI_NODE_MAX, label_nodes=1)
                plan = select_candidates([full_screen], UCI_NODE_MAX, method=method)
                self.assertEqual(ids(plan), [])
                self.assertEqual(plan["budget"]["screenNodeAllocations"], UCI_NODE_MAX)
                self.assertEqual(plan["budget"]["uciNodeAllocationCeiling"], UCI_NODE_MAX)

    def test_uint64_node_ceiling_does_not_bound_seed_or_metadata_ids(self) -> None:
        huge = 10 ** 400
        plan = select_candidates([candidate(1, game_id=huge, ply=huge)], 100, seed=huge)
        self.assertEqual(plan["parameters"]["seed"], huge)
        self.assertEqual(plan["selected"][0]["game_id"], huge)

    def test_input_identity_and_split_validation(self) -> None:
        for name, value in (("source_sha256", "foo"), ("source_sha256", "A" * 64),
                            ("candidate_id", ""), ("split", "holdout"),
                            ("position_key", ""), ("leakage_key", "")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                candidate(1, **{name: value})

    def test_rank_normalization_midranks_zeros_and_constant_signals(self) -> None:
        self.assertEqual(normalized_ranks([0, 0, 0]), [0, 0, 0])
        self.assertEqual(normalized_ranks([4, 4]), [1, 1])
        self.assertEqual(normalized_ranks([0, 2, 2, 4]), [0, 0.5, 0.5, 1])
        self.assertEqual(normalized_ranks([]), [])
        with self.assertRaises(ValueError):
            normalized_ranks([float("nan")])

    def test_empty_pool_and_zero_budget(self) -> None:
        plan = select_candidates([], 0)
        self.assertEqual(plan["selected"], [])
        self.assertEqual(plan["budget"]["totalNodeAllocations"], 0)

    def test_screening_is_charged_for_duplicates_holdout_and_unselected(self) -> None:
        pool = [candidate(1), candidate(2, position_key="position-1"),
                candidate(3, split="validation")]
        plan = select_candidates(pool, 35)
        self.assertEqual(plan["budget"]["screenNodeAllocations"], 15)
        self.assertEqual(plan["budget"]["labelNodeAllocations"], 20)
        self.assertEqual(plan["budget"]["totalNodeAllocations"], 35)
        self.assertEqual(plan["excludedCounts"], {"duplicate_position": 1, "held_out_split": 1})

    def test_overspent_screen_budget_rejected_even_without_eligible_positions(self) -> None:
        with self.assertRaisesRegex(ValueError, "screening"):
            select_candidates([candidate(1, split="test")], 4)

    def test_candidates_can_be_skipped_when_label_cost_does_not_fit(self) -> None:
        pool = [candidate(1, label_nodes=100), candidate(2, label_nodes=10)]
        plan = select_candidates(pool, 25)
        self.assertEqual(ids(plan), ["candidate-002"])
        self.assertEqual(plan["budget"]["unallocatedNodes"], 5)
        self.assertEqual(plan["excluded"]["candidate-001"], "remaining_budget")

    def test_training_game_overlapping_any_holdout_split_is_excluded(self) -> None:
        pool = [candidate(1), candidate(2, game_id=1, split="validation"), candidate(3)]
        plan = select_candidates(pool, 100)
        self.assertEqual(ids(plan), ["candidate-003"])
        self.assertEqual(plan["excluded"]["candidate-001"], "held_out_game")

    def test_holdout_nnue_input_transpositions_excluded_across_games(self) -> None:
        pool = [candidate(1), candidate(2, split="test", leakage_key="input-1"), candidate(3)]
        plan = select_candidates(pool, 100)
        self.assertEqual(ids(plan), ["candidate-003"])
        self.assertEqual(plan["excluded"]["candidate-001"], "held_out_nnue_input")

    def test_explicit_holdout_protects_keys_and_not_consulted_for_signals_or_cost(self) -> None:
        pool = [candidate(1), candidate(2), candidate(3)]
        holdout = [candidate(4, split="validation", leakage_key="input-1", screen_nodes=900)]
        plan = select_candidates(pool, 55, holdout=holdout)
        altered = [candidate(4, split="validation", leakage_key="input-1", screen_nodes=800,
                             teacher_scores_cp=[0, 30000], evaluator_scores_cp=[-500, 20000])]
        repeated = select_candidates(pool, 55, holdout=altered)
        self.assertEqual(plan, repeated)
        self.assertEqual(plan["budget"]["screenNodeAllocations"], 15)
        self.assertNotIn("candidate-001", ids(plan))
        with self.assertRaisesRegex(ValueError, "cannot be training"):
            select_candidates(pool, 55, holdout=[candidate(5)])

    def test_holdout_does_not_influence_training_frequency_or_normalization(self) -> None:
        pool = [candidate(1), candidate(2), candidate(3)]
        base = select_candidates(pool, 100)
        held_out = candidate(4, split="validation", teacher_scores_cp=[-10000, 10000],
                             features=[10] * 30 + [102], evaluator_scores_cp=[-15000, 15000])
        enlarged = select_candidates(pool + [held_out], 105)
        self.assertEqual(base["selected"], enlarged["selected"])
        self.assertEqual(base["lineage"]["featureFrequencySha256"],
                         enlarged["lineage"]["featureFrequencySha256"])

    def test_game_identity_is_namespaced_by_source_hash(self) -> None:
        pool = [candidate(1), candidate(2, source_sha256="b" * 64, game_id=1, split="test")]
        self.assertEqual(ids(select_candidates(pool, 100)), ["candidate-001"])

    def test_game_fingerprint_blocks_same_game_copied_across_sources(self) -> None:
        pool = [candidate(1, game_fingerprint="c" * 64),
                candidate(2, source_sha256="b" * 64, game_id=999, split="test",
                          game_fingerprint="c" * 64), candidate(3)]
        plan = select_candidates(pool, 100)
        self.assertEqual(ids(plan), ["candidate-003"])
        self.assertEqual(plan["excluded"]["candidate-001"], "held_out_game")
        with self.assertRaisesRegex(ValueError, "game_fingerprint"):
            candidate(4, game_fingerprint="invalid")

    def test_source_game_holdout_protection_survives_mixed_fingerprint_presence(self) -> None:
        for training_hash, holdout_hash in (("c" * 64, None), (None, "c" * 64)):
            pool = [candidate(1, game_fingerprint=training_hash),
                    candidate(2, game_id=1, split="validation", game_fingerprint=holdout_hash),
                    candidate(3)]
            with self.subTest(training_hash=training_hash, holdout_hash=holdout_hash):
                plan = select_candidates(pool, 100)
                self.assertEqual(ids(plan), ["candidate-003"])
                self.assertEqual(plan["excluded"]["candidate-001"], "held_out_game")

    def test_source_game_cap_survives_mixed_fingerprint_presence(self) -> None:
        pool = [candidate(1, game_fingerprint="c" * 64), candidate(2, game_id=1)]
        plan = select_candidates(pool, 100)
        self.assertEqual(len(ids(plan)), 1)
        self.assertEqual(plan["excludedCounts"], {"game_cap": 1})
        self.assertEqual(len(ids(select_candidates(pool, 100, max_per_game=0))), 2)

    def test_copied_game_links_cover_missing_metadata_on_both_ends(self) -> None:
        pool = [candidate(1),
                candidate(2, game_id=1, game_fingerprint="c" * 64),
                candidate(3, source_sha256="b" * 64, game_id=3, game_fingerprint="c" * 64)]
        holdout = [candidate(4, source_sha256="b" * 64, game_id=3, split="test")]
        plan = select_candidates(pool, 100, holdout=holdout)
        self.assertEqual(ids(plan), [])
        self.assertEqual(plan["excludedCounts"], {"held_out_game": 3})
        self.assertEqual(plan, select_candidates(list(reversed(pool)), 100,
                                                holdout=list(reversed(holdout))))

    def test_copied_game_cap_uses_union_of_source_and_fingerprint_links(self) -> None:
        pool = [candidate(1), candidate(2, game_id=1, game_fingerprint="c" * 64),
                candidate(3, source_sha256="b" * 64, game_id=3, game_fingerprint="c" * 64),
                candidate(4, source_sha256="b" * 64, game_id=3)]
        groups = group_source_games(pool)
        self.assertEqual(len(set(groups.values())), 1)
        self.assertEqual(groups, group_source_games(list(reversed(pool))))
        plan = select_candidates(pool, 200)
        self.assertEqual(len(ids(plan)), 1)
        self.assertEqual(plan["excludedCounts"], {"game_cap": 3})
        self.assertEqual(plan["selected"][0]["gameGroup"],
                         {"source_sha256": "a" * 64, "game_id": 1})

    def test_conflicting_fingerprints_for_one_source_game_rejected(self) -> None:
        pool = [candidate(1, game_fingerprint="c" * 64),
                candidate(2, game_id=1, game_fingerprint="d" * 64)]
        with self.assertRaisesRegex(ValueError, "conflicting game_fingerprint"):
            select_candidates(pool, 100)
        with self.assertRaisesRegex(ValueError, "conflicting game_fingerprint"):
            select_candidates(pool[:1], 100,
                              holdout=[candidate(2, game_id=1, split="validation",
                                                 game_fingerprint="d" * 64)])

    def test_direct_dataclass_construction_cannot_bypass_cost_validation(self) -> None:
        invalid = replace(candidate(1), label_nodes=-100)
        with self.assertRaisesRegex(ValueError, "label_nodes"):
            select_candidates([invalid], 100)

    def test_duplicate_ids_fail_and_position_duplicates_have_stable_representative(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            select_candidates([candidate(1), candidate(1)], 100)
        pool = [candidate(2, position_key="position-1"), candidate(1)]
        self.assertEqual(ids(select_candidates(pool, 100)), ["candidate-001"])

    def test_game_cap_and_opt_out(self) -> None:
        pool = [candidate(1), candidate(2, game_id=1)]
        limited = select_candidates(pool, 100)
        self.assertEqual(len(ids(limited)), 1)
        self.assertEqual(limited["excludedCounts"], {"game_cap": 1})
        unlimited = select_candidates(pool, 100, max_per_game=0)
        self.assertEqual(len(ids(unlimited)), 2)

    def test_density_prefers_cost_effective_equal_utility_candidates(self) -> None:
        pool = [candidate(1, teacher_scores_cp=[], evaluator_scores_cp=[], label_nodes=40),
                candidate(2, teacher_scores_cp=[], evaluator_scores_cp=[], label_nodes=20)]
        plan = select_candidates(pool, 50, method="rarity")
        self.assertEqual(ids(plan), ["candidate-002"])
        self.assertEqual(plan["budget"]["totalNodeAllocations"], 30)

    def test_seeds_change_ties_but_input_order_changes_nothing(self) -> None:
        pool = [candidate(index, teacher_scores_cp=[], evaluator_scores_cp=[])
                for index in range(12)]
        for method in ("random", "rarity", "rift"):
            with self.subTest(method=method):
                plan = select_candidates(pool, 160, method=method, seed=1)
                self.assertEqual(plan, select_candidates(list(reversed(pool)), 160, method=method, seed=1))
                self.assertNotEqual(ids(plan), ids(select_candidates(pool, 160, method=method, seed=2)))

    def test_signal_weight_ablations_change_the_selection(self) -> None:
        pool = [candidate(1, teacher_scores_cp=[0, 500], evaluator_scores_cp=[0, 0]),
                candidate(2, teacher_scores_cp=[0, 0], evaluator_scores_cp=[0, 500])]
        instability = select_candidates(pool, 30, weights={"rarity": 0, "instability": 1, "disagreement": 0})
        disagreement = select_candidates(pool, 30, weights={"rarity": 0, "instability": 0, "disagreement": 1})
        self.assertEqual(ids(instability), ["candidate-001"])
        self.assertEqual(ids(disagreement), ["candidate-002"])

    def test_bad_weights_methods_and_budget_flags_rejected(self) -> None:
        for parameters in ({"teacher_budget": -1}, {"teacher_budget": True}, {"seed": -1},
                           {"method": "unknown"}, {"max_per_game": -1},
                           {"weights": {"rarity": 1}},
                           {"weights": {"rarity": 0, "instability": 0, "disagreement": 0}},
                           {"weights": {"rarity": -1, "instability": 1, "disagreement": 1}},
                           {"weights": {"rarity": 1e308, "instability": 1e308, "disagreement": 1e308}}):
            options = {"teacher_budget": 100, **parameters}
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                select_candidates([candidate(1)], **options)

    def test_selected_lineage_canonical_pool_and_candidate_hash(self) -> None:
        pool = [candidate(1), candidate(2)]
        plan = select_candidates(pool, 100)
        self.assertEqual(plan["lineage"]["candidatePoolSha256"],
                         canonical_hash([item.record() for item in pool]))
        for selected in plan["selected"]:
            original = next(item for item in pool if item.candidate_id == selected["candidate_id"])
            self.assertEqual(selected["candidateSha256"], canonical_hash(original.record()))

    def test_randomized_budget_and_no_leakage_invariants(self) -> None:
        generator = random.Random(20261013)
        for trial in range(50):
            pool = [candidate(index, screen_nodes=generator.randrange(10),
                              label_nodes=generator.randrange(1, 80),
                              game_id=index // 3,
                              split="validation" if index % 11 == 0 else "training")
                    for index in range(30)]
            screen_cost = sum(item.screen_nodes for item in pool)
            budget = screen_cost + generator.randrange(300)
            held_out_games = {item.game_key for item in pool if item.split != "training"}
            for method in ("random", "rarity", "rift"):
                with self.subTest(trial=trial, method=method):
                    plan = select_candidates(pool, budget, method=method, seed=trial)
                    ledger = plan["budget"]
                    selected = [Candidate.from_mapping({name: value for name, value in record.items()
                                                        if name in candidate(1).record()})
                                for record in plan["selected"]]
                    self.assertLessEqual(ledger["totalNodeAllocations"], budget)
                    self.assertEqual(ledger["totalNodeAllocations"] + ledger["unallocatedNodes"], budget)
                    self.assertEqual(ledger["labelNodeAllocations"], sum(item.label_nodes for item in selected))
                    self.assertEqual(ledger["screenNodeAllocations"], screen_cost)
                    self.assertEqual(len({item.game_key for item in selected}), len(selected))
                    self.assertFalse(held_out_games & {item.game_key for item in selected})

    def test_synthetic_demo_holds_identical_total_cost_and_does_not_claim_chess_gain(self) -> None:
        report = demo()
        self.assertIn("not-chess-evidence", report["kind"])
        self.assertEqual({item["budget"]["totalNodeAllocations"] for item in report["reports"].values()}, {10400})
        self.assertEqual({len(item["selectedIds"]) for item in report["reports"].values()}, {8})
        self.assertIn("hand-aligned", report["warning"])
        self.assertEqual(report, demo())

    def test_cli_jsonl_lineage_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pool.jsonl"
            output = Path(directory) / "plan.json"
            source.write_text(json.dumps(candidate(1).record()) + "\n", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--candidates", str(source), "--teacher-budget", "30", "--output", str(output)]), 0)
                self.assertEqual(main(["--demo", "--output", str(output)]), 2)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["pool"]["selectedPositions"], 1)
            self.assertEqual(len(report["files"]["candidatesSha256"]), 64)
            self.assertEqual(len(report["files"]["selectorSha256"]), 64)

    def test_cli_rejects_bad_records_and_unintentional_demo_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pool.jsonl"
            source.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ":1: candidate must"):
                read_candidates(source)
            source.write_text("NaN\n", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--candidates", str(source), "--teacher-budget", "30"]), 2)
                self.assertEqual(main(["--demo", "--teacher-budget", "30"]), 2)
                self.assertEqual(main([]), 2)

    def test_cli_huge_numeric_inputs_fail_cleanly_and_uint64_boundary_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pool.jsonl"
            cases = [{"teacher_scores_cp": [10 ** 400]},
                     {"teacher_scores_cp": [-(10 ** 400)]},
                     {"evaluator_scores_cp": [10 ** 400]},
                     {"evaluator_scores_cp": [-(10 ** 400)]},
                     {"label_nodes": 10 ** 400}, {"screen_nodes": 10 ** 400},
                     {"label_nodes": UCI_NODE_MAX + 1}]
            for overrides in cases:
                record = {**candidate(1).record(), **overrides}
                source.write_text(json.dumps(record) + "\n", encoding="utf-8")
                errors = io.StringIO()
                with self.subTest(overrides=overrides), redirect_stderr(errors):
                    self.assertEqual(main(["--candidates", str(source), "--teacher-budget", str(UCI_NODE_MAX)]), 2)
                self.assertIn("RIFT planning failed", errors.getvalue())
                self.assertNotIn("OverflowError", errors.getvalue())
            source.write_text(json.dumps(candidate(1, screen_nodes=0, label_nodes=UCI_NODE_MAX).record()) + "\n",
                              encoding="utf-8")
            for value in (UCI_NODE_MAX + 1, 10 ** 400):
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["--candidates", str(source), "--teacher-budget", str(value)]), 2)
            stream = io.StringIO()
            with redirect_stdout(stream):
                self.assertEqual(main(["--candidates", str(source), "--teacher-budget", str(UCI_NODE_MAX)]), 0)
            report = json.loads(stream.getvalue())
            self.assertEqual(report["budget"]["totalNodeAllocations"], UCI_NODE_MAX)

    def test_cli_demo_stdout_is_parseable_json(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(main(["--demo"]), 0)
        self.assertEqual(json.loads(stream.getvalue()), demo())


if __name__ == "__main__":
    unittest.main()
