#!/usr/bin/env python3
"""Deterministic RIFT label-budget planner; never launches a teacher or training.

Input JSONL contains cheap candidate signals, not the future deep labels. Costs
are UCI node limits (planning units), not measured wall time or consumed nodes.
Use --demo for a synthetic, deliberately signal-aligned plumbing ablation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
FEATURE_COUNT_PER_PERSPECTIVE = 40_960
UCI_NODE_MAX = (1 << 64) - 1
METHODS = ("random", "rarity", "rift")
SIGNALS = ("rarity", "instability", "disagreement")


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def node_allocation(value: object, name: str, minimum: int = 0) -> int:
    result = integer(value, name, minimum)
    if result > UCI_NODE_MAX:
        raise ValueError(f"{name} exceeds the UCI uint64 node ceiling (2**64 - 1)")
    return result


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number representable as a float") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def fen_descriptor(fen: str) -> tuple[tuple[int, ...], str, str]:
    """Derive HalfKP features and dedup/leakage keys without python-chess.

    This checks FEN structure, not chess legality. White feature IDs occupy
    [0, 40960); black IDs occupy [40960, 81920). Dedup uses placement, turn,
    rights and EP; leakage protection uses placement and turn, exactly the
    current NNUE inputs (draw clocks/rights/EP are not network features).
    """
    if not isinstance(fen, str):
        raise ValueError("fen must be a string")
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in ("w", "b"):
        raise ValueError("fen must contain six fields and a valid side to move")
    if fields[2] != "-" and (not fields[2] or len(set(fields[2])) != len(fields[2])
                             or any(symbol not in "KQkq" for symbol in fields[2])):
        raise ValueError("fen has invalid castling rights")
    if fields[3] != "-" and (len(fields[3]) != 2 or fields[3][0] not in "abcdefgh"
                             or fields[3][1] not in "36"):
        raise ValueError("fen has invalid en-passant square")
    if not fields[4].isdigit() or not fields[5].isdigit() or int(fields[5]) < 1:
        raise ValueError("fen has invalid move counters")
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError("fen must have eight ranks")
    pieces: dict[int, str] = {}
    for row, rank in enumerate(ranks):
        file_index = 0
        previous_digit = False
        for symbol in rank:
            if symbol in "12345678":
                if previous_digit:
                    raise ValueError("fen has consecutive empty-square digits")
                file_index += int(symbol)
                previous_digit = True
            elif symbol in "PNBRQKpnbrqk" and file_index < 8:
                pieces[(7 - row) * 8 + file_index] = symbol
                file_index += 1
                previous_digit = False
            else:
                raise ValueError("fen contains an invalid rank")
        if file_index != 8:
            raise ValueError("fen rank does not contain eight squares")
    if list(pieces.values()).count("K") != 1 or list(pieces.values()).count("k") != 1:
        raise ValueError("fen must contain exactly one king per side")
    features: list[int] = []
    for perspective, king_symbol in enumerate(("K", "k")):
        king = next(square for square, symbol in pieces.items() if symbol == king_symbol)
        oriented_king = king if perspective == 0 else king ^ 56
        for square, symbol in pieces.items():
            if symbol.lower() == "k":
                continue
            relative_color = int(symbol.isupper() != (perspective == 0))
            bucket = relative_color * 5 + "pnbrq".index(symbol.lower())
            oriented_square = square if perspective == 0 else square ^ 56
            features.append(perspective * FEATURE_COUNT_PER_PERSPECTIVE
                            + oriented_king * 640 + bucket * 64 + oriented_square)
    canonical_rights = "".join(symbol for symbol in "KQkq" if symbol in fields[2]) or "-"
    position_key = canonical_hash([fields[0], fields[1], canonical_rights, fields[3]])
    leakage_key = canonical_hash(fields[:2])
    return tuple(sorted(features)), position_key, leakage_key


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source_sha256: str
    game_id: int
    ply: int
    split: str
    features: tuple[int, ...]
    position_key: str
    leakage_key: str
    screen_nodes: int
    label_nodes: int
    teacher_scores_cp: tuple[float, ...] = ()
    evaluator_scores_cp: tuple[float, ...] = ()
    fen: str | None = None
    game_fingerprint: str | None = None

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "Candidate":
        # Future deep targets are deliberately forbidden in a selector input.
        allowed = {"candidate_id", "source_sha256", "game_id", "ply", "split", "features",
                   "position_key", "leakage_key", "screen_nodes", "label_nodes",
                   "teacher_scores_cp", "evaluator_scores_cp", "fen", "game_fingerprint"}
        if unknown := set(record) - allowed:
            raise ValueError(f"unexpected candidate fields: {', '.join(sorted(unknown))}")
        source = record.get("source_sha256")
        if not isinstance(source, str) or len(source) != 64 or any(
                character not in "0123456789abcdef" for character in source):
            raise ValueError("source_sha256 must be a lowercase SHA256 hex digest")
        fingerprint = record.get("game_fingerprint")
        if fingerprint is not None and (not isinstance(fingerprint, str) or len(fingerprint) != 64
                                       or any(character not in "0123456789abcdef" for character in fingerprint)):
            raise ValueError("game_fingerprint must be a lowercase SHA256 hex digest")
        game_id = integer(record.get("game_id"), "game_id")
        ply = integer(record.get("ply"), "ply")
        candidate_id = record.get("candidate_id", f"{source}:{game_id}:{ply}")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a nonempty string")
        split = record.get("split")
        if split not in ("training", "validation", "test"):
            raise ValueError("split must be training, validation or test")
        fen = record.get("fen")
        descriptor = fen_descriptor(fen) if fen is not None else None  # type: ignore[arg-type]
        supplied_features = record.get("features")
        if supplied_features is not None:
            if not isinstance(supplied_features, list):
                raise ValueError("features must be a list")
            features = tuple(sorted(set(integer(value, "feature") for value in supplied_features)))
            if any(value >= 2 * FEATURE_COUNT_PER_PERSPECTIVE for value in features):
                raise ValueError("feature is outside the two-perspective HalfKP range")
            if descriptor is not None and features != descriptor[0]:
                raise ValueError("features disagree with the FEN descriptor")
        elif descriptor is not None:
            features = descriptor[0]
        else:
            raise ValueError("candidate needs fen or features")
        keys: list[str] = []
        for name, offset in (("position_key", 1), ("leakage_key", 2)):
            key = record.get(name, descriptor[offset] if descriptor else None)
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} must be a nonempty string when fen is absent")
            if descriptor is not None and key != descriptor[offset]:
                raise ValueError(f"{name} disagrees with the FEN descriptor")
            keys.append(key)
        score_lists: list[tuple[float, ...]] = []
        for name in ("teacher_scores_cp", "evaluator_scores_cp"):
            scores = record.get(name, [])
            if not isinstance(scores, list):
                raise ValueError(f"{name} must be a list")
            converted = tuple(finite_number(value, name) for value in scores)
            if any(abs(value) >= 32_000 for value in converted):
                raise ValueError(f"{name} must contain CP scores, not saturated mate sentinels")
            score_lists.append(converted)
        return cls(candidate_id, source, game_id, ply, str(split), features, *keys,
                   node_allocation(record.get("screen_nodes"), "screen_nodes"),
                   node_allocation(record.get("label_nodes"), "label_nodes", 1),
                   *score_lists, fen, fingerprint)  # type: ignore[arg-type]

    @property
    def game_key(self) -> tuple[str, int]:
        # Source-local identity is always valid, even if optional fingerprints
        # are present on only some rows. Cross-file copies are merged below.
        return self.source_sha256, self.game_id

    def record(self) -> dict[str, object]:
        return {"candidate_id": self.candidate_id, "source_sha256": self.source_sha256,
                "game_id": self.game_id, "ply": self.ply, "split": self.split,
                "features": list(self.features), "position_key": self.position_key,
                "leakage_key": self.leakage_key, "screen_nodes": self.screen_nodes,
                "label_nodes": self.label_nodes, "teacher_scores_cp": list(self.teacher_scores_cp),
                "evaluator_scores_cp": list(self.evaluator_scores_cp), "fen": self.fen,
                "game_fingerprint": self.game_fingerprint}


def group_source_games(candidates: Sequence[Candidate]) -> dict[tuple[str, int], tuple[str, int]]:
    """Union source/game identities linked by a canonical whole-game hash.

    Optional hashes add links; they never replace the source-local identity.
    Mixed metadata and transitive copies therefore share one exclusion/cap
    group. Conflicting hashes for one source-local game are a provenance error.
    Component representatives are lexical minima, independent of row order.
    """
    parents = {candidate.game_key: candidate.game_key for candidate in candidates}
    source_fingerprints: dict[tuple[str, int], str] = {}
    fingerprint_sources: dict[str, tuple[str, int]] = {}

    def find(key: tuple[str, int]) -> tuple[str, int]:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for candidate in candidates:
        fingerprint = candidate.game_fingerprint
        if fingerprint is None:
            continue
        key = candidate.game_key
        previous = source_fingerprints.setdefault(key, fingerprint)
        if previous != fingerprint:
            raise ValueError("conflicting game_fingerprint values for one source/game identity")
        linked = fingerprint_sources.setdefault(fingerprint, key)
        left, right = find(key), find(linked)
        if left != right:
            parents[max(left, right)] = min(left, right)
    return {key: find(key) for key in parents}


def normalized_ranks(values: Sequence[float]) -> list[float]:
    """Training-pool empirical midranks; true zero stays zero; all-positive ties=1."""
    values = [finite_number(value, "normalization signal") for value in values]
    if any(value < 0 for value in values):
        raise ValueError("normalization requires finite nonnegative signals")
    if not values:
        return []
    counts = Counter(values)
    ranks: dict[float, float] = {}
    lower = 0
    for value in sorted(counts):
        equal = counts[value]
        ranks[value] = (0.0 if value == 0 else 1.0 if len(counts) == 1 else
                        (lower + (equal - 1) / 2) / max(1, len(values) - 1))
        lower += equal
    return [ranks[value] for value in values]


def select_candidates(candidates: Sequence[Candidate], teacher_budget: int, *,
                      method: str = "rift", seed: int = 1,
                      weights: Mapping[str, float] | None = None,
                      max_per_game: int = 1,
                      holdout: Sequence[Candidate] = ()) -> dict[str, object]:
    """Return an auditable plan, without obtaining labels or modifying inputs.

    Screening cost is charged for *every* pool row, even duplicates/held-out
    rows. Holdout descriptors are consulted only for keys, never for signals.
    Every selected position gets a fresh search with its requested label_nodes;
    screen searches cannot supply a warm hash or offset this label allocation.
    """
    node_allocation(teacher_budget, "teacher_budget")
    integer(seed, "seed")
    integer(max_per_game, "max_per_game")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    supplied_weights = dict(weights) if weights is not None else dict.fromkeys(SIGNALS, 1.0)
    if set(supplied_weights) != set(SIGNALS):
        raise ValueError("weights must specify rarity, instability and disagreement")
    converted_weights = {name: finite_number(value, f"weight {name}")
                         for name, value in supplied_weights.items()}
    if any(value < 0 for value in converted_weights.values()) or sum(converted_weights.values()) <= 0:
        raise ValueError("weights must be nonnegative with a positive sum")
    weight_total = sum(converted_weights.values())
    # Finite terms can overflow their sum; reject rather than emitting NaN.
    if not math.isfinite(weight_total):
        raise ValueError("weight sum must be finite")
    normalized_weights = {name: value / weight_total for name, value in converted_weights.items()}
    # The dataclass is public: do not let direct construction bypass validation.
    candidates = [Candidate.from_mapping(candidate.record()) for candidate in candidates]
    holdout = [Candidate.from_mapping(candidate.record()) for candidate in holdout]
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")
    ordered = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    screening_cost = sum(candidate.screen_nodes for candidate in ordered)
    if screening_cost > teacher_budget:
        raise ValueError("screening node allocations alone exceed the teacher budget")
    held_out = [candidate for candidate in candidates if candidate.split != "training"] + list(holdout)
    if any(candidate.split == "training" for candidate in holdout):
        raise ValueError("explicit holdout descriptors cannot be training records")
    game_groups = group_source_games(list(candidates) + list(holdout))
    held_out_games = {game_groups[candidate.game_key] for candidate in held_out}
    held_out_inputs = {candidate.leakage_key for candidate in held_out}
    exclusions: dict[str, str] = {}
    eligible: list[Candidate] = []
    seen_positions: set[str] = set()
    for candidate in ordered:
        if candidate.split != "training":
            reason = "held_out_split"
        elif game_groups[candidate.game_key] in held_out_games:
            reason = "held_out_game"
        elif candidate.leakage_key in held_out_inputs:
            reason = "held_out_nnue_input"
        elif candidate.position_key in seen_positions:
            reason = "duplicate_position"
        else:
            seen_positions.add(candidate.position_key)
            eligible.append(candidate)
            continue
        exclusions[candidate.candidate_id] = reason
    frequency = Counter(feature for candidate in eligible for feature in candidate.features)
    raw: dict[str, list[float]] = {name: [] for name in SIGNALS}
    for candidate in eligible:
        raw["rarity"].append(math.fsum(1 / math.sqrt(1 + frequency[feature])
                                            for feature in candidate.features) /
                              len(candidate.features) if candidate.features else 0.0)
        raw["instability"].append(max((abs(right - left) for left, right in
                                       zip(candidate.teacher_scores_cp, candidate.teacher_scores_cp[1:])),
                                      default=0.0))
        raw["disagreement"].append(max(candidate.evaluator_scores_cp) - min(candidate.evaluator_scores_cp)
                                 if candidate.evaluator_scores_cp else 0.0)
    normalized = {name: normalized_ranks(values) for name, values in raw.items()}
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(eligible):
        signals = {name: normalized[name][index] for name in SIGNALS}
        score = (signals["rarity"] if method == "rarity" else
                 math.fsum(normalized_weights[name] * signals[name] for name in SIGNALS))
        tie_key = hashlib.sha256(f"rift-v1:{seed}:{candidate.candidate_id}".encode("utf-8")).hexdigest()
        rows.append({"candidate": candidate, "raw": {name: raw[name][index] for name in SIGNALS},
                     "normalized": signals, "score": score,
                     "density": score / candidate.label_nodes, "tieKey": tie_key})
    rows.sort(key=(lambda row: (row["tieKey"], row["candidate"].candidate_id)) if method == "random" else
              (lambda row: (-row["density"], -row["score"], row["tieKey"], row["candidate"].candidate_id)))
    remaining = teacher_budget - screening_cost
    selected: list[dict[str, object]] = []
    game_counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        candidate = row["candidate"]
        assert isinstance(candidate, Candidate)
        group = game_groups[candidate.game_key]
        if max_per_game and game_counts[group] >= max_per_game:
            exclusions[candidate.candidate_id] = "game_cap"
        elif candidate.label_nodes > remaining:
            exclusions[candidate.candidate_id] = "remaining_budget"
        else:
            remaining -= candidate.label_nodes
            game_counts[group] += 1
            selected.append({**candidate.record(), "candidateSha256": canonical_hash(candidate.record()),
                             "gameGroup": {"source_sha256": group[0], "game_id": group[1]},
                             "signalsRaw": row["raw"], "signalsNormalized": row["normalized"],
                             "utility": row["score"] if method != "random" else None,
                             "utilityPerNode": row["density"] if method != "random" else None,
                             "tieKey": row["tieKey"]})
    selected_cost = sum(int(record["label_nodes"]) for record in selected)
    return {"schemaVersion": SCHEMA_VERSION, "algorithm": "RIFT-v1-prototype", "method": method,
            "parameters": {"seed": seed, "normalizedWeights": normalized_weights,
                           "maxPerGameGroup": max_per_game, "normalization": "eligible-pool-midranks-v1",
                           "gameGrouping": "source-game-and-canonical-fingerprint-union-v1",
                           "ranking": "seeded-order" if method == "random" else "utility-per-requested-node",
                           "labelSearch": "fresh-clear-hash-no-screen-node-reuse"},
            "lineage": {"candidatePoolSha256": canonical_hash([candidate.record() for candidate in ordered]),
                        "holdoutDescriptorsSha256": canonical_hash(sorted(
                            [{"source_sha256": candidate.source_sha256, "game_id": candidate.game_id,
                              "game_fingerprint": candidate.game_fingerprint,
                              "leakage_key": candidate.leakage_key} for candidate in held_out],
                            key=lambda record: (record["source_sha256"], record["game_id"],
                                                record["leakage_key"], record["game_fingerprint"] or ""))),
                        "featureFrequencySha256": canonical_hash(sorted(frequency.items())),
                        "gameGroupingSha256": canonical_hash(sorted(game_groups.items()))},
            "budget": {"teacherNodeLimit": teacher_budget, "screenNodeAllocations": screening_cost,
                       "uciNodeAllocationCeiling": UCI_NODE_MAX,
                       "labelNodeAllocations": selected_cost,
                       "totalNodeAllocations": screening_cost + selected_cost, "unallocatedNodes": remaining},
            "pool": {"rows": len(candidates), "eligibleUniquePositions": len(eligible),
                     "selectedPositions": len(selected), "seenHalfKpFeatures": len(frequency)},
            "selected": selected, "excluded": exclusions,
            "excludedCounts": dict(sorted(Counter(exclusions.values()).items()))}


def read_candidates(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("candidate must be a JSON object")
                candidates.append(Candidate.from_mapping(record))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return candidates


def demo() -> dict[str, object]:
    """Synthetic ablation: arbitrary features and a handcrafted hidden payoff.

    The payoff is never passed to select_candidates. Its correlation with our
    selection signals is intentionally constructed, not empirical evidence.
    """
    candidates: list[Candidate] = []
    hidden_payoff: dict[str, float] = {}
    for index in range(24):
        rare = index >= 16
        unstable = index % 4 == 3
        disagreement = index % 6 == 5
        candidate_id = f"toy-{index:02}"
        candidates.append(Candidate.from_mapping({
            "candidate_id": candidate_id, "source_sha256": "a" * 64, "game_id": index, "ply": 20,
            "split": "training", "features": [100, 1000 + index if rare else 200],
            "position_key": candidate_id, "leakage_key": candidate_id,
            "screen_nodes": 100, "label_nodes": 1000,
            "teacher_scores_cp": [20, 180 if unstable else 22],
            "evaluator_scores_cp": [20, 170 if disagreement else 25],
        }))
        hidden_payoff[candidate_id] = float(1 + rare + 2 * unstable + 2 * disagreement)
    reports: dict[str, object] = {}
    for method in METHODS:
        plan = select_candidates(candidates, 10_400, method=method, seed=20261013)
        selected = plan["selected"]
        assert isinstance(selected, list)
        ids = [str(record["candidate_id"]) for record in selected]
        reports[method] = {"budget": plan["budget"], "selectedIds": ids,
                           "syntheticPayoffSum": sum(hidden_payoff[candidate_id] for candidate_id in ids)}
    return {"schemaVersion": SCHEMA_VERSION, "kind": "synthetic-plumbing-ablation-not-chess-evidence",
            "warning": "Payoff was hand-aligned with selection signals; no model, teacher or chess strength measured.",
            "reports": reports}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--holdout", type=Path, help="Validation/test descriptors with no deep targets")
    parser.add_argument("--teacher-budget", type=int)
    parser.add_argument("--method", choices=METHODS, default="rift")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-per-game", type=int, default=1, help="0 disables the source-game cap")
    for name in SIGNALS:
        parser.add_argument(f"--weight-{name}", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="New plan file; never overwrite an existing artifact")
    args = parser.parse_args(argv)
    try:
        if args.demo:
            if args.candidates or args.holdout or args.teacher_budget is not None:
                raise ValueError("--demo cannot be combined with input candidates/holdout/budget")
            report = demo()
        else:
            if args.candidates is None or args.teacher_budget is None:
                raise ValueError("--candidates and --teacher-budget are required without --demo")
            candidates = read_candidates(args.candidates)
            holdout = read_candidates(args.holdout) if args.holdout else []
            report = select_candidates(candidates, args.teacher_budget, method=args.method, seed=args.seed,
                                       weights={name: getattr(args, f"weight_{name}") for name in SIGNALS},
                                       max_per_game=args.max_per_game, holdout=holdout)
            report["files"] = {"candidatesSha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
                               "holdoutSha256": hashlib.sha256(args.holdout.read_bytes()).hexdigest()
                               if args.holdout else None,
                               "selectorSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as destination:
                destination.write(payload)
        else:
            print(payload, end="")
        return 0
    except (ValueError, OSError) as error:
        print(f"RIFT planning failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
