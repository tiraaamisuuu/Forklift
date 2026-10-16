#!/usr/bin/env python3
"""Audit compact HalfKP-v1 data coverage and position distributions."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess

from nnue_dataset import (
    active_features_from_packed,
    iter_compact_records,
    sha256_file,
    unpack_piece_codes,
    write_json_atomic,
)


FEATURE_COUNT = 64 * 640
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}


def structurally_possible_feature_mask() -> tuple[bool, ...]:
    """Necessary occupancy rules, not a certificate of reachable chess positions.

    HalfKP allocates king x relative-piece x piece-square slots. No non-king
    piece can occupy the anchor king square; neither pawn bucket can occur on
    rank one/eight. Other legality/reachability restrictions are not modeled.
    """
    return tuple(
        square != king and not (bucket in (0, 5) and square // 8 in (0, 7))
        for king in range(64)
        for bucket in range(10)
        for square in range(64)
    )


def _frequency_summary(frequencies: list[int]) -> dict[str, object]:
    seen_frequencies = [frequency for frequency in frequencies if frequency > 0]
    seen = len(seen_frequencies)
    return {
        "featureCount": len(frequencies),
        "seen": seen,
        "unseen": len(frequencies) - seen,
        "fraction": seen / len(frequencies) if frequencies else 0.0,
        "lowFrequencyIncludesZero": True,
        **{f"atMost{limit}": sum(frequency <= limit for frequency in frequencies)
           for limit in (1, 10, 100)},
        **{f"seenAtMost{limit}": sum(0 < frequency <= limit for frequency in frequencies)
           for limit in (1, 10, 100)},
        "seenFrequency": {
            "minimum": min(seen_frequencies, default=0),
            "median": percentile(seen_frequencies, 0.5),
            "p95": percentile(seen_frequencies, 0.95),
            "maximum": max(seen_frequencies, default=0),
        },
    }


def feature_coverage(frequencies: list[int]) -> dict[str, object]:
    """Preserve legacy raw-slot metrics and add a clearly bounded denominator."""
    if len(frequencies) != FEATURE_COUNT:
        raise ValueError("HalfKP frequency array must contain 40,960 slots")
    mask = structurally_possible_feature_mask()
    possible = [frequency for frequency, allowed in zip(frequencies, mask) if allowed]
    excluded = [frequency for frequency, allowed in zip(frequencies, mask) if not allowed]
    return {
        **_frequency_summary(frequencies),
        "denominator": "all-allocated-feature-slots",
        "structurallyPossible": {
            **_frequency_summary(possible),
            "denominator": "slots-passing-necessary-occupancy-rules",
            "rules": ["nonking-piece-off-anchor-king-square", "pawns-off-back-ranks"],
            "legalityCertified": False,
            "excludedFeatureCount": len(excluded),
            "excludedSeen": sum(frequency > 0 for frequency in excluded),
            "excludedActivations": sum(excluded),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-positions", type=int, default=0,
                        help="Zero audits every record")
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def labelled_bucket(value: int, boundaries: tuple[tuple[int, str], ...], fallback: str) -> str:
    magnitude = abs(value)
    for maximum, label in boundaries:
        if magnitude <= maximum:
            return label
    return fallback


def position_distribution(board_bytes: bytes) -> tuple[str, str]:
    material = {chess.WHITE: 0, chess.BLACK: 0}
    phase = 0
    for code in unpack_piece_codes(board_bytes):
        if code == 0:
            continue
        color = chess.WHITE if code <= 6 else chess.BLACK
        piece_type = code if code <= 6 else code - 6
        material[color] += PIECE_VALUES.get(piece_type, 0)
        phase += PHASE_WEIGHTS.get(piece_type, 0)
    phase_name = "opening" if phase >= 18 else ("middlegame" if phase >= 7 else "endgame")
    imbalance = material[chess.WHITE] - material[chess.BLACK]
    material_name = labelled_bucket(
        imbalance,
        ((50, "balanced"), (300, "small"), (700, "medium")),
        "large",
    )
    return phase_name, material_name


def analyze_records(paths: list[Path], maximum_positions: int = 0) -> dict[str, object]:
    frequencies = [0] * FEATURE_COUNT
    king_squares = {"white": [0] * 64, "black": [0] * 64}
    phases: Counter[str] = Counter()
    material: Counter[str] = Counter()
    eval_magnitudes: Counter[str] = Counter()
    results: Counter[str] = Counter()
    score_signs: Counter[str] = Counter()
    ply_buckets: Counter[str] = Counter()
    positions = 0

    for record in iter_compact_records(paths):
        codes = unpack_piece_codes(record.board_bytes)
        king_squares["white"][codes.index(chess.KING)] += 1
        king_squares["black"][codes.index(chess.KING + 6)] += 1
        for perspective in (chess.WHITE, chess.BLACK):
            for feature in active_features_from_packed(record.board_bytes, perspective):
                frequencies[feature] += 1

        phase, material_bucket = position_distribution(record.board_bytes)
        phases[phase] += 1
        material[material_bucket] += 1
        eval_magnitudes[labelled_bucket(
            record.score_cp,
            ((99, "0-99"), (299, "100-299"), (699, "300-699")),
            "700+",
        )] += 1
        results["win" if record.result > 0 else ("loss" if record.result < 0 else "draw")] += 1
        score_signs["positive" if record.score_cp > 0 else
                    ("negative" if record.score_cp < 0 else "zero")] += 1
        ply_buckets[labelled_bucket(
            record.ply,
            ((20, "0-20"), (40, "21-40"), (80, "41-80"), (120, "81-120")),
            "121+",
        )] += 1
        positions += 1
        if maximum_positions and positions >= maximum_positions:
            break

    return {
        "positions": positions,
        "halfKpActivations": sum(frequencies),
        "featureCoverage": feature_coverage(frequencies),
        "kingSquareCounts": king_squares,
        "phaseCounts": dict(sorted(phases.items())),
        "materialImbalanceCounts": dict(sorted(material.items())),
        "teacherEvalMagnitudeCounts": dict(sorted(eval_magnitudes.items())),
        "teacherScoreSignCounts": dict(sorted(score_signs.items())),
        "resultCounts": dict(sorted(results.items())),
        "plyCounts": dict(sorted(ply_buckets.items())),
    }


def main() -> int:
    args = parse_args()
    if args.max_positions < 0:
        raise SystemExit("--max-positions cannot be negative")
    missing = [str(path) for path in args.data if not path.is_file()]
    if missing:
        raise SystemExit("dataset does not exist: " + ", ".join(missing))
    report = {
        "schemaVersion": 1,
        "datasetFormat": "HalfKP-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "inputs": [{
            "path": str(path.resolve()),
            "sizeBytes": path.resolve().stat().st_size,
            "sha256": sha256_file(path.resolve()),
        } for path in args.data],
        "maximumPositions": args.max_positions,
        "diagnostics": analyze_records(args.data, args.max_positions),
    }
    write_json_atomic(args.output, report)
    coverage = report["diagnostics"]["featureCoverage"]
    possible = coverage["structurallyPossible"]
    print(f"positions={report['diagnostics']['positions']} raw_slot_coverage={coverage['fraction']:.2%} "
          f"seen={coverage['seen']}/{coverage['featureCount']} "
          f"structurally_possible_coverage={possible['fraction']:.2%} "
          f"seen_possible={possible['seen']}/{possible['featureCount']} "
          f"excluded_seen={possible['excludedSeen']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
