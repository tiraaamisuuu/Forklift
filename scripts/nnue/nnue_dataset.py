#!/usr/bin/env python3
"""Versioned compact dataset primitives for HalfKP-v1 training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import BinaryIO, Iterable

import chess


SHARD_MAGIC = b"TNNDS1\0\0"
SHARD_VERSION = 1
SHARD_HEADER = struct.Struct("<8sIIQ")
# 64 four-bit piece codes, teacher cp, game result, side to move, game id, ply.
SHARD_RECORD = struct.Struct("<32shbBIH")
MAX_NON_KING_PIECES = 30


@dataclass(frozen=True)
class ShardRecord:
    board_bytes: bytes
    score_cp: int
    result: int
    turn: chess.Color
    game_id: int
    ply: int


@dataclass(frozen=True)
class ShardHeader:
    count: int
    record_size: int = SHARD_RECORD.size
    version: int = SHARD_VERSION


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def orient_square(square: chess.Square, perspective: chess.Color) -> int:
    if perspective == chess.WHITE:
        return square
    return chess.square(chess.square_file(square), 7 - chess.square_rank(square))


def active_features(board: chess.Board, perspective: chess.Color) -> list[int]:
    king_square = board.king(perspective)
    if king_square is None:
        raise ValueError("training position has no king")
    oriented_king = orient_square(king_square, perspective)
    features: list[int] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        piece_bucket = piece.piece_type - 1
        relative_color = 0 if piece.color == perspective else 1
        bucket = relative_color * 5 + piece_bucket
        oriented_piece = orient_square(square, perspective)
        features.append(oriented_king * 640 + bucket * 64 + oriented_piece)
    features.sort()
    return features


def piece_code(piece: chess.Piece | None) -> int:
    if piece is None:
        return 0
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)


def pack_board(board: chess.Board) -> bytes:
    packed = bytearray(32)
    for square in chess.SQUARES:
        code = piece_code(board.piece_at(square))
        byte_index = square // 2
        if square % 2 == 0:
            packed[byte_index] = code
        else:
            packed[byte_index] |= code << 4
    return bytes(packed)


def unpack_piece_codes(board_bytes: bytes) -> list[int]:
    if len(board_bytes) != 32:
        raise ValueError("packed board must contain exactly 32 bytes")
    codes: list[int] = []
    for value in board_bytes:
        codes.append(value & 0x0F)
        codes.append((value >> 4) & 0x0F)
    if any(code > 12 for code in codes):
        raise ValueError("packed board contains an invalid piece code")
    return codes


def active_features_from_packed(board_bytes: bytes, perspective: chess.Color) -> list[int]:
    codes = unpack_piece_codes(board_bytes)
    king_code = chess.KING + (0 if perspective == chess.WHITE else 6)
    try:
        king_square = codes.index(king_code)
    except ValueError as error:
        raise ValueError("packed training position has no perspective king") from error

    oriented_king = orient_square(king_square, perspective)
    features: list[int] = []
    for square, code in enumerate(codes):
        if code == 0:
            continue
        color = chess.WHITE if code <= 6 else chess.BLACK
        piece_type = code if code <= 6 else code - 6
        if piece_type == chess.KING:
            continue
        relative_color = 0 if color == perspective else 1
        bucket = relative_color * 5 + (piece_type - 1)
        features.append(oriented_king * 640 + bucket * 64 + orient_square(square, perspective))
    features.sort()
    return features


def encode_record(board: chess.Board, score_cp: int, result: float,
                  game_id: int, ply: int) -> bytes:
    bounded_score = max(-32_000, min(32_000, int(score_cp)))
    discrete_result = -1 if result < 0 else (1 if result > 0 else 0)
    if not 0 <= game_id <= 0xFFFFFFFF:
        raise ValueError("game id exceeds compact shard range")
    if not 0 <= ply <= 0xFFFF:
        raise ValueError("ply exceeds compact shard range")
    return SHARD_RECORD.pack(
        pack_board(board), bounded_score, discrete_result,
        1 if board.turn == chess.WHITE else 0, game_id, ply,
    )


def decode_record(payload: bytes) -> ShardRecord:
    if len(payload) != SHARD_RECORD.size:
        raise ValueError("invalid compact shard record size")
    board_bytes, score_cp, result, turn, game_id, ply = SHARD_RECORD.unpack(payload)
    if turn not in (0, 1):
        raise ValueError("compact shard has invalid side to move")
    if result not in (-1, 0, 1):
        raise ValueError("compact shard has invalid game result")
    codes = unpack_piece_codes(board_bytes)
    if codes.count(chess.KING) != 1 or codes.count(chess.KING + 6) != 1:
        raise ValueError("compact shard position must contain exactly one king per side")
    return ShardRecord(board_bytes, score_cp, result, bool(turn), game_id, ply)


def read_shard_header(source: BinaryIO) -> ShardHeader:
    payload = source.read(SHARD_HEADER.size)
    if len(payload) != SHARD_HEADER.size:
        raise ValueError("truncated compact shard header")
    magic, version, record_size, count = SHARD_HEADER.unpack(payload)
    if magic != SHARD_MAGIC:
        raise ValueError("invalid compact shard magic")
    if version != SHARD_VERSION:
        raise ValueError(f"unsupported compact shard version: {version}")
    if record_size != SHARD_RECORD.size:
        raise ValueError("compact shard record size does not match this trainer")
    return ShardHeader(count=count, record_size=record_size, version=version)


class BinaryShardWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._output = path.open("w+b")
        self._count = 0
        self._output.write(SHARD_HEADER.pack(
            SHARD_MAGIC, SHARD_VERSION, SHARD_RECORD.size, 0,
        ))

    @property
    def count(self) -> int:
        return self._count

    def write(self, board: chess.Board, score_cp: int, result: float,
              game_id: int, ply: int) -> None:
        self._output.write(encode_record(board, score_cp, result, game_id, ply))
        self._count += 1

    def write_record(self, record: ShardRecord) -> None:
        self._output.write(SHARD_RECORD.pack(
            record.board_bytes, record.score_cp, record.result,
            1 if record.turn == chess.WHITE else 0, record.game_id, record.ply,
        ))
        self._count += 1

    def close(self) -> None:
        if self._output.closed:
            return
        self._output.flush()
        self._output.seek(0)
        self._output.write(SHARD_HEADER.pack(
            SHARD_MAGIC, SHARD_VERSION, SHARD_RECORD.size, self._count,
        ))
        self._output.flush()
        self._output.close()

    def __enter__(self) -> "BinaryShardWriter":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            # Windows readers/antivirus can briefly deny replacement of an open file.
            time.sleep(0.02 * 2 ** attempt)


def iter_compact_records(paths: Iterable[Path]) -> Iterable[ShardRecord]:
    for path in paths:
        with path.open("rb") as source:
            header = read_shard_header(source)
            for _ in range(header.count):
                payload = source.read(header.record_size)
                yield decode_record(payload)
            if source.read(1):
                raise ValueError(f"compact shard contains trailing bytes: {path}")
