#!/usr/bin/env python3
"""Build immutable, game-disjoint float32 activation caches for frozen-head studies.

The cache stores the *float training model* accumulators, not dequantized engine
accumulators. Training and test membership never depends on a teacher label.
The final manifest is the completion marker; a partial directory is not a cache.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Callable

import numpy as np
import torch

from initialization import load_initial_weights
from nnue_dataset import (
    SHARD_HEADER, SHARD_RECORD, active_features_from_packed, decode_record,
    read_shard_header, sha256_file, write_json_atomic,
)
from train import HalfKpV1, collate, score_to_win_probability


SCHEMA_VERSION = 1
RECORD_DTYPE = np.dtype([
    ("board", "u1", (32,)), ("score", "<i2"), ("result", "i1"),
    ("turn", "u1"), ("game", "<u4"), ("ply", "<u2"),
])
ARRAY_FILES = {
    "activations": "x.npy", "target_cp": "cp.npy", "target_wdl": "wdl.npy",
    "teacher_cp": "teacher_cp.npy", "game_ids": "game_ids.npy",
    "source_indices": "rows.npy", "boards": "board_bytes.npy",
    "turns": "turn.npy", "identity": "identity.npy",
}


def effective_input_identity(first: list[int], second: list[int]) -> bytes:
    """Hash exact ordered feature bags, including perspective and bag boundaries.

    Unlike raw board+turn, this also catches color-flipped vertically mirrored
    positions which are identical inputs to this HalfKP representation.
    """
    digest = hashlib.sha256()
    for bag in (first, second):
        digest.update(struct.pack("<I", len(bag)))
        digest.update(np.asarray(bag, dtype="<u2").tobytes())
    return digest.digest()


def validation_partition(game_id: int, seed: int) -> str:
    value = struct.pack("<QQ", seed, game_id)
    return "validation" if hashlib.sha256(value).digest()[0] & 1 == 0 else "test"


def _source(path: Path) -> tuple[np.memmap, dict]:
    path = path.resolve(strict=True)
    with path.open("rb") as handle:
        header = read_shard_header(handle)
    expected = SHARD_HEADER.size + header.count * SHARD_RECORD.size
    if path.stat().st_size != expected:
        raise ValueError(f"compact shard size/count mismatch: {path}")
    if header.count == 0:
        raise ValueError(f"compact shard is empty: {path}")
    records = np.memmap(path, mode="r", offset=SHARD_HEADER.size,
                        shape=(header.count,), dtype=RECORD_DTYPE)
    return records, {"path": str(path), "sha256": sha256_file(path),
                     "count": header.count, "bytes": expected}


def _close_arrays(arrays: dict[str, np.ndarray]) -> None:
    for value in arrays.values():
        if isinstance(value, np.memmap):
            if value.flags.writeable:
                value.flush()
            value._mmap.close()


def _new_arrays(directory: Path, count: int, hidden: int) -> dict[str, np.memmap]:
    directory.mkdir()
    specifications = {
        "activations": ("<f4", (count, hidden * 2)),
        "target_cp": ("<f4", (count,)), "target_wdl": ("<f4", (count,)),
        "teacher_cp": ("<i2", (count,)), "game_ids": ("<u4", (count,)),
        "source_indices": ("<i8", (count,)), "boards": ("u1", (count, 32)),
        "turns": ("u1", (count,)), "identity": ("u1", (count, 32)),
    }
    return {name: np.lib.format.open_memmap(
        directory / ARRAY_FILES[name], mode="w+", dtype=dtype, shape=shape,
    ) for name, (dtype, shape) in specifications.items()}


def build_cache(
    train_path: Path, validation_path: Path, initial_weights: Path,
    output_dir: Path, *, train_count: int = 250_000,
    validation_count: int = 32_000, test_count: int = 32_000,
    seed: int = 20261015, hidden: int = 256, feature_factorization: bool = True,
    batch_size: int = 4096, device: str = "cpu", cpu_threads: int = 4,
    max_seconds: float = 900.0,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Sample and cache frozen activations, failing rather than reusing outputs.

    Source-wide game IDs must be disjoint. Selected effective inputs are unique
    within and across all splits. This does not claim full-source input overlap
    auditing: only selected input identities are compared.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (train_count, validation_count, test_count, hidden,
                         batch_size, cpu_threads)):
        raise ValueError("cache sizes, batch size, hidden width and threads must be positive integers")
    if not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max seconds must be finite and positive")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"cache output directory must not already exist: {output_dir}")
    train_path = train_path.resolve(strict=True)
    validation_path = validation_path.resolve(strict=True)
    initial_weights = initial_weights.resolve(strict=True)
    if train_path.samefile(validation_path):
        raise ValueError("training and validation sources must differ")
    if any(output_dir == path or output_dir in path.parents
           for path in (train_path, validation_path, initial_weights)):
        raise ValueError("cache output must not contain a source file")
    started = time.monotonic()
    source_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns)
                    for path in (train_path, validation_path, initial_weights)}

    def check_time() -> None:
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("activation cache exceeded its wall-clock budget")

    def report(**values: object) -> None:
        check_time()
        if progress:
            progress({"elapsedSeconds": round(time.monotonic() - started, 3), **values})

    train_records = validation_records = None
    arrays: dict[str, dict[str, np.memmap]] = {}
    try:
        train_records, train_source = _source(train_path)
        validation_records, validation_source = _source(validation_path)
        check_time()
        train_games = np.unique(train_records["game"])
        validation_games = np.unique(validation_records["game"])
        overlapping_games = np.intersect1d(train_games, validation_games, assume_unique=True)
        if len(overlapping_games):
            raise ValueError(f"source game IDs overlap ({len(overlapping_games)} games): "
                             f"{overlapping_games[:8].tolist()}")
        report(stage="source-audit", trainingGames=len(train_games),
               validationGames=len(validation_games), overlappingGames=0)
        if train_count > len(train_records) or validation_count + test_count > len(validation_records):
            raise ValueError("requested cache counts exceed available source rows")
        torch.set_num_threads(cpu_threads)
        # Do not permit initialization to perturb the caller's CPU shuffle RNG.
        with torch.random.fork_rng(devices=[]):
            model = HalfKpV1(hidden, feature_factorization=feature_factorization)
        initialization = load_initial_weights(model, initial_weights)
        model.eval().requires_grad_(False).to(torch.device(device))
        output_dir.mkdir(parents=True, exist_ok=False)
        counts = {"train": train_count, "validation": validation_count, "test": test_count}
        for split, count in counts.items():
            arrays[split] = _new_arrays(output_dir / split, count, hidden)
        completed = dict.fromkeys(counts, 0)
        skipped_duplicates = dict.fromkeys(counts, 0)
        examined = dict.fromkeys(counts, 0)
        seen_inputs: set[bytes] = set()
        selected_games: dict[str, set[int]] = {split: set() for split in counts}
        pending: dict[str, list[tuple]] = {split: [] for split in counts}
        partitions: dict[int, str] = {}
        rng = np.random.default_rng(seed)

        def flush(split: str) -> None:
            rows = pending[split]
            if not rows:
                return
            batch = collate([(first, second, float(max(-3000, min(3000, record.score_cp))),
                              score_to_win_probability(max(-3000, min(3000, record.score_cp)), 400.0),
                              float(record.score_cp))
                             for _, record, first, second, _ in rows])
            with torch.inference_mode():
                first, first_offsets, second, second_offsets = (
                    value.to(device) for value in batch[:4])
                activation = torch.cat((model.accumulator(first, first_offsets),
                                        model.accumulator(second, second_offsets)), dim=1)
                values = activation.cpu().numpy()
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("frozen activations must be finite float32 values")
            begin, end = completed[split], completed[split] + len(rows)
            destination = arrays[split]
            destination["activations"][begin:end] = values
            destination["target_cp"][begin:end] = batch[4].numpy()
            destination["target_wdl"][begin:end] = batch[5].numpy()
            for position, (source_index, record, _, _, identity) in enumerate(rows, begin):
                destination["teacher_cp"][position] = record.score_cp
                destination["game_ids"][position] = record.game_id
                destination["source_indices"][position] = source_index
                destination["boards"][position] = np.frombuffer(record.board_bytes, dtype=np.uint8)
                destination["turns"][position] = int(record.turn)
                destination["identity"][position] = np.frombuffer(identity, dtype=np.uint8)
                selected_games[split].add(record.game_id)
            completed[split] = end
            pending[split] = []
            report(stage="activations", split=split, completed=end, requested=counts[split])

        # The permutation includes a deterministic reserve pool; reject duplicates
        # until each exact count is filled or exhaust the source and fail.
        for source_name, records in (("train", train_records), ("validation", validation_records)):
            for iteration, source_index in enumerate(rng.permutation(len(records))):
                if iteration % 1024 == 0:
                    check_time()
                record = decode_record(records[source_index].tobytes())
                if source_name == "train":
                    split = "train"
                else:
                    split = partitions.setdefault(record.game_id, validation_partition(record.game_id, seed))
                if completed[split] + len(pending[split]) == counts[split]:
                    if source_name == "train" or all(
                        completed[name] + len(pending[name]) == counts[name]
                        for name in ("validation", "test")
                    ):
                        break
                    continue
                examined[split] += 1
                first = active_features_from_packed(record.board_bytes, record.turn)
                second = active_features_from_packed(record.board_bytes, not record.turn)
                identity = effective_input_identity(first, second)
                if identity in seen_inputs:
                    skipped_duplicates[split] += 1
                    continue
                seen_inputs.add(identity)
                pending[split].append((int(source_index), record, first, second, identity))
                if len(pending[split]) >= batch_size:
                    flush(split)
            for split in (("train",) if source_name == "train" else ("validation", "test")):
                flush(split)
                if completed[split] != counts[split]:
                    raise ValueError(f"not enough distinct {split} inputs: "
                                     f"{completed[split]} < {counts[split]}")
        for first_split, second_split in (("train", "validation"), ("train", "test"),
                                          ("validation", "test")):
            if selected_games[first_split] & selected_games[second_split]:
                raise AssertionError("selected games overlap despite source/partition audit")
        manifests = {}
        for split, values in arrays.items():
            _close_arrays(values)
            files = {}
            for name, filename in ARRAY_FILES.items():
                path = output_dir / split / filename
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                files[name] = {"path": f"{split}/{filename}", "sha256": sha256_file(path),
                               "shape": list(value.shape), "dtype": str(value.dtype)}
                value._mmap.close()
                check_time()
            manifests[split] = {"count": counts[split], "games": len(selected_games[split]),
                                "examinedRows": examined[split],
                                "excludedDuplicateInputs": skipped_duplicates[split], "files": files}
        arrays.clear()
        for path, old_stat in source_stats.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != old_stat:
                raise ValueError(f"source changed while cache was built: {path}")
        manifest = {
            "schemaVersion": SCHEMA_VERSION, "status": "complete",
            "createdUtc": datetime.now(timezone.utc).isoformat(),
            "sourceWeights": {"path": str(initial_weights),
                              "sha256": initialization["sha256"],
                              "initialization": initialization},
            "sources": {"train": train_source, "validation": validation_source},
            "configuration": {"seed": seed, "hidden": hidden,
                              "featureFactorization": feature_factorization,
                              "targetScale": 600.0, "wdlScale": 400.0, "resultWeight": 0.0,
                              "teacherClipCp": [-3000, 3000], "activationDtype": "float32",
                              "perspectiveOrder": "side-to-move-first",
                              "batchSize": batch_size, "device": str(device),
                              "cpuThreads": cpu_threads, "maxSeconds": max_seconds},
            "audit": {"fullSourceGameIdsDisjoint": True,
                      "sourceGameCounts": {"train": len(train_games), "validation": len(validation_games)},
                      "selectedGameIdsDisjoint": True,
                      "selectedEffectiveInputsDisjoint": True,
                      "effectiveInputIdentity": "sha256(length+uint16-le-first,length+uint16-le-second)",
                      "inputAuditScope": "selected rows only; not complete source input overlap",
                      "validationPartition": "sha256(<uint64-le seed,uint64-le game_id>)[0] low bit"},
            "splits": manifests, "elapsedSeconds": time.monotonic() - started,
            "environment": {"torchVersion": torch.__version__, "numpyVersion": np.__version__},
        }
        check_time()
        write_json_atomic(output_dir / "manifest.json", manifest)
        return manifest
    finally:
        for values in arrays.values():
            # Some maps may already have been flushed/closed before an error.
            for value in values.values():
                if isinstance(value, np.memmap) and not value._mmap.closed:
                    if value.flags.writeable:
                        value.flush()
                    value._mmap.close()
        for records in (train_records, validation_records):
            if records is not None:
                records._mmap.close()


def load_cache(manifest_path: Path, split: str, *, verify_file_hashes: bool = True) -> dict[str, np.ndarray]:
    """Open one complete split read-only, checking provenance file hashes by default."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("unsupported or incomplete activation cache")
    split = "validation" if split == "tune" else split
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown cache split: {split}")
    entry = manifest["splits"][split]
    arrays: dict[str, np.ndarray] = {}
    try:
        for name in ARRAY_FILES:
            descriptor = entry["files"][name]
            path = (manifest_path.parent / descriptor["path"]).resolve(strict=True)
            if manifest_path.parent not in path.parents:
                raise ValueError("cache array path escapes cache directory")
            if verify_file_hashes and sha256_file(path) != descriptor["sha256"]:
                raise ValueError(f"cache array hash mismatch: {path}")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays[name] = value
            if (list(value.shape) != descriptor["shape"] or str(value.dtype) != descriptor["dtype"]
                    or len(value) != entry["count"]):
                raise ValueError(f"cache array shape/dtype/count mismatch: {path}")
        return arrays
    except BaseException:
        _close_arrays(arrays)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=250_000)
    parser.add_argument("--validation-count", type=int, default=32_000)
    parser.add_argument("--test-count", type=int, default=32_000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--feature-factorization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    args = parser.parse_args()
    manifest = build_cache(
        args.train, args.validation, args.initial_weights, args.output_dir,
        train_count=args.train_count, validation_count=args.validation_count,
        test_count=args.test_count, seed=args.seed, hidden=args.hidden,
        feature_factorization=args.feature_factorization, batch_size=args.batch_size,
        device=args.device, cpu_threads=args.cpu_threads, max_seconds=args.max_seconds,
        progress=lambda value: print(json.dumps(value), flush=True),
    )
    print(json.dumps({"manifest": str(args.output_dir.resolve() / "manifest.json"),
                      "elapsedSeconds": manifest["elapsedSeconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
