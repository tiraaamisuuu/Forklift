"""Cheap dataset-path guards; deliberately no dataset reads or ML imports.

Distinct paths are necessary, not sufficient, for an independent holdout.
Copied files, hard links, shared games and equivalent model inputs require a
separate content/provenance audit. This module does not certify any of those.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def validate_dataset_paths(
    training_paths: Sequence[Path],
    validation_paths: Sequence[Path] | None = None,
) -> None:
    """Reject repeated/resolved aliases before allocating training resources.

    Resolution may inspect filesystem metadata, but never reads file contents,
    requires datasets to exist, scans directories or starts a worker/engine.
    Path equality follows the host platform's case-sensitivity conventions.
    """
    if not training_paths:
        raise ValueError("at least one training dataset path is required")
    if validation_paths is not None and not validation_paths:
        raise ValueError("an explicit validation dataset list cannot be empty")

    def unique_paths(paths: Sequence[Path], split: str) -> set[Path]:
        seen: set[Path] = set()
        for path in paths:
            try:
                resolved = Path(path).resolve()
            except (OSError, RuntimeError) as error:
                # Python <= 3.12 uses RuntimeError for symlink loops. Translate
                # resolution failures so the trainer can reject without writes.
                raise ValueError(f"cannot resolve {split} dataset path: {path}") from error
            if resolved in seen:
                raise ValueError(f"duplicate {split} dataset path: {resolved}")
            seen.add(resolved)
        return seen

    training = unique_paths(training_paths, "training")
    validation = unique_paths(validation_paths or (), "validation")
    overlap = training & validation
    if overlap:
        shared = min(overlap, key=str)
        raise ValueError(f"training and validation dataset paths overlap: {shared}")
