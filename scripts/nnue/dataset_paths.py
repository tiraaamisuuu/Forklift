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


def validate_initial_weights_path(
    initial: Path, output: Path,
    metrics: Path | None = None, manifest: Path | None = None,
) -> None:
    """Keep warm-start weights outside every artifact the trainer may write.

    Metadata-only, including hard-link identities for existing targets. Reserve
    temporary names as well as final files and the whole epoch directory so a
    pre-training progress update cannot destroy the source before it is loaded.
    """
    try:
        source = initial.resolve()
        metrics_path = metrics or output.with_suffix(".metrics.json")
        paths = {
            output, output.with_suffix(".pt"),
            output.with_suffix(".checkpoint.pt"),
            output.with_suffix(".best.checkpoint.pt"),
            output.with_suffix(".progress.json"),
            metrics_path, metrics_path.with_suffix(".csv"),
            manifest or output.with_suffix(".manifest.json"),
        }
        periodic = output.with_suffix(".checkpoints").resolve()
        if source == periodic or periodic in source.parents:
            raise ValueError("initial weights must not be inside this run's checkpoint directory")
        # An outside source can still be hard-linked to a pre-existing epoch
        # temporary. Inspect only this run's small checkpoint directory, never
        # dataset contents; nonexistent future files cannot already be aliases.
        if periodic.is_dir():
            paths.update(periodic.glob("epoch-*.pt"))
            paths.update(periodic.glob("epoch-*.pt.tmp"))
        paths |= {path.with_name(path.name + ".tmp") for path in paths}
        for artifact in paths:
            if source == artifact.resolve():
                raise ValueError("initial weights must not overlap this run's output artifacts")
            try:
                same_file = source.samefile(artifact)
            except FileNotFoundError:
                same_file = False
            if same_file:
                raise ValueError("initial weights must not share a file with output artifacts")
    except (OSError, RuntimeError) as error:
        raise ValueError(f"cannot resolve initial/output paths: {error}") from error
