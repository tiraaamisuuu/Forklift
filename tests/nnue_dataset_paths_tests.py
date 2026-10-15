#!/usr/bin/env python3
"""Small stdlib-only path checks; no datasets, Torch or chess engines needed."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

from dataset_paths import validate_dataset_paths, validate_initial_weights_path  # noqa: E402


class DatasetPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_distinct_missing_files_do_not_require_dataset_reads(self) -> None:
        validate_dataset_paths(
            [self.root / "train.nnuebin"], [self.root / "validation.nnuebin"],
        )
        self.assertEqual(list(self.root.iterdir()), [])

    def test_repeated_training_path_is_rejected(self) -> None:
        path = self.root / "train.nnuebin"
        with self.assertRaisesRegex(ValueError, "duplicate training"):
            validate_dataset_paths([path, path])

    def test_lexical_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate training"):
            validate_dataset_paths([
                self.root / "train.nnuebin",
                self.root / "unused" / ".." / "train.nnuebin",
            ])

    def test_repeated_validation_path_is_rejected(self) -> None:
        path = self.root / "validation.nnuebin"
        with self.assertRaisesRegex(ValueError, "duplicate validation"):
            validate_dataset_paths([self.root / "train.nnuebin"], [path, path])

    def test_cross_split_overlap_is_rejected(self) -> None:
        path = self.root / "train.nnuebin"
        with self.assertRaisesRegex(ValueError, "paths overlap"):
            validate_dataset_paths([path], [path])

    def test_cross_split_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "paths overlap"):
            validate_dataset_paths(
                [self.root / "train.nnuebin"],
                [self.root / "unused" / ".." / "train.nnuebin"],
            )

    def test_smoke_position_split_can_omit_validation_paths(self) -> None:
        validate_dataset_paths([self.root / "train.nnuebin"])

    def test_missing_training_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one training"):
            validate_dataset_paths([])

    def test_explicit_empty_validation_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            validate_dataset_paths([self.root / "train.nnuebin"], [])

    def test_resolution_failures_are_validation_errors(self) -> None:
        for error in (OSError("unavailable"), RuntimeError("symlink loop")):
            with self.subTest(error=type(error).__name__), \
                    mock.patch.object(Path, "resolve", side_effect=error):
                with self.assertRaisesRegex(ValueError, "cannot resolve training"):
                    validate_dataset_paths([self.root / "train.nnuebin"])

    def test_initial_weights_can_be_outside_all_output_paths(self) -> None:
        validate_initial_weights_path(self.root / "base.pt", self.root / "new" / "model.nnue")

    def test_initial_weights_cannot_be_a_derived_artifact_or_temporary(self) -> None:
        output = self.root / "model.nnue"
        for suffix in (".nnue", ".pt", ".checkpoint.pt", ".best.checkpoint.pt",
                       ".progress.json", ".metrics.json", ".metrics.csv", ".manifest.json"):
            artifact = output.with_suffix(suffix)
            for source in (artifact, artifact.with_name(artifact.name + ".tmp")):
                with self.subTest(source=source):
                    with self.assertRaisesRegex(ValueError, "overlap"):
                        validate_initial_weights_path(source, output)

    def test_initial_weights_cannot_be_custom_metrics_or_manifest(self) -> None:
        source = self.root / "base.pt"
        output = self.root / "new.nnue"
        for options in ({"metrics": source}, {"manifest": source},
                        {"metrics": source.with_suffix(".json")}):
            candidate = source.with_suffix(".csv") if options.get("metrics") == source.with_suffix(".json") else source
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    validate_initial_weights_path(candidate, output, **options)

    def test_initial_weights_cannot_be_in_periodic_checkpoint_directory(self) -> None:
        output = self.root / "model.nnue"
        with self.assertRaisesRegex(ValueError, "checkpoint directory"):
            validate_initial_weights_path(
                self.root / "model.checkpoints" / "epoch-0004.pt", output,
            )

    def test_existing_output_hardlink_to_initial_weights_is_rejected(self) -> None:
        source = self.root / "base.pt"
        output = self.root / "model.nnue"
        source.touch()
        try:
            output.hardlink_to(source)
        except OSError as error:
            self.skipTest(f"hard links unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "share a file"):
            validate_initial_weights_path(source, output)

    def test_existing_periodic_temporary_hardlink_is_rejected(self) -> None:
        source = self.root / "base.pt"
        source.touch()
        periodic = self.root / "model.checkpoints"
        periodic.mkdir()
        try:
            (periodic / "epoch-0001.pt.tmp").hardlink_to(source)
        except OSError as error:
            self.skipTest(f"hard links unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "share a file"):
            validate_initial_weights_path(source, self.root / "model.nnue")


if __name__ == "__main__":
    unittest.main()
