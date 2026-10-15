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

from dataset_paths import validate_dataset_paths  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
