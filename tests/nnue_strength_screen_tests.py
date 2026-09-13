"""Freeze the predeclared NNUE match contracts."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/nnue"))
from run_strength_screen import screen_plan


class StrengthPlanTests(unittest.TestCase):
    def test_confirmation_changes_seeds_and_tests_classical_only(self):
        original = screen_plan(False)
        confirmation = screen_plan(True)
        self.assertTrue(set(item[4] for item in original).isdisjoint(item[4] for item in confirmation))
        self.assertEqual([item[2] for item in confirmation], [800, 400])
        self.assertEqual([item[3] for item in confirmation], ["30+0.3", "60+0.6"])
        self.assertFalse(any(item[5] for item in confirmation))

    def test_original_screen_is_unchanged(self):
        plan = screen_plan(False)
        self.assertEqual([item[2] for item in plan], [400, 400])
        self.assertEqual([item[5] for item in plan], [True, False])
        self.assertEqual([item[4] for item in plan], [20260912, 20260913])


if __name__ == "__main__":
    unittest.main()
