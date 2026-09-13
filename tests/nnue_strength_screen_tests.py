"""Freeze the predeclared NNUE match contracts."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/nnue"))
from run_strength_screen import screen_plan


class StrengthPlanTests(unittest.TestCase):
    def test_overnight_is_new_evidence_with_longer_control(self):
        plan = screen_plan(False, True)
        self.assertEqual([p[2] for p in plan], [800, 1200])
        self.assertEqual(plan[1][3], "90+0.9")
        previous = {p[4] for p in screen_plan(False) + screen_plan(True)}
        self.assertTrue(previous.isdisjoint(p[4] for p in plan))
        self.assertFalse(any(p[5] for p in plan))
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
