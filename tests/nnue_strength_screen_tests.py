"""Freeze the predeclared NNUE match contracts."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/nnue"))
from run_strength_screen import screen_plan


class StrengthPlanTests(unittest.TestCase):
    def test_paused_restart_uses_its_fresh_predeclared_seed(self):
        resumed = screen_plan(False, recover_long=True, paused_restart=True)
        previous = screen_plan(False, True) + screen_plan(False, recover_long=True)
        self.assertEqual(resumed[0], screen_plan(False, True)[0])
        self.assertEqual(resumed[1][2:4], (1200, "90+0.9"))
        self.assertEqual(resumed[1][4], 202610121)
        self.assertNotIn(resumed[1][4], [p[4] for p in previous])
    def test_long_recovery_preserves_short_stage_and_changes_long_seed(self):
        original = screen_plan(False, True)
        recovery = screen_plan(False, recover_long=True)
        self.assertEqual(recovery[0], original[0])
        self.assertEqual(recovery[1][2:4], (1200, "90+0.9"))
        self.assertEqual(recovery[1][4], 202610120)
        self.assertNotIn(recovery[1][4], [p[4] for p in original])
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
