import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_phase5_single_key as phase5


class _State:
    def __init__(self, *, max_xy, max_xyz, z):
        self.max_xy_norm_mm = float(max_xy)
        self.max_xyz_norm_mm = float(max_xyz)
        self.z_mm = float(z)


class Phase8XYWorkspaceTests(unittest.TestCase):
    def test_phase5_35_mm_cap_is_preserved_when_state_requests_it(self):
        state = _State(max_xy=35.0, max_xyz=80.0, z=0.0)
        self.assertAlmostEqual(
            phase5._effective_xy_command_budget_mm(state),
            35.0,
            places=9,
        )

    def test_phase8_xy_radius_is_derived_from_80_mm_xyz_sphere(self):
        state = _State(max_xy=80.0, max_xyz=80.0, z=-66.0)
        expected = math.sqrt(80.0**2 - 66.0**2)
        self.assertAlmostEqual(
            phase5._effective_xy_command_budget_mm(state),
            expected,
            places=9,
        )
        self.assertGreater(expected, 35.0)

    def test_phase8_xy_radius_tightens_automatically_at_max_descent(self):
        state = _State(max_xy=80.0, max_xyz=80.0, z=-70.0)
        expected = math.sqrt(80.0**2 - 70.0**2)
        self.assertAlmostEqual(
            phase5._effective_xy_command_budget_mm(state),
            expected,
            places=9,
        )
        self.assertGreater(expected, 0.0)
        self.assertLess(expected, 40.0)


if __name__ == "__main__":
    unittest.main()
