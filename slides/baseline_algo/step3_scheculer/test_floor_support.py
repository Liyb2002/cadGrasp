"""Floor shear must be compressive, bounded, and act at the original point."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step3_scheculer import floor_support as F
from step3_scheculer import verification as V


class FloorSupportTests(unittest.TestCase):
    def test_shear_bound_tension_and_contact_moment(self):
        point = COORD.polar(np.array([.25, -.5, 0.]))
        com = COORD.polar(np.array([0., 0., 1.]))
        columns = F.columns(point, com)

        def feasible(force, free_moment=None):
            force = COORD.polar(force)
            torque = np.cross(point-com, force)
            if free_moment is not None:
                torque += COORD.axial(free_moment)
            result = linprog(np.zeros(4), A_eq=columns.T,
                             b_eq=np.r_[force, torque], bounds=(0, None), method='highs')
            return result.success

        self.assertTrue(feasible([16., 16., 1.]))
        self.assertTrue(feasible([F.COEFFICIENT, 0., 1.]))
        self.assertFalse(feasible([F.COEFFICIENT+1., 0., 1.]))
        self.assertFalse(feasible([1., 0., 0.]))
        self.assertFalse(feasible([0., 0., -1.]))
        self.assertFalse(feasible([0., 0., 1.], [0., 0., 1.]))

    def test_independent_supply_reconstruction_includes_all_floor_rays(self):
        problem = V.C.Problem.__new__(V.C.Problem)
        problem.floor = COORD.polar(np.array([.125, -.25, 0.]))
        problem.domain = NS(com=COORD.polar(np.array([0., 0., .5])))
        problem.scale = np.array([1., 1., 1., 2., 2., 2.])
        problem.floor_columns = V.C.U.floor(F.columns(problem.floor, problem.domain.com),problem.scale)
        supply = V.Supply(problem, [])
        np.testing.assert_array_equal(supply.owners, [-1]*4+[-2])
        np.testing.assert_array_equal(supply.raw[:,6],[0,0,0,0,-1])
        for i in range(5):
            np.testing.assert_array_equal(np.array(supply.exact(i), float), supply.raw[i])


if __name__ == '__main__':
    unittest.main()
