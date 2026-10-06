
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
import numpy as np
from scipy.spatial import ConvexHull
from step33 import demand_hull, perimeter, S, F, ROOT


class ConvexPerimeters(unittest.TestCase):
    def test_non_circular_exact_enclosure(self):
        points=np.array([[0.,0.],[.04,0.],[.04,.012],[0.,.012],[.02,.006]])
        polygon,indices,area=demand_hull(points)
        self.assertEqual(len(polygon),4)
        self.assertAlmostEqual(area,.04*.012)
        mesh=S.unpack(perimeter(polygon))
        ground=mesh.vertices[np.abs(mesh.vertices[:,2])<1e-10,:2]
        actual=ConvexHull(ground)
        self.assertAlmostEqual(actual.volume,area,places=10)
        np.testing.assert_allclose(ground.min(0),[0,0],atol=1e-10)
        np.testing.assert_allclose(ground.max(0),[.04,.012],atol=1e-10)
        # The interior is hollow, while the exact four extreme points remain.
        self.assertAlmostEqual(S.solid(mesh).volume()*S.SCALE**3,
                               (.04*.012-.03*.002)*.005,places=11)

    def test_all_saved_points_covered_by_actual_ground_material(self):
        points=np.load(ROOT/'objects/B/poses/pose_3/floor_contact.npz')['floor_demands_xy_m']
        polygon,indices,area=demand_hull(points)
        mesh=S.unpack(perimeter(polygon))
        ground=mesh.vertices[np.abs(mesh.vertices[:,2])<1e-10,:2]
        equations=ConvexHull(ground).equations
        self.assertLessEqual(np.max(points@equations[:,:2].T+equations[:,2]),1e-9)
        self.assertAlmostEqual(ConvexHull(ground).volume,area,places=10)

    def test_cut_extreme_point_cannot_be_claimed_covered(self):
        points=np.array([[0.,0.],[.04,0.],[.04,.012],[0.,.012]])
        polygon,_,_=demand_hull(points)
        cut=F.md.Manifold.cube((.003/S.SCALE,.003/S.SCALE,.01/S.SCALE)).translate((-.001/S.SCALE,-.001/S.SCALE,-.001/S.SCALE))
        mesh=S.unpack(perimeter(polygon)-cut)
        ground=mesh.vertices[np.abs(mesh.vertices[:,2])<1e-10,:2]
        equations=ConvexHull(ground).equations
        self.assertGreater(np.max(points@equations[:,:2].T+equations[:,2]),1e-4)


if __name__=='__main__':unittest.main()
