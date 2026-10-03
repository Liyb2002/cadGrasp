"""The contact hull covers a demand without material filling its interior."""
import unittest
import numpy as np
from shapely.geometry import Polygon
import trimesh

from step4_connect_support.baseline_current import open_ground_frame as F,build_coupled_saddle as S
from step4_connect_support.baseline_current.convex_foot import landing


class OpenFrameTests(unittest.TestCase):
    def test_hollow_frame_retains_ground_convex_hull(self):
        xy=np.array([[-.05,-.04],[.05,-.04],[.05,.04],[-.05,.04]])
        frame,_,report=F.frame(xy,np.eye(3),np.zeros(3),lambda x:x)
        ground=landing(frame,np.eye(3),np.zeros(3))
        self.assertLess(ground.area/Polygon(xy).area,.15)
        self.assertLess(ground.convex_hull.symmetric_difference(Polygon(xy)).area,1e-12)
        center=trimesh.creation.box([.02,.02,.004]);center.apply_translation([0,0,.002])
        self.assertLess(abs((S.solid(center)^frame).volume())*S.SCALE**3,1e-14)
        self.assertEqual(len(frame.decompose()),1)
        self.assertTrue(report['interior_is_not_material'])

    def test_ground_wrench_at_interior_is_combination_of_corners(self):
        # For one friction-ray direction, force and torque are affine in the
        # contact point. Ground-hull containment does not require a filled sole.
        xy=np.array([[-.05,-.04],[.05,-.04],[.05,.04],[-.05,.04]])
        points=np.c_[xy,np.zeros(4)];weights=np.array([.1,.2,.3,.4])
        for force in ([0.,0,1],[64.,0,1],[-64.,0,1],[0,64.,1],[0,-64.,1]):
            force=np.asarray(force);center=weights@points
            rhs=np.r_[force,np.cross(center,force)]
            lhs=weights@np.array([np.r_[force,np.cross(point,force)] for point in points])
            np.testing.assert_allclose(lhs,rhs,atol=1e-12)

    def test_disconnected_non_extreme_ground_fragment_is_unnecessary(self):
        xy=np.array([[-.05,-.04],[.05,-.04],[.05,.04],[-.05,.04]])
        cutters=[]
        for x in (-.009,.009):
            box=trimesh.creation.box([.004,.012,.01]);box.apply_translation([x,.04,.002])
            cutters.append(S.solid(box))
        obstacle=F.union(cutters)
        frame,_,_=F.frame(xy,np.eye(3),np.zeros(3),lambda x:x-obstacle)
        self.assertEqual(len(frame.decompose()),1)
        ground=landing(frame,np.eye(3),np.zeros(3))
        self.assertLess(Polygon(xy).difference(ground.convex_hull.buffer(1e-10)).area,1e-12)


if __name__=='__main__':unittest.main()
