"""Ground-plane intersections survive support unions and ring expansion."""
import sys
from pathlib import Path
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import step33

class GroundRingExpansionTests(unittest.TestCase):
    def test_crossing_solid_has_ground_section_without_ground_vertices(self):
        mesh=trimesh.creation.box(extents=[2,2,2])
        _,passed,covered=step33.ground_hull_coverage(mesh,np.eye(4),np.array([[.9,.9],[-.9,-.9]]))
        self.assertTrue(passed)
        self.assertEqual(covered,2)
        _,passed,covered=step33.ground_hull_coverage(mesh,np.eye(4),np.array([[1.1,0.]]))
        self.assertFalse(passed)
        self.assertEqual(covered,0)

    def test_floating_solid_has_no_ground_coverage(self):
        mesh=trimesh.creation.box(extents=[1,1,1]);mesh.apply_translation([0,0,2])
        self.assertEqual(step33.ground_hull_coverage(mesh,np.eye(4),np.zeros((1,2))),([],False,0))

    def test_expansion_changes_outer_extent_but_keeps_ring_height(self):
        polygon=np.array([[-.01,-.01],[.01,-.01],[.01,.01],[-.01,.01]])
        ring=step33.S.unpack(step33.perimeter(polygon,expansion=.004))
        np.testing.assert_allclose(ring.bounds[:,:2],[[-.014,-.014],[.014,.014]],atol=1e-12)
        np.testing.assert_allclose(ring.bounds[:,2],[0,.005],atol=1e-12)

if __name__=='__main__':unittest.main()
