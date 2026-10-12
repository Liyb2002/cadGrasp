from pathlib import Path
from unittest.mock import patch
import sys,unittest
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
from component_membership import robust_contains,winding_numbers


class MembershipTests(unittest.TestCase):
    def test_ambiguous_ray_inside_is_resolved_without_accepting_outside(self):
        mesh=trimesh.creation.box();points=np.array([[.499999,0.,0.],[.500001,0.,0.]])
        with patch('component_membership.contains_points',return_value=np.zeros(2,bool)):
            np.testing.assert_array_equal(robust_contains(mesh,points),[True,False])

    def test_nested_cavity_stays_outside_material(self):
        outer=trimesh.creation.box([2.,2.,2.]);inner=trimesh.creation.box();inner.invert()
        mesh=trimesh.util.concatenate([outer,inner]);points=np.array([[0.,0.,0.],[.75,0.,0.],[1.1,0.,0.]])
        with patch('component_membership.contains_points',return_value=np.zeros(3,bool)):
            np.testing.assert_array_equal(robust_contains(mesh,points),[False,True,False])

    def test_closed_component_orientation_does_not_change_membership(self):
        mesh=trimesh.creation.box();mesh.invert()
        with patch('component_membership.contains_points',return_value=np.zeros(1,bool)):
            self.assertTrue(robust_contains(mesh,[[.1,.2,.3]])[0])

    def test_vertex_probe_is_unresolved(self):
        mesh=trimesh.creation.box()
        self.assertFalse(np.isfinite(winding_numbers(mesh.triangles,mesh.vertices[:1]))[0])


if __name__=='__main__':unittest.main()
