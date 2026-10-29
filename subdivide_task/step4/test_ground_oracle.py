"""Physical regressions for sparse contact hulls and complete installed stock."""
import unittest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step4.ground_oracle import ground_enclosure


def row(matrix=None):
    return dict(id='task',world_from_support=np.eye(4) if matrix is None else matrix,
        hull=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]))


class GroundOracleTests(unittest.TestCase):
    def test_sparse_corner_pads_enclose_pressure_without_filling_hull(self):
        pads=[G.box([x-.025,y-.025,0],[x+.025,y+.025,.01]) for x in (-.6,.6) for y in (-.6,.6)]
        mesh=G.mesh(G.union(pads))
        result=ground_enclosure(mesh,[row()])[0]
        self.assertTrue(result['passed'])
        self.assertGreater(result['actual_ground_triangle_count'],0)
        self.assertLess(mesh.volume,.001)
        self.assertFalse(mesh.contains([[0.,0.,.005]])[0])

    def test_small_contact_hull_rejects_even_if_stock_covers_pressure_from_above(self):
        top=G.box([-.7,-.7,.2],[.7,.7,.21]);leg=G.box([-.02,-.02,0],[.02,.02,.2])
        result=ground_enclosure(G.mesh(top+leg),[row()])[0]
        self.assertFalse(result['passed'])
        self.assertGreater(result['pressure_hull_gap_m'],.4)

    def test_contact_hull_cannot_hide_material_below_another_installation(self):
        mesh=G.mesh(G.box([-.7,-.7,0],[.7,.7,.02])+G.box([-.01,-.01,-.1],[.01,.01,.01]))
        result=ground_enclosure(mesh,[row()])[0]
        self.assertFalse(result['passed'])
        self.assertLess(result['minimum_world_z_m'],-.09)

    def test_tangent_point_is_not_a_real_ground_patch(self):
        mesh=trimesh.creation.icosphere(subdivisions=1,radius=1.)
        mesh.apply_translation([0,0,1])
        result=ground_enclosure(mesh,[row()])[0]
        self.assertFalse(result['passed'])
        self.assertEqual(result['actual_ground_triangle_count'],0)

    def test_turned_support_uses_its_actual_new_ground_face(self):
        mesh=G.mesh(G.box([0,-.7,0],[1.4,.7,1.4]))
        turn=np.eye(4);turn[:3,:3]=Rotation.from_euler('y',-90,degrees=True).as_matrix()
        turn[0,3]=.7
        result=ground_enclosure(mesh,[row(turn)])[0]
        self.assertTrue(result['passed'])
        self.assertGreater(result['actual_ground_triangle_count'],0)


if __name__=='__main__':unittest.main()
