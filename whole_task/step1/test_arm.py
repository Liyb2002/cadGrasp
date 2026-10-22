"""Triangle against a convex hull: the exact clip in arm.py."""
import unittest
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from whole_task.step1.arm import clipped,Check


class ClipTests(unittest.TestCase):
    def setUp(self):
        h=ConvexHull(np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],float))
        self.n,self.d=h.equations[:,:3],h.equations[:,3]

    def test_triangle_through_the_cube_with_no_vertex_inside(self):
        self.assertTrue(clipped(np.array([[-3,0,0],[3,.5,0],[3,-.5,0]],float),self.n,self.d))

    def test_triangle_past_a_corner(self):
        # Its vertices lie beyond different planes, yet it misses the cube.
        self.assertFalse(clipped(np.array([[2.5,0,0],[0,2.5,0],[2.5,2.5,0]],float),self.n,self.d))


class TableTests(unittest.TestCase):
    def test_tool_body_must_clear_the_table_beyond_the_contact(self):
        check=Check.__new__(Check)
        check.mesh=trimesh.Trimesh(vertices=[[0,0,0],[0,1,0],[0,0,1]],faces=[[0,1,2]],process=False)
        check.table_bounds=np.array([[-2,-2,-.02],[2,2,0.]])
        check.length=.1;check.needle_radius=.002;check.head_radius=.02
        axis=np.array([-np.cos(np.radians(20)),0,-np.sin(np.radians(20))]);W=np.eye(4)
        self.assertFalse(check.tool_above_table(0,axis,W))
        W[2,3]=.01
        self.assertTrue(check.tool_above_table(0,axis,W))
        check.head_radius=.06
        self.assertFalse(check.tool_above_table(0,axis,W))

    def test_arm_collision_uses_the_finite_table_footprint(self):
        check=Check.__new__(Check);check.mesh=trimesh.creation.box();check.mesh.apply_translation([10,10,10])
        check.placed={};check.table_bounds=np.array([[-1,-1,-.1],[1,1,0.]])
        check.table=trimesh.creation.box(bounds=check.table_bounds).triangles
        vertices=trimesh.creation.box(extents=[.2,.2,.2]).vertices;hull=ConvexHull(vertices)
        check.hulls=[(hull.equations[:,:3],hull.equations[:,3],np.zeros(3),np.linalg.norm(vertices,axis=1).max())]
        class Arm:
            point=np.array([0,0,.05])
            def forward(self,q,geometry=False):return None,None,[(np.eye(3),self.point)]
        check.arm=Arm();self.assertTrue(check.hits(None,np.eye(4)))
        check.arm.point=np.array([2,0,.05]);self.assertFalse(check.hits(None,np.eye(4)))
        check.arm.point=np.array([0,0,.2]);self.assertFalse(check.hits(None,np.eye(4)))


if __name__=='__main__':
    unittest.main()
