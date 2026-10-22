"""Exact reach of a translating needle and head on boxes with known answers."""
import unittest
import numpy as np
import trimesh
from whole_task.step1 import reach as R


def scene(pillar=None):
    parts=[trimesh.creation.box(bounds=[[0,0,0],[1,1,1]])]
    if pillar is not None:parts.append(trimesh.creation.box(bounds=pillar))
    return trimesh.util.concatenate(parts)


class ReachTests(unittest.TestCase):
    down=np.array([0.,0,-1])

    def top_face(self,mesh):
        return int(np.flatnonzero((mesh.face_normals[:,2]>.9)&(mesh.triangles_center[:,2]>.99))[0])

    def test_open_top_is_reached(self):
        mesh=scene();ok,needle,head=R.certify(mesh,self.top_face(mesh),self.down,.3,.15,.002,.005)
        self.assertTrue(ok);self.assertEqual(needle,np.inf);self.assertEqual(head,np.inf)

    def test_head_blocked_by_pillar_behind_the_needle(self):
        # The pillar starts beyond the needle (z > 1.3) and is ~0.1 from the needle line.
        mesh=scene([[1.1,.4,1.5],[1.2,.6,2.]]);f=self.top_face(mesh)
        line=R.clearance(mesh.triangles,R.test_points(mesh)[f],self.down,.3)
        xy=R.test_points(mesh)[f][:,:2];lo=np.array([1.1,.4]);hi=np.array([1.2,.6])
        expected=np.linalg.norm(np.maximum(np.maximum(lo-xy,xy-hi),0),axis=1)
        np.testing.assert_allclose(line,expected,atol=1e-12)
        self.assertFalse(R.certify(mesh,f,self.down,.3,float(expected.min())+1e-6,.002,.005)[0])
        self.assertTrue(R.certify(mesh,f,self.down,.3,float(expected.min())-1e-6,.002,.005)[0])

    def test_obstacle_beside_the_needle_only(self):
        # Within the needle length only the line matters, however wide the head.
        mesh=scene([[1.05,.4,1.05],[1.2,.6,1.25]]);f=self.top_face(mesh)
        self.assertTrue(R.certify(mesh,f,self.down,.3,.5,.002,.005)[0])

    def test_line_through_the_object_is_blocked(self):
        mesh=scene([[.2,.2,1.4],[.8,.8,1.5]]);f=self.top_face(mesh)
        self.assertFalse(R.certify(mesh,f,self.down,1.,.01,.002,.005)[0])

    def test_needle_body_rubbing_a_wall_is_blocked(self):
        # A slab 20-25 mm behind the tip, just beside the needle line: the line misses it,
        # a needle body thicker than the gap does not.
        mesh=scene([[1.001,0.,1.02],[1.2,1.,1.025]]);f=self.top_face(mesh)
        gap=float(R.clearance(mesh.triangles,R.test_points(mesh)[f],self.down,.005).min())
        self.assertGreater(gap,0)
        self.assertTrue(R.certify(mesh,f,self.down,.3,.001,gap/2,.005)[0])
        self.assertFalse(R.certify(mesh,f,self.down,.3,.001,gap*2,.005)[0])

    def test_back_face_is_not_reached(self):
        mesh=scene();f=self.top_face(mesh)
        self.assertFalse(R.certify(mesh,f,-self.down,.3,.01,.002,.005)[0])

    def test_screen_agrees_on_open_box(self):
        mesh=scene();screen=R.Screen(mesh,.3,.15,.002,.01,200000)
        reached=screen(self.down)
        np.testing.assert_array_equal(reached,(mesh.face_normals[:,2]>.9))

if __name__=='__main__':
    unittest.main()
