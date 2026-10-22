"""Trace actual surface edges, including depth and shared-edge ray contacts."""
import unittest
import numpy as np
import trimesh

from .task_boundary import boundary_loops,sample_loop,candidate_region
from .find_tasks_video import contact
from .workspace import points_inside_box,box_edges


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.Trimesh([[0,0,0],[1,0,.2],[1,1,.8],[0,1,.3]],
                                  [[0,1,2],[0,2,3]],process=False)

    def test_find_tasks_displays_full_candidate_before_exclusive_assignment(self):
        arrays=dict(reach=np.array([[True],[True],[False]]),target=np.ones(3,bool),
            surface_inside_box=np.ones((3,1),bool),face_task=np.array([0,1,-1]))
        np.testing.assert_array_equal(candidate_region(arrays,0),[True,True,False])
        arrays['surface_inside_box'][1,0]=False
        with self.assertRaisesRegex(ValueError,'outside the 3D work box'):candidate_region(arrays,0)

    def test_nonplanar_boundary_omits_internal_diagonal_and_closes(self):
        loops=boundary_loops(self.mesh,[True,True]);self.assertEqual(len(loops),1)
        loop=loops[0];vertices=loop['vertex_ids']
        self.assertEqual(vertices[0],vertices[-1]);self.assertEqual(len(vertices),5)
        edges={tuple(sorted(e)) for e in zip(vertices[:-1],vertices[1:])}
        self.assertEqual(edges,{(0,1),(1,2),(2,3),(0,3)})
        self.assertGreater(np.ptp(self.mesh.vertices[vertices,2]),.7)

    def test_sampling_visits_every_corner_in_3d_and_stays_on_its_owner_face(self):
        loop=boundary_loops(self.mesh,[True,True])[0]
        axes=np.array([[0.,0.,-1.],[0.,-.1,-1.]])
        axes/=np.linalg.norm(axes,axis=1)[:,None]
        path=sample_loop(self.mesh,loop,axes,37)
        self.assertEqual(len(path['points']),38)
        self.assertEqual(set(path['edge_indices']),set(range(4)))
        np.testing.assert_array_equal(path['points'][0],path['points'][-1])
        np.testing.assert_allclose(np.linalg.norm(path['axes'],axis=1),1)
        for point,edge,u,owner in zip(path['points'],path['edge_indices'],path['edge_parameters'],path['faces']):
            a,b=loop['vertex_ids'][edge:edge+2]
            np.testing.assert_allclose(point,(1-u)*self.mesh.vertices[a]+u*self.mesh.vertices[b])
            self.assertTrue({int(a),int(b)}<=set(self.mesh.faces[owner]))
        for vertex in loop['vertex_ids']:
            self.assertLess(np.linalg.norm(path['points']-self.mesh.vertices[vertex],axis=1).min(),1e-12)

    def test_hole_and_outer_contour_are_both_returned(self):
        vertices=[[x,y,0.] for x,y in [(-2,-2),(2,-2),(2,2),(-2,2),(-1,-1),(1,-1),(1,1),(-1,1)]]
        faces=[]
        for i in range(4):
            j=(i+1)%4;faces.extend([[i,j,4+j],[i,4+j,4+i]])
        mesh=trimesh.Trimesh(vertices,faces,process=False)
        loops=boundary_loops(mesh,np.ones(8,bool))
        self.assertEqual(len(loops),2)
        np.testing.assert_allclose([loop['length_m'] for loop in loops],[16,8])

    def test_lobes_touching_at_a_vertex_keep_every_boundary_edge(self):
        mesh=trimesh.Trimesh([[0,0,0],[1,0,0],[0,1,0],[-1,0,.2],[0,-1,.2]],
                            [[0,1,2],[0,3,4]],process=False)
        loops=boundary_loops(mesh,[True,True]);self.assertEqual(len(loops),1)
        ids=loops[0]['vertex_ids'];self.assertEqual(ids[0],ids[-1])
        edges=[tuple(sorted(e)) for e in zip(ids[:-1],ids[1:])]
        self.assertEqual(len(edges),6);self.assertEqual(len(set(edges)),6)
        self.assertEqual(set(edges),{(0,1),(1,2),(0,2),(0,3),(3,4),(0,4)})
        self.assertEqual(set(loops[0]['face_ids']),{0,1})

    def test_surface_edge_is_first_hit_instead_of_far_side(self):
        front=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        back=front+[0,0,-1.]
        # A floating-point offset from the exact shared boundary must not skip it.
        point=np.array([.5,.5+1e-14,0.]);direction=np.array([0.,0.,-1.])
        face,hit=contact(np.array([front,back]),point-.4*direction,direction)
        self.assertEqual(face,0);np.testing.assert_allclose(hit,point,atol=1e-12)

    def test_box_has_eight_3d_corners_and_rejects_same_xy_outside_depth(self):
        bounds=np.array([[-1,-2,-3],[1,2,3]])
        self.assertEqual(np.unique(np.asarray(box_edges(bounds)).reshape(-1,3),axis=0).shape,(8,3))
        np.testing.assert_array_equal(points_inside_box([[0,0,0],[0,0,4],[0,0,-4]],bounds),[True,False,False])


if __name__=='__main__':unittest.main()
