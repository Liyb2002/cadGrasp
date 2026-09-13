"""Separate head-to-ground solids, opposing directions and installation order."""
from pathlib import Path
import sys
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step5_connect_support import layout as L,solids as S,motion as M,ring as R
from step5_connect_support.surface_check import surface_distances
from step2_local_support.surface import areas
from step2_local_support import insertion as D


def contact(mesh,normal,index):
    face=int(np.argmax(mesh.face_normals@normal));tri=mesh.triangles[face];center=tri.mean(axis=0)
    tri=(center+.3*(tri-center))[None]
    return dict(candidate_id=f'C{index}',candidate_index=index,center_m=center,center_face=face,
        radius_m=float(np.linalg.norm(tri-center,axis=2).max()),source_faces=np.array([face]),
        triangles_m=tri,triangle_areas_m2=areas(tri))


class ConnectionTests(unittest.TestCase):
    def test_two_separate_supports_enter_from_opposite_sides_and_enclose_ground_demand(self):
        mesh=trimesh.creation.box([1., 1., 1.]);mesh.apply_translation([0, 0, 1.])
        contacts=[contact(mesh,[-1,0,0],0),contact(mesh,[1,0,0],1)]
        directions=[dict(certified_directions=D.normalize(isolated=[a])) for a in [0.,180.]]
        points=np.array([[-.6,-.6],[.6,-.6],[.6,.6],[-.6,.6]])
        result,modules=L.search(mesh,contacts,directions,points,np.zeros(3),.01)
        self.assertTrue(result['passed'],result)
        self.assertEqual(len(modules),2)
        self.assertEqual(result['bearings_deg'],[0.,180.])
        self.assertEqual(result['installation']['installation_order'],[0,1])
        self.assertTrue(result['installation']['final_supports_disjoint'])
        self.assertTrue(result['ring_coverage']['passed'])
        self.assertEqual(result['ring']['expansion'],1.)
        for module,patch in zip(modules,contacts):
            self.assertTrue(module['solid']['one_solid'])
            self.assertLess(surface_distances(module['joined'],patch['triangles_m'].reshape(-1,3)).max(),1e-10)

    def test_isolated_angles_are_not_lost_by_a_regular_grid(self):
        allowed=D.normalize([[14.01,14.02]],isolated=[2.499999999999986])
        candidates=L.candidate_angles(allowed)
        self.assertIn(2.499999999999986,candidates)
        self.assertIn(14.015,candidates)
        self.assertTrue(all(D.contains(allowed,a) for a in candidates))

    def test_precedence_can_require_reverse_order_and_detect_cycles(self):
        self.assertEqual(L.installation_order(3,[(2,0),(0,1)]),[2,0,1])
        self.assertIsNone(L.installation_order(2,[(0,1),(1,0)]))

    def test_individually_clear_paths_can_have_no_joint_installation_order(self):
        obstacle=trimesh.creation.box([.2, .2, .2]);obstacle.apply_translation([0, 3, 1])
        modules=[]
        for x,a in [(1.,[1.,0,0]),(-1.,[-1.,0,0])]:
            solid=trimesh.creation.box([.2, .2, .4]);solid.apply_translation([x, 0, .2])
            modules.append(dict(joined=solid,parts=[solid],plan=dict(direction=np.array(a))))
        result=L.check_assembly(obstacle,modules)
        self.assertTrue(all(r['passed'] for r in result['object_sweeps']))
        self.assertTrue(result['final_supports_disjoint'])
        self.assertEqual(set(map(tuple,result['precedence_edges'])),{(0,1),(1,0)})
        self.assertIsNone(result['installation_order'])
        self.assertFalse(result['passed'])

    def test_ring_arcs_can_touch_but_cannot_overlap(self):
        pad=np.array([[0,0],[1,0],[1,1],[0,1]])
        first=dict(ground_polygons_xy_m=[pad])
        self.assertTrue(L.pads_separate([first,dict(ground_polygons_xy_m=[pad+[1,0]])],1e-12))
        self.assertFalse(L.pads_separate([first,dict(ground_polygons_xy_m=[pad+[.9,0]])],1e-12))

    def test_complete_hull_coverage_does_not_replace_complete_ring_coverage(self):
        points=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        ring=R.make(points,1.,1.)
        plans=[dict(ground_polygons_xy_m=[p]) for p in ring['edge_strips_xy_m']]
        self.assertTrue(R.check(ring,plans,1.)['passed'])
        incomplete=plans[:-1]
        for plan in incomplete:
            xy=np.concatenate(plan['ground_polygons_xy_m'])
            plan['ground_corners_m']=np.c_[xy,np.zeros(len(xy))]
        self.assertTrue(L.ground_check(incomplete,points,np.zeros(3),1.)['passed'])
        self.assertFalse(R.check(ring,incomplete,1.)['passed'])

    def test_boundary_ray_uses_withdrawal_and_can_miss_from_outside(self):
        square=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        ring=R.make(square,1.,1.)
        np.testing.assert_allclose(R.ray_exit(ring,[0,0],[-1,0])[0],[-1,0])
        self.assertIsNone(R.ray_exit(ring,[-2,0],[-1,0]))

    def test_duplicate_contact_heads_remain_overlapped_after_depth_search(self):
        mesh=trimesh.creation.box([1., 1., 1.]);mesh.apply_translation([0, 0, 1.])
        first=contact(mesh,[-1,0,0],0);second={**first,'candidate_id':'C1','candidate_index':1}
        directions=[dict(certified_directions=D.normalize(isolated=[0.]))]*2
        points=np.array([[-.6,-.6],[.6,-.6],[.6,.6],[-.6,.6]])
        result,modules=L.search(mesh,[first,second],directions,points,np.zeros(3),.01)
        self.assertEqual(result['status'],'no_feasible_contact_backing')
        self.assertFalse(result['passed'])
        self.assertGreater(result['head_precheck']['collisions'][0]['head_intersection_m3'],0)
        self.assertEqual(result['attempts'],[])

    def test_clear_endpoint_does_not_prove_clear_trajectory(self):
        mesh=trimesh.creation.box([1, 1, 1]);mesh.apply_translation([0, 0, 1.])
        part=trimesh.creation.box([.2, .2, .2]);part.apply_translation([1, 0, 1.])
        result=M.sweep_check(mesh,[part],np.array([1.,0.,0.]),3.)
        self.assertLess(result['per_part_final_intersection_m3'][0],1e-12)
        self.assertFalse(result['passed'])

    def test_surface_distance_preserves_thin_faces(self):
        shift=np.array([.04,-.08,.1]);vertices=np.array([[0.,0,0],[.1,0,0],[.1,1e-8,0]])+shift
        mesh=trimesh.Trimesh(vertices,[[0,1,2]],process=False);center=vertices.mean(axis=0)
        np.testing.assert_allclose(surface_distances(mesh,[center,center+[0,0,1e-5]]),[0.,1e-5],atol=1e-16)


if __name__=='__main__':unittest.main()
