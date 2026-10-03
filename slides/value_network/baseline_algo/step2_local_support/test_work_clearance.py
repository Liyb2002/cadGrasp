"""Regressions for early rejection of boundaries, backing solids and unknowns."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import circles as P, work_volume as W, work_clearance as C
from step2_local_support.test_work_volume import box


class ContactClearanceTests(unittest.TestCase):
    def test_omitted_access_volume_admits_the_same_previously_blocked_head(self):
        mesh=trimesh.Trimesh([[0,0,1],[1,0,1],[0,1,1]],[[0,1,2]],process=False)
        work=W.WorkVolume(mesh.triangles+np.array([0,0,.05]),[[0,0,1]],[17],30.,1.)
        patch={0:mesh.triangles[0]}
        self.assertFalse(C.ContactClearance(mesh,.1,work).check(patch)['passed'])
        check=C.ContactClearance(mesh,.1,None).check(patch)
        self.assertTrue(check['passed'])
        self.assertFalse(check['enforced']);self.assertFalse(check['verified'])
        self.assertTrue(C.apply_constraint(dict(valid=True,status='valid'),check)['valid'])

    def test_shared_work_edge_rejects_neighbor_without_modifying_its_patch(self):
        mesh=trimesh.Trimesh([[0,0,1],[1,0,1],[0,1,1],[1,1,1]],[[0,1,2],[1,3,2]],process=False)
        work=W.WorkVolume(mesh.triangles[[0]],[[0,0,1]],[0],30.,1.)
        polygons={1:mesh.triangles[1].copy()};before=polygons[1].copy()
        result=C.ContactClearance(mesh,.01,work).check(polygons)
        self.assertFalse(result['passed'])
        self.assertEqual(result['classification'],'circular_cone_intersection')
        self.assertFalse(result['interface_check']['passed'])
        np.testing.assert_array_equal(polygons[1],before)
        eligible=C.apply_constraint(dict(valid=True,status='valid',radius_m=.7),result)
        self.assertFalse(eligible['valid']);self.assertEqual(eligible['radius_m'],.7)

    def test_clear_interface_does_not_exempt_a_blocked_finite_backing(self):
        mesh=trimesh.Trimesh([[0,0,1],[1,0,1],[0,1,1]],[[0,1,2]],process=False)
        work=W.WorkVolume(mesh.triangles+np.array([0,0,.05]),[[0,0,1]],[17],30.,1.)
        self.assertTrue(work.check_surface(mesh.triangles)['passed'])
        self.assertFalse(C.ContactClearance(mesh,.1,work).check({0:mesh.triangles[0]})['passed'])
        self.assertTrue(C.ContactClearance(mesh,.01,work).check({0:mesh.triangles[0]})['passed'])

    def test_intersecting_bound_cannot_reject_a_clear_nonconvex_union(self):
        work=W.WorkVolume([[[-.01,-.01,0],[.01,-.01,0],[0,.01,0]]],[[0,0,1]],[0],30.,1.)
        mesh=box([0,0,-1])
        result=C.ContactClearance(mesh,.01,work).check_parts([box([-2,0,.1]),box([2,0,.1])])
        self.assertTrue(result['passed'])
        self.assertEqual(result['method'],'actual_joined_head_cells')

    def test_unresolved_clearance_is_ineligible(self):
        for classification in ['solver_unresolved','visibility_or_envelope_unresolved']:
            result=C.apply_constraint(dict(valid=True,status='valid'),dict(passed=False,classification=classification))
            self.assertFalse(result['valid']);self.assertEqual(result['status'],'work_volume_unresolved')

    def test_cone_or_reachability_change_invalidates_geometry_signature(self):
        data=dict(geometry=dict(vertices_m=[],faces=[],work_face_ids=[]),frame=dict(moment_origin_m=[0,0,0]),
                  load=dict(cone_half_deg=30.,max_force=1.),reachability=dict(ray_offset_m=1e-5))
        initial=P.geometry_signature(SimpleNamespace(data=data))
        for group,key,value in [('load','cone_half_deg',20.),('reachability','ray_offset_m',1e-6)]:
            changed=copy.deepcopy(data);changed[group][key]=value
            self.assertNotEqual(initial,P.geometry_signature(SimpleNamespace(data=changed)))


if __name__=='__main__':unittest.main()
