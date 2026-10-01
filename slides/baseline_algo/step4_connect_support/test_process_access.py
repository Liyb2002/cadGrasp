"""Regression checks for Step3-compatible working-surface exclusions."""
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh

from step4_connect_support import process_access as A, working_surface as W


def task():
    mesh=trimesh.creation.box(extents=[1,1,1])
    mesh.apply_translation([0,0,-.5])
    ids=np.flatnonzero(mesh.face_normals[:,2]>.9)
    return SimpleNamespace(pose='pose_1',
        domain=SimpleNamespace(mesh=mesh,work_ids=ids))


class WorkingSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.triangle=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        self.task=task()

    def test_shared_edge_allowed(self):
        neighbor=np.array([[0.,0,0],[1,0,0],[.5,-1,0]])
        result=W.intersection(self.triangle,neighbor,1.)
        self.assertTrue(result['passed'])
        self.assertEqual(result['classification'],'shared_edge_or_vertex_only')

    def test_shared_vertex_allowed(self):
        neighbor=np.array([[0.,0,0],[-1,0,0],[0,-1,1]])
        self.assertTrue(W.intersection(self.triangle,neighbor,1.)['passed'])

    def test_small_coplanar_overlap_away_from_centroid_rejected(self):
        patch=np.array([[.01,.01,0],[.02,.01,0],[.01,.02,0]])
        result=W.intersection(self.triangle,patch,1.)
        self.assertFalse(result['passed'])
        self.assertGreater(result['minimum_work_edge_distance_m'],.001)
        self.assertLess(result['equality_residual_m'],1e-12)

    def test_transverse_interior_crossing_rejected(self):
        patch=np.array([[.2,.2,-1],[.2,.2,1],[.4,.2,0]])
        self.assertFalse(W.intersection(self.triangle,patch,1.)['passed'])

    def test_positive_gap_allowed(self):
        self.assertTrue(W.intersection(self.triangle,self.triangle+[0,0,1e-5],1.)['passed'])

    def test_thin_interior_overlap_is_not_mislabelled_shared_edge(self):
        patch=np.array([[.1,0,0],[.9,0,0],[.5,1e-6,0]])
        result=W.intersection(self.triangle,patch,1.)
        self.assertFalse(result['passed'])
        self.assertGreater(result['minimum_work_edge_distance_m'],W.TOLERANCE_M)

    def test_owner_contacts_use_exact_step3_work_face_mask(self):
        mesh=self.task.domain.mesh
        nonwork=int(np.flatnonzero(mesh.face_normals[:,2]<.5)[0])
        work=int(self.task.domain.work_ids[0])
        contacts=[dict(candidate_id=str(i),triangles_m=mesh.triangles[[i]],source_faces=[i])
            for i in (nonwork,work)]
        rows=A.contact_checks([self.task],[contacts],owner_only=True)
        self.assertTrue(rows[0]['passed'])
        self.assertFalse(rows[1]['passed'])

    def test_closed_support_containing_work_face_rejected(self):
        body=trimesh.creation.box(extents=[2,2,2])
        result=W.check(body,self.task)
        self.assertFalse(result['passed'])
        self.assertTrue(result['contained_work_face_ids'])

    def test_support_above_work_face_does_not_enable_ray_gate(self):
        body=trimesh.creation.box(extents=[.2,.2,.2])
        body.apply_translation([0,0,.5])
        guard=A.Guard.__new__(A.Guard)
        guard.case=SimpleNamespace(tasks=[self.task])
        guard.bases=np.eye(3)[None];guard.offsets=np.zeros((1,3))
        guard.contact_checks=[];guard.root_checks=[]
        result=guard.verify(body)
        self.assertTrue(result['passed'])
        self.assertFalse(result['processing_ray_volume_enforced'])

    def test_idle_contact_on_work_face_rejected(self):
        work=int(self.task.domain.work_ids[0])
        triangle=self.task.domain.mesh.triangles[[work]]
        own=dict(candidate_id='idle',triangles_m=triangle+[0,0,1],source_faces=[0])
        tasks=[self.task,SimpleNamespace(pose='pose_2',domain=self.task.domain)]
        bases=np.repeat(np.eye(3)[None],2,axis=0)
        offsets=np.array([[0.,0,0],[0,0,-1.]])
        rows=A.contact_checks(tasks,[[],[own]],bases,offsets)
        row=next(r for r in rows if r['pose']=='pose_1')
        self.assertFalse(row['passed'])
        self.assertFalse(row['owner_contact'])


if __name__=='__main__':
    unittest.main()
