"""Regression tests for real sharing and all-pose descent semantics."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import trimesh
from step3_scheculer import shared_dsl as S, contact_dsl as D
from step2_local_support import geometry as G
from step4_connect_support.fixture_view import cells_for


def tasks():
    mesh=trimesh.creation.box(extents=[1.,1.,1.]);mesh.apply_translation([0,0,.5])
    angle=np.pi/6
    rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    result=[]
    for k,r in enumerate([np.eye(3),rotation]):
        transform=np.eye(4);transform[:3,:3]=r;transform[:3,3]=[k*.3,k*.1,0.]
        view=mesh.copy();view.apply_transform(transform)
        result.append(SimpleNamespace(pose=f'pose_{k+1}',floor=np.array([0.,0.,0.]),scale=np.ones(6),
            targets=np.zeros((2,6)),supply=lambda cs:np.eye(7),
            domain=SimpleNamespace(mesh=view,com=view.center_mass,
                work_ids=np.array([4+k],int),data={'frame':{'T_world_mesh':transform.tolist()}})))
    return result


class SharingTests(unittest.TestCase):
    def compiler(self):
        return S.JointCompiler(tasks(),dict(device='cpu',gpu_iterations=600))

    def test_same_installed_contact_and_solid_after_reorientation(self):
        c=self.compiler();p=D.Program((D.Patch(7,(.5,0.,.5),.08),))
        canonical=c.compile(p);self.assertIsNotNone(canonical)
        place=S.placement(c.tasks);depth=.01*c.scale
        native=cells_for(canonical[0],c.tasks[0].domain,G.vertex_offsets(c.tasks[0].domain.mesh,depth)[0])
        for task,view,b,o in zip(c.tasks,c.views,np.asarray(place['bases']),np.asarray(place['offsets'])):
            contact=view.compile(p)[0]
            self.assertEqual(contact['candidate_id'],'SH0007')
            np.testing.assert_allclose(contact['triangles_m']@b+o,canonical[0]['triangles_m'],atol=1e-12)
            cells=cells_for(contact,task.domain,G.vertex_offsets(task.domain.mesh,depth)[0])
            for actual,expected in zip(cells,native):np.testing.assert_allclose(actual@b+o,expected,atol=1e-12)

    def test_work_faces_are_excluded_for_every_pose(self):
        c=self.compiler()
        self.assertFalse(set(c.canonical.polygons)&set(np.concatenate([t.domain.work_ids for t in c.tasks])))

    def test_global_delete_removes_one_physical_head_from_every_view(self):
        c=self.compiler();p=D.Program((D.Patch(7,(.5,0.,.5),.05),D.Patch(8,(-.5,0.,.5),.05)))
        for view in c.views:
            self.assertEqual([r['candidate_id'] for r in view.compile(p.delete(0))],['SH0008'])
        # Count is one, not one per pose; the joint objective has one count term.
        values=c.loss(p.delete(0),[0])[1]
        self.assertAlmostEqual(values['head_count'],.002)
        self.assertEqual(len(values['per_pose_force']),2)

    def test_second_pose_load_changes_joint_loss_and_gradient(self):
        c=self.compiler();p=D.Program((D.Patch(7,(.5,0.,.5),.05),))
        calls=[]
        def solve(rays,targets):
            calls.append(targets.copy())
            return np.full((len(rays),len(targets)),float(targets[0,0]))
        with patch.object(c.backend,'solve',side_effect=solve):
            first=c.loss(p,[0])[0]
            c.tasks[1].targets[0,0]=2.
            c.force_cache.clear()
            second=c.loss(p,[0])[0]
        self.assertGreater(second,first)
        self.assertEqual(len(calls),4)
        self.assertEqual(calls[-1][0,0],2.)

    def test_descent_repairs_shared_parameter_for_nonfirst_pose(self):
        c=self.compiler();p=D.Program((D.Patch(7,(.5,.2,.5),.05),))
        c.tasks[0].supply=lambda cs:np.zeros((1,7))
        c.tasks[1].supply=lambda cs:np.array([[0.,cs[0]['center_m'][1],0.,0.,0.,0.,0.]])
        def solve(rays,targets):
            return np.array([[float(r[0,1]**2)]*len(targets) for r in rays])
        with patch.object(c.backend,'solve',side_effect=solve):
            initial=c.loss(p,[0])[0]
            q,history=D.descend(c,p,[0],steps=4)
            self.assertLess(c.loss(q,[0])[0],initial)
            self.assertTrue(any(h['accepted'] for h in history))
            self.assertNotEqual(q.patches[0].center,p.patches[0].center)

    def test_false_registration_cannot_make_heads_shared_by_label(self):
        data=tasks()
        data[1].domain.mesh.apply_translation([.1,0,0])
        with self.assertRaises(AssertionError):S.placement(data)

    def test_force_cache_cannot_restore_an_old_shared_identity(self):
        from step3_scheculer.run_dsl import Search
        from step3_scheculer import run_dsl as R
        search=Search.__new__(Search)
        first=dict(candidate_id='SH0001',triangles_m=np.zeros((1,3,3)),source_faces=np.array([0]))
        current=dict(first,candidate_id='SH0002')
        search.compiler=SimpleNamespace(compile=lambda p:[current])
        search.task=SimpleNamespace(targets=np.zeros((32768,6)),supply=lambda cs:np.eye(7))
        import hashlib
        key=hashlib.sha256(first['triangles_m'].tobytes()+first['source_faces'].tobytes()).hexdigest()
        search.lp_cache={key:dict(contacts=[first],passed=True,covered=32768,mask=np.ones(32768,bool),info={})}
        row=search.classify(D.Program((D.Patch(2,(0,0,0),.1),)))
        self.assertEqual(row['contacts'][0]['candidate_id'],'SH0002')

    def test_closeness_is_soft_and_perpendicular_exits_remain_allowed(self):
        self.assertAlmostEqual(D.alignment([[1,0,0],[0,1,0]]),1.)
        c=self.compiler();p=D.Program((D.Patch(7,(.5,0.,.5),.05),))
        value,terms=c.loss(p,[0],exits=True)
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(terms['alignment'],0.)

if __name__=='__main__':unittest.main()
