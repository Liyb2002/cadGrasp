"""Physical frame/scale and per-load protection checks for the new model."""
import unittest
from common import *
from model import Layout, Model
from search import lost_protected, Search
from classify import classify
from projection import cone_projection as stable_projection


class Invariants(unittest.TestCase):
    def test_scaled_boolean_transform_matches_rigid_mesh(self):
        mesh=C.trimesh.creation.box(extents=[.04,.02,.03])
        transform=C.trimesh.transformations.rotation_matrix(.61,[.2,.7,.4])
        transform[:3,3]=[.035,-.017,.022]
        a=transform_solid(C.S.solid(mesh),transform)
        b=C.S.solid(transform_mesh(mesh,transform))
        self.assertLess(C.material_volume(a-b)+C.material_volume(b-a),1e-15)

    def test_juxtapose_keeps_native_orientation_and_ground_height(self):
        model=object.__new__(Model)
        a=C.trimesh.transformations.rotation_matrix(.3,[0.,1.,0.]);a[:3,3]=[0.,0.,.06]
        b=C.trimesh.transformations.rotation_matrix(-.5,[1.,0.,0.]);b[:3,3]=[0.,0.,.08]
        model.native=np.array([a,b])
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[0.,0.,1.],[0.,0.,1.]]),np.array([0,1]),(0,1))
        result=model.juxtapose(layout,1,0)
        np.testing.assert_allclose(a @ result.placements[1],b,atol=1e-15)
        frame=tangent_frame(model.floor_normal(result,1))
        result.placements[1,:3,3]+=.04*frame[:,0]
        actual=a @ result.placements[1]
        np.testing.assert_allclose(actual[:3,:3],b[:3,:3],atol=1e-15)
        self.assertAlmostEqual(actual[2,3],b[2,3],places=14)

    def test_wrench_moments_are_invariant_to_task_lateral_shift(self):
        rotation=C.trimesh.transformations.rotation_matrix(.47,[.3,.8,.2])[:3,:3]
        p=np.array([.03,.01,-.04]);com=np.array([.01,-.02,.02]);normal=np.array([.4,.3,.8])
        shift=np.array([.06,-.07,0.])
        np.testing.assert_allclose(np.cross(rotation @ p+shift-(rotation @ com+shift),rotation @ normal),
                                   rotation @ np.cross(p-com,normal),atol=1e-15)

    def test_protection_compares_specific_loads_not_equal_counts(self):
        anchor={'masks':{0:np.array([True,False,True])}}
        trial={'masks':{0:np.array([False,True,True]),1:np.ones(3,dtype=bool)}}
        self.assertEqual(lost_protected(anchor,trial),1)

    def test_layout_key_handles_noncontiguous_active_indices(self):
        layout=Layout(np.repeat(np.eye(4)[None],4,axis=0),np.ones((4,3)),np.arange(4),(0,2,3))
        self.assertEqual(layout.key(),layout.copy().key())
        changed=layout.copy();changed.placements[2,0,3]=.001
        self.assertNotEqual(layout.key(),changed.key())

    def test_layout_cost_uses_physical_centers_not_coordinate_origins(self):
        model=object.__new__(Model)
        model.mesh=C.trimesh.creation.box(extents=[.04,.02,.03])
        model.mesh.apply_translation([.08,-.02,.01])
        q=np.array([np.eye(4),C.trimesh.transformations.rotation_matrix(.8,[0,0,1])])
        q[1,:3,3]=model.mesh.center_mass-q[1,:3,:3] @ model.mesh.center_mass
        layout=Layout(q,np.ones((2,3)),np.arange(2),(0,1))
        self.assertGreater(np.linalg.norm(q[1,:3,3]),.01)
        self.assertLess(model.layout_span(layout),1e-14)

    def test_actual_geometry_slots_include_translation(self):
        from types import SimpleNamespace
        layout=Layout(np.eye(4)[None],np.ones((1,3)),np.array([0]),(0,))
        choices=[]
        for i,kind in enumerate(['direction','direction','translation']):
            trial=layout.copy();trial.placements[0,0,3]=i+1
            choices.append((kind,trial,dict(step_degrees=i+1)))
        model=SimpleNamespace(proxy=lambda layout,targets:dict(loss=layout.placements[0,0,3],sum_loss=0.,span_m=0.))
        selected,_=Search(model,None,finalists=2).shortlist(choices,[])
        self.assertEqual({r[3] for r in selected},{'direction','translation'})

    def test_juxtapose_slots_try_distinct_hosts(self):
        from types import SimpleNamespace
        layout=Layout(np.eye(4)[None],np.ones((1,3)),np.array([0]),(0,))
        choices=[]
        for i,host in enumerate(['pose_2','pose_2','pose_3']):
            trial=layout.copy();trial.placements[0,0,3]=i+1
            choices.append(('juxtapose',trial,dict(guest_index=0,host=host)))
        model=SimpleNamespace(proxy=lambda layout,targets:dict(loss=layout.placements[0,0,3],sum_loss=0.,span_m=0.))
        selected,_=Search(model,None,finalists=2).shortlist(choices,[])
        self.assertEqual({r[5]['host'] for r in selected},{'pose_2','pose_3'})

    def test_batch_primal_classification_matches_independent_original_lps(self):
        rng=np.random.default_rng(7)
        rays=np.vstack([np.eye(7),rng.random((8,7))])
        demands=np.vstack([rng.random((30,7)), -np.ones((2,7))])
        mask,info=classify(rays,demands)
        expected=np.array([C.C.W.solve(rays,target) is not None for target in demands])
        np.testing.assert_array_equal(mask,expected)
        self.assertLessEqual(info['maximum_primal_replay_residual'],2e-9)

    def test_large_opposing_reactions_have_precise_batch_witnesses(self):
        rays=np.eye(7)
        rays[1]=[-1.,1e-7,0.,0.,0.,0.,0.]
        demands=np.ones((20,7));demands[:,0]=np.linspace(-.2,.2,20)
        mask,info=classify(rays,demands)
        self.assertTrue(mask.all())
        self.assertEqual(info['equilibrium_lps'],1)
        self.assertLessEqual(info['maximum_primal_replay_residual'],2e-9)

    def test_unresolved_lp_retries_preserve_original_equilibrium(self):
        from unittest.mock import patch
        rays=np.eye(7)
        rays[1]=[-1.,1e-7,0.,0.,0.,0.,0.]
        demands=np.vstack([np.ones(7),-np.ones(7)])
        with patch.object(C.C.W,'solve',side_effect=RuntimeError('Unresolved equilibrium residual')):
            mask,info=classify(rays,demands)
        np.testing.assert_array_equal(mask,[True,False])
        self.assertEqual(info['numerical_lp_retries'],2)
        self.assertLessEqual(info['maximum_primal_replay_residual'],2e-9)

    def test_polar_projection_matches_independent_nnls(self):
        from scipy.optimize import nnls
        rng=np.random.default_rng(19)
        for _ in range(5):
            rays=rng.normal(size=(18,7));rays[:,2]=abs(rays[:,2])+.1
            target=rng.normal(size=7)
            weights,_=nnls(rays.T,target,maxiter=2000)
            expected=.5*np.sum((weights @ rays-target)**2)
            result=stable_projection(rays,target)
            self.assertAlmostEqual(result['loss'],expected,places=8)
            self.assertLess(result['kkt_max_violation'],1e-7)
            self.assertTrue((result['coefficients']>=0).all())

    def test_polar_projection_handles_duplicate_full_span(self):
        rays=np.vstack([np.eye(7),-np.eye(7),np.eye(7),np.zeros((1,7))])
        result=stable_projection(rays,np.arange(7)-3.)
        self.assertLess(result['loss'],1e-16)
        self.assertLess(result['kkt_max_violation'],1e-7)


if __name__=='__main__':
    unittest.main()
