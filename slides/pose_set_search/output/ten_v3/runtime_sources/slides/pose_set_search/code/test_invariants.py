"""Physical frame/scale and per-load protection checks for the new model."""
import unittest
from common import *
from model import Layout, Model
from search import lost_protected, Search


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


if __name__=='__main__':
    unittest.main()
