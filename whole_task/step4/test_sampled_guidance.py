import unittest
import numpy as np
import trimesh
from types import SimpleNamespace
from collections import OrderedDict
from whole_task.step3 import independent_geometry as G
from whole_task.operations.geometry import OperationModel
from whole_task.operations.state import Layout
from .sampled_guidance import swept_contains,probe_indices,SampledObjective


class SampledGeometryTests(unittest.TestCase):
    def test_material_guidance_preserves_old_ribs_and_matches_exact_local_union(self):
        from .sampled_guidance import compose_material,mesh_contains
        parent=G.box([0.,0.,0.],[1.,.2,.1])+G.box([.9,0.,0.],[1.,1.,.1])
        raw=G.box([.1,.1,0.],[.4,.4,.2])+G.box([.2,.2,0.],[.25,.8,.1])
        edit=G.box([.1,.05,-.1],[.45,.45,.3]);protected=G.box([.3,.1,0.],[.5,.3,.2])
        patches=G.box([.2,.3,0.],[.3,.6,.1])
        pts=np.random.default_rng(8).uniform([-.1,-.1,-.05],[1.1,1.1,.25],(2000,3))
        contains=lambda value:mesh_contains(G.mesh(value),pts)
        r,p,e,c,b=map(contains,(raw,parent,edit,protected,patches))
        expected=(parent-(edit-protected))+(raw^(edit-protected))+(raw^patches)
        actual=compose_material(r,p,e,c,b)
        def exact_contains(value):
            # The coplanar Boolean reference has shared triangle edges.
            # Trimesh ray parity retries them in random directions, making
            # this reference fluctuate. Oriented solid angle also preserves
            # cavity shells and is deterministic away from the boundary.
            triangles=G.mesh(value).triangles
            a,b,c=[triangles[None,:,k]-pts[:,None] for k in range(3)]
            la,lb,lc=[np.linalg.norm(v,axis=2) for v in (a,b,c)]
            dot=lambda u,v:np.einsum('nfi,nfi->nf',u,v)
            denominator=la*lb*lc+dot(a,b)*lc+dot(b,c)*la+dot(c,a)*lb
            winding=2*np.arctan2(dot(a,np.cross(b,c)),denominator).sum(1)
            return np.abs(winding)>2*np.pi
        np.testing.assert_array_equal(actual,exact_contains(expected))
        self.assertGreater(np.count_nonzero(actual&~r),0)
        retained=parent^protected
        expected=(raw^parent)+retained+(raw^((edit+patches)-protected))
        np.testing.assert_array_equal(compose_material(r,p,e,c,b,'free_material',contains(retained)),exact_contains(expected))

    def test_interior_probes_do_not_invent_contacts_on_leading_box_faces(self):
        from .sampled_guidance import surface_probes
        body=trimesh.creation.box([.1,.12,.08]);triangles=body.triangles
        direction=np.array([1.,2.,3.]);direction/=np.linalg.norm(direction)
        normals=np.repeat(body.face_normals,4,axis=0)
        leading=np.repeat(body.face_normals@direction>0.,4)
        corners=np.concatenate([triangles,triangles.mean(1)[:,None]],axis=1).reshape(-1,3)
        old_keep=~swept_contains(body,direction,1.,corners+1e-5*normals)
        new_keep=~swept_contains(body,direction,1.,surface_probes(triangles)+1e-5*normals)
        self.assertTrue((old_keep&leading).any())
        self.assertFalse((new_keep&leading).any())
        self.assertTrue(new_keep[~leading].all())

    def test_actual_failed_member_is_adjusted_without_touching_an_outside_family(self):
        from .gradient_repair import repair_owners
        model=SimpleNamespace(ids=list('ABCDE'),repair_endpoint_indices={'C':[17],'E':[3]},
            repair_pressure_indices={'D':[4]})
        self.assertEqual(repair_owners(model,[0,1],[0,1,2,3]),[0,1,2,3])

    def test_illegal_exit_does_not_receive_guidance_coverage(self):
        objective=SampledObjective.__new__(SampledObjective)
        objective.cache=OrderedDict();objective.calls=0
        objective.model=SimpleNamespace(check_deadline=lambda:None)
        q=Layout(np.eye(4)[None],np.eye(4)[None],np.array([[1.,0.,0.]]))
        self.assertIsNone(objective.evaluate(q))
    def test_oblique_ray_occupancy_matches_analytic_swept_box(self):
        body=trimesh.creation.box([.1,.12,.08])
        body.apply_transform(trimesh.transformations.translation_matrix([.1,.2,.3]))
        direction=np.array([1.,2.,3.]);direction/=np.linalg.norm(direction)
        low=np.minimum(body.bounds[0],body.bounds[0]+.4*direction)-.03
        high=np.maximum(body.bounds[1],body.bounds[1]+.4*direction)+.03
        points=np.random.default_rng(17).uniform(low,high,(1500,3))
        enter=((points-body.bounds[1])/direction).max(1)
        leave=((points-body.bounds[0])/direction).min(1)
        expected=np.maximum(enter,0)<np.minimum(leave,.4)
        np.testing.assert_array_equal(swept_contains(body,direction,.4,points),expected)

    def test_nonconvex_notch_is_preserved_in_sweep_guidance(self):
        body=G.mesh(G.box([-.1,-.1,.2],[0.,.1,.3])+G.box([0.,-.1,.2],[.1,0.,.3]))
        points=np.random.default_rng(17).uniform([-.14,-.14,.17],[.14,.14,.73],(1500,3))
        expected=(np.abs(points[:,:2])<.1).all(1)&((points[:,0]<0)|(points[:,1]<0))&(points[:,2]>.2)&(points[:,2]<.7)
        np.testing.assert_array_equal(swept_contains(body,np.array([0.,0.,1.]),.4,points),expected)
        point=np.array([[.05,.05,.5]])
        self.assertFalse(swept_contains(body,np.array([0.,0.,1.]),.4,point)[0])

    def test_actual_failed_full_endpoint_is_added_to_stratified_guidance(self):
        model=OperationModel.__new__(OperationModel)
        model.tolerance=1e-10;model.evaluations=0;model.path_cache={}
        support=G.mesh(G.box([-1.,-1.,0.],[1.,1.,.2]))
        body=G.mesh(G.box([-.1,-.1,.2],[.1,.1,.4]))
        targets=np.zeros((100,6));targets[:,2]=1.;targets[37,0]=1.
        task=dict(id='C',index=0,body=body,body_world=body,material=np.eye(4),
            domain=SimpleNamespace(work_ids=np.array([],int),com=body.center_mass),
            endpoints=dict(endpoint_need_wrench=targets),work=G.box([4.,4.,0.],[5.,5.,1.]),
            direction=np.array([0.,0.,1.]),length=1.)
        layout=Layout(np.eye(4)[None],np.eye(4)[None],np.array([[0.,0.,1.]]))
        candidate=dict(layout=layout,support=support,tasks=[task])
        with self.assertRaisesRegex(ValueError,'reduced endpoint rejection'):
            model.audit(candidate,full=False,continuous=False,all_outer=True)
        self.assertNotIn(37,probe_indices(100,24))
        self.assertIn(37,probe_indices(100,24,model.repair_endpoint_indices['C']))
        self.assertTrue((probe_indices(35,24,model.repair_endpoint_indices['C'])<35).all())

    def test_pressure_projection_moves_only_participants_tangent_to_fixed_floors(self):
        from unittest.mock import patch
        from scipy.spatial.transform import Rotation
        from .gradient_repair import global_pressure_feasible,project_pressure
        q=Layout(np.tile(np.eye(4),(2,1,1)),np.tile(np.eye(4),(2,1,1)),np.tile([0.,0.,1.],(2,1)))
        q.material[1,:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix();q.material[1,2,3]=.01
        model=SimpleNamespace(records=[{},{}],world_data=lambda *a:dict(
            endpoints=dict(endpoint_need_wrench=None),domain=SimpleNamespace(com=None)))
        pressure=np.array([[-.001,0.],[.001,0.],[0.,.001]])
        with patch('whole_task.step4.gradient_repair.G.pressure_points',return_value=pressure):
            self.assertFalse(global_pressure_feasible(model,q))
            projected=project_pressure(model,q,[0])
            self.assertIsNotNone(projected)
            self.assertTrue(global_pressure_feasible(model,projected))
        np.testing.assert_array_equal(projected.material[1],q.material[1])
        self.assertEqual(projected.material[0,2,3],q.material[0,2,3])


if __name__=='__main__':unittest.main()
