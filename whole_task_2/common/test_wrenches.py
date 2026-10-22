"""Paired mechanics remain usable without an independent pose/load stage."""
import unittest
import numpy as np
import trimesh
from .wrenches import build_domain,demand,sample_domain,endpoint_demands
from .task_loads import TaskLoads
from .transforms import rigid_transform


def domain(model='local_inward_normal_cone',transform=None):
    mesh=trimesh.creation.box(extents=[.1,.1,.1])
    ids=np.flatnonzero(mesh.face_normals[:,2]>.9)
    transform=rigid_transform(np.eye(3),[0,0,.2]) if transform is None else transform
    pose=dict(id='example',world_from_object=transform,working_face_ids=ids,
        filled_face_ids=[],status='current_search_pose',geometry_audit=dict(geometrically_passed_face_ids=ids))
    station=dict(world_gravity_m_s2=[0,0,-9.81],tool_to_surface_axis_world=[0,0,-1])
    physical=dict(mass_kg=.3,com_mesh_frame=mesh.center_mass)
    policy=dict(force_direction_model=model,direction_half_angle_deg=25 if model!='fixed_world_needle_axis' else 0,
        reaction_target_order=['Fx','Fy','Fz','tau_x','tau_y','tau_z'],
        maximum_processing_force_mg=.5,continuous_cover=dict(azimuth_sides=16,polar_intervals=2),
        visibility_ray_offset_m=1e-5,ray_batch_size=128)
    return build_domain(mesh,pose,station,physical,policy)


class WrenchMathTests(unittest.TestCase):
    def test_known_force_and_moment_sign(self):
        np.testing.assert_array_equal(demand([0,2,0],[1,0,0],[0,0,0],[0,0,-1]),[-1,0,1,0,0,2])

    def test_translation_keeps_com_wrench(self):
        a=domain();b=domain(transform=rigid_transform(np.eye(3),[.4,-.2,.7]))
        np.testing.assert_allclose(a.evaluate(0,.2,.3,.4,False)['need_wrench'],
            b.evaluate(0,.2,.3,.4,False)['need_wrench'],atol=1e-14)

    def test_world_fixed_axis_does_not_rotate_with_the_object(self):
        t=rigid_transform([[0,-1,0],[1,0,0],[0,0,1]],[.1,.2,.3])
        a=domain('fixed_world_needle_axis',t);row=a.evaluate(0,.2,.3,.4,False)
        np.testing.assert_allclose(row['force_push_mg'],[0,0,-.4],atol=1e-14)
        np.testing.assert_allclose(row['need_wrench'][3:],-np.cross(row['pt_m']-a.com,row['force_push_mg']),atol=1e-14)

    def test_in_memory_adapter_recomputes_paired_targets(self):
        d=domain();s=sample_domain(d,64,17)
        task=TaskLoads('example',d,s['targets_scaled'],s['wrench_scale'],s['targets_7d'],s['source_face_id'],
            s['parameters'],s['work_face_index'],s['geometrically_screened'],float(s['weight_per_sample']))
        np.testing.assert_allclose(task.forward_nodes(np.arange(64)),s['targets_7d'],atol=1e-13)
        np.testing.assert_allclose(s['need_wrench_SI'],s['need_wrench']*.3*9.81,atol=1e-13)

    def test_continuous_outer_targets_retain_gravity_and_pairing(self):
        d=domain();e=endpoint_demands(d)
        np.testing.assert_array_equal(e['endpoint_need_wrench'][0],np.r_[-d.gravity,np.zeros(3)])
        force=-e['endpoint_need_wrench'][1:,:3]-d.gravity
        point=d.mesh.vertices[e['source_vertex_ids'][1:]]
        np.testing.assert_allclose(e['endpoint_need_wrench'][1:,3:],-np.cross(point-d.com,force),atol=1e-14)

    def test_world_cone_ignores_rotated_surface_normals(self):
        t=rigid_transform([[0,0,1],[0,1,0],[-1,0,0]],[.1,.2,.3])
        d=domain('world_needle_direction_cone',t)
        sample=sample_domain(d,256,17,check_visibility=False)
        unit=sample['force_push_mg']/np.linalg.norm(sample['force_push_mg'],axis=1)[:,None]
        self.assertTrue((unit@d.axis>=np.cos(np.deg2rad(25))-1e-14).all())
        self.assertTrue((unit@d.inward[0]<.5).all())
        np.testing.assert_allclose(sample['need_wrench'][:,3:],
            -np.cross(sample['pt_m']-d.com,sample['force_push_mg']),atol=1e-14)

    def test_world_cone_outer_hull_covers_interior_pairs(self):
        from scipy.optimize import linprog
        d=domain('world_needle_direction_cone')
        outer=endpoint_demands(d)['endpoint_need_wrench']
        rows=sample_domain(d,8,91,check_visibility=False)['need_wrench']
        A=np.vstack([outer.T,np.ones(len(outer))])
        for row in rows:
            result=linprog(np.zeros(len(outer)),A_eq=A,b_eq=np.r_[row,1.],bounds=(0,None),method='highs')
            self.assertTrue(result.success)
            np.testing.assert_allclose(A@result.x,np.r_[row,1.],atol=1e-9)


if __name__=='__main__':unittest.main()
