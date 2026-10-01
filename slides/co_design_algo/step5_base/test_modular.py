"""Rest-frame assembly, unilateral docking and stationary-base regressions."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import trimesh
from scipy.optimize import linprog
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import withdrawal as W, installation as INIT
from step1.needs import ContinuousNeeds
from step5_base import rectangular_dock as J
from step6_connect_support import modular_workflow as VIDEO


class ModularTests(unittest.TestCase):
    def test_rest_plane_matches_transformed_coordinates_for_real_b(self):
        domain=ContinuousNeeds.read(Path(__file__).resolve().parent.parent/'output/B/pose_2/step_1_needs/needs.json')
        scene=INIT.scene('B',domain);T=np.asarray(scene['T_initial_from_task']);plane=np.asarray(scene['floor_plane'])
        rest=trimesh.transform_points(domain.mesh.vertices,T)
        np.testing.assert_allclose(rest[:,2],domain.mesh.vertices@plane[:3]+plane[3],atol=1e-13)
        self.assertGreater(np.linalg.norm(plane[:3]-[0,0,1]),.1)

    def test_initial_floor_allows_task_downward_direction(self):
        obj=trimesh.creation.box(extents=[2,2,2]).apply_translation([1,0,1])
        cat=dict(vectors=[[0,0,-1]],global_allowed_directions=W.normalize([0]),
                 preferred_withdrawal_direction=None,
                 installation=dict(floor_plane=[1,0,0,0],contact_floor_clearance_m=.0015))
        head=trimesh.creation.box(extents=[.1,.1,.1]).apply_translation([2.1,0,1])
        self.assertTrue(W.Analyzer(obj,.02,cat).test([head],[0,0,-1])['clear'])
        legacy=dict(cat);legacy.pop('installation')
        self.assertFalse(W.Analyzer(obj,.02,legacy).test([head],[0,0,-1])['clear'])

    def test_contact_on_initial_floor_is_rejected_before_head_sweep(self):
        obj=trimesh.creation.box(extents=[2,2,2]).apply_translation([1,0,1])
        ids=np.flatnonzero(obj.face_normals[:,0]<-.9);tri=obj.triangles[ids]
        contact=dict(candidate_id='floor',candidate_index=0,triangles_m=tri,source_faces=ids,
                     triangle_areas_m2=obj.area_faces[ids],radius_m=.2,center_m=tri.reshape(-1,3).mean(0))
        cat=dict(vectors=[[0,0,1]],global_allowed_directions=W.normalize([0]),preferred_withdrawal_direction=None,
                 installation=dict(floor_plane=[1,0,0,0],contact_floor_clearance_m=.0015))
        result=W.Analyzer(obj,.02,cat).analyze(contact)
        self.assertFalse(result['has_certified_direction'])
        self.assertEqual(result['analysis']['checks'][0]['reason'],'contact_on_initial_ground_region')

    def test_socket_has_stop_but_no_withdrawal_latch(self):
        _,_,points,forces=J.joint([0,0,.1])
        self.assertTrue(np.all(forces[:,2]>=0))
        for target,expected in (([0,0,1],True),([0,0,-1],False)):
            r=linprog(np.zeros(len(points)),A_eq=forces.T,b_eq=target,bounds=(0,None),method='highs')
            self.assertEqual(r.success,expected)

    def test_three_body_interface_uses_equal_opposite_reactions(self):
        domain=SimpleNamespace(mesh=trimesh.creation.box(extents=[.1,.1,.1]),com=np.zeros(3))
        floor=dict(original_pivot_m=np.array([0.,0.,0.]))
        footprint=dict(pads_xy_m=[[[-1,-1],[1,-1],[1,1],[-1,1]]])
        matrix,_=J.matrix(domain,[],floor,footprint,J.description([.2,0,.1]),64.)
        self.assertEqual(matrix.shape[0],18)
        np.testing.assert_array_equal(matrix[:6,4:24],0)
        np.testing.assert_array_equal(matrix[6:12,4:24],-matrix[12:,4:24])
        np.testing.assert_array_equal(matrix[:12,24:],0)

    def test_whole_object_docking_sweep_catches_remote_obstacle(self):
        obj=trimesh.creation.box(extents=[.1,.1,.1]).apply_translation([0,0,.15])
        blue=trimesh.creation.box(extents=[.02,.02,.02]).apply_translation([.1,0,.15])
        remote=trimesh.creation.box(extents=[.1,.1,.02]).apply_translation([0,0,.4])
        self.assertFalse(J.docking_check(obj,blue,[remote])['passed'])
        remote.apply_translation([1,0,0])
        self.assertTrue(J.docking_check(obj,blue,[remote])['passed'])

    def test_transport_rest_to_task_and_high_rotation_clearance(self):
        obj=trimesh.creation.box(extents=[.1,.2,.1]).apply_translation([0,0,.05])
        blue=trimesh.creation.box(extents=[.02,.02,.03]).apply_translation([.15,0,.1])
        base=trimesh.creation.box(extents=[.3,.3,.03]).apply_translation([.4,0,.015])
        report=dict(installation=dict(T_initial_from_task=np.eye(4).tolist()),
                    d0_withdrawal_direction=[0,0,1],initial_module_sweep=dict(length_m=.15))
        plan=VIDEO.transport(SimpleNamespace(mesh=obj),blue,base,report)
        self.assertTrue(plan['transport_geometry_verified'])
        np.testing.assert_allclose(VIDEO.pose_at(0,plan),plan['T_initial_from_task'])
        np.testing.assert_allclose(VIDEO.pose_at(14,plan),np.eye(4),atol=1e-15)
        for t in np.linspace(7,10,21):
            points=trimesh.transform_points(np.vstack([obj.vertices,blue.vertices]),VIDEO.pose_at(t,plan))
            self.assertGreater(points[:,2].min(),base.bounds[1,2])



if __name__=='__main__':unittest.main()
