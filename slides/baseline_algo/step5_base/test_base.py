"""Regression cases that distinguish demand rings from projected-object bases."""
from pathlib import Path
import sys
import unittest
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import trimesh
from shapely.geometry import Polygon
from step5_base import base as C
from step6_connect_support import direction_first as X
from step5_connect_support import belt_geometry as B, whole_assembly as A
from step5_connect_support.fixtures import contacts
from step2_local_support import withdrawal as W


class BaseTests(unittest.TestCase):
    def test_smaller_geometric_base_is_not_selected_when_bearing_fails(self):
        mesh=trimesh.creation.box([1.,1.,1.]);mesh.apply_translation([0,0,1.])
        domain=SimpleNamespace(mesh=mesh,work_ids=[],com=mesh.center_mass)
        required=np.array([[-.3,-.3],[.3,-.3],[.3,.3],[-.3,.3]])
        floor=dict(floor_demands_xy_m=required,continuous_floor_enclosure_xy_m=required,
                   original_pivot_m=np.zeros(3))
        candidates=list(C.candidate_polygons(required,np.zeros(3),Polygon(),1.))[:2]
        catalogue=dict(vectors=[[-1.,0,0]],preferred_withdrawal_direction=[-1.,0,0],
                       global_allowed_directions=W.normalize([0]))
        directions=dict(mode=W.MODE,direction_catalogue=catalogue,
            contacts=[dict(certified_directions=W.normalize([0]))],common_directions=W.normalize([0]))
        failed=dict(sampled_passed=True,continuous_passed=False,attempts=[])
        passed=dict(sampled_passed=True,continuous_passed=True,sufficient_friction_coefficient=64.,attempts=[])
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(C,'OUTPUTS',Path(folder)), \
                patch.object(C.A,'read_inputs',return_value=(domain,contacts(mesh),{},floor,directions,None,[])), \
                patch.object(C,'candidate_polygons',return_value=iter(candidates)), \
                patch.object(C,'FRICTION_WITNESSES',(64.,)), \
                patch.object(C.M,'bearing',side_effect=[(failed,{}),(passed,dict(scale=np.ones(6),equilibrium_matrix=np.zeros((12,1))))]) as bearing, \
                patch.object(C.M,'grounded_matrix',return_value=(np.zeros((12,1)),{})):
            report=C.build('fixture')
            self.assertTrue(report['passed'])
            self.assertTrue(report['bearing_verified'])
            self.assertEqual(report['base']['offset_m'],candidates[1][1]['offset_m'])
            self.assertEqual(len(report['rejected_bearing_candidates']),1)
            self.assertEqual(bearing.call_count,2)
            self.assertIn('bearing.npz',report['artifacts'])

    def test_high_overhang_does_not_enlarge_horizontal_floor_ring(self):
        mesh=trimesh.creation.box([4.,4.,.1]); mesh.apply_translation([0,0,2])
        required=np.array([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]])
        obstacle=C.shadow(mesh,np.array([1.,0,0]),.04,.02,10.)
        self.assertTrue(obstacle.is_empty)
        material,record=next(C.candidate_polygons(required,np.zeros(3),obstacle,1.))
        self.assertEqual(record['offset_m'],0.)
        self.assertAlmostEqual(record['footprint_area_m2'],.25)
        self.assertLess(max(material.bounds),.3)

    def test_low_obstacle_ray_cuts_ring_but_preserves_demand_hull(self):
        mesh=trimesh.creation.box([.1,.1,.1]);mesh.apply_translation([0,0,.05])
        required=np.array([[-.3,-.3],[.3,-.3],[.3,.3],[-.3,.3]])
        obstacle=C.shadow(mesh,np.array([1.,0,0]),.04,.01,10.)
        material,record=next(C.candidate_polygons(required,np.zeros(3),obstacle,1.))
        self.assertEqual(len(material.interiors),0)  # corridor opens the ring
        parts,_=C.extrude(material,record['height_m'])
        self.assertTrue(A.footprint(parts,np.zeros(3),required,1.)[0]['passed'])
        scene=X.SweptScene(mesh,np.array([1.,0,0]),.01)
        self.assertTrue(all(scene.clear(p,ground=True) for p in parts))
        self.assertTrue(B.union_parts(parts,1.)[1]['one_solid'])

    def test_opening_that_loses_required_extreme_is_rejected(self):
        required=np.array([[-.3,-.3],[.3,-.3],[.3,.3],[-.3,.3]])
        barrier=Polygon([[-3,-3],[0,-3],[0,3],[-3,3]])
        with patch.object(C,'OFFSET_FRACTIONS',np.array([0.])):
            self.assertEqual(list(C.candidate_polygons(required,np.zeros(3),barrier,1.)),[])

    def test_step6_retains_exact_fixed_base_and_direction(self):
        mesh=trimesh.creation.box([1.,1.,1.]);mesh.apply_translation([0,0,1.])
        work=np.flatnonzero(mesh.face_normals[:,0]>.9)
        domain=SimpleNamespace(mesh=mesh,work_ids=work,com=mesh.center_mass)
        selected=contacts(mesh)
        required=np.array([[-.7,-.7],[.7,-.7],[.7,.7],[-.7,.7]])
        floor=dict(floor_demands_xy_m=required,continuous_floor_enclosure_xy_m=required,
                   original_pivot_m=np.zeros(3))
        material,record=next(C.candidate_polygons(required,np.zeros(3),Polygon(),1.))
        parts,polygons=C.extrude(material,record['height_m']);record['pads_xy_m']=polygons
        direction=np.array([-1.,0,0])
        catalogue=dict(vectors=[direction.tolist()],preferred_withdrawal_direction=direction.tolist(),
                       global_allowed_directions=W.normalize([0]))
        directions=dict(mode=W.MODE,direction_catalogue=catalogue,
            contacts=[dict(certified_directions=W.normalize([0]))],common_directions=W.normalize([0]))
        saved=dict(passed=True,base=record,withdrawal_direction=direction.tolist(),direction_id=0)
        with patch.object(X,'base_candidates',side_effect=AssertionError('Step6 must not design a base')):
            report,module=X.search(domain,selected,floor,.01,24,direction_record=directions,
                                   base_solution=(saved,parts))
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['base'],record)
        actual=[p for p,label in zip(module['parts'],module['labels']) if label.startswith('ground_strip_')]
        for a,b in zip(actual,parts):
            np.testing.assert_array_equal(a.vertices,b.vertices)
        np.testing.assert_array_equal(report['withdrawal_direction'],direction)

    def test_failed_base_still_retains_heads_for_step6_diagnostics(self):
        mesh=trimesh.creation.box([1.,1.,1.]);mesh.apply_translation([0,0,1.])
        domain=SimpleNamespace(mesh=mesh,work_ids=[],com=mesh.center_mass)
        catalogue=dict(vectors=[[-1.,0,0]],preferred_withdrawal_direction=[-1.,0,0],
                       global_allowed_directions=W.normalize([0]))
        directions=dict(mode=W.MODE,direction_catalogue=catalogue,
            contacts=[dict(certified_directions=W.normalize([0]))],common_directions=W.normalize([0]))
        stages=[]
        report,module=X.search(domain,contacts(mesh),{},.01,24,
            progress=lambda stage,parts,labels: stages.append(stage),
            direction_record=directions,base_solution=(dict(passed=False),[]))
        self.assertEqual(report['status'],'step5_base_not_verified')
        self.assertIsNone(module)
        self.assertEqual(stages,['heads'])


if __name__=='__main__': unittest.main()
