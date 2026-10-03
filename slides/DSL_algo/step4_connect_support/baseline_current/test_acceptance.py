"""Step4 accepts geometry independently of historical force replay."""
import contextlib
import copy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from step4_connect_support.baseline_current import acceptance as A, build_coupled_saddle as S


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.creation.box(extents=[.02,.02,.02])
        self.mesh.apply_translation([0,0,.01])
        self.hull=[[-.01,-.01],[.01,-.01],[.01,.01],[-.01,.01]]
        self.case=SimpleNamespace(poses=['pose_1'],demands=[self.hull],
            schedule=dict(passed=False,covered_counts=[1]))
        self.row=dict(pose='pose_1',min_fixture_z_m=0.,max_active_contact_gap_m=0.,
            maximum_missing_head_cell_volume_m3=0.,withdrawal=dict(clear=True),
            independent_head_withdrawal=dict(clear=True),actual_ground_hull_xy_m=self.hull,
            coupled_equilibrium_passed=False,maximum_equilibrium_residual=10.,
            minimum_reaction_coefficient=-1.)
        self.report=dict(checks=[self.row],solid=dict(one_solid=True,component_count=1),
            placement=dict(bases=[np.eye(3)],offsets=[np.zeros(3)]),maximum_spatial_span_m=.03)

    def evaluate(self):
        return A.evaluate(self.case,self.report,self.mesh,dict(passed=True))

    def test_force_replay_does_not_veto_or_relabel_step3(self):
        original=copy.deepcopy(self.case.schedule)
        result=self.evaluate()
        self.assertTrue(result['passed'])
        self.assertFalse(result['step3_passed'])
        self.assertFalse(result['step4_force_torque_enforced'])
        self.assertEqual(self.case.schedule,original)

    def test_old_size_target_does_not_veto_geometry(self):
        self.report['maximum_spatial_span_m']=.019
        result=self.evaluate()
        self.assertTrue(result['passed'])
        self.assertTrue(result['geometry_excluding_size_passed'])
        self.assertFalse(result['size_limit_enforced'])
        self.assertEqual(result['failures'],[])

    def test_missing_ground_hull_remains(self):
        self.case.demands=[np.asarray(self.hull)*2]
        result=self.evaluate()
        self.assertFalse(result['passed'])
        self.assertEqual(result['failures'][0]['check'],'ground_hull_coverage')

    def test_collision_and_contact_checks_remain(self):
        for key,value in [('min_fixture_z_m',-.001),('max_active_contact_gap_m',.001),
                          ('maximum_missing_head_cell_volume_m3',1e-8),
                          ('withdrawal',dict(clear=False)),('independent_head_withdrawal',dict(clear=False))]:
            with self.subTest(key=key):
                row=copy.deepcopy(self.row);row[key]=value
                self.assertFalse(all(A.pose_checks(row).values()))

    def test_geometry_verifier_never_calls_force_solver(self):
        obj=trimesh.creation.box(extents=[.02,.02,.02])
        obj.apply_translation([0,0,.03])
        task=SimpleNamespace(pose='pose_1',domain=SimpleNamespace(mesh=obj),targets=np.zeros((2,6)))
        contacts=[[dict(triangles_m=np.array([[[-.01,-.01,.02],[.01,-.01,.02],[.01,.01,.02]]]))]]
        heads=[[[self.mesh.vertices.copy()]]]
        directions=np.array([[1.,0.,0.]])
        sweep=S.swept_solid(obj,-S.SWEEP_LENGTH*directions[0])
        with patch.object(S,'bearing_rays',side_effect=AssertionError('Force assembly must not run')), \
             patch.object(S.Q,'BatchSolver',side_effect=AssertionError('Force solve must not run')), \
             contextlib.redirect_stdout(io.StringIO()):
            checks,certificate=S.verify([task],contacts,heads,directions,np.array([np.eye(3)]),
                np.zeros((1,3)),self.mesh,[sweep],check_equilibrium=False)
        self.assertTrue(all(A.pose_checks(checks[0]).values()))
        self.assertIsNone(checks[0]['coupled_equilibrium_passed'])
        self.assertEqual(checks[0]['lp_count'],0)
        self.assertFalse(any(k.endswith('_weights') for k in certificate))


if __name__=='__main__':unittest.main()
