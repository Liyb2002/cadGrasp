"""Physical invariants of grounding, independent seating and execution."""
import unittest
import numpy as np
from .mobile import MobileTaskModel
from fixture_state import installed_state


class MobileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=MobileTaskModel()
        cls.layout,_=cls.model.initial()

    def test_only_actual_ground_contact_provides_floor_reactions(self):
        m=self.model;q=self.layout.copy()
        for k in q.active:
            np.testing.assert_allclose(m.tasks[k].domain.mesh.vertices[m.floor_vertices[k]],m.tasks[k].floor,atol=1e-13)
            self.assertAlmostEqual(m.tasks[k].floor[2],0.,places=12)
            self.assertEqual(len(m.floor_columns(q,k)),5)
        q.state_offsets[1,2]=.02
        self.assertEqual(len(m.floor_columns(q,1)),1)
        self.assertEqual(m.floor_columns(q,1)[0,-1],-1.)
        q.state_offsets[1,2]=-.001
        with self.assertRaises(ValueError):m.check_station(q)

    def test_each_pose_executes_at_same_COM_xy_after_search_translation(self):
        m=self.model;q=self.layout.copy()
        q.state_offsets[:,:2]=[[.1,-.2],[-.15,.1],[.2,.1],[-.1,-.1]]
        q.state_offsets[:,2]=[0.,.01,.03,.02]
        for k in q.active:
            row=m.execution(q,k)
            np.testing.assert_allclose(row['object_COM_world_m'][:2],m.center_xy,atol=1e-12)
            np.testing.assert_allclose(np.array(row['world_from_support'])@q.placements[k],row['world_from_object'],atol=1e-12)
            self.assertAlmostEqual(row['object_floor_gap_m'],q.state_offsets[k,2])
            world_head=np.array(row['world_from_object'])@np.r_[m.work_rays[k].head,1]
            np.testing.assert_allclose(world_head[:3],row['head_world_m'],atol=1e-12)

    def test_two_seats_sharing_search_host_can_recenter_independently(self):
        m=self.model;q=self.layout.copy();k=1
        U=m.native[k].copy();U[:2,3]+=[.06,-.02]
        q.hosts[k]=0;q.placements[k]=np.linalg.inv(m.native[0])@U
        q.directions[k]=m.native[0,:3,:3].T@m.native[k,:3,:3]@q.directions[k]
        rows=[m.execution(q,j)for j in (0,1)]
        self.assertFalse(np.allclose(rows[0]['world_from_support'],rows[1]['world_from_support']))
        for row in rows:np.testing.assert_allclose(row['object_COM_world_m'][:2],m.center_xy,atol=1e-12)

    def test_height_cannot_bypass_reach_or_change_force_direction(self):
        m=self.model;q=self.layout.copy();q.state_offsets[0,2]=2.
        with self.assertRaisesRegex(ValueError,'reach'):m.check_station(q)
        for task in m.tasks:
            targets=task.forward_nodes(np.arange(len(task.targets)))
            np.testing.assert_allclose(targets,task.targets_7d,atol=1e-13,rtol=0)
            np.testing.assert_allclose(task.domain.axis,[0,-2**-.5,-2**-.5],atol=1e-15)

    def test_continuous_pressure_envelope_contains_all_samples_at_new_heights(self):
        from system_floor import system_floor_demands
        from base_oracle import cover
        m=self.model
        for height in (0.,.03,.15):
            q=self.layout.copy();q.state_offsets[:,2]=height
            for k,row in enumerate(m.base_oracle.rows(q)):
                com=row['U'][:3,:3]@m.mesh.center_mass+row['U'][:3,3]
                points=system_floor_demands(m.tasks[k].targets/m.tasks[k].scale,com)['floor_demands_xy_m']
                self.assertTrue(cover(points,row['hull'])[0])
                self.assertLess(len(m.base_oracle.loads[k]),len(m.tasks[k].targets)/80)

    def test_original_pass_cannot_stop_search_after_new_physical_gate_fails(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from material_evaluator import MaterialSample,MaterialEvaluator
        from .complete_gate import MobileEvaluator
        item=MaterialSample(self.layout.copy(),0.,dict(passed=True,connected_passed=True),0.,
            result=dict(masks={k:np.ones(3,bool)for k in self.layout.active}),construction=dict(passed=True))
        self.assertTrue(item.completed)
        checker=MobileEvaluator(SimpleNamespace(model=self.model))
        with patch.object(MaterialEvaluator,'construct',return_value=item), \
             patch('whole_task.step4.complete_gate.K.D.export_exact_obj'), \
             patch('whole_task.step4.complete_gate.condition_core',side_effect=ValueError('blocked finite work access')):
            result=checker.construct(item)
        self.assertFalse(result.completed)
        self.assertIn('blocked finite work access',result.construction['physical_certificate_error'])


if __name__=='__main__':unittest.main()
