"""Operation acceptance contracts, including allowed larger contact patches."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from step3_scheculer import operation_dsl as F


def optimizer():
    o=object.__new__(F.Optimizer)
    contact=lambda name:dict(candidate_id=name,triangle_areas_m2=np.array([.01]))
    o.tasks=[SimpleNamespace(domain=SimpleNamespace(mesh=SimpleNamespace(area=1.),data={'frame':{'T_world_mesh':np.eye(4)}}))]*2
    o.incumbent=F.State(((contact('A'),contact('B')),(contact('C'),)),np.array([np.eye(3)]*2),np.zeros((2,3)),(F.ray((0,0,-1),0),F.ray((0,-1,0),1)))
    o.witness=('old witness',);o.check={'passed':True}
    o.checker=SimpleNamespace(check=Mock(return_value={'passed':True,'per_pose':[{'covered':32768}]*2}))
    o.construct=Mock(return_value=(F.I.ROOT/'accepted',{'passed':True},None));o.event=Mock()
    return o


class OperationsTests(unittest.TestCase):
    def test_larger_head_and_deletion_can_be_committed(self):
        o=optimizer();c=dict(o.incumbent.groups[0][0],triangle_areas_m2=np.array([.02]))
        candidate=F.with_group(o.incumbent,0,[c]);self.assertTrue(o.accept(candidate,'merge_two_heads'))
        self.assertIs(o.incumbent,candidate)

    def test_area_over_budget_rejected_before_certification(self):
        o=optimizer();old=o.incumbent;c=dict(old.groups[0][0],triangle_areas_m2=np.array([.021]))
        self.assertFalse(o.accept(F.with_group(old,0,[c]),'resize_then_delete'))
        self.assertIs(o.incumbent,old);o.checker.check.assert_not_called()

    def test_numerical_error_is_not_a_failed_incumbent(self):
        o=optimizer();old=o.incumbent;certificate=o.witness
        o.checker.check.side_effect=RuntimeError('HiGHS unresolved')
        self.assertFalse(o.accept(F.with_group(old,0,old.groups[0][:1]),'exchange_then_delete'))
        self.assertIs(o.incumbent,old);self.assertIs(o.witness,certificate)

    def test_new_body_failure_preserves_qualified_geometry(self):
        o=optimizer();old=o.incumbent;certificate=o.witness;o.construct.return_value=None
        self.assertFalse(o.accept(F.with_group(old,0,old.groups[0][:1]),'merge_two_heads'))
        self.assertIs(o.incumbent,old);self.assertIs(o.witness,certificate)

    def test_area_change_without_actual_objective_improvement_is_not_committed(self):
        o=optimizer();old=o.incumbent;c=dict(old.groups[0][0],triangle_areas_m2=np.array([.02]))
        self.assertFalse(o.accept(F.with_group(old,0,[c,old.groups[0][1]]),'resize'))
        self.assertIs(o.incumbent,old)

    def test_latest_copied_constructor_is_used(self):
        from step4_connect_support.baseline_current.envelope_growth import EnvelopeGrow
        self.assertTrue(issubclass(F.Grow,EnvelopeGrow))
        self.assertFalse(F.Grow.recheck_export)

if __name__=='__main__':unittest.main()
