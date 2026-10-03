"""Feasible-incumbent contracts independent of the surrogate optimizer."""
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from step3_scheculer import feasible_dsl as F


def state(ids=(("A","B"),("C",)), directions=((0,0,-1),(0,-1,0))):
    return F.State(tuple(tuple({'candidate_id':v} for v in row) for row in ids),
                   np.array([np.eye(3)]*len(ids)),np.zeros((len(ids),3)),
                   tuple(F.ray(d,i) for i,d in enumerate(directions)))


def optimizer():
    o=object.__new__(F.Optimizer)
    o.tasks=[SimpleNamespace(domain=SimpleNamespace(data={'frame':{'T_world_mesh':np.eye(4)}})) for _ in range(2)]
    o.incumbent=state();o.witness=('original certificate',);o.check={'passed':True}
    o.checker=SimpleNamespace(check=Mock(return_value={'passed':True,'per_pose':[{'covered':32768}]*2}))
    o.construct=Mock(return_value=(F.I.ROOT/'accepted',{'passed':True},None))
    o.event=Mock()
    return o


class FeasibleDSLTests(unittest.TestCase):
    def test_distinct_exits_and_independent_heads_are_valid_state(self):
        s=state()
        self.assertEqual(s.physical_count,3)
        self.assertEqual(s.shared_count,0)
        self.assertFalse(np.allclose(s.paths[0]['initial_object_exit_world'],s.paths[1]['initial_object_exit_world']))

    def test_optional_partial_sharing_counts_one_physical_instance(self):
        s=state((("A","B"),("A","C")))
        self.assertEqual(s.physical_count,3)
        self.assertEqual(s.shared_count,1)

    def test_force_or_exit_failure_retains_complete_incumbent(self):
        o=optimizer();old=o.incumbent;witness=o.witness
        o.checker.check.return_value={'passed':False,'per_pose':[{'covered':32767}]}
        self.assertFalse(o.accept(F.with_group(old,0,old.groups[0][:1]),'delete'))
        self.assertIs(o.incumbent,old);self.assertIs(o.witness,witness)
        o.construct.assert_not_called()

    def test_full_constructor_failure_rolls_back_even_after_local_success(self):
        o=optimizer();old=o.incumbent;witness=o.witness
        o.construct.return_value=None
        self.assertFalse(o.accept(F.with_group(old,0,old.groups[0][:1]),'delete'))
        self.assertIs(o.incumbent,old);self.assertIs(o.witness,witness)

    def test_proposal_that_only_improves_surrogate_is_not_committed(self):
        o=optimizer();old=o.incumbent
        self.assertFalse(o.accept(replace(old,offsets=old.offsets+.01),'surrogate'))
        self.assertIs(o.incumbent,old);o.checker.check.assert_not_called()

    def test_feasible_improvement_commits_with_witness(self):
        o=optimizer();candidate=F.with_group(o.incumbent,0,o.incumbent.groups[0][:1])
        self.assertTrue(o.accept(candidate,'delete'))
        self.assertIs(o.incumbent,candidate)
        self.assertIs(o.witness,o.construct.return_value)
        self.assertIn('witness',o.event.call_args.kwargs)
        self.assertTrue(o.event.call_args.kwargs['checks']['passed'])

    def test_neither_objective_can_get_worse(self):
        self.assertFalse(F.improves((5,.2),(4,.3)))
        self.assertFalse(F.improves((5,.2),(6,.1)))
        self.assertFalse(F.improves((5,.2),(5,.2)))
        self.assertTrue(F.improves((5,.2),(4,.2)))
        self.assertTrue(F.improves((5,.2),(5,.1)))

    def test_descent_error_does_not_erase_feasible_solution(self):
        o=optimizer();o.initialize=Mock();o.delete=Mock()
        o.parameter_descent=Mock(side_effect=RuntimeError('bad proposal'))
        o.align_paths=Mock();o.shared_proposals=Mock();o.publish=Mock()
        o.parameter_descent.__name__='parameter_descent'
        old=o.incumbent;o.optimize()
        self.assertIs(o.incumbent,old);o.publish.assert_called_once()
        o.shared_proposals.assert_called_once()

if __name__=='__main__':unittest.main()

class SeatingTests(unittest.TestCase):
    def test_reseating_preserves_local_solution_and_clears_foreign_roots(self):
        import trimesh
        from unittest.mock import patch
        from step3_scheculer import feasible_seating as T
        mesh=trimesh.creation.box(extents=(.04,.04,.04));mesh.apply_translation([0,0,.025])
        tasks=[SimpleNamespace(pose='pose_'+str(i),targets=np.zeros((1,6)),scale=np.ones(6),
               domain=SimpleNamespace(mesh=mesh,com=np.zeros(3))) for i in range(2)]
        source=state(directions=((0,0,-1),(0,0,-1)))
        source=replace(source,offsets=np.array([[0,0,0],[-.06,0,0]]))
        root=trimesh.creation.box(extents=(.006,.006,.006));root.apply_translation([.06,0,.03])
        builder=lambda *_:([[ [root.vertices] ],[ [root.vertices] ]],{},0.)
        with patch.object(T,'pressure_centers',return_value=(np.zeros((1,2)),None)):
            result=T.restore(tasks,source,builder)
        self.assertIsNotNone(result)
        repaired,records=result;self.assertGreater(len(records),0)
        self.assertIs(repaired.groups,source.groups);self.assertIs(repaired.paths,source.paths)
        np.testing.assert_array_equal(repaired.bases,source.bases)
        np.testing.assert_array_equal(source.offsets,[[0,0,0],[-.06,0,0]])
        np.testing.assert_allclose(repaired.offsets[:,2],0.,atol=1e-12)
