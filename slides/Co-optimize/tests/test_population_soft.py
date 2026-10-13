"""Sampling probabilities, protected poses and the fixed-slot contract."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import tempfile,unittest
import numpy as np
from test_population32 import SmallModel,initial,MaterialSample,membership,centered_seat
from whole_search.reuse_first import registered
from population_soft import (failure_weights, weighted_pose, problem_weights,
    action_probabilities, discrete_move, differential_move, replace_guest,
    preserves_poses, physical_rank, PopulationSoftSearch,carry_installation,joint_discrete_move,align_guest_centers,native_slide,restore_failed_ground_contacts)


def evidence(q, failures=(50,20,10,0)):
    return dict(layout=q, masks={k:np.arange(100)>=failures[k] for k in q.active},volume_cm3=1.)


def sample(q, failures=(50,20,10,0)):
    return MaterialSample(q,1.,dict(enclosure_passed=False,passed=False,gap_m=1.),
                          1.,'test','',{},result=evidence(q,failures))


class SoftContracts(unittest.TestCase):
    def test_ground_repair_only_moves_failed_airborne_guest_and_keeps_role(self):
        m=SmallModel();q=centered_seat(m,initial(),1,0);q=centered_seat(m,q,2,0)
        q.placements[1,2,3]+=.002;q.placements[2,2,3]+=.003
        m.workpiece_height=lambda value,k:value.placements[k,2,3]
        repaired,owners=restore_failed_ground_contacts(m,q,evidence(q,(0,20,0,0)))
        self.assertEqual(owners,[1]);self.assertAlmostEqual(repaired.placements[1,2,3],0.)
        self.assertTrue(preserves_poses(q,repaired,[0,2,3]));self.assertEqual(membership(q),membership(repaired))

    def test_exact_failure_normalization_excludes_success(self):
        q=initial(); weights=failure_weights(q,evidence(q))
        self.assertEqual(list(weights.values()),[.5,.2,.1,0.])
        rng=SimpleNamespace(choice=Mock(return_value=0))
        weighted_pose(q.active,weights,rng)
        np.testing.assert_allclose(rng.choice.call_args.kwargs['p'],[.625,.25,.125])
        self.assertEqual(rng.choice.call_args.args[0],[0,1,2])

    def test_stale_masks_rejected(self):
        q=initial(); before=evidence(q); other=q.copy();other.directions[0]=[1.,0.,0.]
        with self.assertRaises(ValueError):failure_weights(other,before)

    def test_all_force_passed_is_held_without_explicit_oracle_exception(self):
        m=SmallModel(); m.poses=['A','B','C','D']; item=sample(initial(),(0,0,0,0))
        item.oracle['certificates']=[dict(pose='A',other_pose='C',gap_m=.02)]
        weights,mode=problem_weights(m,item)
        self.assertEqual(sum(weights.values()),0.);self.assertIsNone(weighted_pose(item.layout.active,weights,np.random.default_rng(42)))
        weights,mode=problem_weights(m,item,True)
        self.assertEqual(weights,{0:.02,1:0.,2:.02,3:0.})

    def test_count_changes_sampling_only(self):
        m=SmallModel(); q=initial(); weights=dict.fromkeys(q.active,1.)
        self.assertEqual(action_probabilities(q,1,weights),{'create':1.})
        one=centered_seat(m,q,1,0); two=centered_seat(m,one,2,0)
        self.assertGreater(action_probabilities(one,3,weights)['create'],action_probabilities(two,3,weights)['create'])
        self.assertLess(action_probabilities(one,1,weights)['restore'],action_probabilities(two,1,weights)['restore'])
        self.assertEqual(physical_rank(sample(q)),physical_rank(sample(two)))

    def test_role_exchange_preserves_number_and_native_rotations(self):
        m=SmallModel(); q=centered_seat(m,initial(),1,0)
        moved,kept=replace_guest(m,q,1,2)
        self.assertTrue(kept);self.assertEqual(membership(moved),(3,1))
        self.assertTrue(registered(moved,1));self.assertFalse(registered(moved,2))
        moved,kept=replace_guest(m,q,1,0)
        self.assertFalse(kept);self.assertEqual(membership(moved),(3,1))
        for k in moved.active:m.validate_placement(moved,k)

    def test_discrete_and_differential_never_move_successful_parameters(self):
        m=SmallModel(); q=centered_seat(m,initial(),1,0);weights={0:0.,1:.5,2:.2,3:0.}
        rng=np.random.default_rng(42)
        for _ in range(100):
            moved,detail=discrete_move(m,q,weights,rng)
            self.assertTrue(preserves_poses(q,moved,[0,3]))
        population=[q.copy()for _ in range(32)];keys=[x.key()for x in population]
        moved,detail=differential_move(m,q,population,7,weights,rng,0)
        self.assertEqual(detail['selected_pose'],1)
        self.assertTrue(preserves_poses(q,moved,[0,2,3]));self.assertNotIn(7,detail['donor_slots'])
        self.assertEqual(len(set(detail['donor_slots'])),2);self.assertEqual(keys,[x.key()for x in population])

    def test_shared_installation_move_is_protected(self):
        q=initial();q.hosts[1]=0;other=q.copy();other.state_offsets[0,2]=.01
        self.assertFalse(preserves_poses(q,other,[0]))

    def test_oracle_conflict_can_activate_force_success_after_user_relaxes_freeze(self):
        m=SmallModel();m.poses=['A','B','C','D'];item=sample(initial(),(0,20,0,0))
        item.oracle['certificates']=[dict(pose='A',other_pose='C',gap_m=.02)]
        weights,mode=problem_weights(m,item,True)
        self.assertEqual(mode,'original_force_plus_oracle_failure_gap')
        np.testing.assert_allclose(list(weights.values()),[.2,.2,.2,0.])

    def test_force_loss_retains_severity_information(self):
        few=sample(initial(),(10,0,0,0));few.loss=1.
        many=sample(initial(),(50,0,0,0));many.loss=.0001
        self.assertLess(physical_rank(many),physical_rank(few))

    def test_shared_oracle_failure_allows_coupled_adjustment_when_enabled(self):
        search=SimpleNamespace(model=SmallModel(),frontier_width=32,allow_oracle_fallback=True,
            allow_coupled_adjustment=True)
        controller=PopulationSoftSearch(search)
        self.assertEqual(controller.adjustable_poses(sample(initial(),(0,20,0,0))),[0,1,2,3])

    def test_group_move_removes_old_installation_instead_of_orphaning_guests(self):
        m=SmallModel();q=centered_seat(m,initial(),1,0)
        moved=centered_seat(m,q,0,2);moved=carry_installation(m,q,moved,0)
        self.assertNotIn(0,moved.hosts[list(moved.active)])
        self.assertEqual(moved.hosts[0],moved.hosts[1])
        for k in moved.active:m.validate_placement(moved,k)

    def test_group_role_exchange_keeps_restored_pose_registered(self):
        m=SmallModel();q=centered_seat(m,initial(),1,0);q=centered_seat(m,q,3,0)
        moved,_=replace_guest(m,q,1,0)
        moved=carry_installation(m,q,moved,0,exclude=(1,))
        self.assertTrue(registered(moved,1));self.assertEqual(membership(moved),membership(q))
        self.assertNotIn(0,moved.hosts[list(moved.active)])

    def test_joint_roles_are_chosen_in_mutation_and_preserve_native_tasks(self):
        m=SmallModel();q=initial();layouts=[];rng=np.random.default_rng(42)
        for _ in range(32):
            trial,detail=joint_discrete_move(m,q,{0:.5,1:.2,2:.1,3:0.},rng,0)
            self.assertFalse(registered(trial,0));self.assertEqual(detail['selected_pose'],0)
            self.assertFalse(detail['contact_matching_run'])
            for k in trial.active:m.validate_placement(trial,k)
            layouts.append(trial.key())
        self.assertGreater(len(set(layouts)),1);self.assertEqual(membership(q),(4,0))

    def test_center_height_is_adjusted_after_xy_initialization(self):
        from fixture_state import installed_state
        m=SmallModel();m.native[0,2,3]=.35
        q=centered_seat(m,initial(),1,0)
        def center(t,k):
            U=installed_state(m.native,t,t.hosts[k])@t.placements[k]
            return U[:3,:3]@m.mesh.center_mass+U[:3,3]
        self.assertNotAlmostEqual(center(q,0)[2],center(q,1)[2])
        moved=align_guest_centers(m,q)
        np.testing.assert_allclose(center(moved,0),center(moved,1))
        self.assertEqual(membership(moved),membership(q));m.validate_placement(moved,1)

    def test_feasible_initial_population_stops_without_preassigned_roles(self):
        m=SmallModel();q=initial()
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory)/'case/step4/step4.2/experiment';out.mkdir(parents=True)
            search=SimpleNamespace(model=m,frontier_width=32,out=out,iterations=8,stop_when_feasible=True,
                rng=np.random.default_rng(42),record=Mock(),process_snapshot=Mock())
            c=PopulationSoftSearch(search)
            def ready(t,*args,**kwargs):
                item=sample(t,(0,0,0,0));item.oracle.update(passed=True,connected_passed=True)
                item.construction={'passed':True};item.evaluation=dict(with_full_support={'volume_litres':1.},material_full_cm3=1.)
                return item
            c.preview=ready;c.checked=lambda item:item
            c.certify=lambda items:setattr(c,'incumbent',items[0])
            result=c.solve(evidence(q,(0,0,0,0)))
            self.assertEqual(search.frontier_statistics['rounds'],0)
            self.assertEqual(search.frontier_statistics['configured_rounds'],8)
            self.assertEqual(len(c.slots),32);self.assertEqual(membership(result['layout']),(4,0))
            self.assertEqual(c.audit,[])

    def test_full_validation_keeps_good_raw_move_when_adjustment_has_false_failures(self):
        search=SimpleNamespace(model=SmallModel(),frontier_width=32)
        c=PopulationSoftSearch(search);raw=sample(initial(),(0,0,0,0));raw.loss=1e-8
        bad=sample(initial(),(10,0,0,0));bad.loss=0.
        c.refine_preview=Mock(return_value=bad);c.checked=lambda item:item
        self.assertIs(c.checked_adjustment(raw,0,0),raw)

    def test_native_slide_changes_seat_without_forcing_a_different_installation(self):
        m=SmallModel();q=initial();moved=native_slide(m,q,1,0,np.random.default_rng(42))
        self.assertEqual(moved.hosts[1],1);self.assertFalse(registered(moved,1))
        np.testing.assert_allclose(moved.placements[1,:3,:3],np.eye(3));m.validate_placement(moved,1)

    def test_broader_geometry_eligibility_does_not_accept_failed_force(self):
        m=SmallModel();m.poses=['A','B','C','D']
        c=PopulationSoftSearch(SimpleNamespace(model=m,frontier_width=32,allow_geometry_validation=True))
        item=sample(initial());item.result['masks']={k:np.arange(32768)>=1000 for k in item.layout.active}
        self.assertTrue(c.evaluator.space_candidate(item));self.assertFalse(item.completed)
        item.oracle.update(passed=True,connected_passed=True);item.construction={'passed':True}
        self.assertFalse(item.completed)  # Original demands must still pass.

    def test_two_rounds_keep_32_slots_one_adjustment_or_hold(self):
        m=SmallModel(); q=initial()
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory)/'case/step4/step4.2/experiment';out.mkdir(parents=True)
            search=SimpleNamespace(model=m,frontier_width=32,out=out,iterations=2,
                rng=np.random.default_rng(42),record=Mock(),process_snapshot=Mock())
            c=PopulationSoftSearch(search)
            def preview(t,operation='parent',parent='',detail=None,result=None):
                item=sample(t);item.operation=operation;item.detail=detail or {};return item
            c.preview=preview;c.checked=lambda item:item;c.certify=Mock()
            c.refine_preview=Mock(side_effect=lambda raw,iteration,maximum:raw)
            with patch('population_soft.freeze_population_measure'):c.solve(evidence(q))
            self.assertEqual(len(c.slots),32);self.assertEqual(c.refine_preview.call_count,64)
            for row in c.audit:
                self.assertEqual(len(row['events']),32)
                for event in row['events']:
                    self.assertNotIn('error',event)
                    self.assertEqual(event['adjustment_passes'],1)
                    self.assertNotIn(event['selected_pose'],event['protected_pose_indices'])
                    np.testing.assert_allclose(list(event['pose_probabilities'].values()),[.625,.25,.125,0.])
            c=PopulationSoftSearch(search);c.preview=lambda t,*a,**kw:sample(t,(0,0,0,0))
            c.checked=lambda item:item;c.certify=Mock();c.refine_preview=Mock()
            search.out=Path(directory)/'held/step4/step4.2/experiment';search.out.mkdir(parents=True)
            with patch('population_soft.freeze_population_measure'):c.solve(evidence(q,(0,0,0,0)))
            self.assertEqual(c.refine_preview.call_count,0);self.assertEqual(c.branch_counts['hold'],64)
            self.assertTrue(all(e['adjustment_passes']==0 for r in c.audit for e in r['events']))


if __name__=='__main__':unittest.main()
