"""Mechanical counterexamples, shared reactions and independent LP equivalence."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step3_scheculer import passive_support as U, verification as V
C=V.C


class PassiveSupportTests(unittest.TestCase):
    def test_hard_rest_pass_does_not_require_full_work_load_coverage(self):
        from types import SimpleNamespace
        domain=SimpleNamespace(gravity=np.array([0.,-1.,0.]))
        full=np.vstack([U.heads([COORD.wrench([0,0,1,0,0,0])],np.ones(6)),U.floor([],np.ones(6))])
        self.assertTrue(C.gravity_check(full,domain,np.ones(6))['passed'])
        mask=C.J.classify(full,np.array([COORD.wrench([0,0,1,0,0,0]),COORD.wrench([1,0,1,0,0,0])]))[0]
        np.testing.assert_array_equal(mask,[True,False])

    def setUp(self):
        self.scale=np.ones(6)
        self.slack=U.floor(np.empty((0,6)),self.scale)

    def test_downward_head_cannot_borrow_the_object_floor_reaction(self):
        # Floor at x=-1; downward head at x=-2. Object-only equilibrium
        # accepts floor=2, head=1. That would lift the massless support.
        floor=np.array([COORD.wrench([0.,0.,1.,0.,1.,0.])])
        head=np.array([COORD.wrench([0.,0.,-1.,0.,-2.,0.])])
        target=np.array(COORD.wrench([0.,0.,1.,0.,0.,0.]))
        self.assertIsNotNone(C.W.solve(np.vstack([floor,head]),target))
        lifted=np.vstack([U.floor(floor,self.scale),U.heads(head,self.scale)])
        self.assertIsNone(C.W.solve(lifted,target))
        self.assertFalse(C.J.classify(lifted,target[None])[0][0])

    def test_downward_and_upward_heads_cooperate_across_groups(self):
        down=U.heads([COORD.wrench([1,0,-1,0,0,0])],self.scale)
        up=U.heads([COORD.wrench([0,0,1,0,0,0])],self.scale)
        full=C.I.merge_columns(self.slack,down,up)
        for magnitude in (.5,1.,2.):
            target=np.array(COORD.wrench([1.,0.,magnitude,0.,0.,0.]))
            witness=C.W.solve(full,target)
            self.assertIsNotNone(witness)
            ids=witness['indices']; weights=np.array(witness['coefficients'])
            np.testing.assert_allclose(weights@full[ids],U.target(target),atol=1e-12)
            self.assertIsNone(C.W.solve(C.I.merge_columns(self.slack,up),target))
        report=C.verify_classification(full,np.array([COORD.wrench([1,0,1,0,0,0])]),np.array([True]))
        self.assertAlmostEqual(report['sample_checks'][0]['passive_support']['support_floor_normal_mg'],1.)

    def test_reallocate_instead_of_rejecting_one_bad_solution(self):
        floor=np.array([COORD.wrench([0.,0.,1.,0.,1.,0.])])
        heads=np.array([COORD.wrench([0.,0.,-1.,0.,-2.,0.]),COORD.wrench([0.,0.,1.,0.,-1.,0.])])
        full=np.vstack([U.floor(floor,self.scale),U.heads(heads,self.scale)])
        self.assertIsNotNone(C.W.solve(full,np.array(COORD.wrench([0,0,1,0,0,0]))))

    def test_zero_reaction_and_unloaded_support_are_allowed(self):
        full=U.floor([COORD.wrench([0,0,1,0,0,0])],self.scale)
        self.assertIsNotNone(C.W.solve(full,np.array(COORD.wrench([0,0,1,0,0,0]))))
        self.assertIsNotNone(C.W.solve(full,np.zeros(6)))

    def test_lift_matches_direct_original_equations_and_inequality(self):
        rng=np.random.default_rng(47)
        raw=rng.normal(size=(18,6)); raw[:3,1]=1.
        vertical=raw[:,1].copy();vertical[:3]=0.
        lifted=np.vstack([U.floor(raw[:3],self.scale),U.heads(raw[3:],self.scale)])
        targets=rng.normal(size=(30,6))
        actual=C.J.classify(lifted,targets)[0]
        for target,passed in zip(targets,actual):
            direct=linprog(np.zeros(len(raw)),A_eq=raw.T,b_eq=target,
                           A_ub=-vertical[None],b_ub=[0.],bounds=(0,None),method='highs')
            self.assertEqual(bool(passed),direct.success)

    def test_seven_coordinate_cone_and_certificate_math(self):
        full=np.eye(7)
        H=C.W.cone(full)
        self.assertTrue(C.halfspace_membership(H,np.ones((1,7)))[0])
        self.assertFalse(C.halfspace_membership(H,-np.ones((1,7)))[0])
        initial=np.ones(7)
        proof=C.W.precise_basis(full,initial,initial)
        self.assertEqual(len(proof['indices']),7)
        self.assertIsNotNone(C.W.exact_separator(full,-initial))

    def test_real_b_pose2_downward_candidate_is_not_coverage(self):
        from step1.cases import selected_pose
        if not (C.OUTPUTS/'B/pose_2/step2_local_support/circles.npz').exists():
            self.skipTest('Requires the B pose2 Step2 catalogue')
        with selected_pose('pose_2'):
            problem=C.Problem('B')
            index=next(i for i,row in enumerate(problem.geometry['patches']) if row['id']=='C023')
            supply=V.Supply(problem,[problem.candidate(index)])
            gravity=np.array(COORD.wrench([0.,0.,1.,0.,0.,0.]))*problem.scale
            self.assertIsNotNone(C.W.solve(supply.full[:,:6],gravity))
            self.assertIsNone(C.W.solve(supply.full,gravity))
            self.assertFalse(C.J.classify(supply.full,problem.targets)[0].any())


if __name__=='__main__':unittest.main()
