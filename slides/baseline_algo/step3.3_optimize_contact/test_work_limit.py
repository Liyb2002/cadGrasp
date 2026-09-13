"""A physical forbidden half-space caps an otherwise free contact expansion."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step3_scheculer.stage_imports import load_stage
A=load_stage('optimize','adjust')


class WorkLimitTests(unittest.TestCase):
    def test_omitting_access_allows_expansion_across_the_former_boundary(self):
        problem=self.problem()
        problem.work_clearance.work=None
        maximum,_=problem.maximum_radius()
        self.assertGreater(maximum,1.5)
        check=problem.geometry(1.01)['row']['geometry_check']['work_volume_check']
        self.assertTrue(check['passed']);self.assertFalse(check['enforced']);self.assertFalse(check['verified'])

    def problem(self):
        # Downward normal, with an X-aligned first edge so the polygonal circle
        # still has a vertex exactly on the tested x=0 work boundary.
        mesh=trimesh.Trimesh([[-4,-3,1],[4,-3,1],[0,5,1]],[[1,0,2]],process=False)
        work=A.W.WorkVolume([[[0,-3,0],[0,3,0],[0,0,4]]],[[1,0,0]],[17],30.,8.)
        problem=A.SizeProblem.__new__(A.SizeProblem)
        problem.surface=A.P.SurfaceCircles(mesh,{0:mesh.triangles[0]})
        problem.pool=problem.surface.pool(0);problem.center=np.array([-1.,0.,1.]);problem.seed=0
        problem.initial_radius=.4;problem.cap=2.;problem.tolerance=1e-5
        problem.clearance=A.P.LocalClearance(mesh,.01)
        problem.work_clearance=A.WC.ContactClearance(mesh,.01,work,problem.clearance.offsets)
        problem.work_safe_radius=0.;problem.fixed_area=0.;problem.minimum_area=0.
        problem.domain=type('Domain',(),{'mesh':mesh})();problem.cache={}
        return problem

    def test_expansion_stops_before_work_volume_and_shrinking_stays_clear(self):
        problem=self.problem()
        maximum,limit=problem.maximum_radius()
        self.assertGreater(maximum,.999);self.assertLess(maximum,1.)
        self.assertTrue(limit['reason'].startswith('work_volume_'))
        self.assertTrue(problem.geometry(.2)['row']['geometry_valid'])
        self.assertEqual(problem.geometry(.2)['row']['geometry_check']['work_volume_check']['method'],'nested_head_containment')
        self.assertTrue(problem.work_clearance.check(problem.geometry(.2)['patch'])['passed'])
        with self.assertRaisesRegex(ValueError,'work_volume_'):
            problem.score(1.01)

    def test_touching_the_work_boundary_never_reaches_force_scoring(self):
        problem=self.problem()
        # The rightmost circle vertex is exactly on x=0, the work source plane.
        with patch.object(A.C.J,'classify',side_effect=AssertionError('Blocked size reached force scoring')) as classify:
            with self.assertRaisesRegex(ValueError,'work_volume_'):
                problem.score(1.)
            classify.assert_not_called()
        self.assertFalse(problem.geometry(1.)['row']['geometry_check']['work_volume_check']['passed'])

    def test_clear_contact_cannot_exempt_backing_that_touches_the_work_boundary(self):
        problem=self.problem();mesh=problem.domain.mesh
        # The contact is above the source plane; its downward backing touches it.
        work=A.W.WorkVolume(mesh.triangles+np.array([0,0,-.01]),[[0,0,-1]],[17],30.,8.)
        self.assertTrue(work.check_surface(mesh.triangles)['passed'])
        problem.work_clearance=A.WC.ContactClearance(mesh,.01,work,problem.clearance.offsets)
        with self.assertRaisesRegex(ValueError,'work_volume_'):
            problem.score(problem.initial_radius)
        self.assertEqual(problem.work_safe_radius,0.)

    def test_unresolved_expansion_cannot_raise_the_admissible_radius(self):
        for classification in ['solver_unresolved','visibility_or_envelope_unresolved']:
            with self.subTest(classification=classification):
                problem=self.problem()
                self.assertTrue(problem.geometry(problem.initial_radius)['row']['geometry_valid'])
                unknown=dict(passed=False,classification=classification)
                with patch.object(problem.work_clearance,'check',return_value=unknown):
                    maximum,limit=problem.maximum_radius()
                    self.assertEqual(maximum,problem.initial_radius)
                    self.assertEqual(problem.work_safe_radius,problem.initial_radius)
                    self.assertEqual(limit['reason'],'work_volume_unresolved')
                    with self.assertRaisesRegex(ValueError,'work_volume_unresolved'):
                        problem.score(.6)


if __name__=='__main__':unittest.main()
