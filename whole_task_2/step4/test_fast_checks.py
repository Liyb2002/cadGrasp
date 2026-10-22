"""The Step4 broad phase must reproduce the unchanged all-Boolean Step3 check."""
import unittest
import numpy as np
import trimesh
from whole_task_2.step3 import exit_geometry as X
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step4 import fast_checks as F


class FastChecksTests(unittest.TestCase):
    def setUp(self):
        self.body=trimesh.creation.icosphere(subdivisions=2,radius=.02)

    def check(self,support,direction=(.3,.1,1.),length=.2):
        d=np.asarray(direction,float);d/=np.linalg.norm(d);path=[('T',self.body,d,length)]
        fast,fast_bad=F.continuous_checks(G.solid(support),path,1e-10)
        slow,slow_bad=X.continuous_checks(G.solid(support),path,1e-10)
        self.assertEqual(fast[0]['passed'],slow[0]['passed'])
        self.assertAlmostEqual(fast[0]['continuous_overlap_upper_bound_m3'],
            slow[0]['continuous_overlap_upper_bound_m3'],delta=1e-15)
        self.assertEqual(sorted(f for _,f,_ in fast_bad),sorted(f for _,f,_ in slow_bad))
        return fast[0]

    def test_crossing_support(self):
        support=trimesh.creation.box(extents=[.05,.05,.01]);support.apply_translation([.01,0,.08])
        self.assertFalse(self.check(support)['passed'])

    def test_support_enclosing_whole_path_needs_the_point_test(self):
        support=trimesh.creation.box(extents=[.6,.6,.6]);support.apply_translation([0,0,.1])
        row=self.check(support);self.assertFalse(row['passed'])
        self.assertEqual(row['prisms_exactly_disjoint_by_broad_phase'],0)

    def test_distant_support_skips_every_prism_exactly(self):
        support=trimesh.creation.box(extents=[.05,.05,.05]);support.apply_translation([.3,0,0])
        row=self.check(support);self.assertTrue(row['passed'])
        self.assertEqual(row['prisms_exactly_disjoint_by_broad_phase'],row['leading_face_count'])

    def test_installed_for_step4_models(self):
        from whole_task_2.operations import geometry as O
        from whole_task_2.step3.membranes import MembraneModel
        self.assertIs(O.continuous_checks,F.continuous_checks)
        self.assertIs(MembraneModel.refine_exits,F.refine_exits)


if __name__=='__main__':
    unittest.main()
