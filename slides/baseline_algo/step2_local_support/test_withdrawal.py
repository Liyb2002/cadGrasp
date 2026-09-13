"""Whole-ray obstruction, 3-D work/floor policy and monotone common sets."""
import sys
from pathlib import Path
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import withdrawal as W

class WithdrawalTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.creation.box(extents=[2., 2., 2.]);self.mesh.apply_translation([0, 1.1, 0])
        self.catalogue=dict(vectors=[[1.,0,0],[-1.,0,0],[0,0,1]],global_allowed_directions=W.normalize([0,1,2]),preferred_withdrawal_direction=[1.,0,0])
        self.analyzer=W.Analyzer(self.mesh,.02,self.catalogue)

    def test_continuous_ray_hits_remote_obstacle_despite_clear_start_and_end(self):
        head=trimesh.creation.box(extents=[.1, .1, .1]);head.apply_translation([-2., 1., 0])
        self.assertFalse(self.analyzer.test([head],[1.,0,0])['clear'])
        self.assertTrue(self.analyzer.test([head],[-1.,0,0])['clear'])
        self.assertTrue(self.analyzer.test([head],[0,0,1.])['clear'])

    def test_face_contact_allows_outward_and_tangent_but_not_inward(self):
        head=trimesh.creation.box(extents=[.1, .2, .2]);head.apply_translation([1.05, 1., 0])
        self.assertTrue(self.analyzer.test([head],[1.,0,0])['clear'])
        self.assertTrue(self.analyzer.test([head],[0,0,1.])['clear'])
        self.assertFalse(self.analyzer.test([head],[-1.,0,0])['clear'])
        self.assertFalse(self.analyzer.test([head],[0,-1.,0])['clear'])

    def test_work_side_and_floor_are_locked_before_selecting_any_head(self):
        ids=np.flatnonzero(self.mesh.face_normals[:,0]>.9)
        cat=W.make_catalogue(self.mesh,ids,[]);v=np.asarray(cat['vectors'])
        alive=v[cat['global_allowed_directions']['ids']]
        self.assertTrue(np.all(alive[:,0]<=W.NORMAL_TOL));self.assertTrue(np.all(alive[:,1]>=-W.NORMAL_TOL))
        self.assertTrue(np.any(alive[:,1]>.1));self.assertTrue(len(cat['work_face_locked_ids'])>0)

    def test_downward_facing_head_can_withdraw_horizontally_above_floor(self):
        head=trimesh.creation.box(extents=[.2, .05, .2]);head.apply_translation([0, .075, 0])
        self.assertTrue(self.analyzer.test([head],[1.,0,0])['clear'])
        self.assertFalse(self.analyzer.test([head],[0,1.,0])['clear'])

    def test_all_cells_must_share_the_same_ray(self):
        a=trimesh.creation.box(extents=[.1, .2, .2]);a.apply_translation([1.05, 1., 0])
        b=a.copy();b.apply_translation([-2.1, 0, 0])
        self.assertFalse(self.analyzer.test([a,b],[1.,0,0])['clear'])
        self.assertTrue(self.analyzer.test([a,b],[0,0,1.])['clear'])
        self.assertEqual(W.common([dict(certified_directions=W.normalize([0,1])),dict(certified_directions=W.normalize([1,2]))],W.normalize([0,1,2])),W.normalize([1]))

if __name__=='__main__':unittest.main()
