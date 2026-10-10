"""Real acceptance protects solved poses and distinguishes plateau progress."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import unittest
import numpy as np
from local_descent import accept_step,step41_state
from types import SimpleNamespace
from tempfile import TemporaryDirectory

class LocalTests(unittest.TestCase):
    def state(self,rows):return dict(masks=[np.array(r,bool) for r in rows])
    def test_protect_solved_pose(self):
        self.assertFalse(accept_step(self.state([[1,1],[0,0]]),self.state([[1,0],[1,1]]),1.,0.))
    def test_real_plateau_deficit_progress(self):
        s=self.state([[1,0]])
        self.assertTrue(accept_step(s,s,1.,.9))
        self.assertFalse(accept_step(s,s,1.,1.))
    def test_coverage_regression_rejected(self):
        self.assertFalse(accept_step(self.state([[1,1,0]]),self.state([[1,0,0]]),1.,0.))
    def test_step41_saved_state(self):
        with TemporaryDirectory() as tmp:
            base=Path(tmp);folder=base/'step4/step4.1/data';folder.mkdir(parents=True)
            np.savez(folder/'pose_1.npz',force_mask=[True,False],supply_7d=np.eye(7))
            search=SimpleNamespace(base=base,group=dict(poses=['pose_1']),exact_calls=0)
            d=np.array([[1.,0.,0.]])
            result=step41_state(search,d);d[0,0]=0
            self.assertEqual(result['directions'][0,0],1.)
            self.assertEqual(result['counts'],[1])
if __name__=='__main__':unittest.main()
