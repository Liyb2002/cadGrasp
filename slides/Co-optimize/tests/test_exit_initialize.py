
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
from co_common import *
from step41 import initialize_direction
class Initialize(unittest.TestCase):
    def test_each_pose_native_up(self):
        first=trimesh.transformations.rotation_matrix(.7,[1,0,0]);second=trimesh.transformations.rotation_matrix(-.5,[0,1,0])
        directions=[]
        for T in [first,second]:
            d,world=initialize_direction(T);directions.append(d)
            np.testing.assert_allclose(world,[0,0,1],atol=1e-15)
            np.testing.assert_allclose(T[:3,:3]@d,[0,0,1],atol=1e-15)
            self.assertAlmostEqual(np.linalg.norm(d),1.)
        self.assertFalse(np.allclose(*directions))
    def test_continuous_sweep_partition_and_reversible_cut(self):
        obj=trimesh.creation.box([.01,.01,.01]);seedmesh=trimesh.creation.box([.03,.03,.03]);seed=S.solid(seedmesh)
        sweep=S.solid(S.swept_solid(obj,[0,0,.04]));removed=seed^sweep;remaining=seed-sweep
        self.assertAlmostEqual(material_volume(seed),material_volume(removed)+material_volume(remaining),places=12)
        self.assertLess(material_volume(remaining^sweep),1e-14)
        opposite=S.solid(S.swept_solid(obj,[0,0,-.04]));new=seed-opposite
        self.assertGreater(material_volume(new-remaining),0.)
    def test_each_panel_cut_is_independent(self):
        obj=trimesh.creation.box([.01,.01,.01]);seed=S.solid(trimesh.creation.box([.03,.03,.03]))
        up=S.solid(S.swept_solid(obj,[0,0,.04]));down=S.solid(S.swept_solid(obj,[0,0,-.04]))
        own_up=seed-(seed-up);own_down=seed-(seed-down)
        self.assertGreater(material_volume(own_down-own_up),0.)
        self.assertGreater(material_volume(own_up-own_down),0.)
        self.assertAlmostEqual(material_volume(seed),material_volume(seed-up)+material_volume(own_up),places=12)
if __name__=='__main__':unittest.main()
