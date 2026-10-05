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
if __name__=='__main__':unittest.main()
