import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from types import SimpleNamespace
from worst_wrench_descent import farthest_load,direct_gradient
from physics_guided_cone_iterative import cone_projection
from physics_guided_geometry import tangent_frames

class WorstWrenchTests(unittest.TestCase):
    def test_safe_bounds_find_exhaustive_worst_in_multiple_cones(self):
        rng=np.random.default_rng(19)
        tasks=[SimpleNamespace(targets=rng.normal(size=(40,7))) for _ in range(2)]
        rays=[np.eye(7)[:3],-np.eye(7)[:4]]
        search=SimpleNamespace(states=[(t,None) for t in tasks],
            projection=lambda state,k,i:cone_projection(rays[k],tasks[k].targets[i]))
        state=dict(masks=[np.zeros(40,bool),np.zeros(40,bool)])
        found,_=farthest_load(search,state)
        expected=max((cone_projection(rays[k],t.targets[i])['loss'],k,i) for k,t in enumerate(tasks) for i in range(40))
        self.assertAlmostEqual(found['loss'],expected[0],places=10)
        self.assertEqual((found['pose_index'],found['load_index']),expected[1:])
    def test_geometry_finite_difference_matches_known_direction_derivative(self):
        directions=np.array([[0.,0.,1.]])
        normal=np.array([[0.,0.,1.]])
        vector=np.array([.3,-.4,.2])
        search=SimpleNamespace(normals=normal,exact=lambda d:dict(directions=d,counts=[0]),
            projection=lambda state,k,i:dict(loss=float((state['directions'][0]@vector)**2)))
        target=dict(loss=.04,pose_index=0,load_index=0)
        g,probes=direct_gradient(search,None,directions,target,1e-5)
        expected=2*.2*vector@tangent_frames(directions)[0]
        np.testing.assert_allclose(g[0],expected,atol=1e-8)
        self.assertTrue(all(p['formula']=='central difference' for p in probes))
    def test_boundary_uses_feasible_one_sided_difference(self):
        directions=np.array([[1.,0.,0.]])
        search=SimpleNamespace(normals=np.array([[0.,0.,1.]]),exact=lambda d:dict(directions=d,counts=[0]),
            projection=lambda state,k,i:dict(loss=float(1+state['directions'][0,2])))
        g,probes=direct_gradient(search,None,directions,dict(loss=1.,pose_index=0,load_index=0),1e-5)
        np.testing.assert_allclose(g[0],np.array([0.,0.,1.])@tangent_frames(directions)[0],atol=1e-8)
        self.assertTrue(any('difference at boundary' in p['formula'] for p in probes))

if __name__=='__main__':unittest.main()
