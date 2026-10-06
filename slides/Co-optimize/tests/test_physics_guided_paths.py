
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _bootstrap
from co_common import *
from exit_clearance import ExitClearance
from physics_guided_geometry import tangent_frames
from physics_guided_paths import MaterialPathModel


class MaterialPathTests(unittest.TestCase):
    def test_graph_tracks_existing_material_and_path_gradient(self):
        seed=trimesh.creation.box([.06,.04,.04])
        moving=trimesh.creation.box([.01,.01,.01])
        points=np.array([[-.029,0,0],[.029,0,0]])
        normals=np.array([[1.,0,0],[-1.,0,0]])
        model=MaterialPathModel(seed,ExitClearance(moving),points,normals,.04,.06)
        anchor=trimesh.creation.box([.035,.05,.05]);anchor.apply_translation([-.015,0,0])
        d=np.array([[0.,0.,1.]])
        values,jac=model.linearize(d,tangent_frames(d),anchor)
        z=np.array([[.01,-.01]])
        cost,gradient=model.cost(values,jac,z)
        self.assertTrue(np.isfinite(cost).all())
        self.assertTrue((cost>=0).all())
        for k in range(2):
            delta=np.zeros_like(z);delta[0,k]=1e-6
            finite=(model.cost(values,jac,z+delta)[0]-model.cost(values,jac,z-delta)[0])/2e-6
            np.testing.assert_allclose(finite,gradient[:,0,k],atol=1e-8)

    def test_initially_disconnected_material_has_no_invented_bridge(self):
        a=trimesh.creation.box([.02,.02,.02]);a.apply_translation([-.03,0,0])
        b=trimesh.creation.box([.02,.02,.02]);b.apply_translation([.03,0,0])
        seed=trimesh.util.concatenate([a,b]);moving=trimesh.creation.box([.005,.005,.005])
        p=np.array([[-.021,0,0],[.021,0,0]]);n=np.array([[-1.,0,0],[1.,0,0]])
        model=MaterialPathModel(seed,ExitClearance(moving),p,n,.02,.08)
        d=np.array([[0.,0.,1.]])
        values,jac=model.linearize(d,tangent_frames(d),a)
        self.assertEqual(model.fixed_cost[0],0.)
        self.assertEqual(model.fixed_cost[1],2.)
        self.assertEqual(model.path_matrix[1].nnz,0)


if __name__=='__main__':unittest.main()
