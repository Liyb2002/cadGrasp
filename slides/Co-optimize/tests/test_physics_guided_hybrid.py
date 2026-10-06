"""Real-dual feedback, immutable pool proposal legality and nominal shadowing checks."""

import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
import numpy as np
from physics_guided_cone import cone_projection
from physics_guided_dual import certificate_candidates,deficit_contact_values
from hybrid_directions import project_common,fibonacci
from physics_guided_contact_sweep import NominalContactSweep
from co_common import trimesh
from exit_clearance import ExitClearance
from physics_guided_geometry import tangent_frames


class HybridTests(unittest.TestCase):
    def test_value_comes_from_actual_missing_force(self):
        full=np.eye(7)[[0]];target=np.array([1.,1.,0,0,0,0,0])
        projection=cone_projection(full,target)
        candidates=np.vstack([np.eye(7)[1],np.eye(7)[0],-np.eye(7)[1]])
        weights,info=deficit_contact_values([candidates],[(0,projection)])
        np.testing.assert_allclose(weights,[1.,0.,0.],atol=1e-12)
        self.assertAlmostEqual(info['maximum_fixed_allocation_gain'],.5)
        recovered=cone_projection(np.vstack([full,candidates[0]]),target)
        self.assertLess(recovered['loss'],1e-20)

    def test_positive_ray_rescaling_does_not_change_feedback(self):
        projection=cone_projection(np.eye(7)[[0]],np.array([1.,1.,0,0,0,0,0]))
        rays=np.eye(7)[[1,2]]
        a,_=deficit_contact_values([rays],[(0,projection)])
        b,_=deficit_contact_values([rays*np.array([100.,.001])[:,None]],[(0,projection)])
        np.testing.assert_allclose(a,b)

    def test_certificate_scan_finds_hidden_extreme_in_saved_order(self):
        targets=np.zeros((100,7));targets[:,0]=1.;targets[:,1]=np.linspace(.01,.2,100)
        targets[53,1]=2.
        ids,info=certificate_candidates(targets,np.ones(100,bool),lambda i:cone_projection(np.eye(7)[[0]],targets[i]))
        self.assertEqual(ids[0],53)
        self.assertEqual(info['scanned'],100)
        self.assertLess(info['projected'],100)

    def test_shared_tendencies_remain_floor_legal_including_antipodes(self):
        normals=np.vstack([np.eye(3),-np.eye(3)])
        for a in np.vstack([fibonacci(30),normals]):
            directions=project_common(a,normals)
            np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1.,atol=1e-12)
            self.assertTrue(np.all(np.sum(directions*normals,axis=1)>=-1e-12))

    def test_nominal_core_cost_does_not_impose_padding_on_allowed_contacts(self):
        mesh=trimesh.creation.box([.02,.02,.02]);point=np.array([[.01,0,0]])
        model=NominalContactSweep(ExitClearance(mesh),point,np.array([[1.,0,0]]),.06,.02)
        clear=model.distances(np.array([-1.,0,0]))[0]
        blocked=model.distances(np.array([1.,0,0]))[0]
        self.assertLess(clear,.002)
        self.assertGreater(blocked,.1)
        self.assertEqual(model.margin,0.)

    def test_nominal_trajectory_sensitivity(self):
        mesh=trimesh.creation.box([.02,.02,.02]);points=np.array([[.01,.006,0]])
        model=NominalContactSweep(ExitClearance(mesh),points,np.array([[1.,0,0]]),.06,.02)
        d=np.array([[.8,.4,.4472135955]]);d/=np.linalg.norm(d,axis=1)[:,None]
        frames=tangent_frames(d);values,jac=model.linearize(d,frames)
        step=1e-4
        for axis in range(2):
            plus=d[0]+step*frames[0,:,axis];plus/=np.linalg.norm(plus)
            minus=d[0]-step*frames[0,:,axis];minus/=np.linalg.norm(minus)
            finite=(model.distances(plus)-model.distances(minus))/(2*step)
            np.testing.assert_allclose(finite,jac[:,0,axis],rtol=.02,atol=1e-4)


if __name__=='__main__':unittest.main()
