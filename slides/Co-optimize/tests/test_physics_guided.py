"""Review checks for physical-envelope and geometric direction sensitivities."""

import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from physics_guided import *
from physics_guided_objective import acquisition_equilibrium


class GeometryGradientTests(unittest.TestCase):
    def test_normal_cost_gradient(self):
        normals=np.array([[1.,0,0],[0,1.,0],[-1.,0,0]])
        d=np.array([[.2,.7,.68],[-.5,.6,.62]]);d/=np.linalg.norm(d,axis=1)[:,None]
        cost,gradient=acquisition(d,normals)
        for i in range(2):
            for k in range(3):
                delta=np.zeros_like(d);delta[i,k]=1e-6
                finite=(acquisition(d+delta,normals)[0]-acquisition(d-delta,normals)[0])/2e-6
                np.testing.assert_allclose(gradient[:,i,k],finite,atol=1e-8,rtol=1e-5)

    def test_spherical_update_and_jacobian(self):
        d=np.array([[0.,0.,1.],[.6,0.,.8]])
        frame=tangent_frames(d);z=np.array([[.1,-.2],[.03,.15]])
        updated=retract(d,frame,z)
        np.testing.assert_allclose(np.linalg.norm(updated,axis=1),1.,atol=1e-14)
        jac=retraction_jacobian(d,frame,z)
        for i in range(2):
            for k in range(2):
                delta=np.zeros_like(z);delta[i,k]=1e-6
                finite=(retract(d,frame,z+delta)-retract(d,frame,z-delta))/2e-6
                np.testing.assert_allclose(finite[i],jac[i,:,k],atol=1e-9)

    def test_sweep_distance_captures_nonlocal_shadowing(self):
        # The upper box side is tangential to +Z. Normal-only guidance cannot
        # see the wider lower box moving through its outward bearing material.
        upper=trimesh.creation.box([.02,.02,.01]);upper.apply_translation([0,0,.04])
        lower=trimesh.creation.box([.05,.02,.01])
        pole=trimesh.creation.box([.005,.005,.04]);pole.apply_translation([0,0,.02])
        combined=trimesh.util.concatenate([upper,lower])
        p=np.array([[.01,0,.04]]);n=np.array([[1.,0,0]])
        a=SweepDistanceModel(ExitClearance(upper),p,n,.08,.05)
        b=SweepDistanceModel(ExitClearance(combined),p,n,.08,.05)
        self.assertLess(a.distances(np.array([0.,0.,1.]))[0],0.)
        self.assertGreater(b.distances(np.array([0.,0.,1.]))[0],0.)

    def test_full_sweep_distance_derivative_stability(self):
        mesh=trimesh.creation.box([.02,.02,.02])
        points=np.array([[.018,.009,.035],[.028,.015,.055]])
        d=np.array([[.6,.2,.77]]);d/=np.linalg.norm(d,axis=1)[:,None]
        frames=tangent_frames(d)
        a=SweepDistanceModel(ExitClearance(mesh),points,np.zeros_like(points),.08,.02,depth=0.,step=2e-4)
        values,gradient=a.linearize(d,frames)
        for step in [2e-4,1e-4]:
            for axis in range(2):
                plus=d[0]+step*frames[0,:,axis];plus/=np.linalg.norm(plus)
                minus=d[0]-step*frames[0,:,axis];minus/=np.linalg.norm(minus)
                finite=(a.distances(plus)-a.distances(minus))/(2*step)
                np.testing.assert_allclose(finite,gradient[:,0,axis],rtol=2e-3,atol=1e-4)

    def test_sweep_local_cost_jacobian(self):
        values=np.array([[.1,-.2],[-.05,.03]])
        jac=np.array([[[.2,.3],[-.4,.1]],[[.3,-.1],[.2,.5]]])
        z=np.array([[.02,-.01],[.01,.03]])
        cost,g=distance_cost(values,jac,z)
        for i in range(2):
            for k in range(2):
                dz=np.zeros_like(z);dz[i,k]=1e-6
                finite=(distance_cost(values,jac,z+dz)[0]-distance_cost(values,jac,z-dz)[0])/2e-6
                np.testing.assert_allclose(finite,g[:,i,k],atol=1e-8)


class PhysicsEnvelopeTests(unittest.TestCase):
    def test_envelope_gradient_and_positive_ray_scaling(self):
        rays=np.eye(7)[:2];floor=np.empty((0,7));target=np.array([1.,2.,0,0,0,0,0])
        costs=np.array([.1,.2]);physical=acquisition_equilibrium(floor,rays,target,costs)
        self.assertLess(np.linalg.norm(physical['residual']),1e-12)
        for j in range(2):
            delta=np.zeros(2);delta[j]=1e-6
            fd=(acquisition_equilibrium(floor,rays,target,costs+delta)['value']-
                acquisition_equilibrium(floor,rays,target,costs-delta)['value'])/2e-6
            self.assertAlmostEqual(fd,physical['cost_gradient'][j],places=8)
        scaled=acquisition_equilibrium(floor,rays*np.array([.01,100.])[:,None],target,costs)
        self.assertAlmostEqual(scaled['value'],physical['value'],places=12)

    def test_expensive_missing_contact_prefers_residual(self):
        target=np.array([1.,0,0,0,0,0,0])
        physical=acquisition_equilibrium(np.empty((0,7)),np.eye(7)[:1],target,np.array([100.]))
        self.assertAlmostEqual(physical['value'],1.)
        self.assertEqual(physical['reactions'][0],0.)


if __name__=='__main__':unittest.main()
