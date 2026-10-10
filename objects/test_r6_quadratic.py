import unittest
from pathlib import Path
import numpy as np
from r6_demand import R6Demand
from r6_quadratic import QuadraticR6,features,feature_jacobian,moment_loss


class QuadraticR6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain=R6Demand.read(Path(__file__).parent/'B/poses/pose_1/angle_30/r6_domain.json')
        cls.compiled=QuadraticR6(cls.domain)

    def test_forward_all_angles_and_gravity(self):
        for angle in [15,30,60]:
            d=R6Demand.read(Path(__file__).parent/f'B/poses/pose_1/angle_{angle}/r6_domain.json')
            p=np.array([[.2,.3,0.,0.,0.],[0.,1.,d.alpha,.7,.5],[1.,0.,d.alpha,2.1,.2]])
            np.testing.assert_allclose(QuadraticR6(d).spherical([0,1,2],p,False)['need_wrench'],
                                      d.evaluate([0,1,2],p,False)['need_wrench'],atol=1e-14,rtol=0)

    def test_polynomial_chain_rule(self):
        x=np.array([.23,.31,.03,-.07,-.21]); j=feature_jacobian(x[0],x[1],x[2:])
        np.testing.assert_allclose(self.compiled.matrices[3]@j,
                                  self.domain.jacobian_cartesian(3,*x[:2],x[2:]),atol=1e-14)
        for k in range(5):
            e=np.eye(5)[k]*1e-6
            a,b=x+e,x-e
            np.testing.assert_allclose(j[:,k],(features(a[0],a[1],a[2:])-features(b[0],b[1],b[2:]))/2e-6,atol=1e-10)

    def test_quadratic_integral_and_gradient(self):
        rng=np.random.default_rng(7); a=rng.normal(size=(7,3)); q=np.eye(7)-a@np.linalg.pinv(a)
        scale=np.r_[np.ones(3),np.full(3,7.)]; h=self.compiled.cell_quadratic(4,q,scale)
        z=features([.1,.2,.3],[.4,.2,.1],rng.normal(size=(3,3)))
        b=np.c_[(z@self.compiled.matrices[4].T)*scale,np.zeros(3)]
        w=np.array([.2,.3,.5]); m=np.einsum('k,ki,kj->ij',w,z,z)
        self.assertAlmostEqual(float(moment_loss(h,m)),float(w@(.5*np.sum((b@q)**2,axis=1))),12)
        direction=rng.normal(size=10); eps=1e-6
        numerical=(.5*(z[0]+eps*direction)@h@(z[0]+eps*direction)-.5*(z[0]-eps*direction)@h@(z[0]-eps*direction))/(2*eps)
        self.assertAlmostEqual(float(direction@h@z[0]),float(numerical),7)

    def test_seventh_equation_retained(self):
        q=np.eye(7); q[0,6]=q[6,0]=.3
        h=self.compiled.cell_quadratic(0,q)
        np.testing.assert_allclose(h,self.compiled.matrices[0].T@self.compiled.matrices[0])


if __name__=='__main__': unittest.main()
