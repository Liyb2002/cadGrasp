"""Continuous-domain boundaries and exact geometric certificates."""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step3_scheculer import verification as V


class VerificationTests(unittest.TestCase):
    def domain(self):
        mesh=NS(triangles=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]))
        return NS(domain=NS(mesh=mesh,work_ids=np.array([0]),normals=np.array([[0.,0.,1.]]),
                            com=np.zeros(3),gravity=np.array([0.,0.,-1.]),k=.5,half_angle=np.pi/6))

    def test_cap_interior_is_not_replaced_by_its_rim(self):
        problem=self.domain()
        # Maximizing -need_z chooses d=+z, which is inside the cap, not on its rim.
        value=V.domain_extrema(problem,np.array([np.asarray([0.,0.,-1.,0.,0.,0.])]))[0]
        self.assertAlmostEqual(value['maximum'],-.5)
        self.assertEqual(value['magnitude_mg'],.5)

    def test_zero_magnitude_endpoint_is_included(self):
        value=V.domain_extrema(self.domain(),np.array([np.asarray([0.,0.,1.,0.,0.,0.])]))[0]
        self.assertAlmostEqual(value['maximum'],1.)
        self.assertEqual(value['magnitude_mg'],0.)

    def test_rational_basis_checks_exact_equilibrium(self):
        matrix=[[F(1),F(1,10)],[F(1,3),F(1)]]
        target=[F(2),F(3)]
        coefficients=V.rational_solve(matrix,target)
        self.assertEqual([sum(a*b for a,b in zip(row,coefficients)) for row in matrix],target)
        self.assertTrue(all(v>0 for v in coefficients))

    def test_horizontal_obstruction_is_not_a_3d_wrap_angle_test(self):
        n=np.array([[.2,0.,-1.],[-.2,0.,-1.],[0.,.2,-1.],[0.,-.2,-1.]])
        n/=np.linalg.norm(n,axis=1)[:,None]
        problem=NS(domain=NS(mesh=NS(face_normals=n)))
        contact={'source_faces':np.arange(4)}
        proof=V.horizontal_obstruction(problem,contact)
        self.assertTrue(proof['horizontal_translation_obstructed'])
        self.assertAlmostEqual(proof['largest_projected_normal_gap_degrees'],90.)

    def test_coefficient_error_bound_does_not_accept_small_residual_alone(self):
        from step3_scheculer.enclosure import basis_membership
        targets=np.ones((3,6))
        targets[1,0]=-1e-12
        targets[2,0]=1e-12
        passed,_=basis_membership(np.eye(6),targets)
        np.testing.assert_array_equal(passed,[True,False,False])

    def test_tangent_outer_cap_contains_axis_rim_and_interior_after_rotation(self):
        from scipy.spatial import ConvexHull
        from step3_scheculer import enclosure as E
        n=np.array([[.3,.4,np.sqrt(.75)]])
        e1,e2=E.frame(n);alpha=np.pi/6
        for sides,bands in [(8,1),(16,2),(32,4)]:
            vertices=E.cap_vertices(n,alpha,sides,bands)[0]
            hull=ConvexHull(vertices)
            theta,phi=np.meshgrid(np.linspace(0,alpha,41),np.linspace(0,2*np.pi,157))
            points=np.cos(theta.ravel())[:,None]*n+np.sin(theta.ravel())[:,None]*(
                np.cos(phi.ravel())[:,None]*e1+np.sin(phi.ravel())[:,None]*e2)
            self.assertLessEqual(np.max(points@hull.equations[:,:3].T+hull.equations[:,3]),1e-12)
        # A one-band box used rho=sin(alpha)/cos(pi/sides) even at z=1.
        # Tangency excludes that artificial radial/axial combination.
        vertices=E.cap_vertices(n,alpha,8,1)[0]
        old_maximum=np.hypot(1,np.sin(alpha)/np.cos(np.pi/8))
        self.assertLess(np.linalg.norm(vertices,axis=1).max(),old_maximum-.05)


if __name__=='__main__':
    unittest.main()
