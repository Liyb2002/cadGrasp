"""Independent derivative, inverse-map and physical-domain checks."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from r6_demand import R6Demand, expression_from_domain


class R6ExpressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(__file__).parent/"B/poses/pose_1"

    def domain(self, angle=30):
        data = json.loads((self.folder/f"angle_{angle}/needs.json").read_text())
        expression = expression_from_domain(data, "test-only")
        domain = R6Demand(expression, data["geometry"])
        parameters = np.tile([1/3,1/3,0.,0.,0.], (len(domain.a),1))
        reachable = domain.evaluate(np.arange(len(domain.a)), parameters)["reachable"]
        face = int(np.flatnonzero(reachable)[0])
        expression["zero_demand_witness"] = dict(work_face_index=face, parameters=parameters[face].tolist())
        return domain, data

    def test_forward_and_eliminated_membership_at_all_three_angles(self):
        rng = np.random.default_rng(9173)
        for angle in [15,30,60]:
            domain, _ = self.domain(angle)
            count = 90; face = np.arange(count)%len(domain.a)
            uv = rng.random((count,2)); uv[uv.sum(axis=1)>1] = 1-uv[uv.sum(axis=1)>1]
            parameters = np.c_[uv, rng.uniform(.01,domain.alpha,count),
                               rng.uniform(0,2*np.pi,count), rng.uniform(.005,.5,count)]
            values = domain.evaluate(face, parameters)
            self.assertTrue(values["reachable"].any())
            self.assertTrue(domain.contains(values["need_wrench"][values["reachable"]]).all())
            boundary = np.asarray([[0.,0.,0.,0.,0.], [1.,0.,domain.alpha,0.,.5],
                                   [0.,1.,domain.alpha,2*np.pi,.5]])
            values = domain.evaluate(np.full(3,face[0]), boundary, False)
            self.assertTrue(domain.contains(values["need_wrench"],False).all())

    def test_cartesian_jacobian_against_independent_finite_differences(self):
        domain, _ = self.domain()
        face = 3; variables = np.array([.2,.3,.04,-.03,-.2]); step = 1e-6
        jac = domain.jacobian_cartesian(face,*variables[:2],variables[2:])
        def forward(x):
            r = domain.a[face]+x[0]*domain.e[face]+x[1]*domain.f[face]
            return np.r_[-domain.gravity-x[2:], -np.cross(r,x[2:])]
        for column in range(5):
            delta = np.eye(5)[column]*step
            numerical = (forward(variables+delta)-forward(variables-delta))/(2*step)
            np.testing.assert_allclose(jac[:,column],numerical,atol=2e-10,rtol=1e-8)

    def test_spherical_jacobian_and_zero_force_rank(self):
        domain, _ = self.domain()
        face = 2; p = np.array([.2,.3,.17,.73,.25]); step = 1e-6
        jac = domain.jacobian_spherical(face,p)
        for column in range(5):
            delta = np.eye(5)[column]*step
            numerical = (domain.evaluate(face,p+delta,False)["need_wrench"]-
                         domain.evaluate(face,p-delta,False)["need_wrench"])/(2*step)
            np.testing.assert_allclose(jac[:,column],numerical,atol=2e-10,rtol=1e-8)
        p[-1] = 0.
        zero = domain.jacobian_spherical(face,p)
        np.testing.assert_array_equal(zero[:,:4],0.)
        self.assertEqual(np.linalg.matrix_rank(zero),1)

    def test_impossible_torsion_and_force_are_rejected(self):
        domain, _ = self.domain()
        values = domain.evaluate(0,np.array([.2,.3,.1,.7,.25]),False)
        original = values["need_wrench"]
        impossible = original.copy(); impossible[3:] += .1*values["force_push_mg"]
        self.assertFalse(bool(domain.contains(impossible,False)))
        impossible = original.copy(); impossible[:3] = -domain.gravity-3*values["force_push_mg"]
        self.assertFalse(bool(domain.contains(impossible,False)))
        self.assertTrue(bool(domain.contains(np.r_[-domain.gravity,[0.,0.,0.]],False)))
        self.assertFalse(bool(domain.contains(np.r_[-domain.gravity,[0.,0.,.01]],False)))

    def test_rigid_translation_preserves_com_wrenches(self):
        first, data = self.domain()
        shifted = copy.deepcopy(data); delta = np.array([.13,-.07,.01])
        shifted["geometry"]["vertices_m"] = (np.asarray(data["geometry"]["vertices_m"])+delta).tolist()
        shifted["frame"]["moment_origin_m"] = (np.asarray(data["frame"]["moment_origin_m"])+delta).tolist()
        transform = np.asarray(data["frame"]["T_world_mesh"]); transform[:3,3] += delta
        shifted["frame"]["T_world_mesh"] = transform.tolist()
        shifted["load"]["gravity_application_point_m"] = shifted["frame"]["moment_origin_m"]
        second = R6Demand(expression_from_domain(shifted,"test-only"),shifted["geometry"])
        p = np.array([.2,.3,.17,.73,.25])
        a,b = first.evaluate(0,p,False),second.evaluate(0,p,False)
        np.testing.assert_allclose(a["need_wrench"],b["need_wrench"],atol=1e-13,rtol=0)
        np.testing.assert_allclose(b["pt_m"]-a["pt_m"],delta,atol=1e-13,rtol=0)

    def test_thin_triangle_inverse_remains_stable(self):
        path = Path(__file__).parent/"C2/poses/pose_11/angle_30/needs.json"
        data = json.loads(path.read_text())
        domain = R6Demand(expression_from_domain(data,"test-only"),data["geometry"])
        edges = np.stack([domain.e,domain.f],axis=-1)
        self.assertGreater(np.linalg.cond(edges).max(),1e5)
        np.testing.assert_allclose(domain.barycentric@edges,np.broadcast_to(np.eye(2),(len(edges),2,2)),
                                   atol=5e-10,rtol=0)
        face = int(np.argmax(np.linalg.cond(edges)))
        parameters = np.array([.23,.31,.2,.7,.21])
        value = domain.evaluate(face,parameters,False)["need_wrench"]
        p = -value[:3]-domain.gravity
        r = (np.cross(value[3:],domain.n[face])+domain.h[face]*p)/(domain.n[face]@p)
        uv = domain.barycentric[face]@(r-domain.a[face])
        np.testing.assert_allclose(uv,parameters[:2],atol=1e-8,rtol=0)


if __name__ == "__main__":
    unittest.main()
