"""Discriminating checks for force reference frames and contact assumptions."""
import unittest
import mujoco
import numpy as np
from cases import smoke, evaluate, random_cases
from scene import build
from equilibrium import solve as solve_quasistatic, geometric_contact_check
from video import apply_load


class PhysicsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.report=build()

    def test_free_bodies_and_unilateral_normal_only_support(self):
        m=self.model;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        self.assertEqual(m.nv,12);self.assertEqual(m.neq,0)
        self.assertTrue(np.all(m.dof_damping==0));self.assertTrue(np.all(m.dof_frictionloss==0))
        contacts=[c for c in d.contact if set(m.geom_bodyid[c.geom])=={1,2}]
        self.assertGreater(len(contacts),0)
        self.assertEqual({int(c.dim) for c in contacts},{1})
        self.assertGreater(self.report['object_tetrahedra']['count'],1)
        self.assertTrue(self.report['object_tetrahedra']['boundary_matches_source_triangles'])

    def test_off_com_force_produces_correct_free_body_acceleration(self):
        # Independent Newton/Euler reference, with collision and gravity disabled.
        m=self.model;d=mujoco.MjData(m)
        oldgravity=m.opt.gravity.copy();oldtype=m.geom_contype.copy();oldaff=m.geom_conaffinity.copy()
        try:
            m.opt.gravity[:]=0;m.geom_contype[:]=0;m.geom_conaffinity[:]=0
            mujoco.mj_forward(m,d)
            b=m.body('object').id;F=np.array([.71,-.43,.29]);arm=np.array([.021,-.014,.032])
            mujoco.mj_applyFT(m,d,F,np.zeros(3),d.xipos[b]+arm,b,d.qfrc_applied)
            mujoco.mj_forward(m,d)
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacBodyCom(m,d,jp,jr,b)
            np.testing.assert_allclose(jp@d.qacc,F/m.body_mass[b],atol=1e-9)
            R=d.ximat[b].reshape(3,3);I=R@np.diag(m.body_inertia[b])@R.T
            np.testing.assert_allclose(jr@d.qacc,np.linalg.solve(I,np.cross(arm,F)),atol=1e-9)
        finally:
            m.opt.gravity[:]=oldgravity;m.geom_contype[:]=oldtype;m.geom_conaffinity[:]=oldaff

    def test_physical_load_domain_and_reproducibility(self):
        cases=smoke();self.assertEqual(len(cases),10)
        for c in cases:self.assertLessEqual(np.linalg.norm(c['force_body_mg']),.5+1e-12)
        a=random_cases(3,41);b=random_cases(3,41);self.assertEqual(a,b)
        self.assertNotEqual(a,random_cases(3,42))
        invalid=dict(cases[0]);invalid['parameters']=list(invalid['parameters']);invalid['parameters'][-1]=.501
        with self.assertRaises(ValueError):evaluate(invalid)

    def test_static_check_cannot_balance_bunny_on_numerical_floor_force_couples(self):
        m=self.model;case=dict(smoke()[0],force_body_mg=[0.,0.,0.])
        self.assertEqual(solve_quasistatic(m,case)[0]['status'], 'equilibrium_feasible')
        oldaff=m.geom_conaffinity.copy()
        try:
            m.geom_conaffinity[(m.geom_bodyid>0)&(m.geom_contype>0)]=4
            self.assertEqual(solve_quasistatic(m,case)[0]['status'], 'equilibrium_unresolved')
        finally:m.geom_conaffinity[:]=oldaff

    def test_constant_force_and_arrow_do_not_follow_motion_or_disappear(self):
        m=self.model;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        point=np.array([.01,-.03,.12]);force=np.array([1.,2.,-.5])
        for t in [0.,1.,3.,4.5,6.,100.]:
            d.qpos[:3]=[.1,.2,.3];d.qpos[3:7]=[np.cos(.3),0.,0.,np.sin(.3)]
            mujoco.mj_forward(m,d)
            apply_load(m,d,1,point,force)
            expected=np.zeros(m.nv)
            mujoco.mj_applyFT(m,d,force,np.zeros(3),point,1,expected)
            np.testing.assert_allclose(d.qfrc_applied,expected,atol=1e-12)
            np.testing.assert_array_equal(d.qfrc_applied[:3],force)
            np.testing.assert_allclose(d.qfrc_applied[3:6],d.xmat[1].reshape(3,3).T@np.cross(point-d.xpos[1],force),atol=1e-12)

    def test_quasistatic_certificate_and_impossible_upward_load(self):
        model=self.model;case=smoke()[0]
        for weight in [True,False]:
            result,_=solve_quasistatic(model,case,support_weight=weight)
            self.assertEqual(result['status'],'equilibrium_feasible')
            self.assertLess(result['max_force_balance_residual_N'],1e-7)
            self.assertEqual(result['dynamic_integration_steps'],0)
            data=mujoco.MjData(model);mujoco.mj_forward(model,data);cache={}
            for reaction in result['reaction_witness']:
                contact=data.contact[reaction['contact_index']]
                self.assertTrue(geometric_contact_check(model,data,contact,cache)['valid'])
        impossible=dict(case,force_body_mg=[0.,0.,2.])
        result,_=solve_quasistatic(model,impossible)
        self.assertEqual(result['status'],'equilibrium_unresolved')


if __name__=='__main__':unittest.main()
