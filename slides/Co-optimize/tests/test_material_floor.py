from pathlib import Path
from types import SimpleNamespace
import sys,unittest
import numpy as np

CO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(CO/'helper_func'),str(CO/'step4.2')]
import _bootstrap
from oracle_search import OracleModel
from whole_search.model import Layout
from material_floor import project_floor_layout
from material_evaluator import floor_height_proposal
from fixture_state import installed_state
from system_floor import system_floor_demands,transform_points


class FloorProjectionTests(unittest.TestCase):
    def test_contact_floor_bound_prevents_a_pressure_only_descent_from_losing_a_port(self):
        m=OracleModel.__new__(OracleModel);m.allow_z_translation=True
        m.native=np.repeat(np.eye(4)[None],3,axis=0);m.native[:,2,3]=.01
        m.native[1,:3,:3]=[[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]]
        m.mesh=SimpleNamespace(center_mass=np.zeros(3));m.points=np.array([[.01,0.,0.]])
        loads=[np.array([[0.,0.,1.,0.,0.,0.]]) for _ in range(3)];loads[2][0,4]=-.045
        m.base_oracle=SimpleNamespace(loads=loads)
        m.contact_delta=SimpleNamespace(state=lambda t:SimpleNamespace(available={k:np.array([True])for k in t.active}))
        q=Layout(np.repeat(np.eye(4)[None],3,axis=0),np.tile([0.,0.,1.],(3,1)),np.array([0,1,0]),(0,1,2))
        q.placements[2,0,3]=-.05;q.state_offsets[1,2]=.1
        unprotected=project_floor_layout(m,q,fixed_seats=(2,))
        protected=project_floor_layout(m,q,protect_ports=True,fixed_seats=(2,))
        self.assertIsNotNone(unprotected);self.assertIsNotNone(protected)
        self.assertAlmostEqual(unprotected.state_offsets[1,2],0.)
        self.assertAlmostEqual(protected.state_offsets[1,2],.03)
        np.testing.assert_array_equal(protected.placements,q.placements)
        for k in protected.active:
            p=transform_points(m.points,protected.placements[k])
            for host in [0,1]:
                T=installed_state(m.native,protected,host)
                self.assertGreaterEqual(float((p@T[2,:3]+T[2,3]).min()),-1e-9)

    def test_seat_motion_fits_pressure_when_both_registered_states_stay_grounded(self):
        m=OracleModel.__new__(OracleModel)
        m.native=np.repeat(np.eye(4)[None],3,axis=0);m.native[:,2,3]=.01
        m.native[1,:3,:3]=[[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]]
        m.allow_z_translation=True;m.mesh=SimpleNamespace(center_mass=np.zeros(3))
        m.base_oracle=SimpleNamespace(loads=[np.array([[0.,0.,1.,0.,0.,0.]])]*3)
        q=Layout(np.repeat(np.eye(4)[None],3,axis=0),np.tile([0.,0.,1.],(3,1)),np.array([0,1,0]),(0,1,2))
        q.placements[2,0,3]=-.05
        t=project_floor_layout(m,q,fixed_ground=(0,1))
        self.assertIsNotNone(t)
        np.testing.assert_array_equal(t.state_offsets,q.state_offsets)
        self.assertAlmostEqual(t.placements[2,0,3],-.01)
        np.testing.assert_array_equal(q.placements[2,:3,3],[-.05,0.,0.])
        for k in t.active:
            m.validate_placement(t,k)
            T=installed_state(m.native,t,t.hosts[k]);com=transform_points(m.mesh.center_mass[None],T@t.placements[k])[0]
            pressure=transform_points(system_floor_demands(m.base_oracle.loads[k],com)['floor_demands_world_m'],np.linalg.inv(T))
            for h in [0,1]:
                B=installed_state(m.native,t,h)
                self.assertGreaterEqual(float((pressure@B[2,:3]+B[2,3]).min()),-1e-9)


if __name__=='__main__':unittest.main()
