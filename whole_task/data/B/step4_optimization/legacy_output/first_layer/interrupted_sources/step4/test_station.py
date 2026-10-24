"""Regressions for the adaptations that differ from the reference Step4."""
from types import SimpleNamespace
import unittest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .adapter import TaskModel, FiniteAccess
from .search import shift_seat, rehost, lock_seats
from whole_search.model import Layout
from fixture_state import installed_state


class StationTests(unittest.TestCase):
    def setUp(self):
        native=np.repeat(np.eye(4)[None],3,axis=0)
        native[:,:3,:3]=Rotation.from_euler('zyx',[[0,0,0],[40,15,20],[-30,50,10]],degrees=True).as_matrix()
        native[:,:3,3]=[[0,0,.2],[.1,-.2,.3],[-.2,.1,.25]]
        self.model=SimpleNamespace(native=native)
        self.model.validate_placement=lambda q,k:TaskModel.validate_placement(self.model,q,k)
        self.model.check_station=lambda q:TaskModel.check_station(self.model,q)
        self.model.floor_normal=lambda q,k:native[q.hosts[k],:3,:3].T@np.array([0,0,1])
        self.layout=Layout(np.repeat(np.eye(4)[None],3,axis=0),np.array([R.T@[0,0,1]for R in native[:,:3,:3]]),
                           np.arange(3),(0,1,2))

    def test_registered_seats_remain_exact_identities(self):
        canonical=lock_seats(self.model,self.layout)
        np.testing.assert_array_equal(canonical.placements,self.layout.placements)
        moved=shift_seat(self.model,self.layout,1,[.01,0,0])
        for k in moved.active:
            np.testing.assert_array_equal(moved.placements[k,:3,:3],np.eye(3))

    def test_uncoupled_world_translation_is_rejected(self):
        q=self.layout.copy();q.state_offsets[1,0]=.01
        with self.assertRaisesRegex(ValueError,'robot work pose'):
            self.model.validate_placement(q,1)

    def test_coupled_translation_preserves_actual_world_task(self):
        delta=np.array([.01,-.02,.03])
        q=shift_seat(self.model,self.layout,1,delta)
        np.testing.assert_allclose(q.placements[1,:3,3],delta,atol=1e-15)
        self.model.check_station(q)
        np.testing.assert_array_equal(q.placements[0],self.layout.placements[0])

    def test_rehosting_preserves_pose_and_world_exit(self):
        before=self.model.native[1,:3,:3]@self.layout.directions[1]
        q=rehost(self.model,self.layout,1,0)
        self.model.check_station(q)
        self.assertEqual(q.hosts[1],0)
        np.testing.assert_allclose(self.model.native[0,:3,:3]@q.directions[1],before,atol=1e-15)

    def test_shared_installation_shift_couples_all_its_seats(self):
        q=rehost(self.model,self.layout,1,0)
        moved=shift_seat(self.model,q,1,[.01,.02,-.005])
        self.model.check_station(moved)
        for k in (0,1):
            np.testing.assert_allclose(moved.placements[k,:3,3]-q.placements[k,:3,3],[.01,.02,-.005],atol=1e-15)
        np.testing.assert_array_equal(moved.placements[2],q.placements[2])

    def test_finite_access_never_adopts_legacy_infinite_cap(self):
        mesh=trimesh.creation.box(extents=[.1,.1,.1]);ids=np.flatnonzero(mesh.face_normals[:,2]>.9)
        work=FiniteAccess(mesh,ids,[0,0,.1])
        self.assertIs(work.solid(.5),work.solid(1000))
        np.testing.assert_array_equal(work.contains_points([[0,0,.075],[0,0,.04],[0,0,.11]]),[True,False,False])
        self.assertEqual(work.required_length([[0,0,-1000]],1000),0.)

    def test_contact_core_simplification_preserves_clearance(self):
        from whole_task.step3.legacy import kernel as K
        def box(extents,center):
            mesh=trimesh.creation.box(extents=extents);mesh.apply_translation(center)
            return K.S.solid(mesh)
        seed=box([.1,.1,.1],[0,0,0])-box([.08,.08,.08],[0,0,0])
        nominal=box([.2,.04,.04],[.08,0,0])
        padded=box([.22,.05,.05],[.08,0,0])
        core=(seed-nominal)^box([.01,.1,.1],[.045,0,0])
        old=((seed-padded)+core)-nominal
        old=old-(padded-core)
        simple=(seed-padded)+core
        self.assertLess(K.material_volume(old-simple)+K.material_volume(simple-old),1e-14)
        self.assertLess(K.material_volume(simple^nominal),1e-14)
        self.assertLess(K.material_volume((simple-core)^padded),1e-14)
        self.assertLess(K.material_volume(core-simple),1e-14)

    def test_work_head_remains_fixed_under_coupled_move(self):
        h=np.array([.2,-.4,.5]);k=1
        local=np.linalg.inv(self.model.native[k])@np.r_[h,1]
        q=shift_seat(self.model,self.layout,k,[.02,0,0])
        actual=installed_state(self.model.native,q,q.hosts[k])@q.placements[k]@local
        np.testing.assert_allclose(actual[:3],h,atol=1e-15)


if __name__=='__main__':unittest.main()
