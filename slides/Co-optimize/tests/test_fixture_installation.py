"""Independent installed-state motion, zero-margin floors and saved transforms."""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import sys, tempfile, unittest
import numpy as np
import trimesh

COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/'helper_func'),str(COOPT/'step4.2')]
import _bootstrap
from whole_search.model import Layout, Model
from whole_search.delta_guidance import ContactDelta
from oracle_search import OracleModel
from pose_oracle import PoseOracleConstraints
from base_oracle import BaseOracle
from fixture_state import installed_state, installed_states
from ground_support import CONTACT_CLEARANCE_M, allowed_contacts
from translation.installation import shift_installation, installation_gradient
from whole_step4_render import NominalBuilder
from whole_search.common import C, unpack_solid


class InstallationTests(unittest.TestCase):
    def fixture(self):
        model=OracleModel.__new__(OracleModel)
        model.native=np.repeat(np.eye(4)[None],2,axis=0)
        model.native[1,:3,:3]=[[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]]
        model.allow_z_translation=True;model.support_floor_clearance_m=CONTACT_CLEARANCE_M
        model.poses=['A','B']
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),
                      np.array([[0.,0.,1.],[1.,0.,0.]]),np.arange(2),(0,1))
        return model,layout

    def test_installed_motion_moves_whole_shared_state_and_preserves_task_rotations(self):
        model,q=self.fixture();q.hosts[:]=0
        q.placements[1,:3,:3]=model.native[1,:3,:3]
        before=installed_states(model.native,q)
        trial=shift_installation(model,q,[0],[.02,-.03,.04])
        for k in q.active:
            old=before[q.hosts[k]]@q.placements[k]
            new=installed_state(model.native,trial,trial.hosts[k])@trial.placements[k]
            np.testing.assert_allclose(new[:3,3]-old[:3,3],[.02,-.03,.04])
            np.testing.assert_array_equal(new[:3,:3],old[:3,:3])
        np.testing.assert_array_equal(trial.placements,q.placements)
        np.testing.assert_array_equal(q.state_offsets,np.zeros((2,3)))

    def test_cache_and_save_distinguish_installations_with_identical_object_seats(self):
        model,q=self.fixture();trial=shift_installation(model,q,[1],[.01,.02,.03])
        self.assertNotEqual(q.key(),trial.key())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'layout.npz';np.savez_compressed(path,**trial.arrays(model.native))
            with np.load(path) as saved:
                loaded=Layout.from_arrays(saved)
                np.testing.assert_array_equal(saved['native_world_original'],model.native)
                np.testing.assert_allclose(saved['native_world'],installed_states(model.native,trial))
            self.assertEqual(trial.key(),loaded.key())
        legacy={k:v for k,v in q.arrays().items() if k!='state_offsets'}
        self.assertEqual(Layout.from_arrays(legacy).key(),q.key())

    def test_other_state_floor_releases_contacts_without_rebuilding_geometry_pairs(self):
        model,q=self.fixture()
        model.points=np.array([[-.001,0.,.0002],[.0001,0.,.0001]])
        model.sources=np.arange(2);model.allowed=[np.arange(2)]*2
        model.mesh=SimpleNamespace(face_normals=np.tile([0.,0.,-1.],(2,1)))
        model.material_at_points=lambda *a:np.ones(2,bool)
        model.locks=lambda *a:np.zeros(2,bool)
        engine=ContactDelta(model);old=engine.state(q);updates=engine.pair_updates
        np.testing.assert_array_equal(old.available[0],[False,True])
        trial=shift_installation(model,q,[1],[0.,0.,.002])
        np.testing.assert_array_equal(engine.state(trial).available[0],[True,True])
        self.assertEqual(engine.pair_updates,updates)
        np.testing.assert_array_equal(old.available[0],[False,True])

    def test_zero_margin_keeps_arbitrarily_close_and_on_floor_contacts(self):
        model,q=self.fixture()
        self.assertEqual(CONTACT_CLEARANCE_M,0.)
        model.points=np.array([[0.,0.,0.],[1e-7,0.,1e-7],[-1e-4,0.,.02]])
        np.testing.assert_array_equal(allowed_contacts(model,q,0),[True,True,False])

    def test_installed_floor_cache_changes_but_common_work_cone_does_not(self):
        model,q=self.fixture()
        work=SimpleNamespace(planes=np.array([[[-1.,0,0,0]]]),counts=np.array([1]),
                             face_ids=np.array([0]),half_angle_deg=30.)
        model.work_rays=[work,work]
        oracle=BaseOracle.__new__(BaseOracle);oracle.model=model;oracle.constraint_cache=OrderedDict()
        old=oracle.pose_constraints(q)
        trial=shift_installation(model,q,[1],[.02,.03,.04]);new=oracle.pose_constraints(trial)
        self.assertIsNot(old,new)
        np.testing.assert_allclose(new.floor_planes_common[1,3]-old.floor_planes_common[1,3],.04)
        np.testing.assert_array_equal(new.poses[1].work_common,old.poses[1].work_common)
        np.testing.assert_allclose(new.poses[1].T_world_object[:3,3]-old.poses[1].T_world_object[:3,3],
                                   [.02,.03,.04])

    def test_installation_gradient_treats_world_axes_equally(self):
        model,q=self.fixture();q.state_offsets[:,2]=.1
        objective=lambda t:float(np.sum((t.state_offsets[0]-[.02,.03,.04])**2))
        g=installation_gradient(model,q,(0,),1e-4,objective)
        np.testing.assert_allclose(g,[-.04,-.06,.12],atol=1e-12)
        lowered=shift_installation(model,q,[0],[0.,0.,-1.])
        self.assertAlmostEqual(model.workpiece_height(lowered,0),0.)

    def test_airborne_installation_removes_object_floor_reactions(self):
        model,q=self.fixture();model.floors=[np.arange(42).reshape(6,7)]*2
        raised=shift_installation(model,q,[0],[0.,0.,.01])
        self.assertEqual(len(model.floor_columns(q,0)),6)
        self.assertEqual(len(model.floor_columns(raised,0)),1)
        model.validate_placement(raised,0)

    def test_restoring_rotating_reuse_preserves_existing_airborne_position(self):
        model,q=self.fixture();q.hosts[1]=0
        q.placements[1]=np.linalg.inv(model.native[0])@model.native[1]
        q.placements[1,:3,3]=[.01,.02,.03];q.state_offsets[0]=[.04,.05,.06]
        q.directions[1]=[0.,0.,1.]
        old_direction=model.native[q.hosts[1],:3,:3]@q.directions[1]
        before=[installed_state(model.native,q,q.hosts[k])@q.placements[k] for k in q.active]
        trial=model.restore_registered(q,1)
        self.assertEqual(trial.hosts[1],1)
        np.testing.assert_allclose(model.native[trial.hosts[1],:3,:3]@trial.directions[1],old_direction)
        self.assertGreater(trial.directions[1]@model.floor_normal(trial,1),0.)
        np.testing.assert_array_equal(trial.placements[1],np.eye(4))
        for k in q.active:
            np.testing.assert_allclose(installed_state(model.native,trial,trial.hosts[k])@trial.placements[k],before[k])

    def test_nominal_floor_clipping_tracks_new_installation(self):
        model,q=self.fixture();q.active=(0,)
        model.registered_seed=C.S.solid(trimesh.creation.box([.02]*3))
        builder=NominalBuilder(model);builder.part=lambda *a:C.union([])
        old=unpack_solid(builder.solid(q))
        raised=shift_installation(model,q,[0],[0.,0.,.02])
        new=unpack_solid(builder.solid(raised))
        self.assertAlmostEqual(new.volume,2*old.volume,places=12)
        self.assertGreaterEqual((new.vertices@installed_state(model.native,raised,0)[:3,:3].T+
                                 installed_state(model.native,raised,0)[:3,3])[:,2].min(),-1e-10)


if __name__=='__main__':unittest.main()
