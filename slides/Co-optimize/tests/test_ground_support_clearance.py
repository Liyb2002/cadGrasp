"""Final-floor clearance removes reaction generators and is visible to descent."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
import trimesh

COOPT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COOPT/'helper_func'))
import _bootstrap
from oracle_search import OracleModel
from whole_search.model import Layout
from whole_search.delta_guidance import ContactDelta
from whole_search.common import C, unpack_solid
from ground_support import allowed_contacts, contact_event_step
from translation.layout_update import shift, translation_frame, translation_gradient
from continuous_support.projection import project_demands
from whole_step4_render import NominalBuilder


class GroundSupportTests(unittest.TestCase):
    def setUp(self):
        self.model = OracleModel.__new__(OracleModel)
        self.model.support_floor_clearance_m = .005
        self.model.allow_z_translation = True
        self.model.native = np.repeat(np.eye(4)[None], 2, axis=0)
        self.model.native[1, :3, :3] = [[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]
        self.model.points = np.array([[.01, 0., .004], [.01, 0., .005],
                                     [.002, 0., .02], [.006, 0., .012]])
        self.model.sources = np.arange(4)
        self.model.allowed = [np.arange(4)]*2
        self.model.mesh = SimpleNamespace(face_normals=np.tile([0., 0., -1.], (4, 1)))
        self.model.material_at_points = lambda *args: np.ones(len(self.model.points), bool)
        self.model.locks = lambda *args: np.zeros(len(self.model.points), bool)
        self.layout = Layout(np.repeat(np.eye(4)[None], 2, axis=0),
                             np.array([[0., 0., 1.], [1., 0., 0.]]), np.arange(2), (0, 1))

    def test_5mm_boundary_and_other_installed_floor(self):
        # Own ground is z=0. Other installed ground, in fixture coordinates,
        # is x=0. Above the current floor alone is insufficient.
        np.testing.assert_array_equal(allowed_contacts(self.model, self.layout, 0),
                                      [False, True, False, True])
        np.testing.assert_array_equal(self.model.contact_allowed(self.layout, 0),
                                      [False, True, False, True])

    def test_lift_changes_contact_availability_but_keeps_other_floor_rejection(self):
        engine = ContactDelta(self.model)
        before = engine.state(self.layout)
        frame = translation_frame(self.model, self.layout, 0)
        raised = shift(self.model, self.layout, 0, .002*frame[:, 2])
        after = engine.state(raised)
        np.testing.assert_array_equal(after.available[0], [True, True, False, True])
        np.testing.assert_array_equal(before.available[0], [False, True, False, True])

    def test_changing_another_host_updates_floor_filter_without_stale_pair_cache(self):
        engine = ContactDelta(self.model)
        before = engine.state(self.layout)
        changed = self.layout.copy()
        changed.hosts[1] = 0
        np.testing.assert_array_equal(engine.state(changed).available[0], [False, True, True, True])
        np.testing.assert_array_equal(before.available[0], [False, True, False, True])

    def test_floor_event_uses_xyz_transforms_and_includes_losing_contacts(self):
        # +world X gains the point limited by the other (x=0) floor.
        enter = contact_event_step(self.model, self.layout, [0], 0, 1, .001)
        self.assertGreater(enter, .003)
        self.assertLess(enter, .00301)
        # -world X loses the point at x=6mm, even though current z is high.
        exit_ = contact_event_step(self.model, self.layout, [0], 0, -1, .0002)
        self.assertGreater(exit_, .001)
        self.assertLess(exit_, .00101)

    def test_loss_gradient_sees_required_lift_beyond_small_difference(self):
        m = self.model
        m.points = np.array([[0., 0., 0.], [0., 0., .02]])
        m.sources = np.arange(2)
        m.allowed = [np.arange(2)]*2
        m.mesh = SimpleNamespace(face_normals=np.array([[-1., 0., 0.], [0., 0., -1.]]))
        # A yaw demand cannot be supplied by the point floor at zero lever arm.
        raw = np.array([[1., 0., 0., 0., 0., 1.], [0., 0., 1., 0., 0., 0.]])
        m.point_rays = [C.U.heads(raw, np.ones(6))]*2
        m.floors = [C.U.floor(C.FLOOR.columns(np.zeros(3), np.zeros(3)), np.ones(6))]*2
        m.contact_delta = ContactDelta(m)
        q = self.layout.copy(); q.active = (0,)
        target = C.U.target(np.array([[1., 0., 1., 0., 0., 1.]]))

        def loss(layout):
            flags = m.contact_delta.state(layout).available[0]
            rays, _ = m.supply_at_points(0, flags, layout)
            residual = project_demands(rays, target)['residuals']
            return float(.5*np.sum(residual**2))

        self.assertGreater(loss(q), .1)
        self.assertGreater(loss(shift(m, q, 0, np.array([0., 0., .001]))), .1)
        gradient = translation_gradient(m, q, [0], .001, objective=loss)
        self.assertLess(gradient[2], -1.)
        np.testing.assert_allclose(gradient[:2], 0., atol=1e-10)
        raised = shift(m, q, 0, np.array([0., 0., .00501]))
        self.assertLess(loss(raised), 1e-20)
        self.assertEqual(len(m.floor_columns(raised, 0)), 1)  # only no-uplift slack

    def test_nominal_export_is_clipped_to_all_physical_floors_without_base(self):
        box = trimesh.creation.box([.02]*3)
        model = SimpleNamespace(native=self.model.native, support_floor_clearance_m=.005,
                                registered_seed=C.S.solid(box))
        builder = NominalBuilder(model)
        builder.part = lambda *args: C.union([])
        mesh = unpack_solid(builder.solid(self.layout))
        self.assertGreater(mesh.volume, 0.)
        self.assertGreaterEqual(mesh.vertices[:, 2].min(), -1e-10)
        self.assertGreaterEqual(mesh.vertices[:, 0].min(), -1e-10)
        self.assertLessEqual(mesh.vertices[:, 2].max(), .010000001)


if __name__ == '__main__':
    unittest.main()
