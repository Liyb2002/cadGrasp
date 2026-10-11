from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
COOPT = HERE.parents[1]
for path in [COOPT/'step5.2', HERE.parent/'placement_base', COOPT/'step5.4', HERE]:
    sys.path.insert(0, str(path))
import numpy as np
import trimesh
import stage5_common as C
import base_geometry as G
import base_seating as B
from geometry import WorkOnlyBase, enclosure
from policy import POLICY
import z_seating as Z


class WorkOnlyPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = C.Inputs('B')
        cls.data = cls.inputs.load(cls.inputs.cases[2]); B.load_demands(cls.data, cls.inputs)

    def test_all_updated_wrenches_and_relative_placements(self):
        heights = np.linspace(.01, .09, len(self.data['rows']))
        moved = Z.move(self.data, heights)
        for old, new, height in zip(self.data['rows'], moved['rows'], heights):
            expected = old['cloud'][:, :2]-height*old['horizontal_force_ratio']
            np.testing.assert_allclose(new['cloud'][:, :2], expected, atol=1e-14)
            np.testing.assert_allclose(np.linalg.inv(old['fixture_transform'])@old['object_transform'],
                                       np.linalg.inv(new['fixture_transform'])@new['object_transform'], atol=1e-14)
            self.assertEqual(len(new['cloud']), 32768)

    def test_minimum_lift_is_exact_and_nonnegative(self):
        mesh = trimesh.creation.box([.08, .06, .04]); mesh.apply_translation([0., 0., -.03])
        state = np.eye(4); obj = np.eye(4); obj[2, 3] = .1
        body = trimesh.creation.box([.02]*3)
        data = dict(rows=[dict(fixture_transform=state, object_transform=obj)], body=body)
        self.assertAlmostEqual(Z.minimum_lifts(mesh, data)[0], .05, places=12)
        lifted = state.copy(); lifted[2, 3] += .05
        self.assertAlmostEqual(C.F.transform_points(mesh.vertices, lifted)[:, 2].min(), 0., places=12)

    def test_enclosure_is_3d_not_only_xy_projection(self):
        mesh = trimesh.creation.box([2., 2., 2.])
        row = dict(pose='test', fixture_transform=np.eye(4), hull=np.array([[0., 0., -1.1], [.1, 0., -1.1], [0., .1, -1.1]]))
        self.assertFalse(enclosure(mesh, dict(rows=[row]))['passed'])
        row['hull'][:, 2] = -.9
        self.assertTrue(enclosure(mesh, dict(rows=[row]))['passed'])

    def test_floor_success_does_not_imply_contact_coverage(self):
        mesh = trimesh.creation.box([.08, .06, .04]); mesh.apply_translation([0., 0., .02])
        obj = np.eye(4); obj[2, 3] = .1
        row = dict(pose='test', fixture_transform=np.eye(4), object_transform=obj,
                   object_ground=np.empty((0, 2)), hull=np.array([[.1, 0., 0.], [.1, .01, 0.], [.11, 0., 0.]]))
        data = dict(rows=[row], body=trimesh.creation.box([.02]*3))
        check, _ = Z.check(mesh, data)
        self.assertTrue(check['floor_passed']); self.assertFalse(check['passed'])

    def test_work_is_subtracted_but_exit_area_remains_available(self):
        # Real work cones on a simple box. A point behind the work face is
        # in a downward withdrawal sweep, yet remains legal here.
        body = trimesh.creation.box([.02]*3)
        work_ids = np.flatnonzero(body.face_normals[:, 2] > .9)
        task = SimpleNamespace(domain=SimpleNamespace(work_ids=work_ids, data={'load': {'cone_half_deg': 30}}))
        data = dict(fixture=trimesh.creation.box([.08]*3), rows=[dict(pose='test',
            fixture_transform=np.eye(4), object_transform=np.eye(4),
            hull=np.array([[-.03, -.03, 0.], [.03, -.03, 0.], [0., .03, 0.]]))])
        builder = WorkOnlyBase(data, [task], body, padding=0.)
        work_point = G.md.Manifold.cube((1., 1., 1.)).translate((-.5, -.5, 19.5))
        exit_side_point = G.md.Manifold.cube((1., 1., 1.)).translate((-.5, -.5, -20.5))
        self.assertLess(G.volume(builder.free ^ work_point), 1e-14)
        self.assertGreater(G.volume(builder.free ^ exit_side_point), .9e-9)
        from exit_clearance import ExitClearance
        exit_sweep = G.from_legacy(ExitClearance(body).sweep(np.array([0., 0., -.04]), padded=False))
        self.assertGreater(G.volume(exit_sweep ^ exit_side_point), .9e-9)
        self.assertFalse(POLICY.forbid_exit_paths)
        self.assertEqual(POLICY.floor_owner, 'step5.4')
        self.assertGreater(G.unpack(builder.free).vertices[:, 2].min()*-1, .039)

    def test_empty_transfer_has_continuous_radius_certificate(self):
        mesh = trimesh.creation.box([.1, .06, .02])
        from scipy.spatial.transform import Rotation
        first = np.eye(4); first[2, 3] = .01
        last = np.eye(4); last[:3, :3] = Rotation.from_euler('y', 180, degrees=True).as_matrix()
        last[2, 3] = .01
        moved = dict(states=[first, last], rows=[dict(pose='a'), dict(pose='b')])
        report, frames = Z.transfer(mesh, moved)
        self.assertTrue(report['continuous_floor_clearance_proved_for_defined_path'])
        self.assertGreaterEqual(report['transitions'][0]['exact_rotation_minimum_z_bound_mm'], .999999)
        self.assertGreaterEqual(min(C.F.transform_points(mesh.vertices, t)[:, 2].min() for t in frames), -1e-12)

    @classmethod
    def tearDownClass(cls):
        cls.inputs.verify()


if __name__ == '__main__':
    unittest.main()
