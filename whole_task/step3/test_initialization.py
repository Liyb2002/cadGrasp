"""Geometric and physical regressions for the current-pose Step3 adapter."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from whole_task.common.transforms import rigid_transform, points
from .initialize import build_seed, supply, validate_policy
from .legacy import kernel as K
from .work_access import SurfaceSegments, RegisteredWorkVolumes
from .validate import check_binding


POLICY = json.loads(Path(__file__).with_name("policy.json").read_text())
CONE = dict(apex_world_m=[0., 0., .28], axis_world=[0., 0., -1.],
            half_angle_deg=25., usable_half_angle_deg=23.75, reference_axial_distance_m=.12)


def box():
    mesh = trimesh.creation.box(extents=[.1, .1, .1])
    top = np.flatnonzero(mesh.face_normals[:, 2] > .9)
    return mesh, top


def task_at(mesh, ids, rotation=np.eye(3), translation=np.array([0., 0., .2]), axis=np.array([0., 0., -1.])):
    transform = rigid_transform(rotation, translation)
    world = mesh.copy()
    world.apply_transform(transform)
    domain = SimpleNamespace(work_ids=ids, axis=axis, mesh=world, com=points(transform, mesh.center_mass if mesh.is_watertight else mesh.vertices.mean(axis=0)))
    return SimpleNamespace(domain=domain, scale=np.r_[np.ones(3), np.ones(3)/world.extents.max()]), transform, mesh


class AccessTests(unittest.TestCase):
    def test_segments_stop_at_surface_and_head(self):
        mesh, top = box()
        volume = SurfaceSegments(mesh, top, [0, 0, .1])
        np.testing.assert_array_equal(volume.contains_points(
            [[0, 0, .075], [0, 0, .049], [0, 0, .101], [.1, 0, .075]]),
            [True, False, False, False])

    def test_entire_triangle_not_only_centroid_ray(self):
        mesh, top = box()
        volume = SurfaceSegments(mesh, top, [0, 0, .1])
        for t in (0., .1, .5, 1.):
            q = (1-t)*volume.head+t*mesh.triangles[top].reshape(-1, 3)
            self.assertTrue(volume.contains_points(q).all())
        # Full square pyramid: base .1*.1, height .05, volume = base*height/3.
        self.assertAlmostEqual(K.material_volume(volume.solid()), .1*.1*.05/3, places=14)

    def test_slanted_surface_is_the_actual_finite_end(self):
        mesh = trimesh.Trimesh(vertices=[[-.1, -.1, 0], [.1, -.1, .1], [0, .1, .05]],
                              faces=[[0, 1, 2]], process=False)
        head = np.array([0., 0., .2])
        volume = SurfaceSegments(mesh, [0], head)
        surface = mesh.triangles[0].mean(axis=0)
        self.assertTrue(volume.contains_points([head+.99*(surface-head)])[0])
        self.assertFalse(volume.contains_points([head+1.01*(surface-head)])[0])

    def test_unselected_gaps_and_other_poses_are_not_filled(self):
        mesh = trimesh.Trimesh(vertices=[[-2.5, -1, 0], [-1.5, -1, 0], [-2, 1, 0],
                                       [1.5, -1, 0], [2.5, -1, 0], [2, 1, 0]],
                              faces=[[0, 1, 2], [3, 4, 5]], process=False)
        both = SurfaceSegments(mesh, [0, 1], [0, 0, 1])
        q = [[-1, 0, .5], [1, 0, .5], [0, 0, .5]]
        np.testing.assert_array_equal(both.contains_points(q), [True, True, False])
        source = dict(CONE, apex_world_m=[0, 0, 1])
        states = {"A":task_at(mesh, np.array([0]), translation=np.zeros(3)),
                  "B":task_at(mesh, np.array([1]), translation=np.zeros(3))}
        access = RegisteredWorkVolumes(mesh, states, source)
        np.testing.assert_array_equal(access.for_pose("A").contains_points(q), [True, False, False])
        np.testing.assert_array_equal(access.for_pose("B").contains_points(q), [False, True, False])
        self.assertAlmostEqual(K.material_volume(both.solid()),
                               K.material_volume(access.combined_solid()), places=13)

    def test_empty_work_area_has_no_reserved_cone(self):
        mesh, _ = box()
        volume = SurfaceSegments(mesh, [], [0, 0, .1])
        self.assertFalse(volume.contains_points([[0, 0, .075]])[0])
        self.assertEqual(K.material_volume(volume.solid()), 0.)

    def test_each_installation_joins_own_surface_to_fixed_world_head(self):
        mesh, ids = box()
        states = {"A":task_at(mesh, ids),
                  "B":task_at(mesh, ids, Rotation.from_euler('y', 45, degrees=True).as_matrix(),
                              translation=np.array([.2, -.1, .3]))}
        access = RegisteredWorkVolumes(mesh, states, CONE)
        for pose, (task, transform, _) in states.items():
            local = access.for_pose(pose)
            world = SurfaceSegments(task.domain.mesh, ids, CONE["apex_world_m"])
            np.testing.assert_allclose(points(transform, local.head), world.head, atol=1e-14)
            q = .5*world.head+.5*world.triangles.reshape(-1, 3)
            back = (q-transform[:3, 3]) @ transform[:3, :3]
            self.assertTrue(local.contains_points(back).all())
            np.testing.assert_array_equal(local.contains_points(back), world.contains_points(q))

    def test_coplanar_cell_reports_unresolved_geometry(self):
        mesh, top = box()
        with self.assertRaisesRegex(ValueError, "coplanar"):
            SurfaceSegments(mesh, top, [0, 0, .05])


class SupportTests(unittest.TestCase):
    def test_unsupported_physics_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Step3 policy"):
            validate_policy(dict(POLICY, finite_tool_radius_included=True))

    def test_airborne_has_no_floor_reaction_and_keeps_slack(self):
        mesh, ids = box()
        task, transform, _ = task_at(mesh, ids)
        bottom = np.flatnonzero(mesh.face_normals[:, 2] < -.9)
        full, report = supply(task, transform, mesh.triangles[bottom], bottom, POLICY)
        self.assertEqual(report["workpiece_floor_reaction_count"], 0)
        self.assertTrue(np.any(np.all(full == np.r_[np.zeros(6), -1.], axis=1)))
        mask, _ = K.J.classify(full, [[0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0]])
        np.testing.assert_array_equal(mask, [True, False])

    def test_grounded_contacts_are_actual_mesh_points(self):
        mesh, ids = box()
        task, transform, _ = task_at(mesh, ids, translation=np.array([0, 0, .05]))
        full, report = supply(task, transform, np.empty((0, 3, 3)), np.array([], int), POLICY)
        floor = np.array(report["workpiece_floor_contact_points_m"])
        np.testing.assert_allclose(floor[:, 2], 0, atol=1e-15)
        self.assertEqual(report["workpiece_floor_reaction_count"], 16)
        mask, _ = K.J.classify(full, [[0, 0, 1, 0, 0, 0]])
        self.assertTrue(mask.all())

    def test_actual_wrap_carves_finite_work_volume(self):
        mesh, ids = box()
        states = {"A": task_at(mesh, ids)}
        wrapped, allowed, access, geometry = build_seed(mesh, states, POLICY, CONE)
        self.assertTrue(wrapped.is_watertight)
        self.assertGreater(wrapped.volume, 0)
        self.assertLessEqual(geometry["continuous_work_overlap_m3"], 1e-10)
        triangles, sources = K.contact_boundary(mesh, wrapped, allowed)
        self.assertGreater(len(triangles), 0)
        # Material above the working surface is removed; side stock stays.
        self.assertFalse(np.isin(sources, ids).any())
        np.testing.assert_array_equal(wrapped.contains([[0, 0, .051], [.051, 0, 0]]), [False, True])

    def test_stale_source_is_rejected(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder/'pose.json'
            source.write_text('{}')
            with self.assertRaisesRegex(ValueError, "input changed"):
                check_binding(folder, {"input_sha256": {str(source): '0'*64}, "artifact_sha256": {}})


if __name__ == "__main__":
    unittest.main()
