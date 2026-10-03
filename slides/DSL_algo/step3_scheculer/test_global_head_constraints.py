"""Regressions for inactive heads occupying a future work surface or floor."""
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh

from step2_local_support import surface as S
from step3_scheculer.pair_geometry import PairGeometry
from step3_scheculer.sequential_geometry import SequentialGeometry


def problem(mesh, pose, frame=None, work=()):
    return SimpleNamespace(pose=pose, domain=SimpleNamespace(mesh=mesh, work_ids=np.array(work, int),
        data=dict(frame=dict(T_world_mesh=np.eye(4) if frame is None else frame))))


def head_case(transform=None, work=()):
    mesh = trimesh.Trimesh(vertices=[[0, 0, .01], [.02, 0, .01], [0, .02, .01]],
                           faces=[[0, 1, 2]], process=False)
    triangles = S.fan(mesh.triangles[0])
    geometry = SequentialGeometry.__new__(SequentialGeometry)
    transform = np.eye(4) if transform is None else transform
    other_mesh = mesh.copy().apply_transform(transform)
    geometry.problems = [problem(mesh, 'pose_1'), problem(other_mesh, 'pose_2', transform, work)]
    geometry.transforms = [[np.eye(4), transform], [np.linalg.inv(transform), np.eye(4)]]
    geometry.local = [SimpleNamespace(mesh=mesh, scale=1., clearance=SimpleNamespace(offsets=np.tile([0,0,.01], (3,1)))),
                      SimpleNamespace(mesh=other_mesh, scale=1.)]
    entry = dict(owner_task=0, active_tasks=(0,), contact=dict(candidate_id='A',
        triangles_m=triangles, source_faces=np.zeros(len(triangles), int)))
    return geometry, entry


class GlobalHeadConstraintTests(unittest.TestCase):
    def test_inactive_future_work_surface_rejects_head(self):
        geometry, entry = head_case(work=[0])
        check = geometry.all_pose_head_check([entry])
        self.assertTrue(check['per_head_pose'][0]['passed'])
        self.assertFalse(check['per_head_pose'][1]['active'])
        self.assertEqual(check['per_head_pose'][1]['overlapping_work_faces'], [0])
        self.assertFalse(check['passed'])

    def test_inactive_head_contact_cannot_enter_another_floor_band(self):
        transform = np.eye(4); transform[2,3] = -.009
        geometry, entry = head_case(transform)
        row = geometry.all_pose_head_check([entry])['per_head_pose'][1]
        self.assertGreater(row['minimum_head_height_m'], 0)
        self.assertLess(row['minimum_contact_height_m'], S.FLOOR_CLEARANCE_M)
        self.assertFalse(row['passed'])

    def test_surface_clearance_does_not_hide_solid_penetration(self):
        transform = np.diag([1., -1., -1., 1.]); transform[2,3] = .015
        geometry, entry = head_case(transform)
        row = geometry.all_pose_head_check([entry])['per_head_pose'][1]
        self.assertGreater(row['minimum_contact_height_m'], S.FLOOR_CLEARANCE_M)
        self.assertAlmostEqual(row['minimum_head_height_m'], -.005)
        self.assertFalse(row['passed'])

    def test_inactive_but_clear_head_is_allowed_without_becoming_active(self):
        geometry, entry = head_case()
        self.assertTrue(geometry.all_pose_head_check([entry])['passed'])
        self.assertEqual(entry['active_tasks'], (0,))

    def test_global_candidate_surface_excludes_future_work_faces_and_ground(self):
        mesh = trimesh.creation.box(extents=[.1,.1,.1]).apply_translation([0,0,.05])
        transform = np.diag([1.,-1.,-1.,1.]); transform[2,3] = .1
        other = mesh.copy().apply_transform(transform)
        # A side face is usable in the owner task but belongs to the future work surface.
        face = int(np.flatnonzero(np.abs(mesh.face_normals[:,2])<.1)[0])
        tasks = [problem(mesh,'pose_1'), problem(other,'pose_2',transform,[face])]
        local = PairGeometry([tasks[0]], count=8, initialize_candidates=False)
        global_ = PairGeometry([tasks[0]], count=8, initialize_candidates=False, head_exclusion_problems=tasks)
        self.assertIn(face, local.surface.polygons)
        self.assertNotIn(face, global_.surface.polygons)
        local_points = np.concatenate(list(local.surface.polygons.values()))
        global_points = np.concatenate(list(global_.surface.polygons.values()))
        self.assertLess((local_points@transform[:3,:3].T+transform[:3,3])[:,2].min(), S.FLOOR_CLEARANCE_M)
        self.assertGreaterEqual((global_points@transform[:3,:3].T+transform[:3,3])[:,2].min(), S.FLOOR_CLEARANCE_M-1e-12)
        self.assertEqual(len(global_.planes), 1)  # Active-task paths are unchanged.
        self.assertEqual(len(global_.head_planes), 2)


if __name__ == '__main__':
    unittest.main()
