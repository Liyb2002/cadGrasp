"""Physical invariants of operation proposals, including the normal-axis caveat."""
import unittest
import numpy as np
import trimesh
from whole_task.step1.needle_geometry import geometric_mask
from .state import Layout, transform
from .translation import translation
from .support_merge import support_merge
from .pose_change import pose_change
from .exit_direction_change import exit_direction_change


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.q = Layout(np.tile(np.eye(4), (2, 1, 1)), np.tile(np.eye(4), (2, 1, 1)),
                        np.tile([0., 0., 1.], (2, 1)))
        self.head = np.array([.2, -.3, .4])

    def test_seat_translation_countermoves_entire_support_installation(self):
        q = translation(self.q, 1, [.1, -.2, 0])
        fixture_body = q.material[1] @ q.poses[1]
        np.testing.assert_allclose(np.linalg.inv(q.material[1]) @ fixture_body, self.q.poses[1], atol=1e-14)
        np.testing.assert_array_equal(self.q.material, np.tile(np.eye(4), (2, 1, 1)))

    def test_merge_turns_the_actual_fixture_and_preserves_world_pose(self):
        q = support_merge(self.q, 1, 0, self.head, 67., [.01, 0, 0])
        np.testing.assert_allclose(transform(self.head, q.material[1]), self.head + [.01, 0, 0], atol=1e-14)
        np.testing.assert_allclose(np.linalg.inv(q.material[1]) @ q.material[1] @ q.poses[1], self.q.poses[1], atol=1e-14)
        self.assertFalse(np.allclose(q.material[1, :3, :3], np.eye(3)))

    def test_needle_axis_rotation_preserves_every_sector_vertex(self):
        vertices = np.array([[.01, 0, .08], [0, .01, .08], [-.01, 0, .08]])
        mesh = trimesh.Trimesh(vertices, [[0, 1, 2]], process=False)
        ws = dict(maximum_reference_distance_m=.1, half_angle_deg=25.)
        for angle in np.linspace(-180, 180, 19):
            p = pose_change(self.q, 0, angle, [0, 0, 1], [0, 0, 0]).poses[0]
            moved = trimesh.Trimesh(transform(mesh.vertices, p), mesh.faces, process=False)
            self.assertTrue(geometric_mask(moved, [0, 0, 0], [0, 0, 1], ws).all())

    def test_rotation_about_an_area_normal_is_not_an_automatic_sector_proof(self):
        # The selected triangle starts just inside one side of the cone. Its
        # normal is x; rotation about its centroid can move a corner outside.
        vertices = np.array([[.037, -.001, .08], [.037, .001, .08], [.037, 0, .081]])
        mesh = trimesh.Trimesh(vertices, [[0, 1, 2]], process=False)
        ws = dict(maximum_reference_distance_m=.1, half_angle_deg=25.)
        self.assertTrue(geometric_mask(mesh, [0, 0, 0], [0, 0, 1], ws).all())
        # A larger triangle within a narrow sector exposes the failure through
        # its Euclidean radius when turned about the surface-normal line.
        escaped = False
        for angle in np.linspace(0, 360, 73):
            q = pose_change(self.q, 0, angle, [1, 0, 0], vertices.mean(0))
            moved = trimesh.Trimesh(transform(vertices, q.poses[0]), [[0, 1, 2]], process=False)
            escaped |= not geometric_mask(moved, [0, 0, 0], [0, 0, 1], ws).all()
        self.assertTrue(escaped)

    def test_exit_is_normalized_and_downward_moves_rejected(self):
        np.testing.assert_allclose(exit_direction_change(self.q, 0, [0, 2, 2]).exits[0], [0, 2**-.5, 2**-.5])
        for bad in ([0, 0, -1], [0, 0, 0], [np.nan, 0, 1]):
            with self.assertRaises(ValueError):
                exit_direction_change(self.q, 0, bad)

    def test_head_inside_body_is_rejected_even_when_work_triangles_fit_sector(self):
        from .geometry import OperationModel
        model=OperationModel.__new__(OperationModel)
        model.source=trimesh.creation.box(extents=[.01,.01,.05])
        ids=np.flatnonzero(model.source.face_normals[:,2]>.99)
        model.records=[dict(id='test',working_face_ids=ids.tolist())]
        model.workspace=dict(maximum_reference_distance_m=.1,half_angle_deg=25.)
        model.head=np.array([0.,0.,.1]);model.axis=np.array([0.,0.,1.])
        q=Layout(np.eye(4)[None],np.eye(4)[None],np.array([[0.,0.,1.]]))
        q.poses[0,2,3]=.1
        with self.assertRaisesRegex(ValueError,'head center is inside'):
            model.prepare(q)

    def test_local_clearance_contains_the_original_forbidden_prism(self):
        from .clearance import expand_prism
        from whole_task.step3 import independent_geometry as G
        box=trimesh.creation.box(extents=[.02,.01,.03])
        original=G.solid(box)
        expanded=expand_prism(original,1e-6)
        self.assertLess(G.volume(original-expanded),1e-18)
        np.testing.assert_allclose(G.mesh(expanded).bounds,box.bounds+np.array([[-1e-6]*3,[1e-6]*3]),atol=1e-14,rtol=0)


if __name__ == '__main__':
    unittest.main()
