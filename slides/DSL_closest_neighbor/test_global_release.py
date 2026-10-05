import unittest
import numpy as np
from merge_release import Head, cube, cut_head, cut_contact, volume, direction_value, outside_convex, planes, nonempty_parts, subtract_cell_box

class ReleaseGeometryTests(unittest.TestCase):
    def test_shared_material_requires_a_coupled_cut(self):
        a = cube(np.zeros(3), .01)
        b = cube(np.array([.01,0,0]), .01)
        cutter = cube(np.array([.008,0,0]), .004)
        one_label = (a-cutter)+b
        all_labels = (a-cutter)+(b-cutter)
        self.assertAlmostEqual(volume((a+b)-one_label), 0, places=12)
        self.assertGreater(volume((a+b)-all_labels), 0)
        self.assertAlmostEqual(volume(all_labels-((a+b)-cutter)), 0, places=12)

    def test_per_label_removal_can_misidentify_material_still_present(self):
        a = cube(np.zeros(3), .01)
        b = cube(np.array([.01,0,0]), .01)
        clipped = a-cube(np.array([.008,0,0]), .004)
        present = clipped+b
        stale_label_removal = a-clipped
        actual_removal = (a+b)-present
        self.assertGreater(volume(stale_label_removal ^ present), 0)
        self.assertAlmostEqual(volume(actual_removal ^ present), 0, places=12)

    def test_connected_attachment_can_still_damage_a_required_core(self):
        body = cube(np.zeros(3), .01)
        anchor = cube(np.array([.02,0,0]), .01)
        core = cube(np.array([.009,0,0]), .002)
        head = Head('H', ('p:c',), body, body)
        edited = cut_head(head, ((np.array([.008,0,0]), .001),))
        self.assertEqual(len(nonempty_parts(edited.solid+anchor)), 1)
        self.assertGreater(volume(core ^ (head.raw-edited.solid)), 0)

    def test_connected_head_can_lose_its_support_attachment(self):
        body = cube(np.zeros(3), .01)
        anchor = cube(np.array([.02,0,0]), .01)
        self.assertEqual(len(nonempty_parts(body+anchor)), 1)
        head = Head('H', ('p:c',), body, body)
        edited = cut_head(head, ((np.array([.01,0,0]), .012),))
        self.assertEqual(len(nonempty_parts(edited.solid)), 1)
        self.assertEqual(len(nonempty_parts(edited.solid+anchor)), 2)

    def test_disjoint_cell_cutter_does_not_split_convex_seed(self):
        cell = np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
        result = subtract_cell_box(cell, np.array([0,0,5]), .1)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], cell)

    def test_repeated_nonintersecting_cuts_do_not_multiply_contact_triangles(self):
        contact = dict(triangles_m=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float),
                       source_faces=np.array([0]))
        for _ in range(10):
            contact = cut_contact(contact, np.eye(3), np.zeros(3), ((np.ones(3)*5, .1),))
        self.assertEqual(len(contact['triangles_m']), 1)
        self.assertAlmostEqual(contact['triangle_areas_m2'].sum(), .5)

    def test_exclusive_material_loss_has_a_negative_cut_gradient(self):
        head = cube(np.zeros(3), .01)
        fixed = cube(np.array([-.015,0,0]), .01)
        sweep = cube(np.array([.008,0,0]), .008)
        original = Head('H', ('p:c',), head, head)
        fixed_overlap = volume(fixed ^ sweep)
        def loss(radius):
            edited = cut_head(original, ((np.array([.009,0,0]), radius),))
            return fixed_overlap + volume((edited.solid-fixed) ^ sweep)
        self.assertLess((loss(.002)-loss(0))/.002, 0)
        self.assertAlmostEqual(loss(0), volume((head+fixed) ^ sweep), places=12)

    def test_cut_preserves_subset_and_cannot_increase_sweep_collision(self):
        body = cube(np.zeros(3), .01)
        sweep = cube(np.array([.008,0,0]), .008)
        head = Head('H', ('p:c',), body, body)
        cut = cut_head(head, ((np.array([.009,0,0]), .004),))
        self.assertLess(volume(cut.solid), volume(body))
        self.assertLessEqual(volume(cut.solid-body), 1e-15)
        self.assertLessEqual(volume(cut.solid ^ sweep), volume(body ^ sweep))
        self.assertGreater(volume(body-cut.solid), 0)

    def test_direction_sets_empty_and_common(self):
        directions = np.eye(3)
        self.assertFalse(direction_value(directions, [[True,False,False],[False,False,False]])['valid'])
        result = direction_value(directions, [[True,False,True],[False,True,True]])
        self.assertEqual(result['common_count'], 1)
        self.assertAlmostEqual(result['mean_target_angle_deg'], 0)

    def test_contact_hole_is_not_filled_back(self):
        square = np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]], float)
        pieces = outside_convex(square, planes(np.zeros(3), 1))
        area = sum(abs(np.dot(p[:,0], np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1)))/2 for p in pieces)
        self.assertAlmostEqual(area, 12)

if __name__ == '__main__':
    unittest.main()
