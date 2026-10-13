"""Directed PSO proposals and complete-layout historical-best safety."""
from types import SimpleNamespace
import unittest
import numpy as np
from test_population32 import SmallModel, initial, centered_seat
from test_population_soft import sample
from population_soft import pso_move, preserves_poses, PopulationSoftSearch


class PSOContracts(unittest.TestCase):
    def setUp(self):
        self.model = SmallModel()
        self.current = centered_seat(self.model, initial(), 1, 0)
        self.rng = SimpleNamespace(random=lambda size=None:
            np.full(size, .5) if size is not None else 1.)

    def test_motion_is_towards_scored_exemplars_and_preserves_current_role(self):
        personal, social = self.current.copy(), self.current.copy()
        personal.placements[1,0,3] += .04
        social.placements[1,0,3] += .08
        velocity = np.zeros((4,3))
        moved, detail = pso_move(self.model, self.current, personal, social, velocity, self.rng, 1)
        self.assertGreater(moved.placements[1,0,3], self.current.placements[1,0,3])
        self.assertLess(moved.placements[1,0,3], social.placements[1,0,3])
        np.testing.assert_array_equal(moved.hosts, self.current.hosts)
        np.testing.assert_array_equal(moved.placements[:,:3,:3], self.current.placements[:,:3,:3])
        self.assertTrue(preserves_poses(self.current, moved, [0,2,3]))
        self.assertTrue(detail['exemplars_selected_by_complete_layout_physical_rank'])
        for k in moved.active:self.model.validate_placement(moved,k)

    def test_coincident_exemplars_do_not_add_random_translation(self):
        velocity = np.zeros((4,3))
        moved, _ = pso_move(self.model, self.current, self.current, self.current, velocity, self.rng, 1)
        self.assertEqual(moved.key(), self.current.key())
        self.assertFalse(np.any(velocity))

    def test_inertia_retains_direction_without_other_attraction(self):
        velocity = np.zeros((4,3));velocity[1,0] = .04
        moved, _ = pso_move(self.model, self.current, self.current, self.current, velocity, self.rng, 1)
        self.assertGreater(moved.placements[1,0,3], self.current.placements[1,0,3])
        self.assertLess(velocity[1,0], .04)

    def test_exploratory_worse_layout_does_not_erase_personal_best(self):
        controller = PopulationSoftSearch(SimpleNamespace(model=self.model,frontier_width=32))
        best = sample(self.current, (0,0,0,0));best.loss = 0.
        worse = sample(self.current, (0,10,0,0));worse.loss = .1
        controller.pso_best = [best]*32
        controller.update_pso_best(0, worse)
        self.assertIs(controller.pso_best[0], best)

    def test_better_full_force_layout_updates_personal_best(self):
        controller = PopulationSoftSearch(SimpleNamespace(model=self.model,frontier_width=32))
        old = sample(self.current, (0,10,0,0));old.loss = 0.
        better = sample(self.current, (0,0,0,0));better.loss = .01
        controller.pso_best = [old]*32
        controller.update_pso_best(0, better)
        self.assertIs(controller.pso_best[0], better)
        self.assertEqual(controller.pso_best_updates, 1)


if __name__ == '__main__':unittest.main()
