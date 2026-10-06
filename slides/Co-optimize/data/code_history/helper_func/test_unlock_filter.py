import unittest
from physics_guided_unlock_filter import unlock_decision


class UnlockFilterTests(unittest.TestCase):
    def test_first_blocker_can_move_while_force_regresses(self):
        result=unlock_decision(.01,.04,.01,.01,1.,.8)
        self.assertEqual(result['acceptance_reason'],'geometry')

    def test_return_to_old_force_best_cannot_reset_geometric_progress(self):
        result=unlock_decision(.04,.01,.01,.01,.8,1.)
        self.assertFalse(result['accepted'])

    def test_new_global_force_best_can_change_goal(self):
        result=unlock_decision(.04,.009,.01,.01,.8,1.)
        self.assertTrue(result['new_best_force'])

    def test_concurrent_better_incumbent_is_respected(self):
        result=unlock_decision(.04,.009,.008,.01,.8,1.)
        self.assertFalse(result['accepted'])

    def test_nonfinite_progress_is_unresolved(self):
        with self.assertRaises(ValueError):
            unlock_decision(.01,float('nan'),.01,.01,1.,.8)


if __name__=='__main__':unittest.main()
