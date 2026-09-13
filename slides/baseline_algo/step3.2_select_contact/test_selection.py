"""Joint contribution and deterministic selection regression tests."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
M = load_stage('select', 'choose')
C = load_stage('score', 'contribution')
from step3_scheculer import contacts as I


class SelectionTests(unittest.TestCase):
    def test_joint_support_is_not_mask_union(self):
        e = np.eye(6)
        a, b = e[[0,1,2]], e[[2,3,4,5]]
        targets = np.array([np.ones(6),e[0],e[5]])
        ma, _, _ = C.classify(a, targets)
        mb, _, _ = C.classify(b, targets)
        actual, _ = C.J.classify(I.merge_columns(a,b), targets, ma | mb)
        np.testing.assert_array_equal(ma | mb, [False,True,True])
        np.testing.assert_array_equal(actual, [True,True,True])

    def test_same_force_and_moment_coefficients(self):
        full = np.array([[1.,0,0,1,0,0]])
        mask, _ = C.J.classify(full,np.array([[1.,0,0,2,0,0]]))
        self.assertFalse(mask[0])

    def test_rank_uses_joint_count_and_excludes_selected_and_invalid(self):
        rows = [dict(index=0,id='C005',status='scored_joint',covered_count=20,individual_count=0,area_m2=9),
                dict(index=1,id='C001',status='scored_joint',covered_count=10,individual_count=10,area_m2=1),
                dict(index=2,id='C002',status='already_selected',covered_count=None),
                dict(index=3,id='C003',status='rejected_geometry',covered_count=None),
                dict(index=4,id='C004',status='scored_joint',covered_count=20)]
        self.assertEqual(M.rank_candidates(rows),[4,0,1])

    def test_hard_failure_cannot_win_over_a_valid_partial_design(self):
        rows=[dict(index=0,id='C001',status='rejected_rest_equilibrium',covered_count=100),
              dict(index=1,id='C002',status='scored_joint',covered_count=20),
              dict(index=2,id='C003',status='rejected_geometry',covered_count=100)]
        self.assertEqual(M.rank_candidates(rows),[1])

    def test_batch_certificates_match_full_lp(self):
        rng = np.random.default_rng(174)
        full = np.c_[np.ones(18),rng.normal(size=(18,5))]
        targets = np.vstack([rng.normal(size=(180,6)),rng.exponential(size=(20,18))@full])
        actual, info = C.J.classify(full,targets)
        expected = [C.W.solve(full,target) is not None for target in targets]
        np.testing.assert_array_equal(actual,expected)
        self.assertLess(info['equilibrium_lps'],len(targets))

    def test_shared_floor_column_is_not_duplicated(self):
        e = np.eye(6)
        np.testing.assert_array_equal(I.merge_columns(e[:4],e[3:]),e)


if __name__ == '__main__':
    unittest.main()
