"""Parallel workers must reproduce original sample masks and LP evidence."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import parallel_scoring as P
from step3_scheculer.stage_imports import load_stage


class ParallelScoreTests(unittest.TestCase):
    def test_real_candidates_match_serial_classifier_and_verifier(self):
        C = load_stage('score', 'contribution')
        problem = C.Problem('B')
        _, base, _ = problem.load_state()
        ids = np.flatnonzero(problem.data.valid)[[0, 8, 16]]
        jobs = [(int(i), C.columns(problem.domain, problem.data, i, problem.floor, problem.scale),
                 problem.targets, base) for i in ids]
        expected = [P.calculate(job) for job in jobs]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            actual = list(pool.map(P.calculate, jobs))
        for a, b in zip(actual, expected):
            self.assertEqual(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(a[2:], b[2:])


if __name__ == '__main__':
    unittest.main()
