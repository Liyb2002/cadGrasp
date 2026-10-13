"""Rejecting one original demand never shortens certification of accepted ones."""
from pathlib import Path
import sys
import unittest
import numpy as np
COOPT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COOPT/'helper_func'))
from whole_search.classify import classify


class CounterexampleTests(unittest.TestCase):
    def test_one_original_counterexample_rejects_candidate(self):
        mask,info=classify(np.eye(7),-np.eye(7),stop_on_counterexample=True)
        self.assertFalse(mask.all())
        self.assertTrue(info['stopped_on_original_counterexample'])
        self.assertEqual(info['counterexample_index'],0)

    def test_accepted_candidate_still_checks_every_original_demand(self):
        targets=np.tile(np.eye(7),(4682,1))[:32768]
        mask,info=classify(np.eye(7),targets,stop_on_counterexample=True)
        self.assertEqual(len(mask),32768)
        self.assertTrue(mask.all())
        self.assertNotIn('stopped_on_original_counterexample',info)


if __name__=='__main__':unittest.main()
