import unittest
from unittest.mock import patch
import numpy as np
from whole_task_2.step3.legacy import kernel as K
from .force_witness import classify_with_witness


class WitnessTests(unittest.TestCase):
    def test_recorded_bases_cover_every_original_target_with_unchanged_classifier(self):
        heads=np.vstack([np.eye(6),-np.eye(6)])
        columns=np.vstack([np.r_[np.zeros(6),-1.],np.c_[heads,heads[:,2]]])
        rng=np.random.default_rng(21)
        targets=rng.normal(0.,.3,(200,6));targets[:,2]=1.
        original_solver=K.J.C.W.solve
        expected,expected_report=K.J.classify(columns,targets)
        actual,report,basis=classify_with_witness(columns,targets)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(report,expected_report)
        self.assertIs(K.J.C.W.solve,original_solver)
        replay,_=K.J.classify(basis,targets)
        self.assertTrue(replay.all())
        self.assertTrue(all(any(np.array_equal(row,c) for c in columns) for row in basis))

    def test_observer_is_restored_when_classifier_raises(self):
        original=K.J.C.W.solve
        with patch.object(K.J,'classify',side_effect=ValueError('failure')):
            with self.assertRaisesRegex(ValueError,'failure'):
                classify_with_witness(np.eye(7),np.zeros((1,6)))
        self.assertIs(K.J.C.W.solve,original)


if __name__=='__main__':unittest.main()
