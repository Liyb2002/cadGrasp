import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
from optimization.initial_directions import initialize_close_directions

class InitialDirectionsTests(unittest.TestCase):
    def test_strict_common(self):
        normals=np.array([[0,0,1],[1,0,0],[0,1,0.]])
        directions,info=initialize_close_directions(normals)
        self.assertEqual(info['common_direction_status'],'strict_common_direction')
        np.testing.assert_allclose(directions,np.repeat(directions[:1],3,axis=0))
        self.assertGreater(np.min(normals@directions[0]),0)

    def test_boundary_common(self):
        normals=np.array([[0,0,1],[0,0,-1.]])
        directions,info=initialize_close_directions(normals)
        self.assertEqual(info['common_direction_status'],'boundary_only_common_direction')
        np.testing.assert_allclose(directions[0],directions[1])
        np.testing.assert_allclose(directions[:,2],0,atol=1e-12)

    def test_no_common_legal_deterministic_close(self):
        normals=np.vstack([np.eye(3),-np.eye(3)])
        directions,info=initialize_close_directions(normals)
        repeated,again=initialize_close_directions(normals)
        self.assertEqual(info['common_direction_status'],'no_nonzero_common_direction')
        np.testing.assert_array_equal(directions,repeated)
        self.assertEqual(info,again)
        np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1,atol=1e-12)
        self.assertGreaterEqual(np.min(np.sum(normals*directions,axis=1)),-1e-10)
        # Native-up directions have zero mean and cannot attain this agreement.
        self.assertGreater(np.linalg.norm(directions.sum(axis=0)),np.linalg.norm(normals.sum(axis=0))+1)

    def test_invalid_normal(self):
        with self.assertRaises(ValueError):initialize_close_directions([[0,0,0]])

if __name__=='__main__':unittest.main()
