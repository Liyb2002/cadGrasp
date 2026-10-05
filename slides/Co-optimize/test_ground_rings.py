import unittest
import numpy as np
from step33 import circle
class Circles(unittest.TestCase):
    def test_diameter(self):
        p=np.array([[-2.,0],[2,0],[0,1],[0,-1]])
        c,r,s=circle(p);np.testing.assert_allclose(c,[0,0]);self.assertAlmostEqual(r,2)
    def test_triangle(self):
        p=np.array([[0.,0],[2,0],[1,np.sqrt(3)]])
        c,r,s=circle(p);np.testing.assert_allclose(c,[1,1/np.sqrt(3)]);self.assertAlmostEqual(r,2/np.sqrt(3))
    def test_saved_demands(self):
        from step33 import ROOT
        p=np.load(ROOT/'objects/B/poses/pose_3/floor_contact.npz')['floor_demands_xy_m'];c,r,s=circle(p)
        self.assertLessEqual(np.linalg.norm(p-c,axis=1).max(),r+2e-11)
        # A circle through boundary points with center in their convex hull
        # supplies a lower bound equal to its enclosing radius.
        q=p[s];weights=np.linalg.lstsq(np.vstack([q.T,np.ones(len(q))]),np.r_[c,1],rcond=None)[0]
        self.assertTrue(np.all(weights>=-1e-10));np.testing.assert_allclose(weights@q,c,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(q-c,axis=1),r,atol=1e-10)
if __name__=='__main__':unittest.main()
