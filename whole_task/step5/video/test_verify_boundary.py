"""The painter's outer contour must remain distinct from genuine inner holes."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .verify import outer_boundary


class BoundaryTests(unittest.TestCase):
    def test_annulus_outer_loop_is_independent_of_winding_and_world_rotation(self):
        vertices=np.array([[-2.,-2.,0.],[2.,-2.,0.],[2.,2.,0.],[-2.,2.,0.],
                           [-1.,-1.,0.],[-1.,1.,0.],[1.,1.,0.],[1.,-1.,0.]])*.02
        edges=np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4]])
        for rotation in [np.eye(3),Rotation.from_euler('xyz',[31.,87.,-12.],degrees=True).as_matrix()]:
            world=vertices@rotation.T+[.3,-.2,.5]
            outer,count=outer_boundary(world,edges[::-1,::-1],rotation@[0.,0.,1.])
            self.assertEqual(count,2)
            self.assertEqual(set(outer),{0,1,2,3})


if __name__=='__main__':unittest.main()
