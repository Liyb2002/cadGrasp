"""Check that merged feet fill between pads and cannot bridge exclusions."""
import unittest
import numpy as np
from shapely.geometry import Polygon
import trimesh

from step4_connect_support.baseline_current import convex_foot as F, build_coupled_saddle as S


class ConvexFootTests(unittest.TestCase):
    def setUp(self):
        mesh=trimesh.creation.box([.008,.008,.003]);mesh.apply_translation([0,0,.04])
        self.original=S.solid(mesh)
        self.patch=dict(v=mesh.vertices,root=mesh.vertices+[0,0,.008],original=self.original)

    def connected(self,body,required):
        for component in body.decompose():
            if all(abs((v-component).volume())*S.SCALE**3<8e-14 for v in required):return component

    def test_full_convex_sole_covers_space_between_separate_pads(self):
        square=np.array([[-.004,-.004],[.004,-.004],[.004,.004],[-.004,.004]])
        xy=F.hull_xy([square+[-.025,0],square+[.025,0]])
        body=F.loft(self.patch,xy,np.eye(3),np.zeros(3),lambda x:x,self.connected)
        self.assertIsNotNone(body)
        ground=F.landing(body,np.eye(3),np.zeros(3))
        self.assertLess(ground.symmetric_difference(Polygon(xy)).area,1e-12)
        self.assertGreater(ground.area,2*Polygon(square).area)
        self.assertEqual(len(body.decompose()),1)

    def test_split_landing_is_rejected_even_when_head_connects_both_sides(self):
        xy=np.array([[-.03,-.01],[.03,-.01],[.03,.01],[-.03,.01]])
        obstacle=trimesh.creation.box([.006,.03,.008]);obstacle.apply_translation([0,0,.001])
        blocker=S.solid(obstacle)
        body=F.loft(self.patch,xy,np.eye(3),np.zeros(3),lambda x:x-blocker,self.connected)
        self.assertIsNone(body)


if __name__=='__main__':unittest.main()
