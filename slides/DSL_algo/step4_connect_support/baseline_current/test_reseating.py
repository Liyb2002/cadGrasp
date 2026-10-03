"""K-pose frame, ground-halfspace and idle-material regressions."""
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from step4_connect_support.baseline_current import reseating as R,build_coupled_saddle as S


class ReseatingTests(unittest.TestCase):
    def test_tall_body_is_not_compact_just_because_footprint_is_small(self):
        points=trimesh.creation.box(extents=[.02,.03,.5]).vertices
        m=R.span_metrics(points,[np.eye(3),np.diag([1.,-1.,-1.])],np.zeros((2,3)))
        self.assertAlmostEqual(m['maximum_horizontal_span_m'],.03)
        self.assertAlmostEqual(m['maximum_spatial_span_m'],.5)

    def test_two_through_five_pose_floor_and_contact_frames(self):
        for count in range(2,6):
            mesh=trimesh.creation.box(extents=[.021,.037,.024]);mesh.apply_translation([.014,-.022,.063])
            roots=[mesh.vertices+np.array([i*.004,-i*.003,i*.002]) for i in range(count)]
            case=SimpleNamespace(poses=list(range(count)),catalogues=[np.array([[1.,0,0]])]*count,
                root_points=roots,mandatory=[np.vstack([p,[0,0,0],[.04,-.03,0],[-.05,.04,0]]) for p in roots])
            for _,normals in R.normal_families(count):
                row=R.radial_placement(case,normals,np.linspace(-.4,.7,count),np.zeros((count,2)),[0]*count,.025)
                bases=np.asarray(row['bases']);offsets=np.asarray(row['offsets'])
                np.testing.assert_allclose(bases[0],np.eye(3),atol=1e-14)
                np.testing.assert_allclose(offsets[0],np.zeros(3),atol=1e-14)
                for j,(points,b,o) in enumerate(zip(roots,bases,offsets)):
                    transform=np.eye(4);transform[:3,:3]=b.T;transform[:3,3]=o
                    fixture=(np.c_[points,np.ones(len(points))]@transform.T)[:,:3]
                    returned=(np.c_[fixture,np.ones(len(points))]@np.linalg.inv(transform).T)[:,:3]
                    np.testing.assert_allclose(returned,points,atol=1e-14)
                    self.assertAlmostEqual(np.linalg.det(b),1.)
                    for other,origin in zip(bases,offsets):
                        self.assertGreaterEqual(((fixture-origin)@other.T)[:,2].min(),.002-1e-12)
                        demand=case.mandatory[j]@b+o
                        self.assertGreaterEqual(((demand-origin)@other.T)[:,2].min(),-1e-12)

    def test_third_idle_group_can_block_first_pose(self):
        corridor_mesh=trimesh.creation.box(extents=[.2,.02,.02]);corridor=S.solid(corridor_mesh)
        head=trimesh.creation.box(extents=[.01,.01,.01])
        far=head.copy();far.apply_translation([0,.1,0])
        roots=[far,far,head]
        case=SimpleNamespace(root_solids=[S.solid(m) for m in roots],root_points=[m.vertices for m in roots],
            poses=['a','b','c'],menus=[[0],[0],[0]],padded_sweeps=[{0:corridor}]*3,
            sweep_bounds=[{0:corridor_mesh.bounds}]*3,sweep_planes=[{0:ConvexHull(corridor_mesh.vertices).equations}]*3)
        layout=dict(bases=[np.eye(3)]*3,offsets=np.zeros((3,3)))
        menu,checks=R.screen(case,layout)
        self.assertIsNone(menu)
        self.assertTrue(any(t['foreign_pose']=='c' and t['overlap_m3']>R.TOL for t in checks))
        layout['offsets'][2,1]=.3
        menu,checks=R.screen(case,layout)
        self.assertEqual(menu,[[0],[0],[0]])
        self.assertEqual({(t['pose'],t['foreign_pose']) for t in checks},
            {(a,b) for a in case.poses for b in case.poses if a!=b})


if __name__=='__main__':unittest.main()
