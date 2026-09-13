"""Connected circles target 1% area but retain smaller available contacts."""
import sys
from pathlib import Path
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import circles as P


def bent_strip():
    """Four connected panels: neighbor turns 40°, whole patch turns 120°."""
    theta=np.deg2rad([-80,-40,0,40,80])
    vertices=[[np.sin(t),y,3+np.cos(t)] for t in theta for y in (-.5,.5)]
    faces=[]
    for a in range(0,8,2):faces.extend([[a,a+2,a+3],[a,a+3,a+1]])
    return trimesh.Trimesh(vertices,faces,process=False)


class CircleTests(unittest.TestCase):
    def test_whole_patch_bound_catches_nonadjacent_normals(self):
        mesh=bent_strip();seed=2
        center_angles=np.rad2deg(np.arccos(np.clip(mesh.face_normals@mesh.face_normals[seed],-1,1)))
        self.assertLess(center_angles.max(),90.)
        adjacent_angles=np.rad2deg(np.arccos(np.clip(np.sum(mesh.face_normals[mesh.face_adjacency[:,0]]*
                                                             mesh.face_normals[mesh.face_adjacency[:,1]],axis=1),-1,1)))
        self.assertLess(adjacent_angles.max(),90.)
        spread=P.normal_spread(mesh,range(len(mesh.faces)))
        self.assertAlmostEqual(spread['wrap_angle_degrees'],120.)
        self.assertFalse(spread['wrap_limit_satisfied'])
        check=P.LocalClearance(mesh,.01).check({i:p for i,p in enumerate(mesh.triangles)})
        self.assertFalse(check['valid'])
        self.assertEqual(check['status'],'wrap_angle_exceeded')

    def test_wrap_shrinks_same_center_circle_and_keeps_positive_area(self):
        mesh=bent_strip();seed=2;center=mesh.triangles_center[seed]
        surface=P.SurfaceCircles(mesh,{i:p for i,p in enumerate(mesh.triangles)})
        target=.95*mesh.area
        before,original=surface.fit_area(center,seed,target)
        self.assertGreater(P.normal_spread(mesh,before)['wrap_angle_degrees'],90.)
        polygons,report=surface.fit(center,seed,target)
        self.assertTrue(report['radius_shrunk_for_wrap'])
        self.assertEqual(report['area_status'],'smaller_wrap_limit')
        self.assertGreater(report['area_m2'],0.)
        self.assertLess(report['area_m2'],original['area_m2'])
        self.assertLess(report['radius_m'],original['radius_m'])
        self.assertLessEqual(report['wrap_angle_degrees'],90.)
        self.assertIn(seed,polygons)
        rebuilt,_=surface.at_radius(center,seed,report['radius_m'])
        self.assertEqual(set(rebuilt),set(polygons))
        for face in polygons:np.testing.assert_array_equal(polygons[face],rebuilt[face])
        just_larger,_=surface.at_radius(center,seed,report['wrap_radius_bracket_m'][1])
        self.assertFalse(P.normal_spread(mesh,just_larger)['wrap_limit_satisfied'])

    def test_fit_actual_area_on_a_coarse_plane(self):
        mesh=trimesh.Trimesh([[-2,-1,1],[2,-1,1],[0,2,1]],[[0,1,2]],process=False)
        surface=P.SurfaceCircles(mesh,{0:mesh.triangles[0]})
        target=.01*mesh.area
        polygons,report=surface.fit(np.array([0.,0,1]),0,target)
        self.assertEqual(report['status'],'area_fitted')
        actual=sum(P.area(p) for p in polygons.values())
        self.assertLessEqual(abs(actual-target),target*P.AREA_REL_TOL)
        radius=report['radius_m']
        expected=P.CIRCLE_SIDES*.5*radius**2*np.sin(2*np.pi/P.CIRCLE_SIDES)
        self.assertAlmostEqual(actual,expected,places=12)

    def test_small_contact_is_kept_without_borrowing_from_disconnected_neighbor(self):
        small=trimesh.Trimesh([[0,0,1],[.01,0,1],[0,.01,1]],[[0,1,2]],process=False)
        large=trimesh.Trimesh([[-2,-1,1.01],[2,-1,1.01],[0,2,1.01]],[[0,1,2]],process=False)
        mesh=trimesh.util.concatenate([small,large])
        mesh.invert()  # Downward pressure; this test isolates available area.
        surface=P.SurfaceCircles(mesh,{i:p for i,p in enumerate(mesh.triangles)})
        polygons,report=surface.fit(mesh.triangles_center[0],0,.01*mesh.area)
        self.assertEqual(report['status'],'area_smaller')
        self.assertEqual(report['area_status'],'smaller_connected_area')
        self.assertEqual(set(polygons),{0})
        self.assertAlmostEqual(sum(P.area(p) for p in polygons.values()),small.area,places=14)
        self.assertTrue(P.LocalClearance(mesh,.001).check(polygons)['valid'])

    def test_component_join_keeps_smaller_circle_instead_of_oversized_lobe(self):
        # A U-shaped sheet: the ball reaches the nearby opposite leg before
        # it reaches the connecting strip at the bottom. Joining that strip
        # adds a large lobe at once, jumping over the requested contact area.
        xs=[-1.,0.,1.,2.];ys=[0.,.1,3.];vertices=[];faces=[]
        for i in range(3):
            for j in range(2):
                if (i,j)==(1,1):continue
                a=len(vertices)
                vertices.extend([[xs[i],ys[j],1],[xs[i+1],ys[j],1],
                                 [xs[i+1],ys[j+1],1],[xs[i],ys[j+1],1]])
                faces.extend([[a,a+1,a+2],[a,a+2,a+3]])
        mesh=trimesh.Trimesh(vertices,faces,process=True)
        center=np.array([-.1,2.,1.]);seed=int(mesh.nearest.on_surface(center[None])[2][0])
        surface=P.SurfaceCircles(mesh,{i:p for i,p in enumerate(mesh.triangles)})
        polygons,report=surface.fit(center,seed,4.)
        self.assertEqual(report['area_status'],'smaller_before_component_join')
        actual=sum(P.area(p) for p in polygons.values())
        self.assertGreater(actual,3.)
        self.assertLess(actual,4.)
        _,after=surface.at_radius(center,seed,report['radius_m']+1e-5)
        self.assertGreater(after,4.)

    def test_concave_contact_is_allowed_when_its_actual_head_is_clear(self):
        # A concave right-angle surface with free space in its inner corner.
        mesh=trimesh.Trimesh([[0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,1,2],[0,0,2]],
                             [[0,1,2],[0,2,3],[0,3,4],[0,4,5]],process=False)
        polygons={i:p for i,p in enumerate(mesh.triangles)}
        mesh.apply_transform(trimesh.transformations.rotation_matrix(3*np.pi/4,[0,1,0]))
        mesh.apply_translation([0, 5, 0])
        polygons={i:p for i,p in enumerate(mesh.triangles)}
        self.assertAlmostEqual(P.normal_spread(mesh,polygons)['wrap_angle_degrees'],90.)
        self.assertTrue(P.LocalClearance(mesh,.01).check(polygons)['valid'])

    def test_connected_thin_wall_cannot_wrap_onto_its_back(self):
        mesh=trimesh.creation.box([1., .02, 1.]);mesh.apply_translation([0, 1, 0])
        center=np.array([.48,0.,1.01]);face=int(mesh.nearest.on_surface(center[None])[2][0])
        surface=P.SurfaceCircles(mesh,{i:p for i,p in enumerate(mesh.triangles)})
        polygons,report=surface.fit(center,face,.01*mesh.area)
        self.assertEqual(report['status'],'area_fitted')
        self.assertTrue(all(mesh.face_normals[f,2]>=-P.NORMAL_DOT_TOL for f in polygons))
        self.assertLessEqual(report['wrap_angle_degrees'],90.)

    def test_centers_stay_on_the_original_surface(self):
        mesh=trimesh.creation.box([1, 1, 1])
        points,faces=P.surface_centers({i:p for i,p in enumerate(mesh.triangles)},mesh=mesh)
        self.assertEqual(len(points),200)
        self.assertEqual(len(np.unique(points,axis=0)),200)
        residual=((points-mesh.triangles[faces,0])*mesh.face_normals[faces]).sum(axis=1)
        np.testing.assert_allclose(residual,0,atol=1e-13)


if __name__=='__main__':unittest.main()
