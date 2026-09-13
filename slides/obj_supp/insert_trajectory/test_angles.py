"""Discriminating geometric cases for continuous angular-set classification."""
import unittest
import numpy as np
import trimesh
import angles as A
from step1.needs import COORD
from step5_connect_support import ground as F


def contains(intervals, angle):
    return any(a+1e-7 < angle%360 < b-1e-7 for a, b in intervals)


class AngleTests(unittest.TestCase):
    def test_half_circle_and_wraparound(self):
        self.assertEqual(A.local_angles([[1., 0., 0.]])['intervals_deg'], [[90., 270.]])
        self.assertEqual(A.local_angles([[-1., 0., 0.]])['intervals_deg'], [[0., 90.], [270., 360.]])
        self.assertEqual(A.local_angles([[0., 1., 0.]])['intervals_deg'], [[0., 360.]])

    def test_opposed_planes_retain_isolated_tangent_directions(self):
        result = A.local_angles([[1., 0., 0.], [-1., 0., 0.]])
        self.assertEqual(result['intervals_deg'], [])
        self.assertEqual(result['isolated_angles_deg'], [90., 270.])

    def test_configuration_obstacle_uses_insertion_sign(self):
        head = trimesh.creation.box([.2, .2, .2]);head.apply_translation(COORD.polar([.1, 0., 1.]))
        triangle = COORD.polar(np.array([[0., -1., 0.], [0., 1., 0.], [0., 0., 2.]]))
        arcs = A.obstacle_shadow(triangle, head, 2.)
        self.assertTrue(contains(arcs, 1.))
        self.assertFalse(contains(arcs, 180.))

    def test_local_angles_do_not_hide_a_remote_obstacle(self):
        object1 = trimesh.creation.box([1., 1., 1.]);object1.apply_translation(COORD.polar([0., 0., 1.]))
        obstacle = trimesh.creation.box([.4, .5, .5]);obstacle.apply_translation(COORD.polar([-1.5, 0., 1.]))
        mesh = trimesh.util.concatenate([object1, obstacle])
        head = trimesh.creation.box([.1, .2, .2]);head.apply_translation(COORD.polar([-.55, 0., 1.]))
        study = A.AngleStudy(mesh, [head]);result = study.classify([[-1., 0., 0.]])
        self.assertFalse(study.test_angle(0.)['clear'])
        self.assertTrue(study.test_angle(60.)['clear'])
        self.assertTrue(contains(result['clear_intervals_deg'], 60.))
        self.assertTrue(contains(result['geometry_blocked_intervals_deg'], 1.))
        for angle in [1., 12., 30., 60., 89., 271., 300., 330., 359.]:
            sweep=F.swept_pieces([head],A.direction(angle),study.length)[0]
            volume=F.intersection_volume(mesh,sweep,study.scale)
            if contains(result['clear_intervals_deg'], angle):self.assertLess(volume,1e-9)
            if contains(result['geometry_blocked_intervals_deg'], angle):self.assertGreater(volume,1e-9)

    def test_sector_encloses_intermediate_angles(self):
        mesh=trimesh.creation.box([.1,.1,.1]);mesh.apply_translation(COORD.polar([-2.,0.,1.]))
        head=trimesh.creation.box([.1,.1,.1]);head.apply_translation(COORD.polar([0.,0.,1.]))
        study=A.AngleStudy(mesh,[head])
        self.assertTrue(study.test_angle(-12.)['clear'])
        self.assertTrue(study.test_angle(12.)['clear'])
        self.assertIsNotNone(study.envelope_hit(-12.,12.))
        self.assertFalse(study.test_angle(0.)['clear'])

    def test_small_clipped_head_vertices_are_not_merged(self):
        points=np.array([[x,y,z] for x in [0.,2e-9] for y in [0.,3e-9]
                         for z in [0.,.0007]])+np.array([.08,-.04,.03])
        head=A.hull_mesh(points)
        self.assertTrue(head.is_watertight)
        self.assertEqual(len(head.vertices),8)
        self.assertGreater(head.metadata['convex_volume_m3'],0.)


if __name__=='__main__':unittest.main()
