"""Discriminating routing cases: a blocked straight bar has a clear bent solution."""
from pathlib import Path
import sys
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step2_local_support import insertion as D
from step5_connect_support import routing as T,layout as L,solids as S,motion as M
from step2_local_support import work_volume as W
from step5_connect_support.test_connection import contact
from step5_connect_support.surface_check import surface_distances


def setup_route():
    mesh=trimesh.creation.box([.4, .4, .4]);mesh.apply_translation([0, 1., 0])
    patch=contact(mesh,[-1,0,0],0)
    heads=D.Analyzer(mesh,.004).heads(patch)
    y=patch['center_m'][2]
    tri=np.array([[[-.84,y-.04,.3],[-.76,y-.04,.3],[-.80,y+.04,.3]]])
    work=W.WorkVolume(COORD.polar(tri),[[0,1,0]],[9],30.,.4)
    return mesh,patch,heads,work,np.array([-1.4,y])


class RoutingTests(unittest.TestCase):
    def test_clear_surface_and_heads_but_straight_connection_is_blocked_and_bent_one_passes(self):
        mesh,patch,heads,work,anchor=setup_route()
        self.assertTrue(work.check_surface(patch['triangles_m'])['passed'])
        self.assertTrue(work.check_parts(heads)['passed'])
        router=T.Router(mesh,work,0.)
        root_radius,_,start=T.roots(heads)[0]
        target=COORD.lift_floor(anchor,.012)
        self.assertFalse(work.check_parts([T.beam(start,target,root_radius,router.radius)])['passed'])
        route=router.connect(heads,anchor,.02)
        self.assertIsNotNone(route)
        self.assertGreater(len(route['record']['waypoints_m']),2)
        parts=heads+route['parts']
        self.assertGreater(min(p.vertices[:,1].min() for p in route['parts']),0.)
        self.assertTrue(work.check_parts(parts)['passed'])
        self.assertTrue(M.sweep_check(mesh,parts,router.direction)['passed'])
        solid,record=S.union_parts(parts,float(mesh.extents.max()))
        self.assertTrue(record['one_solid'])
        self.assertLess(surface_distances(solid,patch['triangles_m'].reshape(-1,3)).max(),1e-10)

    def test_spatial_roadmap_finds_and_verifies_a_detour(self):
        mesh,patch,heads,work,anchor=setup_route()
        router=T.Router(mesh,work,0.,edge_budget=2000)
        root_radius,_,start=T.roots(heads)[0]
        target=COORD.lift_floor(anchor,.012)
        points=router.roadmap(start,target,root_radius,router.checks)
        self.assertIsNotNone(points)
        self.assertGreater(len(points),2)
        parts=router.polyline(points,root_radius)
        self.assertIsNotNone(parts)
        self.assertTrue(work.check_parts(parts)['passed'])
        self.assertTrue(M.sweep_check(mesh,parts,router.direction)['passed'])

    def test_thinner_backing_preserves_contact_and_avoids_work_volume(self):
        mesh=trimesh.creation.box([1, 1, 1]);mesh.apply_translation([0, 1, 0])
        patch=contact(mesh,[1,0,0],0)
        work=W.WorkVolume(COORD.polar([[[.508,-2,0],[.508,2,0],[.508,0,3]]]),[[1,0,0]],[9],30.,1.)
        before=patch['triangles_m'].copy()
        heads,_,_,check,surface,backing=L.backing_heads(mesh,[patch],.01,work)
        self.assertTrue(surface['passed']);self.assertTrue(check['passed'])
        self.assertTrue(backing['passed'])
        self.assertLess(backing['selected_depth_factor'],1.)
        np.testing.assert_array_equal(before,patch['triangles_m'])
        solid,record=S.union_parts(heads[0],1.)
        self.assertTrue(record['one_solid'])
        self.assertLess(surface_distances(solid,before.reshape(-1,3)).max(),1e-10)

    def test_interface_boundary_contact_is_reported_separately(self):
        work=W.WorkVolume([[[0,0,0],[1,0,0],[0,1,0]]],[[0,0,1]],[5],30.,1.)
        patch=np.array([[[0,0,0],[-1,0,0],[0,-1,0]]])
        check=work.check_surface(patch,[27])
        self.assertFalse(check['passed'])
        self.assertTrue(check['witness']['at_work_surface'])
        self.assertEqual(check['contact_source_face'],27)
        self.assertEqual(check['witness']['work_face_id'],5)


if __name__=='__main__':unittest.main()
