"""Ground requirements constrain coverage without prescribing foot locations."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import trimesh
from shapely.geometry import MultiPoint
from step4_connect_support.baseline_current.coverage_growth import CoverageGrow
from step4_connect_support.baseline_current import test_direct_head_growth as head_tests
from step4_connect_support.baseline_current import build_coupled_saddle as S
from step4_connect_support.baseline_current import growing_support as L

class CoverageTests(unittest.TestCase):
    def fixture(self,demands):
        base=head_tests.DirectHeadTests().growth([[.025,.025,.025],[.055,.04,.03]])
        g=CoverageGrow.__new__(CoverageGrow);g.__dict__.update(base.__dict__)
        g.bases=np.array([np.eye(3)]);g.offsets=np.zeros((1,3))
        obj=trimesh.creation.box([.05,.05,.04]);obj.apply_translation([.04,.04,.03])
        g.case=SimpleNamespace(poses=['pose_1'],demands=[np.asarray(demands)],tasks=[SimpleNamespace(domain=SimpleNamespace(mesh=obj))])
        g.navigation_window=dict(min_m=[0,0,0],max_m=[.08,.08,.08])
        g.original_soles=[];g.core_solids=[]
        g.temp=tempfile.TemporaryDirectory();self.addCleanup(g.temp.cleanup);g.out=Path(g.temp.name)
        return g

    def test_grow_only_contacts_needed_for_coverage(self):
        demands=[[.015,.015],[.06,.015],[.03,.06]]
        g=self.fixture(demands);g.ground()
        self.assertEqual(len(g.terminals),2)
        self.assertEqual(g.feet[0]['centers_xy_m'],[])
        solid,report=g.connect()
        self.assertFalse(report['fixed_ground_targets'])
        points=g.floor_points(solid,0)
        self.assertTrue(g.coverage(points,0)[0])
        self.assertLessEqual(MultiPoint(demands).convex_hull.difference(MultiPoint(points).convex_hull.buffer(1e-10)).area,1e-12)
        self.assertLess(len(g.feet[0]['centers_xy_m']),len(g.ground_candidates[0]))
        for _,core in g.core_solids:
            self.assertLessEqual(abs((core-solid).volume())*S.SCALE**3,8e-14)

    def test_floor_halfspaces_constrain_candidate_centers(self):
        g=self.fixture([[.025,.015],[.06,.015],[.03,.06]])
        # A second installed floor requires fixture x >= 20 mm.
        second=np.array([[0,1,0],[0,0,1],[1,0,0]],float)
        g.bases=np.array([np.eye(3),second]);g.offsets=np.array([[0,0,0],[.02,0,0]])
        region=g.center_region(0)
        self.assertGreaterEqual(region.bounds[0],.02301-1e-12)

    def test_lazy_navigation_rejects_colliding_full_rods(self):
        g=self.fixture([[.015,.015],[.06,.015],[.03,.06]])
        wall=head_tests.box([.032,0,0],[.038,.05,.08])
        g.mask=g.mask-wall;g.reference=S.unpack(g.mask);g.pitch=.008
        solid,report=g.connect_heads()
        self.assertTrue(report['lazy_grid_constructed'])
        self.assertGreater(g.blocked_edges,0)
        self.assertEqual(len(L.components(solid)),1)
        for rod in g.beams:
            self.assertLessEqual(abs((rod-g.mask).volume())*S.SCALE**3,8e-14)

    def test_repeated_growth_has_identical_geometry_and_decisions(self):
        demands=[[.015,.015],[.06,.015],[.03,.06]]
        first=self.fixture(demands);second=self.fixture(demands)
        first.ground();second.ground()
        a,ra=first.connect();b,rb=second.connect()
        self.assertEqual(ra['growth_journal'],rb['growth_journal'])
        np.testing.assert_array_equal(S.unpack(a).vertices,S.unpack(b).vertices)

    def test_volume_legality_cannot_bury_original_contact_vertices(self):
        g=self.fixture([[.015,.015],[.06,.015],[.03,.06]])
        g.contact_vertices=np.array([[.04,.03,.03]])
        g.buried_contact_rejections=0
        self.assertFalse(g.legal(g.rod(np.array([.02,.03,.03]),np.array([.06,.03,.03]))))
        self.assertEqual(g.buried_contact_rejections,1)

if __name__=='__main__':unittest.main()
