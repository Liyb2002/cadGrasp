"""Meaningful fixed-floor, envelope and material-reuse construction checks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step5.2'))
import stage5_common as C
sys.path.insert(0,str(Path(__file__).resolve().parent))
import growth as G

np,trimesh=C.np,C.trimesh


def row(state=None,hull=None,ground=()):
    state=np.eye(4) if state is None else state
    hull=np.array([[-.015,-.012,0.],[.015,-.012,0.],[0.,.015,0.]]) if hull is None else hull
    return dict(pose='test',fixture_transform=state,hull=np.asarray(hull),
                object_ground=np.asarray(ground).reshape(-1,2))


class GrowthTests(unittest.TestCase):
    def test_floor_certificate_is_independent_of_bounding_box(self):
        other=np.eye(4);other[:3,:3]=[[0,0,-1],[0,1,0],[1,0,0]];other[2,3]=-.01
        first=row(hull=[[0.,0.,0.],[.002,0.,0.],[0.,.002,0.]],ground=[[.02,0],[.03,0],[.02,.01]])
        second=row(other);second['pose']='other'
        cert=[c for c in G.floor_certificates([first,second]) if c['task_index']==0 and c['other_index']==1]
        self.assertEqual(len(cert),1)
        self.assertAlmostEqual(cert[0]['separation_gap_m'],.01)
        self.assertTrue(cert[0]['bounding_box_independent'])

    def test_object_ground_is_included_in_separating_bound(self):
        other=np.eye(4);other[:3,:3]=[[0,0,-1],[0,1,0],[1,0,0]];other[2,3]=-.01
        first=row(hull=[[.005,0.,0.],[.006,0.,0.],[.005,.001,0.]],ground=[[0.,0.],[0.,.01],[.02,0.]])
        second=row(other)
        cert=[c for c in G.floor_certificates([first,second]) if c['task_index']==0 and c['other_index']==1]
        self.assertEqual(cert,[])

    def test_bbox_uses_every_object_and_fixture_state(self):
        mesh=trimesh.creation.box(extents=[.01,.02,.03]);fixture=trimesh.creation.box(extents=[.02,.01,.01])
        first=dict(object_transform=np.eye(4),fixture_transform=np.eye(4))
        transform=np.eye(4);transform[:3,3]=[.1,-.1,.05]
        second=dict(object_transform=transform,fixture_transform=transform)
        bounds=C.occupied_bounds(mesh,fixture,[first,second])
        np.testing.assert_allclose(bounds,[[-.01,-.11,-.015],[.11,.01,.065]])

    def test_existing_floor_material_is_fully_reused(self):
        mesh=trimesh.creation.box(extents=[.04,.04,.005]);mesh.apply_translation([0,0,.0025])
        result=G.Grow(mesh,[row()],np.array([[-.04,-.04,0.],[.04,.04,.03]])).run()
        self.assertTrue(result['passed'])
        self.assertEqual(result['actions'],[])
        self.assertAlmostEqual(result['added_material_cm3'],0.)
        self.assertAlmostEqual(result['original_material_removed_cm3'],0.)

    def test_feasible_growth_preserves_material_and_box(self):
        mesh=trimesh.creation.box(extents=[.01,.01,.01]);mesh.apply_translation([0,0,.008])
        bounds=np.array([[-.04,-.04,0.],[.04,.04,.03]])
        result=G.Grow(mesh,[row()],bounds).run()
        self.assertTrue(result['passed'],result['status'])
        self.assertGreater(result['added_material_cm3'],0.)
        self.assertAlmostEqual(result['original_material_removed_cm3'],0.,places=9)
        self.assertAlmostEqual(result['box_volume_increase_cm3'],0.,places=9)
        self.assertLessEqual(max(result['maximum_floor_hull_gap_m']),G.TOL)
        self.assertEqual(len(result['mesh'].split(only_watertight=False)),1)

    def test_rejected_additions_are_reported_as_search_exhaustion(self):
        mesh=trimesh.creation.box(extents=[.01,.01,.01]);mesh.apply_translation([0,0,.008])
        result=G.Grow(mesh,[row()],np.array([[-.04,-.04,0.],[.04,.04,.03]]),legal=lambda added:False).run()
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'],'deterministic_attachment_search_exhausted')
        self.assertEqual(result['actions'],[])

    def test_same_inputs_produce_same_growth_actions(self):
        mesh=trimesh.creation.box(extents=[.01,.01,.01]);mesh.apply_translation([0,0,.008])
        bounds=np.array([[-.04,-.04,0.],[.04,.04,.03]])
        first=G.Grow(mesh,[row()],bounds).run()
        second=G.Grow(mesh,[row()],bounds).run()
        self.assertEqual(first['actions'],second['actions'])
        self.assertAlmostEqual(first['added_material_cm3'],second['added_material_cm3'],places=9)

    def test_floor_region_obeys_every_state_box(self):
        first=np.eye(4);second=np.eye(4);second[0,3]=.02
        bounds=np.array([[-.04,-.04,0.],[.04,.04,.03]])
        region=G.floor_region([first,second],0,bounds)
        self.assertAlmostEqual(float(region[:,0].max()),.02)
        for state in [first,second]:
            world=C.F.transform_points(np.c_[region,np.zeros(len(region))],state)
            self.assertTrue((world>=bounds[0]-1e-12).all())
            self.assertTrue((world<=bounds[1]+1e-12).all())


if __name__=='__main__':
    unittest.main()
