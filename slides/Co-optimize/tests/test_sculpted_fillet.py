"""Analytic regression checks for real fillet deletion and retained flat faces."""
from pathlib import Path
import sys,unittest
from types import SimpleNamespace
import numpy as np
COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/p)for p in ['helper_func','step5.2','step5.3']]
import _bootstrap
from deterministic_base import box,solid,unpack,volume
from sculpted_pruning import edge_fillet_knife
from prune_support import material_components
from surface_flow import flow


class SculptedFilletTests(unittest.TestCase):
    def setUp(self):
        self.stock=box(np.zeros(3),np.full(3,.02))
        knife=edge_fillet_knife(np.array([.02,.02,0.]),np.array([.02,.02,.02]),
            np.array([1.,0.,0.]),np.array([0.,1.,0.]),.002)
        self.result=self.stock-knife

    def test_quarter_circle_removal_matches_analytic_volume(self):
        expected=(1-np.pi/4)*.002**2*(.02-44/35*.004)
        removed=volume(self.stock-self.result)
        self.assertAlmostEqual(removed/expected,1.,delta=.04)
        self.assertLessEqual(volume(self.result-self.stock),1e-10)

    def test_flat_contact_and_broad_interior_are_preserved(self):
        contact=box(np.array([.0195,.004,.004]),np.array([.02,.016,.016]))
        interior=box(np.array([.004,.004,.004]),np.array([.016,.016,.016]))
        self.assertLessEqual(volume(contact-self.result),1e-10)
        self.assertLessEqual(volume(interior-self.result),1e-10)

    def test_actual_mesh_is_closed_connected_and_keeps_axis_bounds(self):
        mesh=unpack(self.result)
        self.assertTrue(mesh.is_watertight)
        self.assertEqual(len(material_components(solid(mesh))),1)
        np.testing.assert_allclose(mesh.bounds,np.array([[0.,0.,0.],[.02,.02,.02]]),atol=1e-10,rtol=0)

    def test_surface_flow_keeps_the_entire_mandatory_body(self):
        body=unpack(box(np.full(3,.1),np.full(3,.12)))
        certificate=SimpleNamespace(mesh=body,layout=SimpleNamespace(active=[0],placements=[np.eye(4)]))
        cloud=np.array([[-.05,-.05,-.05],[.05,.05,.05]])
        result,record=flow(unpack(self.stock),certificate,[np.eye(4)],cloud,mandatory=self.stock)
        self.assertTrue(record['pinned_face_tangents_are_linear'])
        self.assertLessEqual(volume(self.stock-result),1e-10)
        self.assertLessEqual(volume(result-self.stock),1e-10)

    def test_free_surface_flow_deletes_material_and_remains_closed(self):
        body=unpack(box(np.full(3,.1),np.full(3,.12)))
        certificate=SimpleNamespace(mesh=body,layout=SimpleNamespace(active=[0],placements=[np.eye(4)]))
        result,_=flow(unpack(self.stock),certificate,[np.eye(4)],np.array([[-.05]*3,[.05]*3]))
        self.assertLess(volume(result),volume(self.stock))
        self.assertLessEqual(volume(result-self.stock),1e-10)
        self.assertTrue(unpack(result).is_watertight)
        self.assertEqual(len(material_components(result)),1)


if __name__=='__main__':unittest.main()
