import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
from co_common import trimesh,G
import prepare_five_pose_group_ready as helper

class FacetWrapFallbackTests(unittest.TestCase):
    def test_facet_cells_keep_original_contact_plane_and_outward_depth(self):
        mesh=trimesh.creation.box([.2,.1,.08]);depth=.005
        for index,triangle in enumerate(mesh.triangles):
            vertices=helper.facet_cell(mesh,triangle,index,depth)
            np.testing.assert_array_equal(vertices[:3],triangle)
            np.testing.assert_allclose((vertices[3:]-triangle)@mesh.face_normals[index],depth)
            self.assertGreater(G.hull_mesh(vertices).volume,0.)

    def test_unrelated_failures_are_not_reinterpreted_as_offset_errors(self):
        with patch.object(helper,'_original_run',side_effect=RuntimeError('Original load hash mismatch')):
            with self.assertRaisesRegex(RuntimeError,'Original load hash mismatch'):
                helper.surface_run(None,None,None,None,None)

    def test_fallback_restores_shared_functions_even_on_failure(self):
        original_offset=helper.preparation.step32.wrap_offsets;original_cell=G.head_cell
        calls=[]
        def run(*args):
            calls.append(1)
            if len(calls)==1:raise RuntimeError('Full surface offset unresolved; cannot label this as force failure')
            self.assertIsNone(helper.preparation.step32.wrap_offsets(None,.005))
            raise RuntimeError('Numerical union unresolved')
        with patch.object(helper,'_original_run',side_effect=run):
            with self.assertRaisesRegex(RuntimeError,'Numerical union unresolved'):
                helper.surface_run(None,None,None,None,None)
        self.assertIs(G.head_cell,original_cell)
        self.assertIs(helper.preparation.step32.wrap_offsets,original_offset)

if __name__=='__main__':unittest.main()
