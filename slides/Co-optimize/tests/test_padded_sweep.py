"""Conservative repair must contain the continuous reference, without filling concavities."""

import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
import physics_guided
import manifold3d as md
import physics_guided_padded_sweep as kernel
from co_common import S,material_volume

class PaddedSweepTests(unittest.TestCase):
    def test_box_oblique_sweep_matches_convex_reference(self):
        mesh=trimesh.creation.box([.2,.1,.08]);delta=np.array([.1,.05,.04])
        reference=trimesh.convex.convex_hull(np.vstack([mesh.vertices,mesh.vertices+delta]))
        output=kernel.padded_swept_solid(mesh,delta)
        self.assertLess(material_volume(S.solid(output)-S.solid(reference)),1e-12)
        self.assertLess(material_volume(S.solid(reference)-S.solid(output)),1e-12)

    def test_collapsed_prism_is_retained_as_a_conservative_superset(self):
        mesh=trimesh.creation.box([.2,.1,.08]);delta=np.array([.1,.05,.04])
        reference=trimesh.convex.convex_hull(np.vstack([mesh.vertices,mesh.vertices+delta]))
        original=kernel._solid;collapsed=[]
        def inject(vertices,faces,label):
            if 'Padded face prism' in label and not collapsed:
                collapsed.append(label);return md.Manifold()
            return original(vertices,faces,label)
        with patch.object(kernel,'_solid',side_effect=inject):output=kernel.padded_swept_solid(mesh,delta)
        self.assertEqual(output.metadata['padded_sweep_repair']['repaired_prisms'],1)
        self.assertGreater(output.metadata['padded_sweep_repair']['maximum_extra_cube_half_extent_m'],0.)
        self.assertLess(material_volume(S.solid(reference)-S.solid(output)),1e-12)
        self.assertLess(material_volume(S.solid(output)-S.solid(reference)),1e-10)

    def test_nonconvex_sweep_keeps_empty_quadrant(self):
        a=md.Manifold.cube([.2,.1,.1]);b=md.Manifold.cube([.1,.2,.1]);value=a+b
        data=value.to_mesh64();mesh=trimesh.Trimesh(np.asarray(data.vert_properties[:,:3]),np.asarray(data.tri_verts),process=False)
        output=kernel.padded_swept_solid(mesh,[0,0,.1])
        reference=md.Manifold.cube([.2,.1,.2])+md.Manifold.cube([.1,.2,.2])
        ref_mesh=reference.to_mesh64();ref=trimesh.Trimesh(np.asarray(ref_mesh.vert_properties[:,:3]),np.asarray(ref_mesh.tri_verts),process=False)
        self.assertLess(material_volume(S.solid(ref)-S.solid(output)),1e-12)
        self.assertLess(material_volume(S.solid(output)-S.solid(ref)),1e-12)

    def test_zero_displacement(self):
        mesh=trimesh.creation.box();result=kernel.padded_swept_solid(mesh,[0,0,0])
        np.testing.assert_array_equal(result.vertices,mesh.vertices)
        self.assertEqual(result.metadata['padded_sweep_repair']['repaired_prisms'],0)

if __name__=='__main__':unittest.main()
