import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import trimesh
from .sweep_cache import sweep_mesh


class SweepCacheTests(unittest.TestCase):
    def test_byte_exact_geometry_reuse_and_all_geometric_inputs_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            body=trimesh.creation.box();calls=[]
            def build():calls.append(True);return body.copy()
            first=sweep_mesh(directory,body,[0.,0.,1.],1.,build,'source-v1')
            second=sweep_mesh(directory,body,[0.,0.,1.],1.,build,'source-v1')
            np.testing.assert_array_equal(first.vertices,second.vertices)
            np.testing.assert_array_equal(first.faces,second.faces);self.assertEqual(len(calls),1)
            sweep_mesh(directory,body,[0.,1.,0.],1.,build,'source-v1')
            sweep_mesh(directory,body,[0.,0.,1.],2.,build,'source-v1')
            sweep_mesh(directory,body,[0.,0.,1.],1.,build,'source-v2')
            moved=body.copy();moved.vertices[0,0]+=1e-12
            sweep_mesh(directory,moved,[0.,0.,1.],1.,build,'source-v1')
            self.assertEqual(len(calls),5)

    def test_corrupt_archive_is_never_used(self):
        with tempfile.TemporaryDirectory() as directory:
            body=trimesh.creation.box();calls=[]
            def build():calls.append(True);return body.copy()
            sweep_mesh(directory,body,[0.,0.,1.],1.,build,'source-v1')
            manifest=next(Path(directory).glob('*.json'));info=json.loads(manifest.read_text())
            archive=Path(directory)/(info['mesh_sha256']+'.npz');archive.write_bytes(b'broken')
            result=sweep_mesh(directory,body,[0.,0.,1.],1.,build,'source-v1')
            self.assertEqual(len(calls),2);np.testing.assert_array_equal(result.vertices,body.vertices)


if __name__=='__main__':unittest.main()
