"""Compare supporting-plane acceleration with original continuous sweep checks."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'slides/baseline_algo'))
from step2_local_support.withdrawal import Analyzer
from step2_local_support.geometry import hull_mesh

class SupportingPlaneTests(unittest.TestCase):
    def analyzer(self,mesh):
        return Analyzer(mesh,.02,dict(vectors=[[1,0,0]],global_allowed_directions={'ids':[0]},preferred_withdrawal_direction=None,installation={'floor_plane':[0,0,0,0]}))
    def test_supporting_face_matches_reference(self):
        mesh=trimesh.creation.box(extents=[1,1,1]);a=self.analyzer(mesh)
        face=int(np.argmax(mesh.face_normals[:,0]));triangle=mesh.triangles[face]
        head=hull_mesh(np.vstack([triangle,triangle+[.02,0,0]]));head.metadata['source_face']=face
        for direction in ([1,0,0],[0,1,0],[1,1,-1]):
            d=np.array(direction,float);d/=np.linalg.norm(d)
            fast=a.test([head],d);head.metadata.clear();slow=a.test([head],d);head.metadata['source_face']=face
            self.assertEqual(fast['clear'],slow['clear']);self.assertTrue(fast['clear'])
    def test_interior_plane_cannot_skip_collision(self):
        # Two disjoint closed boxes: a head exits the first into the second.
        first=trimesh.creation.box(extents=[1,1,1]);second=first.copy();second.apply_translation([1.5,0,0]);mesh=trimesh.util.concatenate([first,second]);a=self.analyzer(mesh)
        face=int(np.argmax(first.face_normals[:,0]));triangle=mesh.triangles[face]
        head=hull_mesh(np.vstack([triangle,triangle+[.02,0,0]]));head.metadata['source_face']=face
        self.assertFalse(a.test([head],[1,0,0])['clear'])
        self.assertFalse(a.supporting_faces[face])
if __name__=='__main__':unittest.main()
