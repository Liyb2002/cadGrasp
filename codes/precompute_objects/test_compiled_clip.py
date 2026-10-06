"""Compiled clipping must match the original positive-area collision test."""
import sys,unittest
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'slides/baseline_algo'))
from step2_local_support.geometry import Clearance
from codes.precompute_objects.heads.compiled_clip import compiled

class ClipTests(unittest.TestCase):
    @unittest.skipIf(compiled is None,'Optional numba backend unavailable')
    def test_random_convex_cells_match_python(self):
        mesh=trimesh.creation.icosphere(subdivisions=2);check=Clearance(mesh,1e-12);rng=np.random.default_rng(7912)
        for i in range(160):
            center=rng.uniform(-1.5,1.5,3);points=center+rng.uniform(-.4,.4,(10,3))
            self.assertEqual(check.obstruction(points),check.obstruction_python(points),i)
    def test_boundary_tolerance_matches_python(self):
        mesh=trimesh.creation.box();check=Clearance(mesh,1e-12)
        for gap in (0.,1e-13,1e-10,-1e-13,-1e-10,.1,-.1):
            box=trimesh.creation.box(extents=[.1,.3,.3]);box.apply_translation([.55+gap,0,0])
            self.assertEqual(check.obstruction(box.vertices),check.obstruction_python(box.vertices))
if __name__=='__main__':unittest.main()
