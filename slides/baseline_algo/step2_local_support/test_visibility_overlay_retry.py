"""Overlay failure retains the full uncertain access domain and other faces."""
from pathlib import Path
import sys
import unittest
import numpy as np
import trimesh
from shapely.errors import GEOSException
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import visibility as V,work_volume as W
from step2_local_support.visibility_overlay_retry import recover_cells


class OverlayRecoveryTests(unittest.TestCase):
    def test_failed_face_is_retained_as_uncertain_and_other_faces_still_run(self):
        mesh=trimesh.creation.box([1., 1., 1.])
        triangles=np.array([[[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]],
                            [[2.,0.,2.],[3.,0.,2.],[2.,1.,2.]]])
        normals=np.tile([0.,0.,1.],(2,1));e1,e2=W.frame(normals)
        slopes=np.array([[-.2,-.2],[.2,-.2],[0.,.2]])
        def broken(mesh,triangles,*args):
            if len(triangles) and triangles[0,0,0]==0:
                raise GEOSException('non-noded intersection')
            return V.build_cells(mesh,triangles,*args)
        cells,report=recover_cells(broken,mesh,triangles,normals,e1,e2,1.,1e-5,slopes,0)
        self.assertEqual(report['overlay_recovery']['failed_face_count'],1)
        first=[c for c in cells if c[1]==0]
        self.assertEqual(len(first),1)
        np.testing.assert_array_equal(first[0][0],triangles[0])
        np.testing.assert_array_equal(first[0][2],slopes)
        self.assertFalse(first[0][3])
        self.assertTrue(any(c[1]==1 and c[3] for c in cells))
        self.assertEqual([r['work_face_index'] for r in report['work_faces']],[0,1])

    def test_non_geometry_errors_are_not_hidden(self):
        def broken(*args):raise ValueError('invalid inputs')
        empty=np.empty((0,3,3))
        with self.assertRaisesRegex(ValueError,'invalid inputs'):
            recover_cells(broken,None,empty,empty,empty,empty,1.,0.,np.zeros((3,2)),0)


if __name__=='__main__':unittest.main()
