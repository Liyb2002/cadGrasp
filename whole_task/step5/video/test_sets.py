"""Display-path helpers for the two-robot Step5 movie."""
import unittest
import numpy as np
import trimesh
from whole_task.step5.video.sets import face_components


class SetsVideoTests(unittest.TestCase):
    def test_sector_selection_splits_into_edge_connected_pieces(self):
        box=trimesh.creation.box(extents=[1,1,1])
        top=np.flatnonzero(box.face_normals[:,2]>.5);bottom=np.flatnonzero(box.face_normals[:,2]<-.5)
        pieces=face_components(box,np.r_[top,bottom])
        self.assertEqual(sorted(map(sorted,pieces)),sorted([sorted(top.tolist()),sorted(bottom.tolist())]))
        side=np.flatnonzero(box.face_normals[:,0]>.5)
        self.assertEqual(len(face_components(box,np.r_[top,side])),1)


if __name__=='__main__':
    unittest.main()
