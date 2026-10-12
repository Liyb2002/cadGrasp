"""Half-rays, continuous source area and certified self-occlusion sectors."""
from pathlib import Path
import sys,unittest
import numpy as np

COOPT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COOPT/'helper_func'))
import _bootstrap
from co_common import trimesh
from work_access import WorkAccess,unpack_solid
from work_access_shadow import ShadowClippedWorkAccess
from trimesh.ray.ray_pyembree import RayMeshIntersector


class ShadowAccessTests(unittest.TestCase):
    def example(self):
        source=trimesh.Trimesh([[-.001,-.001,0],[.001,-.001,0],[0,.001,0]],[[0,1,2]],process=False)
        occluder=trimesh.creation.box([.006,.012,.002]);occluder.apply_translation([.01,0,.02])
        mesh=trimesh.util.concatenate([source,occluder])
        return mesh,WorkAccess(mesh,[0],30),ShadowClippedWorkAccess(mesh,[0],30)

    def test_remove_occluded_sector_and_keep_visible_forward_ray(self):
        mesh,raw,access=self.example()
        points=np.array([[.03,0,.06],[0,0,.06],[0,0,-.01]])
        np.testing.assert_array_equal(raw.contains_points(points),[True,True,False])
        np.testing.assert_array_equal(access.contains_points(points),[False,True,False])
        self.assertEqual(access.definition()['source_faces_with_shadow_clipping'],1)
        self.assertEqual(len(access.cell_face_ids),len(access.counts))
        self.assertTrue((access.cell_face_ids==0).all())

    def test_never_discards_reachable_original_directions(self):
        mesh,_,access=self.example();ray=RayMeshIntersector(mesh)
        source=mesh.triangles[0];normal=mesh.face_normals[0]
        angles=np.arange(0,360,7)*np.pi/180
        for weights in [[1,0,0],[0,1,0],[0,0,1],[.2,.3,.5]]:
            point=np.asarray(weights)@source
            directions=np.c_[np.tan(np.deg2rad(29))*np.cos(angles),
                np.tan(np.deg2rad(29))*np.sin(angles),np.ones(len(angles))]
            clear=~ray.intersects_any(np.repeat((point+1e-5*normal)[None],len(angles),axis=0),directions)
            targets=point+.08*directions[clear]
            self.assertTrue(access.contains_points(targets).all())

    def test_clipped_off_axis_cells_keep_their_side_planes(self):
        _,_,access=self.example()
        points=np.array([[.03,0,.06],[0,0,.06],[-.03,0,.06],[0,0,-.02]])
        ray=RayMeshIntersector(unpack_solid(access.solid(.10)))
        np.testing.assert_array_equal(ray.contains_points(points),access.contains_points(points))


if __name__=='__main__':unittest.main()
