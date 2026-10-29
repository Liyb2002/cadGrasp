"""Six-side split on boxes with known answers."""
import unittest
import numpy as np
import trimesh
from subdivide_task.step1.areas import VIEWS,Surface,areas,drop_small,fill_enclosed,task_regions,nearest_area,connected_tasks


class AreaTests(unittest.TestCase):
    def test_cube_splits_into_its_six_sides(self):
        mesh=trimesh.creation.box(extents=[1,1,1]);axes=-VIEWS
        # A side paints the faces whose outward normal points against its needle.
        tilt=np.where(mesh.face_normals@axes.T<-.9,0.,np.inf)
        label,summary=areas(mesh,np.arange(len(mesh.faces)),tilt,axes,25.,.5)
        surface=Surface(mesh)
        for j in range(6):
            faces=np.flatnonzero(label==j)
            self.assertEqual(len(faces),2);self.assertEqual(len(surface.pieces(label==j)),1)
            np.testing.assert_allclose(mesh.face_normals[faces]@axes[j],-1,atol=1e-12)

    def test_every_face_goes_to_the_most_head_on_side(self):
        # Every side "reaches" every face; incidence alone decides, still one side per cube face.
        mesh=trimesh.creation.box(extents=[1,1,1]);axes=-VIEWS
        label,_=areas(mesh,np.arange(len(mesh.faces)),np.zeros((len(mesh.faces),6)),axes,25.,.5)
        np.testing.assert_allclose(np.einsum('ij,ij->i',mesh.face_normals,axes[label]),-1,atol=1e-12)

    def test_small_detached_piece_is_removed(self):
        # The pruning helper leaves discarded islands unassigned; hole filling is separate.
        mesh=trimesh.creation.box(extents=[1,1,1]).subdivide().subdivide().subdivide();surface=Surface(mesh)
        label=(mesh.face_normals@VIEWS.T).argmax(1);top=np.flatnonzero(label==2)
        f=top[np.argmin(np.linalg.norm(mesh.triangles_center[top]-[0,0,.5],axis=1))];label[f]=5
        self.assertEqual(drop_small(surface,np.arange(len(mesh.faces)),label,mesh.area_faces,.01*mesh.area),1)
        self.assertEqual(label[f],-1)

    def test_task_regions_remove_small_islands_but_keep_large_disconnected_parts(self):
        meshes=[trimesh.creation.box(extents=[s,s,s]) for s in (1.,.03,.5)]
        for i,mesh in enumerate(meshes):mesh.apply_translation([3*i,0,0])
        mesh=trimesh.util.concatenate(meshes);certified=np.ones((len(mesh.faces),1),bool);before=certified.copy()
        regions,enclosed=task_regions(mesh,np.arange(len(mesh.faces)),certified)
        self.assertTrue(regions[:12,0].all());self.assertFalse(regions[12:24,0].any())
        self.assertTrue(regions[24:,0].all());self.assertFalse(enclosed.any())
        np.testing.assert_array_equal(certified,before)

    def test_partition_keeps_unreachable_and_removed_islands_unassigned(self):
        meshes=[trimesh.creation.box(extents=[s,s,s]) for s in (1.,.03,.5)]
        for i,mesh in enumerate(meshes):mesh.apply_translation([3*i,0,0])
        mesh=trimesh.util.concatenate(meshes);tilt=np.full((len(mesh.faces),1),np.inf);tilt[:24]=0.
        label,_=areas(mesh,np.arange(len(mesh.faces)),tilt,np.array([[0.,0,-1]]),25.,.1)
        self.assertTrue((label[:12]==0).all());self.assertTrue((label[12:]==-1).all())

    def test_unreachable_pocket_belongs_to_its_surrounding_task(self):
        mesh=trimesh.creation.box().subdivide().subdivide().subdivide();surface=Surface(mesh)
        label=(mesh.face_normals@VIEWS.T).argmax(1);top=np.flatnonzero(label==2)
        f=top[np.argmin(np.linalg.norm(mesh.triangles_center[top]-[0,0,.5],axis=1))]
        label[f]=-1
        filled=fill_enclosed(surface,np.arange(len(label)),label,mesh.area_faces,.01*mesh.area)
        self.assertIn(f,filled);self.assertEqual(label[f],2)

    def test_pocket_between_two_tasks_is_not_enclosed(self):
        mesh=trimesh.creation.box().subdivide().subdivide().subdivide();surface=Surface(mesh)
        label=(mesh.face_normals@VIEWS.T).argmax(1);top=np.flatnonzero(label==2)
        f=next(int(f) for f in top if any(label[g]!=2 for g in surface.nb[f]))
        label[f]=-1
        fill_enclosed(surface,np.arange(len(label)),label,mesh.area_faces,.01*mesh.area)
        self.assertEqual(label[f],-1)

    def test_target_boundary_is_not_a_surrounding_task(self):
        mesh=trimesh.creation.box().subdivide().subdivide().subdivide();surface=Surface(mesh)
        label=np.full(len(mesh.faces),2);f=0;label[f]=-1
        outside=surface.nb[f][0];label[outside]=-1
        ids=np.delete(np.arange(len(label)),outside)
        fill_enclosed(surface,ids,label,mesh.area_faces,.01*mesh.area)
        self.assertEqual(label[f],-1)

    def test_task_pocket_does_not_become_certified_reach(self):
        mesh=trimesh.creation.box().subdivide().subdivide().subdivide();ids=np.arange(len(mesh.faces))
        certified=(mesh.face_normals[:,2]>.9)[:,None];top=np.flatnonzero(certified[:,0])
        f=top[np.argmin(np.linalg.norm(mesh.triangles_center[top]-[0,0,.5],axis=1))]
        certified[f]=False;before=certified.copy()
        region,filled=task_regions(mesh,ids,certified)
        self.assertTrue(region[f,0]);self.assertTrue(filled[f,0])
        np.testing.assert_array_equal(certified,before)
        # A hole beyond the configured small-pocket area must remain a hole.
        region,filled=task_regions(mesh,ids,certified,mesh.area_faces[f]/mesh.area/2)
        self.assertFalse(region[f,0]);self.assertFalse(filled[f,0])

    def test_unreachable_disconnected_target_joins_closest_task(self):
        meshes=[trimesh.creation.box() for _ in range(3)]
        meshes[1].apply_translation([4,0,0]);meshes[2].apply_translation([3,0,0])
        mesh=trimesh.util.concatenate(meshes);label=np.repeat([0,1,-1],12)
        nearest_area(Surface(mesh),np.arange(len(label)),label)
        np.testing.assert_array_equal(label[24:],1)

    def test_disconnected_regions_of_one_direction_become_separate_tasks(self):
        meshes=[trimesh.creation.box() for _ in range(3)]
        for i,mesh in enumerate(meshes):mesh.apply_translation([3*i,0,0])
        mesh=trimesh.util.concatenate(meshes);label=np.repeat([0,1,0],12)
        task,components=connected_tasks(mesh,label)
        self.assertEqual(len(components),3)
        self.assertNotEqual(task[0],task[24])
        for k,(parent,faces) in enumerate(components):
            np.testing.assert_array_equal(label[faces],parent)
            np.testing.assert_array_equal(np.flatnonzero(task==k),faces)
        np.testing.assert_array_equal(np.sort(np.concatenate([f for _,f in components])),np.arange(36))

    def test_tasks_require_shared_edges_not_just_a_vertex(self):
        mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,-1,1]],
                             faces=[[0,1,2],[0,3,4],[1,0,5]],process=False)
        task,components=connected_tasks(mesh,np.zeros(3,int))
        self.assertEqual(len(components),2)
        self.assertEqual(task[0],task[2]);self.assertNotEqual(task[0],task[1])

    def test_task_split_excludes_non_target_faces_and_keeps_direction_boundaries(self):
        mesh=trimesh.creation.box();label=np.full(12,-1)
        pair=mesh.face_adjacency[0];label[pair]=[0,1]
        task,components=connected_tasks(mesh,label)
        self.assertEqual(len(components),2)
        np.testing.assert_array_equal(task<0,label<0)
        self.assertNotEqual(task[pair[0]],task[pair[1]])


if __name__=='__main__':
    unittest.main()
