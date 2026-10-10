"""Registered initialization, complete cones and legacy artifact readers."""
import sys,unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from step31 import register,build_seed
from work_access import RegisteredWorkVolumes


def task(ids,angle):
    return SimpleNamespace(domain=SimpleNamespace(work_ids=np.array(ids),data={'load':{'cone_half_deg':angle}}))


class WholeInitializeTests(unittest.TestCase):
    def test_all_world_poses_align_to_the_first_saved_pose(self):
        group=dict(id='test',poses=['pose_3','pose_1','pose_6'])
        states,mesh,record,inputs=register('B',group)
        self.assertEqual(record['reference_pose'],'pose_3')
        target=states['pose_3'][0].domain.mesh.vertices
        for row in record['states']:
            original=states[row['pose']][0].domain.mesh.vertices
            np.testing.assert_allclose(transform_points(original,np.array(row['T_native_world_to_reference_world'])),target,atol=1e-12,rtol=0)
            np.testing.assert_allclose(transform_points(mesh.vertices,np.array(row['T_fixture_to_world'])),original,atol=1e-12,rtol=0)

    def test_deduplicated_union_keeps_larger_original_angle_on_shared_face(self):
        mesh=trimesh.Trimesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],process=False)
        states={'a':(task([0],20),np.eye(4),mesh),'b':(task([0],40),np.eye(4),mesh)}
        access=RegisteredWorkVolumes(mesh,states)
        point=np.array([[.2+10*np.tan(np.radians(35)),.2,10.]])
        self.assertTrue(access.contains_points(point)[0])
        self.assertEqual(len(access.families),1)
        self.assertEqual(access.families[0].half_angle_deg,40)
        self.assertEqual(access.poses['a']['half_angle_deg'],20)

    def test_work_carved_seed_retains_contact_and_excludes_all_work_volume(self):
        mesh=trimesh.creation.box(extents=[.04,.04,.04])
        top=np.flatnonzero(mesh.face_normals[:,2]>.9);side=np.flatnonzero(mesh.face_normals[:,0]>.9)
        states={'top':(task(top,30),np.eye(4),mesh),'side':(task(side,30),np.eye(4),mesh)}
        support,triangles,sources,allowed,access,report=build_seed(mesh,states,.002)
        self.assertGreater(report['material_volume_cm3'],0)
        self.assertGreater(len(triangles),0)
        self.assertTrue(np.isin(sources,allowed).all())
        self.assertFalse(np.isin(sources,np.r_[top,side]).any())
        self.assertLess(report['continuous_work_union_overlap_m3'],1e-10)
        normals=mesh.face_normals[sources]
        self.assertFalse(access.contains_points(triangles.mean(1)+1e-7*normals).any())
        self.assertTrue(report['finite_caps_beyond_entire_uncut_seed'])

    def test_new_artifact_reader_preserves_historical_fixture_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);legacy=base/'step3/step3.2';current=base/'step3/step3.1'
            (legacy/'data').mkdir(parents=True);(legacy/'wrapped_support.obj').touch();(legacy/'data/contacts.npz').touch()
            self.assertEqual(initialized_wrap(base),legacy/'wrapped_support.obj')
            self.assertEqual(initialized_contacts(base),legacy/'data/contacts.npz')
            (current/'data').mkdir(parents=True);(current/'wrapped_support.obj').touch();(current/'data/contacts.npz').touch()
            self.assertEqual(initialized_wrap(base),current/'wrapped_support.obj')
            self.assertEqual(initialized_contacts(base/'step3'),current/'data/contacts.npz')

    def test_b_wrap_remains_work_clear_after_reloading_boolean_boundary(self):
        group=dict(id='pose6+10+13+17+30',poses=['pose_6','pose_10','pose_13','pose_17','pose_30'])
        states,mesh,record,inputs=register('B',group)
        support,triangles,sources,allowed,access,report=build_seed(mesh,states)
        bounds=np.array([mesh.vertices.min(0)-.005,mesh.vertices.max(0)+.005])
        corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
        forbidden,lengths=access.solid_for_points(corners)
        self.assertLessEqual(material_volume(S.solid(support)^forbidden),report['continuous_geometry_tolerance_m3'])
        self.assertTrue(report['exported_float64_boundary_checked'])
        self.assertGreater(len(triangles),0)


if __name__=='__main__':unittest.main()
