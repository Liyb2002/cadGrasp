"""Failure artifacts must show actual material and reproducible intersections."""
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import json
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step5_connect_support import failure_visuals as F,direction_first as X,belt_geometry as B
from step5_connect_support.fixtures import contacts


class FailureVisualTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.creation.box([1., 1., 1.]);self.mesh.apply_translation([0, 0, 1.])
        self.domain=SimpleNamespace(mesh=self.mesh,work_ids=np.array([],int),com=self.mesh.center_mass)
        self.contacts=contacts(self.mesh)
        self.parts,self.labels=X.make_heads(self.mesh,self.contacts,.01)

    def test_rejected_direction_shows_a_real_overlap_and_keeps_all_parts(self):
        event,intersection=F.find_event(self.domain,self.parts,self.labels,[1.,0,0])
        self.assertTrue(event['object_collision_parts'])
        self.assertGreater(intersection.volume,0)
        self.assertFalse(event['accepted_trajectory'])
        p=self.parts[event['witness_part']].copy();p.vertices+=event['translation_m']
        self.assertAlmostEqual(B.Scene(self.mesh).volume(p),event['intersection_volume_m3'],places=12)

    def test_no_sampled_collision_is_not_marked_as_trajectory_success(self):
        event,intersection=F.find_event(self.domain,self.parts,self.labels,[-1.,0,0])
        self.assertIsNone(intersection)
        self.assertFalse(event['accepted_trajectory'])
        self.assertEqual(event['witness_kind'],'no_sampled_penetration_witness')

    def test_failed_frame_member_is_captured_instead_of_discarded(self):
        scene=X.SweptScene(self.mesh,np.array([1.,0,0]),.02)
        events=[];scene.failure_callback=lambda *a:events.append(a)
        scene.context_parts=self.parts;scene.context_labels=self.labels;scene.context_stage='loose_frame'
        member=trimesh.creation.box([.1, .1, .1]);member.apply_translation([0, 0, 1])
        self.assertFalse(scene.clear(member))
        self.assertEqual(len(events),1)
        self.assertEqual(events[0][0],'loose_frame')
        self.assertEqual(len(events[0][1]),len(self.parts)+1)
        np.testing.assert_array_equal(events[0][1][-1].vertices,member.vertices)

    def test_insufficient_clearance_is_distinct_from_material_collision(self):
        member=trimesh.creation.box([.1, .1, .1]);member.apply_translation([-.551, 0, 1])
        event,intersection=F.clearance_event(self.domain,[member],['frame_candidate'],[-1.,0,0],.02,0)
        self.assertEqual(event['witness_kind'],'reserved_clearance_intersection')
        self.assertEqual(event['object_collision_parts'],[])
        self.assertEqual(event['intersection_volume_m3'],0.)
        self.assertGreater(event['clearance_overlap_m3'],0.)
        self.assertTrue(event['clearance_envelope_not_actual_material'])

    def test_static_artifacts_contain_actual_stl_and_offline_viewer(self):
        geometry=dict(attempts=[dict(direction=[1.,0,0])])
        with tempfile.TemporaryDirectory() as folder,patch.object(F.I,'hashes',return_value={}):
            out=Path(folder);report=F.render('test',self.domain,self.contacts,self.parts,self.labels,
                geometry,'fixed_heads_block_all_proposed_sweeps',out,static_only=True)
            self.assertEqual(report['material_stage'],'heads')
            self.assertFalse(report['successful_trajectory_claimed'])
            self.assertTrue((out/'failure.png').is_file())
            self.assertTrue((out/'failure_pose_mm.stl').is_file())
            viewer=(out/'failure_viewer.html').read_text()
            self.assertNotIn('__DATA__',viewer)
            self.assertNotIn('<script src=',viewer)
            original=trimesh.load(out/'failed_shape_mm.stl',process=False)
            np.testing.assert_allclose(original.triangles/1000,trimesh.util.concatenate(self.parts).triangles,atol=1e-12)
            moved=trimesh.load(out/'failure_pose_mm.stl',process=False)
            displacement=np.asarray(report['events'][report['primary_event']]['translation_m'])
            np.testing.assert_allclose(moved.triangles/1000,original.triangles/1000+displacement,atol=1e-12)

    def test_zero_heads_shows_object_without_fabricated_support_or_motion(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(F.I,'hashes',return_value={}):
            out=Path(folder);report=F.no_heads('test',self.domain,out)
            self.assertEqual(report['shape_labels'],[])
            self.assertFalse(report['trajectory_tested'])
            self.assertFalse((out/'failed_shape_mm.stl').exists())
            self.assertTrue((out/'failure.png').is_file())
            viewer=(out/'failure_viewer.html').read_text()
            payload=json.loads(viewer.split('<script id="data" type="application/json">')[1].split('</script>')[0])
            self.assertEqual(payload['shapes'],[[]])
            self.assertTrue(payload['object']['f'])
            self.assertEqual(payload['events'][0]['translation'],[0,0,0])


if __name__=='__main__':unittest.main()
