"""Step3 uses the exclusive Step1.3 partition, retaining each pose's witnesses."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from whole_task_2.step1.common import sha256, write_json
from . import tool_membranes as M


class Step1TaskTests(unittest.TestCase):
    def fixture(self, root, empty_pose=None):
        work = root/'whole_task_2/data/B/step1_reach'; work.mkdir(parents=True)
        config = root/'robots_config/config.json'; config.parent.mkdir()
        config.write_text('{}')
        labels = np.array([0,0,1,2,3,4,5,-1])
        if empty_pose is not None:labels[labels==empty_pose]=-1
        joints = np.broadcast_to(np.arange(6)[None, :, None, None], (8, 6, 4, 7)).copy()
        # All poses could reach all faces; only their exclusive assignments are work.
        regions = np.ones((8,6),bool)
        reach = np.ones((8,6),bool); reach[1,0]=False
        axes = np.zeros((8,6,3)); axes[:,:,0]=np.arange(6)*.1; axes[:,:,2]=-np.sqrt(1-axes[:,:,0]**2)
        np.savez_compressed(work/'plan.npz', face_task=labels,
                            task_reach=regions, reach=reach, reach_axis=axes,
                            joints=joints)
        placement = dict(rotation_world_object=np.eye(3).tolist(), translation_world_m=[0., 0., 0.])
        areas = [dict(id=id, face_ids=np.flatnonzero(labels==j).tolist(), placement=placement)
                 for j,id in enumerate(('+X', '+Y', '+Z', '-X', '-Y', '-Z'))]
        plan = dict(areas=areas, arrays_sha256=sha256(work/'plan.npz'),
                    input_sha256={'robots_config/config.json':sha256(config)})
        write_json(work/'plan.json', plan)
        write_json(work/'validation.json', dict(status='passed', plan_sha256=sha256(work/'plan.json')))

    def test_exclusive_regions_keep_their_own_joint_indices_without_repeated_faces(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            with patch.object(M, 'ROOT', root):
                _, records, _ = M.step1_tasks('B')
            self.assertEqual([r['id'] for r in records], ['+X', '+Y', '+Z', '-X', '-Y', '-Z'])
            for j,r in enumerate(records):
                np.testing.assert_array_equal(r['joints'], np.full((len(r['certified_face_ids']),4,7),j))
                np.testing.assert_allclose(r['axes'][:,0],j*.1)
            self.assertEqual(records[-1]['working_face_ids'],[6])
            faces = [f for r in records for f in r['working_face_ids']]
            self.assertEqual(sorted(faces),list(range(7)))
            self.assertEqual(len(faces),len(set(faces)))

    def test_enclosed_fill_carries_load_without_borrowing_a_tool_witness(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            with patch.object(M, 'ROOT', root):
                _, records, _ = M.step1_tasks('B')
            self.assertEqual(records[0]['working_face_ids'],[0,1])
            self.assertEqual(records[0]['certified_face_ids'],[0])
            self.assertEqual(records[1]['certified_face_ids'],[2])
            self.assertEqual(records[2]['certified_face_ids'],[3])

    def test_a_redundant_pose_needs_no_support_task(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root,empty_pose=5)
            with patch.object(M, 'ROOT', root):
                plan, records, _ = M.step1_tasks('B')
            self.assertEqual(len(plan['areas']),6)
            self.assertEqual([r['id'] for r in records],['+X','+Y','+Z','-X','-Y'])
            self.assertEqual([r['direction_index'] for r in records],list(range(5)))

    def test_a_mismatched_partition_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            work = root/'whole_task_2/data/B/step1_reach'
            import json
            plan = json.loads((work/'plan.json').read_text())
            plan['areas'][1]['face_ids']=[0,2]
            write_json(work/'plan.json',plan)
            write_json(work/'validation.json',dict(status='passed',plan_sha256=sha256(work/'plan.json')))
            with patch.object(M,'ROOT',root), self.assertRaisesRegex(ValueError,'task assignment does not match'):
                M.step1_tasks('B')


class TaskPartitionTests(unittest.TestCase):
    def test_actual_downstream_face_lists_cannot_repeat_a_surface_face(self):
        mesh=trimesh.creation.box()
        with self.assertRaisesRegex(ValueError,'repeat source faces'):
            M.task_partition([dict(working_face_ids=[0,1]),dict(working_face_ids=[1,2])],mesh)
        result=M.task_partition([dict(working_face_ids=[0,1]),dict(working_face_ids=[2])],mesh)
        self.assertEqual(result['assigned_face_count'],3)
        self.assertEqual(result['duplicate_face_count'],0)
        self.assertAlmostEqual(result['assigned_area_fraction'],.25)


if __name__ == '__main__':
    unittest.main()
