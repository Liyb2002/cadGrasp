"""Keep-outs bind current Step1 witnesses without requiring a membrane rebuild."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from whole_task.step1.common import sha256
from . import keepout as K


class Step1KeepoutInputsTests(unittest.TestCase):
    def test_model_uses_current_task_faces_poses_axes_and_joints_without_old_step3(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); objects=root/'objects/B'; objects.mkdir(parents=True)
            mesh=trimesh.creation.box(extents=[.1,.2,.3]); mesh.export(objects/'mesh.stl')
            (objects/'meta.json').write_text('{}')
            outer=root/'outer.npz'
            np.savez_compressed(outer,vertices=mesh.vertices*1.1,faces=mesh.faces)
            pose=np.eye(4);pose[:3,3]=[.1,-.7,.3]
            record=dict(id='+X',working_face_ids=[2,5],certified_face_ids=[2,5],
                pose=pose,axes=np.array([[0.,0.,-1.],[0.,1.,0.]]),
                joints=np.arange(56).reshape(2,4,7),angle_deg=0)
            setup=dict(working_direction_world=[0,0,-1],work_point_m=[0,-.7,.3])
            policy=dict(boolean_volume_tolerance_m3=1e-10,mold_outward_displacement_m=.005)
            with patch.object(K,'ROOT',root),patch.object(K,'inputs_and_policy',
                    return_value=({},[record],setup,policy,outer,{})):
                model=K.load_task_model('B')
            self.assertEqual(model.ids,['+X'])
            saved=model.records[0]
            self.assertEqual(saved['working_face_ids'],[2,5])
            for key in ('pose','axes','joints'):
                np.testing.assert_array_equal(saved[key],record[key])
            self.assertFalse((root/'whole_task/data/B/step3_tool_membranes').exists())

    def test_signature_tracks_step1_inputs_and_ignores_old_membrane_files(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); plan=root/'plan.npz'; plan.write_bytes(b'current task arrays')
            outer=root/'outer.npz'; outer.write_bytes(b'geometry')
            stale=root/'whole_task/data/B/step3_tool_membranes'; stale.mkdir(parents=True)
            def inputs(_):
                return {},[],{}, {},outer,{str(plan):sha256(plan)}
            with patch.object(K,'ROOT',root),patch.object(K,'SOURCES',[]), \
                    patch.object(K,'inputs_and_policy',side_effect=inputs):
                original=K.signature('B')
                (stale/'report.json').write_text(json.dumps({'tasks':['obsolete']}))
                (stale/'initialization.npz').write_bytes(b'old joints')
                self.assertEqual(K.signature('B'),original)
                plan.write_bytes(b'new task assignment')
                self.assertNotEqual(K.signature('B')[1],original[1])


if __name__=='__main__':
    unittest.main()
