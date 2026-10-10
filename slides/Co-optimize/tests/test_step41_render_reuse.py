"""The presentation reuses the validated joint solid without repeating acceptance."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import trimesh
from PIL import Image
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'helper_func'));sys.path.insert(0,str(BASE/'vis_func'))
import step41_render as R
from exit_clearance import ExitClearance
from co_common import S, D, material_volume

class RenderReuseTests(unittest.TestCase):
    def test_joint_contact_checks_remain_and_final_solid_is_reused(self):
        obj=trimesh.creation.box(extents=[.01,.01,.01]);seed=S.solid(trimesh.creation.box(extents=[.02,.02,.02]))-S.solid(obj)
        direction=np.array([0.,0.,1.]);policy=ExitClearance(obj)
        sweep=S.solid(S.swept_solid(obj,.5*direction));padded=policy.sweep(.5*direction)
        allowed=np.flatnonzero(obj.face_normals@direction<=1e-9)
        joint=policy.construct(seed,[sweep],[padded],allowed)
        self.assertTrue(joint['diagnostics']['contact_check_performed'])
        with tempfile.TemporaryDirectory(dir=BASE/'data') as directory:
            here=Path(directory);base=here/'data/object_inputs/Test/pose1'
            (base/'step3/step3.1').mkdir(parents=True);(base/'step3/step3.2/data').mkdir(parents=True);(base/'step3/step3.3').mkdir(parents=True)
            D.export_exact_obj(obj,base/'step3/step3.1/registered_object.obj');D.export_exact_obj(S.unpack(seed),base/'step3/step3.3/support_with_rings.obj')
            np.savez(base/'step3/step3.2/data/contacts.npz',allowed_faces=np.arange(len(obj.faces)))
            group=dict(id='pose1',poses=['pose_1']);computed=dict(policy=policy,seed=seed,full_length_m=.5,final_construction=joint,nominal_sweeps={'pose_1':sweep},padded_sweeps={'pose_1':padded},fan=8)
            with patch.object(R,'HERE',here),patch.object(R,'state',return_value=(SimpleNamespace(inputs=[]),np.eye(4),obj)),patch.object(R,'provenance',return_value=dict(inputs={},code={})),patch.object(R,'draw_pose',side_effect=lambda *a,**k:Image.new('RGB',(20,20),'white')),patch.object(policy,'construct',wraps=policy.construct) as construct:
                # This test covers solid reuse; direction-space rendering has
                # its own filesystem context and is outside this fixture.
                with patch('direction_space.render',side_effect=lambda *a,**k:Image.new('RGB',(20,20),'white').save(base/'step4/step4.1/direction_space.png')):
                    R.render('Test',group,directions={'pose_1':direction},computed=computed)
            self.assertEqual(construct.call_count,1)
            self.assertFalse(construct.call_args.kwargs['check_contacts'])
            out=base/'step4/step4.1';record=json.loads((out/'data/render.json').read_text())
            self.assertTrue(record['reuses_computed_final_construction'])
            self.assertTrue(record['clearance_diagnostics']['contact_check_performed'])
            self.assertEqual(record['clearance_diagnostics'],joint['diagnostics'])
            exported=S.solid(trimesh.load(out/'data/initialization_final_support.obj',force='mesh',process=False))
            self.assertLess(material_volume(exported-joint['remaining'])+material_volume(joint['remaining']-exported),1e-12)

if __name__=='__main__':unittest.main()
