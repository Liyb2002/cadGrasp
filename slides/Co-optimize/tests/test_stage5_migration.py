"""A pruning result must never be published or evaluated as the final solid."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/'helper_func'),str(COOPT/'evaluation')]
import publish_output_images as P
import stage5_shared as S
spec=importlib.util.spec_from_file_location('migration_evaluation',COOPT/'evaluation/run.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)

def record(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))

class MigrationTests(unittest.TestCase):
    def test_step4_reference_verifies_its_exact_layout_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'current';folder.mkdir()
            (folder/'layout.npz').write_bytes(b'current layout')
            (folder/'support_with_base.obj').write_bytes(b'certified support')
            fingerprints={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                          for name in ['layout.npz','support_with_base.obj']}
            record(folder/'preconstructed_base.json',dict(complete=True,
                construction=dict(passed=True),support='support_with_base.obj',source_sha256=fingerprints))
            inputs=SimpleNamespace(protected={});data=dict(selected=folder)
            args=SimpleNamespace(reference_stage='step4')
            with patch.object(S,'ROOT',root):
                path,_=S.constructed_reference(inputs,data,args)
                self.assertEqual(path,folder/'support_with_base.obj')
                (folder/'layout.npz').write_bytes(b'another layout')
                with self.assertRaises(ValueError):S.constructed_reference(inputs,data,args)
    def test_legacy_reference_cannot_borrow_a_different_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            record(root/'case/step5/step5.2/old/data/report.json',dict(complete=True,
                passed=True,selected_step4_experiment='old-layout'))
            inputs=SimpleNamespace(base=root,protected={})
            data=dict(case=dict(id='case',experiment='current-layout'))
            args=SimpleNamespace(reference_stage='step5.2',reference_experiment='old')
            with self.assertRaises(ValueError):S.constructed_reference(inputs,data,args)
    def test_evaluation_rejects_pruning_without_final_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            record(root/'step5/step5.2/demo/data/report.json',dict(stage_type='support_pruning'))
            with self.assertRaises(ValueError):E.final_stage(root,'demo')
    def test_evaluation_prefers_new_final_stage_over_legacy_same_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            record(root/'step5/step5.2/demo/data/report.json',dict(stage_type='support_pruning'))
            record(root/'step5/step5.3/demo/data/report.json',dict(stage_type='final_support_construction'))
            self.assertEqual(E.final_stage(root,'demo'),root/'step5/step5.3/demo')
    def test_direct_publication_does_not_create_an_experiment_subfolder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);numeric=root/'numeric';visual=root/'visual'
            record(numeric/'case/step5/step5.3/demo/data/report.json',
                dict(complete=True,passed=True,poses=['pose_1'],published_images={}))
            folder=visual/'case/step5.3';folder.mkdir(parents=True)
            for name in ['overview.png','all_poses.png','pose_1.png']:(folder/name).write_bytes(b'unchanged image')
            with patch.object(P,'RESULT',numeric),patch.object(P,'VISUAL',visual),patch.object(P,'ROOT',root):
                result=P.publish_case_images('case','demo')
            self.assertEqual(len(result['images']),3)
            self.assertFalse(any(path.is_dir() for path in folder.iterdir()))
    def test_publisher_rejects_pruned_material_as_a_completed_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            record(root/'case/step5/step5.2/demo/data/report.json',
                dict(complete=True,passed=True,stage_type='support_pruning'))
            with patch.object(P,'RESULT',root):
                with self.assertRaises(ValueError):P.image_sources('case','demo',False)

if __name__=='__main__':unittest.main()
