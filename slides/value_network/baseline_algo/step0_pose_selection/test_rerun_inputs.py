import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from step3_scheculer import contacts as I
from run_sequential_batch import rebind_step0_inputs


class RerunInputTests(unittest.TestCase):
    def fixture(self, root):
        group = root/'pose1+2'
        local = group/'step_1_needs/pose_1'
        local.mkdir(parents=True)
        inputs = {}
        for name in ('needs.json', 'samples.json'):
            (local/name).write_text(name)
            inputs[f'removed_cache/pose_1/{name}'] = I.sha256(local/name)
        report = dict(complete=True, load_input_folders=dict(pose_1='removed_cache/pose_1'),
                      provenance=dict(inputs=inputs, code={}), artifacts={})
        I.save(group/'step0_pose_selection/report.json', report)
        return group, local

    def test_missing_cache_rebinds_only_to_identical_saved_bytes(self):
        with TemporaryDirectory() as tmp, patch.object(I, 'ROOT', Path(tmp)):
            group, local = self.fixture(Path(tmp))
            before = {p.name:p.read_bytes() for p in local.iterdir()}
            report = rebind_step0_inputs(group)
            self.assertEqual(report['load_input_folders']['pose_1'], 'pose1+2/step_1_needs/pose_1')
            self.assertEqual(len(report['identical_input_reference_relocation']), 2)
            self.assertEqual(before, {p.name:p.read_bytes() for p in local.iterdir()})
            self.assertFalse((Path(tmp)/'removed_cache').exists())

    def test_mismatched_copy_rejected_without_partial_report_edit(self):
        with TemporaryDirectory() as tmp, patch.object(I, 'ROOT', Path(tmp)):
            group, local = self.fixture(Path(tmp))
            path = group/'step0_pose_selection/report.json'
            before = path.read_bytes()
            (local/'samples.json').write_text('different loads')
            with self.assertRaisesRegex(ValueError, 'byte-identical'):
                rebind_step0_inputs(group)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
