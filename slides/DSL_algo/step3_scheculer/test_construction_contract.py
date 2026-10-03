import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from step3_scheculer import construction_contract as C, contacts as I

class ContractTests(unittest.TestCase):
    def test_local_contact_checks_cannot_accept_step3(self):
        self.assertFalse(C.accept_witness(dict(complete=True,passed=True,force_passed=True,exit_passed=True)))
    def test_complete_witness_must_preserve_full_cores_and_serialization(self):
        row=dict(complete=True,passed=True,constructed=True,export_roundtrip_bit_exact=True,
            minimum_branch_thickness=dict(all_complete_cores_preserved=True),selected_exit_paths=[dict(id=1)])
        self.assertTrue(C.accept_witness(row))
        for key in ['complete','passed','constructed','export_roundtrip_bit_exact']:
            self.assertFalse(C.accept_witness(dict(row,**{key:False})))
        self.assertFalse(C.accept_witness(dict(row,minimum_branch_thickness={})))
    def test_constructor_failure_rejects_step3(self):
        with tempfile.TemporaryDirectory(dir=I.ROOT,prefix='.dsl_contract_test_') as folder:
            group=Path(folder)
            def failed(*args,**kwargs):
                I.save(kwargs['output']/'report.json',dict(complete=True,passed=False,constructed=False,error='no route'))
            proposal=dict(complete=True,local_contact_passed=True,passed=True,provenance=dict(inputs={},code={}))
            with patch('step4_connect_support.try_dsl_growth.construct_witness',side_effect=failed):
                result=C.certify(group,proposal)
            self.assertFalse(result['passed']);self.assertFalse(result['complete_fixture_verified'])
            self.assertFalse(I.check_report(group/'step3_scheculer/dsl_shared/contact_proposal.json')['passed'])
            I.check_report(group/'step3_scheculer/dsl_shared/report.json')
    def test_contact_failure_skips_construction(self):
        with tempfile.TemporaryDirectory(dir=I.ROOT,prefix='.dsl_contract_test_') as folder:
            group=Path(folder)
            with patch('step4_connect_support.try_dsl_growth.construct_witness') as build:
                result=C.certify(group,dict(complete=True,local_contact_passed=False))
            build.assert_not_called();self.assertFalse(result['passed'])
    def test_step4_rejects_a_local_proposal_without_witness(self):
        with tempfile.TemporaryDirectory(dir=I.ROOT,prefix='.dsl_contract_test_') as folder:
            group=Path(folder)
            I.save(group/'step3_scheculer/dsl_shared/report.json',dict(complete=True,passed=False,
                provenance=dict(inputs={},code={})))
            result=C.publish(group)
            self.assertFalse(result['passed'])
            self.assertEqual(result['status'],'not_applicable_step3_rejected')
            self.assertFalse((group/'step4/data/dsl_shared_support/shape.obj').exists())
    def test_changed_witness_is_rejected_before_step4(self):
        with tempfile.TemporaryDirectory(dir=I.ROOT,prefix='.dsl_contract_test_') as folder:
            group=Path(folder);witness=group/'step3_scheculer/dsl_shared/construction_witness'
            witness.mkdir(parents=True)
            shape=witness/'shape.obj';shape.write_text('original witness')
            hashes=I.hashes([shape])
            I.save(group/'step3_scheculer/dsl_shared/report.json',dict(complete=True,passed=True,
                complete_fixture_verified=True,construction_witness_artifacts=hashes,
                provenance=dict(inputs={},code={})))
            shape.write_text('changed geometry')
            with self.assertRaisesRegex(RuntimeError,'stale dependency'):
                C.replay(group)
    def test_step4_materializes_same_mesh_without_construction_search(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=I.ROOT,prefix='.dsl_contract_test_') as folder:
            group=Path(folder);witness=group/'step3_scheculer/dsl_shared/construction_witness'
            witness.mkdir(parents=True)
            (witness/'shape.obj').write_text('immutable verified geometry')
            (witness/'geometry_certificate.npz').write_bytes(b'verified certificate')
            schedule=dict(complete=True,passed=True,complete_fixture_verified=True,
                construction_witness=str((witness/'report.json').relative_to(I.ROOT)),
                provenance=dict(inputs={},code={}))
            I.save(group/'step3_scheculer/dsl_shared/report.json',schedule)
            body=dict(complete=True,passed=True,constructed=True,volume_cm3=1.,space_budget={},
                artifacts={p:I.sha256(witness/p) for p in ['shape.obj','geometry_certificate.npz']})
            I.save(witness/'report.json',body)
            grow=SimpleNamespace(case=SimpleNamespace(output=None),bases=None,offsets=None)
            with patch.object(C,'replay',return_value=(schedule,body,grow,None,{})), \
                 patch('step4_connect_support.try_dsl_growth.construct_witness',side_effect=AssertionError('Unexpected new construction')), \
                 patch('step4_connect_support.boxed_support.draw'):
                result=C.publish(group)
            self.assertTrue(result['passed'])
            self.assertEqual(I.sha256(witness/'shape.obj'),I.sha256(group/'step4/data/dsl_shared_support/shape.obj'))
            I.check_report(group/'step4/data/dsl_shared_support/report.json')
if __name__=='__main__':unittest.main()
