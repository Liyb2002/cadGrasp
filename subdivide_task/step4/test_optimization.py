"""A proposal or partial force screen must never become an accepted state."""
from copy import deepcopy
import unittest
from .optimize import legal_improvement


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        proof=dict(passed=True,original_sample_count=32768,passed_original_sample_count=32768,
                   continuous_outer_target_count=200,passed_continuous_outer_target_count=200)
        self.candidate=dict(volume_cm3=90.,tasks=[dict(id='W01')],audit=dict(passed=True,full_loads=True,
            source_face_prisms_checked=True,continuous_paths=[dict(id='W01',passed=True)],
            tasks=[dict(id='W01',force=proof,floor=dict(passed=True,actual_contact_area_m2=.01,
                all_pressure_centers_on_actual_bottom=True))]))

    def test_reduced_screen_does_not_accept(self):
        self.candidate['audit']['full_loads']=False
        self.assertFalse(legal_improvement(100.,self.candidate))

    def test_missing_continuous_exit_certificate_does_not_accept(self):
        self.candidate['audit']['source_face_prisms_checked']=False
        self.assertFalse(legal_improvement(100.,self.candidate))

    def test_one_failed_or_preflight_only_task_rejects_whole_operation(self):
        for proof in (dict(passed=False),dict(passed=True,preflight_only=True)):
            candidate=deepcopy(self.candidate)
            candidate['tasks'].append(dict(id='W02'))
            candidate['audit']['tasks'].append(dict(id='W02',force=proof))
            self.assertFalse(legal_improvement(100.,candidate))

    def test_empty_missing_or_duplicated_tasks_do_not_accept(self):
        for mode in ('empty','missing','duplicate'):
            candidate=deepcopy(self.candidate)
            if mode=='empty':
                candidate['audit']['tasks']=[]
            elif mode=='missing':
                candidate['tasks'].append(dict(id='W02'))
            else:
                candidate['tasks'].append(dict(id='W02'))
                candidate['audit']['tasks'].append(deepcopy(candidate['audit']['tasks'][0]))
            self.assertFalse(legal_improvement(100.,candidate))

    def test_incomplete_original_or_continuous_load_count_rejects(self):
        for key in ('passed_original_sample_count','passed_continuous_outer_target_count'):
            candidate=deepcopy(self.candidate)
            candidate['audit']['tasks'][0]['force'][key]-=1
            self.assertFalse(legal_improvement(100.,candidate))

    def test_legal_but_larger_or_equal_is_rejected(self):
        for volume in (90.,89.):
            self.assertFalse(legal_improvement(volume,self.candidate))

    def test_flags_without_per_task_exit_or_actual_floor_proof_do_not_accept(self):
        for part in ('exit','floor'):
            candidate=deepcopy(self.candidate)
            if part=='exit':candidate['audit']['continuous_paths']=[]
            else:candidate['audit']['tasks'][0].pop('floor')
            self.assertFalse(legal_improvement(100.,candidate))

    def test_only_fully_certified_smaller_state_accepts(self):
        self.assertTrue(legal_improvement(100.,self.candidate))


if __name__=='__main__':
    unittest.main()
