"""Partial construction is allowed; incomplete coverage must never become success."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import completion as Q
from step4_floor_contact import floor_contact as F
from step5_connect_support import connect as C
from step3_scheculer import run_all
class CompletionTests(unittest.TestCase):
    def test_rest_failure_or_four_heads_cannot_be_complete(self):
        for change in [dict(contact_count=4),dict(round_limit=None),dict(rest_equilibrium_verified=False)]:
            with self.assertRaises(Q.IncompleteSchedule):
                Q.require_passed(dict(self.complete(),**change))

    def test_omitting_access_does_not_omit_force_or_insertion_requirements(self):
        result=dict(self.complete(),process_access_enforced=False,all_contact_heads_clear_of_work_volume=None)
        self.assertEqual(Q.require_passed(result),result)
        for change in [dict(covered_count=99),dict(contact_heads_individually_insertable=False)]:
            with self.assertRaises(Q.IncompleteSchedule):Q.require_passed(dict(result,**change))

    def complete(self):
        return dict(object='test', complete=True, status='continuous_contact_model_verified',
                    contact_count=2,round_limit=3,rest_equilibrium_verified=True,
                    sample_count=100, covered_count=100, continuous_domain_status='verified',
                    passive_support_no_uplift_verified=True,
                    passive_support_constraint=dict(mode='connected_massless_support_no_uplift',enforced=True),
                    continuous_coverage_proved=True, insertion_mode='common_rigid_withdrawal_3d',common_head_withdrawal_verified=True,common_withdrawal_directions={'ids':[0]},contact_heads_individually_insertable=True,
                    all_contact_areas_above_minimum=True,all_contact_heads_clear_of_work_volume=True)

    def test_only_full_verified_result_is_admitted(self):
        self.assertEqual(Q.require_passed(self.complete()),self.complete())
        failures=[dict(passive_support_no_uplift_verified=False),dict(passive_support_constraint={}),dict(common_withdrawal_directions={'ids':[]}),dict(common_head_withdrawal_verified=False),dict(status='round_limit_reached'),dict(status='candidates_exhausted'),
                  dict(status='sampled_complete'),dict(covered_count=99),dict(sample_count=0,covered_count=0),
                  dict(continuous_coverage_proved=False),dict(continuous_domain_status='not_verified'),
                  dict(all_contact_areas_above_minimum=False),dict(all_contact_heads_clear_of_work_volume=False),
                  dict(contact_heads_individually_insertable=False),dict(insertion_mode='one_rigid_support'),dict(complete=False)]
        for changed in failures:
            with self.subTest(changed=changed),self.assertRaises(Q.IncompleteSchedule):
                Q.require_passed(dict(self.complete(),**changed))

    def test_new_search_cannot_reuse_previous_success(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(Q.I,'OUTPUTS',Path(directory)):
            folder=Path(directory)/'test'/'pose_1'/'step3_scheculer';folder.mkdir(parents=True)
            (folder/'schedule.json').write_text(json.dumps(self.complete()))
            for status in [dict(complete=False,status='running'),dict(complete=True,schedule_sha256='stale')]:
                (folder/'status.json').write_text(json.dumps(status))
                with self.assertRaises(Q.IncompleteSchedule):Q.read_passed('test')

    def test_step5_rejects_a_running_or_stale_search(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(C,'OUTPUTS',Path(directory)), \
                patch.object(C.I,'check_report',return_value=self.complete()):
            root=Path(directory)/'test'/'pose_1'/'step3_scheculer';root.mkdir(parents=True)
            (root/'status.json').write_text(json.dumps(dict(complete=False)))
            with self.assertRaisesRegex(RuntimeError,'completed current search'):
                C.read_inputs('test')

    def test_partial_objects_continue_through_step5_and_return_unsuccessful(self):
        for first in (4,5):
            with self.subTest(first=first),patch.object(run_all,'floor_verified',return_value=False), \
                    patch.object(run_all,'connection_verified',side_effect=lambda n:n=='passed'), \
                    patch.object(run_all.subprocess,'run') as invoke:
                invoke.return_value.returncode=0
                self.assertEqual(run_all.run(['failed','passed'],from_step=first),2)
                for call in invoke.call_args_list:
                    self.assertIn('failed',call.args[0]);self.assertIn('passed',call.args[0])
                self.assertTrue(any('/step5_connect_support/' in c.args[0][1] for c in invoke.call_args_list))

    def test_unsuccessful_completed_search_retains_partial_construction_diagnostics(self):
        with patch.object(run_all,'read_passed',side_effect=Q.IncompleteSchedule('exhausted')), \
                patch.object(run_all,'floor_verified',return_value=False), \
                patch.object(run_all,'connection_verified',return_value=False), \
                patch.object(run_all.subprocess,'run') as invoke:
            invoke.side_effect=lambda command,**kwargs: type('Result',(),{
                'returncode':2 if command[1].endswith('/scheduler.py') else 0})()
            self.assertEqual(run_all.run(['failed'],from_step=3),2)
            stages=[call.args[0][1] for call in invoke.call_args_list]
            self.assertTrue(any(path.endswith('/verification.py') for path in stages))
            self.assertTrue(any('/step4_floor_contact/' in path for path in stages))
            self.assertTrue(any('/step5_connect_support/' in path for path in stages))

    def test_default_run_includes_fixed_foot_connector_and_audit(self):
        with patch.object(run_all,'floor_verified',return_value=True), \
                patch.object(run_all,'connection_verified',return_value=True), \
                patch.object(run_all.subprocess,'run') as invoke:
            invoke.return_value.returncode=0
            self.assertEqual(run_all.run(['passed'],from_step=4),0)
            stages=[call.args[0][1] for call in invoke.call_args_list]
            self.assertTrue(any(path.endswith('/step5_connect_support/audit.py') for path in stages))
            self.assertTrue(any(path.endswith('/step5_connect_support/draw.py') for path in stages))

    def test_connection_budget_only_changes_connector_search(self):
        with patch.object(run_all,'floor_verified',return_value=True), \
                patch.object(run_all,'connection_verified',return_value=False), \
                patch.object(run_all.subprocess,'run') as invoke:
            invoke.return_value.returncode=0
            self.assertEqual(run_all.run(['test'],from_step=4,connection_edge_budget=200),2)
            commands=[c.args[0] for c in invoke.call_args_list]
            bounded=[c for c in commands if '--edge-budget' in c]
            self.assertEqual(len(bounded),1)
            self.assertTrue(bounded[0][1].endswith('/step5_connect_support/connect.py'))
            self.assertEqual(bounded[0][-2:],['--edge-budget','200'])
            self.assertTrue(any(c[1].endswith('/step5_connect_support/audit.py') for c in commands))

    def test_invalid_connection_budget_cannot_start_or_clear_outputs(self):
        with patch.object(run_all.subprocess,'run') as invoke:
            with self.assertRaisesRegex(ValueError,'budget must be positive'):
                run_all.run(['test'],from_step=4,connection_edge_budget=0)
            invoke.assert_not_called()

    def test_one_body_search_keeps_audit_and_forwards_budget(self):
        with patch.object(run_all,'connection_verified',return_value=False), \
                patch.object(run_all.subprocess,'run') as invoke:
            invoke.return_value.returncode=0
            self.assertEqual(run_all.run(['test'],from_step=5,connection_edge_budget=200,connection_workers=4),2)
            commands=[c.args[0] for c in invoke.call_args_list]
            self.assertTrue(commands[0][1].endswith('/step5_connect_support/connect.py'))
            self.assertEqual(commands[0][-2:],['--edge-budget','200'])
            self.assertNotIn('--workers',commands[0])
            self.assertTrue(commands[1][1].endswith('/step5_connect_support/audit.py'))
            self.assertTrue(commands[2][1].endswith('/step5_connect_support/draw.py'))

    def test_invalid_connection_workers_cannot_start_outputs(self):
        with patch.object(run_all.subprocess,'run') as invoke:
            with self.assertRaisesRegex(ValueError,'worker count must be positive'):
                run_all.run(['test'],from_step=5,connection_workers=0)
            invoke.assert_not_called()

    def test_force_directions_recomputes_even_with_resume(self):
        with patch.object(run_all.subprocess, 'run') as invoke:
            invoke.return_value.returncode = 0
            self.assertEqual(run_all.run(['test'], from_step=2, through_step=2,
                                         resume=True, force_directions=True), 0)
            commands = [c.args[0] for c in invoke.call_args_list]
            forced = [c for c in commands if '--force' in c]
            self.assertEqual(len(forced), 1)
            self.assertTrue(forced[0][1].endswith('/insertion_directions.py'))

    def test_resume_reuses_completed_failed_search_but_replays_its_audit(self):
        checkpoint=dict(status='candidates_exhausted',continuous_coverage_proved=False)
        with patch.object(run_all,'completed_schedule',return_value=checkpoint), \
                patch.object(run_all.subprocess,'run') as invoke:
            invoke.return_value.returncode=0
            self.assertEqual(run_all.run(['test'],from_step=3,through_step=3,resume=True),2)
            paths=[call.args[0][1] for call in invoke.call_args_list]
            self.assertEqual(len(paths),1)
            self.assertTrue(paths[0].endswith('/step3_scheculer/audit.py'))

    def test_rerun_invalidates_downstream_in_place_without_archiving(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(Q.I,'OUTPUTS',Path(directory)):
            root=Path(directory)/'test'/'pose_1'
            scheduler=root/'step3_scheculer';scheduler.mkdir(parents=True)
            (scheduler/'schedule.json').write_text(json.dumps(dict(self.complete(),
                status='round_limit_reached',continuous_coverage_proved=False)))
            old=root/'step5_connect_support';old.mkdir()
            (old/'support.stl').write_text('historical geometry')
            (old/'load_check.json').write_text('historical counterexample')
            self.assertIsNone(Q.archive_downstream('test'))
            self.assertTrue(old.exists())
            self.assertEqual((old/'load_check.json').read_text(),'historical counterexample')
            self.assertEqual((old/'support.stl').read_text(),'historical geometry')
            self.assertFalse(json.loads((old/'status.json').read_text())['complete'])
            self.assertFalse((root/'history').exists())

    def test_resume_rechecks_completed_upstream_after_waiting_for_case_lock(self):
        from contextlib import ExitStack
        from step2_local_support import circles,insertion_directions,audit,sampling_audit
        for valid in (True,False):
            with self.subTest(valid=valid),ExitStack() as mocks:
                mocks.enter_context(patch.object(circles,'read',side_effect=None if valid else RuntimeError('stale')))
                mocks.enter_context(patch.object(insertion_directions,'read'))
                replay=mocks.enter_context(patch.object(audit,'run'))
                mocks.enter_context(patch.object(sampling_audit,'run'))
                mocks.enter_context(patch.object(run_all,'completed_schedule',return_value=None))
                invoke=mocks.enter_context(patch.object(run_all.subprocess,'run'))
                invoke.return_value.returncode=0
                self.assertEqual(run_all.run(['test'],from_step=1,through_step=3,resume=True),0)
                paths=[call.args[0][1] for call in invoke.call_args_list]
                self.assertEqual(any(path.endswith('/step1/needs.py') for path in paths),not valid)
                self.assertEqual(replay.call_count,int(valid))
                self.assertTrue(any(path.endswith('/step3_scheculer/scheduler.py') for path in paths))


if __name__=='__main__':unittest.main()
