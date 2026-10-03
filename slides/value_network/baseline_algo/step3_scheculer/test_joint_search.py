"""Regressions for simultaneous, append-only, uncovered-weighted Step3."""
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from step3_scheculer.joint_scoring import coverage_summary, weighted_gain, top5_distribution
from step3_scheculer.joint_geometry import JointGeometry
from step3_scheculer.joint_tasks import canonical_poses, output_folder, prepare_tasks, task_set_folder
from step3_scheculer.run_joint import JointSearch, assignment_key, run_case, main
from step3_scheculer.sample_acceptance import SAMPLE_COUNT
from step3_scheculer import contacts as I
from step3_scheculer.joint_prepared import store_prepared, load_prepared
from step3_scheculer.joint_bounds import remaining_pool_certificate


class WeightedScoringTests(unittest.TestCase):
    def test_remaining_fraction_is_a_multiplier_not_a_denominator(self):
        score = weighted_gain([.8, .2], [.9, .3])
        np.testing.assert_allclose(score['value_by_pose'], [.02, .08])
        self.assertAlmostEqual(score['value'], .05)

    def test_weighted_value_controls_both_top_five_and_probabilities(self):
        base = [.8, .2]
        rows = [dict(index=0, id='easy', eligible=True, covered_fractions=[.9, .2]),
                dict(index=1, id='behind', eligible=True, covered_fractions=[.8, .3])]
        ids, probabilities = top5_distribution(rows, base)
        self.assertEqual(ids, [1, 0])
        np.testing.assert_allclose(probabilities, [.8, .2])
        for i in range(2, 8):
            rows.append(dict(index=i, id=f'candidate_{i}', eligible=True,
                             covered_fractions=[.8, .2+i/100]))
        self.assertNotIn(0, top5_distribution(rows, base)[0])

    def test_weights_refresh_and_pose_order_is_irrelevant(self):
        before, after = [.8, .2, .5, 1.], [.9, .3, .6, 1.]
        one = weighted_gain(before, after)
        two = weighted_gain(after, [1., .4, .7, 1.])
        self.assertLess(two['value'], one['value'])
        self.assertAlmostEqual(one['value'], weighted_gain(before[::-1], after[::-1])['value'])
        self.assertEqual(one['uncovered_weights'][-1], 0)

    def test_zero_gain_uniform_fallback_and_illegal_exclusion(self):
        rows = [dict(index=i, id=str(i), eligible=True, covered_fractions=[0.]*5) for i in range(7)]
        rows.append(dict(index=7, id='illegal', eligible=False, covered_fractions=[1.]*5))
        ids, probabilities = top5_distribution(rows, [0.]*5)
        self.assertEqual(ids, list(range(5)))
        np.testing.assert_allclose(probabilities, [.2]*5)
        self.assertEqual(top5_distribution([], [0.]*5)[0], [])

    def test_five_pose_success_requires_every_original_sample(self):
        masks = [np.ones(32768, bool) for _ in range(5)]
        masks[-1][-1] = False
        self.assertFalse(coverage_summary(masks)['all_sampled_complete'])
        masks[-1][-1] = True
        self.assertTrue(coverage_summary(masks)['all_sampled_complete'])

    def test_coverage_regression_and_invalid_numbers_rejected(self):
        for after in ([.4, 1.], [np.nan, 1.], [1.1, 1.]):
            with self.assertRaises(ValueError):
                weighted_gain([.5, .5], after)


def head(index, active):
    return dict(contact=dict(candidate_id=f'H{index}', candidate_index=index,
                             radius_m=.01+index*.001),
                active_tasks=tuple(active), valid=True, area=.01, reason='valid')


class JointGeometryTests(unittest.TestCase):
    def geometry(self):
        geometry = JointGeometry.__new__(JointGeometry)
        geometry.problems = [object() for _ in range(4)]
        geometry.local = [SimpleNamespace(catalogues=[dict(vectors=[0, 1])],
                                         paths=SimpleNamespace(labels=np.array([0])))]*4
        geometry.same_center = lambda candidate, entries: any(e['contact']['candidate_id'] == candidate['contact']['candidate_id'] for e in entries)
        def view(entry, task):
            return dict(entry, valid=task != 1, reason='work_surface_overlap' if task == 1 else 'valid',
                        directions=[[entry.get('direction', 0)]], path_components=[0])
        geometry.view = view
        return geometry

    def test_head_can_serve_three_poses_without_being_legal_in_all_four(self):
        geometry = self.geometry()
        candidate = head(0, [0])
        proposed, check = geometry.propose([], candidate)
        self.assertEqual(proposed['active_tasks'], (0, 2, 3))
        self.assertEqual(candidate['active_tasks'], (0,))  # frozen pool entry
        self.assertEqual(check['per_pose'][1]['reason'], 'work_surface_overlap')

    def test_common_direction_conflict_only_disables_affected_task(self):
        geometry = self.geometry()
        old = dict(head(0, [0]), direction=1)
        proposed, _ = geometry.propose([old], head(1, [0]))
        self.assertEqual(proposed['active_tasks'], (2, 3))
        self.assertEqual(old['active_tasks'], (0,))
        self.assertTrue(geometry.group_check([old, proposed])['passed'])

    def test_duplicate_rejected_and_empty_tasks_do_not_block_search(self):
        geometry = self.geometry()
        candidate = head(0, [0])
        self.assertIsNone(geometry.propose([candidate], candidate)[0])
        self.assertTrue(geometry.group_check([])['passed'])
        self.assertEqual(len(geometry.contacts_by_pose([])), 4)
        with self.assertRaises(RuntimeError):
            geometry.expand(candidate, 1.01)


class AppendOnlySearchTests(unittest.TestCase):
    def test_real_lp_complete_run_exports_and_replays_all_three_tasks(self):
        # Exercise the complete run/export/Step4 path, not only the inner loop.
        full = np.array([[0., 0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., -1.]])
        targets = np.tile([0., 0., 1., 0., 0., 0.], (SAMPLE_COUNT, 1))
        with TemporaryDirectory() as tmp:
            search = JointSearch.__new__(JointSearch)
            search.name, search.poses = 'test', ('pose_1', 'pose_2', 'pose_3')
            search.seed, search.particles, search.max_heads = 0, 1, None
            search.out = Path(tmp)/'step3'
            search.scores, search.recoveries, search.inputs = OrderedDict(), {}, {}
            search.problems = [SimpleNamespace(pose=pose, targets=targets.copy(),
                supply=lambda contacts: full, scale=np.ones(6), floor=np.zeros(3),
                domain=SimpleNamespace(com=np.zeros(3))) for pose in search.poses]
            search.original_targets = [p.targets.copy() for p in search.problems]
            search.geometry = SimpleNamespace(initial=[], count=1, save=Mock(),
                contacts_by_pose=lambda entries: [[], [], []], group_check=lambda entries: dict(passed=True))
            with patch('step3_scheculer.run_joint.output_folder', return_value=Path(tmp)/'step2'):
                report = search.run()
            self.assertTrue(report['result']['passed'])
            self.assertEqual(report['successful_particles'], 1)
            self.assertEqual(report['unique_successful_assignments'], 1)
            self.assertEqual(report['result']['sample_counts'], [SAMPLE_COUNT]*3)
            self.assertEqual(report['result']['heads'], 0)
            self.assertTrue(report['result']['exported_masks_reproduced'])
            search.geometry.save.assert_called_once()
            for pose in search.poses:
                self.assertEqual(I.read_contacts(search.out/f'final_contacts_{pose}.npz'), [])
            with patch('step3_scheculer.contacts.ROOT', Path(tmp)):
                search.floor(report, Path(tmp)/'step4')
            for pose in search.poses:
                with np.load(Path(tmp)/'step4'/f'floor_contact_{pose}.npz') as data:
                    np.testing.assert_array_equal(data['load_wrenches'], targets)
                    self.assertEqual(len(data['floor_demands_xy_m']), SAMPLE_COUNT)

    def test_real_lp_keeps_five_task_reactions_and_no_uplift_separate(self):
        search = JointSearch.__new__(JointSearch)
        search.scores = OrderedDict()
        search.geometry = SimpleNamespace(view=lambda e, k: e)
        # Without the named head each task's foot covers only its first demand.
        # Only task 4 receives the last head. Earlier tasks must stay unchanged.
        foot = np.array([[0., 0., 1., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., -1.]])
        head_ray = np.array([[0., 0., 1., -1., 0., 0., 1.]])
        targets = np.array([[0., 0., 1., 1., 0., 0.], [0., 0., 1., -.5, 0., 0.]])
        search.problems = [SimpleNamespace(targets=targets,
            supply=lambda contacts: np.vstack([foot, head_ray]) if contacts else foot)
            for _ in range(5)]
        first = search.evaluate([head(0, [0, 1, 2, 3])])
        self.assertEqual(first['covered_counts'], [2, 2, 2, 2, 1])
        self.assertFalse(first['all_sampled_complete'])
        completed = search.evaluate([head(0, [0, 1, 2, 3]), head(1, [4])])
        self.assertEqual(completed['covered_counts'], [2]*5)
        self.assertTrue(completed['all_sampled_complete'])
        # Flipping the lifted no-uplift coefficient must block that solution.
        head_ray[0, -1] = -1.
        search.scores.clear()
        blocked = search.evaluate([head(0, range(5))])
        self.assertEqual(blocked['covered_counts'], [1]*5)

    def test_final_assignment_deduplication_ignores_insertion_order(self):
        a = dict(active_ids_by_pose=[['H0', 'H1'], ['H1', 'H2']])
        b = dict(active_ids_by_pose=[['H1', 'H0'], ['H2', 'H1']])
        c = dict(active_ids_by_pose=[['H0', 'H1', 'H2'], ['H1']])
        self.assertEqual(assignment_key(a), assignment_key(b))
        self.assertNotEqual(assignment_key(a), assignment_key(c))

    def search(self, folder, requirements, candidates, budget=None):
        search = JointSearch.__new__(JointSearch)
        search.seed, search.max_heads, search.out = 0, budget, Path(folder)
        search.scores = OrderedDict()
        search.poses = tuple(f'pose_{i+1}' for i in range(len(requirements)))
        search.problems = [SimpleNamespace(
            targets=rules, supply=lambda group: frozenset(c['candidate_id'] for c in group))
            for rules in requirements]
        def propose(entries, candidate):
            if any(e['contact']['candidate_id'] == candidate['contact']['candidate_id'] for e in entries):
                return None, dict(reason='already_selected', per_pose=[])
            return candidate, dict(reason='eligible', per_pose=[])
        search.geometry = SimpleNamespace(initial=candidates, propose=propose,
            view=lambda e, k: e, group_check=lambda entries: dict(passed=True),
            contacts_by_pose=lambda entries: [[e['contact'] for e in entries if k in e['active_tasks']]
                                              for k in range(len(requirements))],
            expand=Mock(side_effect=AssertionError('No resize allowed')))
        return search

    def classify(self, active, requirements, known_covered=None):
        return np.array([set(required) <= active for required in requirements], bool), {}

    def test_completed_task_reuses_witnesses_without_resolving(self):
        with TemporaryDirectory() as tmp, patch('step3_scheculer.run_joint.J.classify', side_effect=self.classify) as classify:
            search = self.search(tmp, [[['H0']], [['H1']]], [head(0, [0]), head(1, [1])])
            base = search.evaluate([head(0, [0])])
            classify.reset_mock()
            actual = search.evaluate([head(0, [0]), head(2, [0]), head(1, [1])], base['masks'])
            self.assertTrue(actual['all_sampled_complete'])
            self.assertEqual(classify.call_count, 1)
            np.testing.assert_array_equal(classify.call_args.kwargs['known_covered'], [False])

    def test_five_poses_have_variable_head_counts_and_unrestricted_sharing(self):
        requirements = [[[], ['H0']], [[], ['H0', 'H1']], [[], ['H1']], [[], ['H2']], [[], ['H2']]]
        candidates = [head(0, [0, 1]), head(1, [1, 2]), head(2, [3, 4]), head(3, [0])]
        with TemporaryDirectory() as tmp, patch('step3_scheculer.run_joint.J.classify', side_effect=self.classify):
            search = self.search(tmp, requirements, candidates)
            entries, result = search.search_particle(0)
            self.assertTrue(result['passed'])
            self.assertEqual(result['heads'], 3)
            self.assertEqual([len(ids) for ids in result['active_ids_by_pose']], [1, 2, 1, 1, 1])
            previous = []
            for record in result['rounds']:
                self.assertEqual(record['contacts'][:-1], previous)
                previous = record['contacts']
                for contact in previous:
                    self.assertEqual(contact['radius_m'], candidates[int(contact['id'][1:])]['contact']['radius_m'])
            self.assertEqual(len(set(result['selected_ids'])), len(entries))
            search.geometry.expand.assert_not_called()

    def test_zero_immediate_gain_can_be_followed_by_complementary_completion(self):
        requirements = [[['H0', 'H1']], [['H0', 'H1']]]
        with TemporaryDirectory() as tmp, patch('step3_scheculer.run_joint.J.classify', side_effect=self.classify):
            search = self.search(tmp, requirements, [head(0, [0, 1]), head(1, [0, 1])])
            _, result = search.search_particle(0)
            self.assertTrue(result['passed'])
            self.assertEqual(result['rounds'][0]['value'], 0)
            self.assertEqual(result['heads'], 2)

    def test_ten_poses_can_select_more_than_five_heads_without_a_budget(self):
        requirements = [[[f'H{i}']] for i in range(10)]
        candidates = [head(i, [i]) for i in range(10)]
        with TemporaryDirectory() as tmp, patch('step3_scheculer.run_joint.J.classify', side_effect=self.classify):
            search = self.search(tmp, requirements, candidates)
            _, result = search.search_particle(0)
            self.assertTrue(result['passed'])
            self.assertEqual(result['heads'], 10)
            self.assertEqual(result['covered_counts'], [1]*10)

    def test_pool_exhaustion_terminates_even_when_every_gain_is_zero(self):
        with TemporaryDirectory() as tmp, patch('step3_scheculer.run_joint.J.classify', side_effect=self.classify):
            search = self.search(tmp, [[['missing']]]*4, [head(i, [i]) for i in range(4)])
            _, result = search.search_particle(0)
            self.assertFalse(result['passed'])
            self.assertEqual(result['status'], 'candidates_exhausted')
            self.assertEqual(result['heads'], 4)

    def test_optional_budget_is_a_stop_condition_not_a_pose_head_count(self):
        with TemporaryDirectory() as tmp, patch('step3_scheculer.run_joint.J.classify', side_effect=self.classify):
            search = self.search(tmp, [[['H0', 'H1']]]*3, [head(0, [0, 1, 2]), head(1, [0, 1, 2])], budget=1)
            _, result = search.search_particle(0)
            self.assertEqual(result['status'], 'head_budget_reached')
            self.assertFalse(result['passed'])

    def test_already_complete_floor_only_state_needs_no_heads(self):
        with TemporaryDirectory() as tmp, patch('step3_scheculer.run_joint.J.classify', side_effect=self.classify):
            search = self.search(tmp, [[[]]]*4, [head(0, [0])])
            _, result = search.search_particle(0)
            self.assertTrue(result['passed'])
            self.assertEqual(result['heads'], 0)

    def test_resume_retains_prefix_and_replays_the_same_random_sequence(self):
        candidates = [head(i, [i]) for i in range(4)]
        requirements = [[[f'H{i}']] for i in range(4)]
        with TemporaryDirectory() as tmp, patch('step3_scheculer.run_joint.J.classify', side_effect=self.classify):
            original = self.search(Path(tmp)/'full', requirements, candidates)
            _, full = original.search_particle(0)
            partial = self.search(Path(tmp)/'resume', requirements, candidates, budget=1)
            _, prefix = partial.search_particle(0)
            partial.resume, partial.max_heads = True, None
            _, completed = partial.search_particle(0)
            self.assertEqual(completed['selected_ids'], full['selected_ids'])
            self.assertEqual(completed['rounds'][0], prefix['rounds'][0])
            self.assertTrue(completed['passed'])
            partial.seed = 99
            with self.assertRaisesRegex(ValueError, 'seed'):
                partial.search_particle(0)


class RemainingPoolTests(unittest.TestCase):
    def search(self, target):
        rays = {'H0': [0., 0., 1., 1., 0., 0., 1.],
                'H1': [0., 0., 1., -1., 0., 0., 1.]}
        slack = [0., 0., 0., 0., 0., 0., -1.]
        problem = SimpleNamespace(pose='pose_1', targets=np.array([target]),
            supply=lambda contacts: np.array([slack]+[rays[c['candidate_id']] for c in contacts]))
        return SimpleNamespace(problems=[problem], geometry=SimpleNamespace(view=lambda e, k: e))

    def test_impossible_pool_requires_an_exact_original_sample_separator(self):
        with TemporaryDirectory() as tmp:
            search = self.search([0., 0., 1., -1., 0., 0.])
            proof = remaining_pool_certificate(search, [], [head(0, [0])],
                dict(masks=[np.array([False])]), Path(tmp), 1)
            self.assertTrue(proof['original_sample'])
            self.assertTrue(proof['exact_separator']['every_original_generator_checked_exactly'])
            self.assertFalse(proof['extra_loads_added'])
            self.assertEqual(proof['arrays_sha256'], I.sha256(Path(tmp)/proof['arrays']))

    def test_zero_gain_complementary_heads_are_not_pruned(self):
        with TemporaryDirectory() as tmp:
            search = self.search([0., 0., 1., 0., 0., 0.])
            proof = remaining_pool_certificate(search, [], [head(0, [0]), head(1, [0])],
                dict(masks=[np.array([False])]), Path(tmp), 1)
            self.assertIsNone(proof)

    def test_uncertified_numerical_failure_cannot_prune(self):
        with TemporaryDirectory() as tmp, \
                patch('step3_scheculer.joint_bounds.C.W.solve', return_value=None), \
                patch('step3_scheculer.joint_bounds.C.W.exact_separator', return_value=None):
            proof = remaining_pool_certificate(self.search([0., 0., 1., 0., 0., 0.]), [],
                [head(0, [0])], dict(masks=[np.array([False])]), Path(tmp), 1)
            self.assertIsNone(proof)


class PreparedGeometryTests(unittest.TestCase):
    def fixture(self, folder):
        problems, rows = [], []
        for index in range(2):
            pose = f'pose_{index+1}'
            matrix = np.eye(4)
            matrix[0, 3] = index
            problems.append(SimpleNamespace(pose=pose, inputs=[], domain=SimpleNamespace(
                mesh=SimpleNamespace(extents=np.ones(3)), data=dict(frame=dict(T_world_mesh=matrix.tolist())))))
            contact = dict(candidate_id=f'{pose}_C001', candidate_index=index, center_face=0,
                center_m=np.array([index, 0., .2]), radius_m=.1,
                triangles_m=np.array([[[index, 0., .2], [index+.1, 0., .2], [index, .1, .2]]]),
                source_faces=np.array([0]), triangle_areas_m2=np.array([.005]))
            contact['center_m'][0] += .2*index
            contact['triangles_m'][:, :, 0] += .2*index
            I.save_contacts(folder/f'candidates_{pose}.npz', [contact])
            I.save(folder/f'candidates_{pose}.json', dict(count=1, candidates=[dict(
                id=contact['candidate_id'], valid=True, reason='valid', area_m2=.005, radius_m=.1)],
                direction_catalogue=dict(vectors=[[1., 0., 0.], [0., 1., 0.]]), path_roadmap={}))
            checks = [dict(passed=True, reason='passed', common_direction_ids=[0, 1] if index == 0 else [1],
                           common_path_components=[3, 5] if index == 0 else [3]) for _ in range(2)]
            rows.append(dict(index=index, id=contact['candidate_id'], eligible=True,
                             active_tasks=[0, 1], per_pose_geometry=checks))
        store_prepared(folder, problems, 1, dict(base=dict(covered_counts=[0, 0]), rows=rows))
        return problems

    def test_prepared_views_preserve_rigid_surfaces_and_group_intersections(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            problems = self.fixture(folder)
            geometry = load_prepared(folder, problems, 1)
            first, _ = geometry.propose([], geometry.initial[0])
            second, _ = geometry.propose([first], geometry.initial[1])
            for check in geometry.group_check([first, second])['per_pose']:
                self.assertEqual(check['common_direction_ids'], [1])
                self.assertEqual(check['common_path_components'], [3])
            np.testing.assert_allclose(geometry.view(first, 1)['contact']['triangles_m'],
                first['contact']['triangles_m']+[1., 0., 0.])

    def test_changed_candidates_inputs_or_geometry_code_reject_cache(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            problems = self.fixture(folder)
            self.assertIsNone(load_prepared(folder, problems, 2))
            self.assertIsNone(load_prepared(folder, problems[::-1], 1))
            with patch('step3_scheculer.joint_prepared.input_hashes', return_value={'changed': 'input'}):
                self.assertIsNone(load_prepared(folder, problems, 1))
            with patch('step3_scheculer.joint_prepared.geometry_sources', return_value=[]):
                self.assertIsNone(load_prepared(folder, problems, 1))
            path = folder/'candidates_pose_1.json'
            path.write_text(path.read_text()+'\n')
            self.assertIsNone(load_prepared(folder, problems, 1))

    def test_mechanics_change_reuses_geometry_but_not_old_scores(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            problems = self.fixture(folder)
            with patch('step3_scheculer.joint_prepared.scoring_sources', return_value=[]):
                geometry = load_prepared(folder, problems, 1)
                self.assertIsNotNone(geometry)
                self.assertFalse(geometry.first_scores_valid)


class JointInputsTests(unittest.TestCase):
    def test_resume_rejects_a_changed_run_identity_before_writing_candidates(self):
        with TemporaryDirectory() as tmp:
            search = JointSearch.__new__(JointSearch)
            search.resume, search.out, search.geometry_folder = True, Path(tmp), Path(tmp)/'step2'
            search.name, search.poses, search.seed, search.particles = 'test', ('pose_1', 'pose_2'), 0, 1
            search.max_heads, search.inputs = None, {}
            search.geometry = SimpleNamespace(count=1, save=Mock())
            (search.out/'run_identity.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Cannot resume'):
                search.run()
            search.geometry.save.assert_not_called()

    def test_cli_defaults_to_all_registered_poses_without_a_head_limit(self):
        poses = tuple(f'pose_{i}' for i in range(1, 11))
        with patch('step3_scheculer.run_joint.task_poses', return_value=poses), \
                patch('step3_scheculer.run_joint.run_case') as run, \
                patch('step3_scheculer.run_joint.sample_pairs') as pairs:
            for flags in ([], ['--all-poses']):
                run.reset_mock()
                main(['B', *flags])
                run.assert_called_once()
                args, selected = run.call_args.args
                self.assertEqual(selected, poses)
                self.assertIsNone(args.max_heads)
                self.assertEqual(args.particles, 10)
                self.assertEqual(args.candidates, 200)
            pairs.assert_not_called()

    def test_cli_explicit_pose_subset_is_one_joint_run(self):
        with patch('step3_scheculer.run_joint.task_poses', return_value=('pose_1', 'pose_3', 'pose_6')), \
                patch('step3_scheculer.run_joint.run_case') as run:
            main(['B', '--poses', '6', '1', '3'])
            self.assertEqual(run.call_args.args[1], ('pose_1', 'pose_3', 'pose_6'))

    def test_cli_only_samples_pairs_when_requested(self):
        cases = [('pose_1', 'pose_3'), ('pose_2', 'pose_8')]
        with patch('step3_scheculer.run_joint.sample_pairs', return_value=(cases, 20260926)) as pairs, \
                patch('step3_scheculer.run_joint.run_case') as run:
            main(['B', '--pairs', '2'])
            pairs.assert_called_once_with('B', 2, 20260926)
            self.assertEqual([call.args[1] for call in run.call_args_list], cases)

    def test_failed_rerun_does_not_leave_old_complete_schedule_or_audit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            def folder(name, poses, stage):
                return root/stage/'joint_weighted'
            step3, step4 = [folder('B', ['1', '6'], s) for s in ('step3_scheculer', 'step0_pose_selection')]
            for path in (step3, step4):
                path.mkdir(parents=True)
            (step3/'schedule.json').write_text(json.dumps(dict(complete=True, result=dict(passed=True))))
            (step4/'joint_check.json').write_text(json.dumps(dict(passed=True)))
            (step4/'joint_result.json').write_text(json.dumps(dict(complete=True, successful_particles=1)))
            args = SimpleNamespace(object='B', seed=0, particles=1, candidates=2, max_heads=None)
            with patch('step3_scheculer.run_joint.output_folder', side_effect=folder), \
                    patch('step3_scheculer.run_joint.JointSearch', side_effect=RuntimeError('input changed')):
                with self.assertRaisesRegex(RuntimeError, 'input changed'):
                    run_case(args, ('pose_1', 'pose_6'))
            self.assertFalse(json.loads((step3/'schedule.json').read_text())['complete'])
            self.assertIsNone(json.loads((step4/'joint_check.json').read_text())['passed'])
            self.assertFalse(json.loads((step4/'joint_result.json').read_text())['complete'])

    def test_paths_are_order_independent_and_separate_from_old_experiments(self):
        self.assertEqual(canonical_poses(['10', 'pose_2', '4']), ('pose_2', 'pose_4', 'pose_10'))
        with self.assertRaises(ValueError):
            canonical_poses(['1', 'pose_1'])
        a = output_folder('B', ['1', '3', '6', '9'], 'step3_scheculer')
        b = output_folder('B', ['9', '6', '3', '1'], 'step3_scheculer')
        self.assertEqual(a, b)
        self.assertEqual(a.parts[-3:], ('pose1+3+6+9', 'step3_scheculer', 'joint_weighted'))

    def test_four_task_inputs_reuse_saved_samples_byte_for_byte(self):
        with TemporaryDirectory() as tmp, \
                patch('step3_scheculer.joint_tasks.OUTPUTS', Path(tmp)), \
                patch('step3_scheculer.pair_tasks.OUTPUTS', Path(tmp)), \
                patch('step3_scheculer.joint_tasks.read_task', side_effect=lambda name, pose, folder: folder):
            for a, b in [('1', '3'), ('6', '9')]:
                for pose in (f'pose_{a}', f'pose_{b}'):
                    folder = task_set_folder('B', [a, b], 'step_1_needs')/pose
                    folder.mkdir(parents=True)
                    for filename in ('samples.json', 'needs.json'):
                        (folder/filename).write_bytes((pose+' '+filename+'\n').encode())
            folders = prepare_tasks('B', ['9', '1', '6', '3'])
            self.assertEqual(len(folders), 4)
            for folder in folders:
                self.assertEqual((folder/'samples.json').read_bytes(), (folder.name+' samples.json\n').encode())
            self.assertFalse((Path(tmp)/'B/pose_1').exists())


if __name__ == '__main__':
    unittest.main()
