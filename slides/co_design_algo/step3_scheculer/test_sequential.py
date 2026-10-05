"""The 3+2 experiment must never borrow the two inactive heads for scoring."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import trimesh

from step3_scheculer.run_sequential import SequentialSearch, top5, best_shared, output_folder
from step3_scheculer.sequential_geometry import SequentialGeometry, active_entries, local_group
from step3_scheculer import terminal_expansion as T
from step2_local_support import surface as S


def entry(i, active, factor=1.):
    return dict(contact=dict(candidate_id=f'H{i}', candidate_index=i, radius_m=np.sqrt(factor)),
                active_tasks=tuple(active), valid=True, reason='valid', area=.01*factor,
                area_fraction=.01*factor, factor=factor)


class SequentialTests(unittest.TestCase):
    def test_top5_uses_single_task_gain_and_excludes_illegal_candidates(self):
        rows = [dict(index=i, id=f'C{i}', eligible=i != 6, covered_count=100+i) for i in range(7)]
        ids, probabilities = top5(rows, 100)
        self.assertEqual(ids, [5, 4, 3, 2, 1])
        np.testing.assert_allclose(probabilities, np.array([5, 4, 3, 2, 1])/15)

    def test_zero_gain_is_uniform_and_empty_pool_stops(self):
        ids, probs = top5([dict(index=i, id=str(i), eligible=True, covered_count=0) for i in range(3)], 0)
        np.testing.assert_allclose(probs, [1/3]*3)
        self.assertEqual(top5([], 0), ([], []))

    def test_shared_head_is_maximum_pose2_coverage_among_legal_heads(self):
        rows = [dict(id='A', eligible=True, covered_count=10),
                dict(id='B', eligible=False, covered_count=100),
                dict(id='C', eligible=True, covered_count=20)]
        self.assertEqual(best_shared(rows)['id'], 'C')
        self.assertIsNone(best_shared([rows[1]]))

    def test_each_task_receives_only_its_three_active_contacts(self):
        search = SequentialSearch.__new__(SequentialSearch)
        search.scores = {}
        entries = [entry(0, [0]), entry(1, [0]), entry(2, [0, 1]), entry(3, [1]), entry(4, [1])]
        search.geometry = SimpleNamespace(view=lambda e, k: e)
        supplies = [Mock(return_value=np.array([[k]])) for k in range(2)]
        search.problems = [SimpleNamespace(supply=s, targets=np.zeros((100, 6))) for s in supplies]
        with patch('step3_scheculer.run_sequential.J.classify', return_value=(np.ones(100, bool), {})):
            score = search.evaluate(entries)
        self.assertTrue(score['both_sampled_complete'])
        self.assertEqual([[c['candidate_id'] for c in s.call_args.args[0]] for s in supplies],
                         [['H0', 'H1', 'H2'], ['H2', 'H3', 'H4']])
        self.assertEqual(len(search.scores), 2)

    def test_inactive_head_does_not_restrict_active_direction_intersection(self):
        geometry = SimpleNamespace(catalogues=[dict(vectors=[0, 1])], paths=SimpleNamespace(labels=np.array([0])))
        heads = [dict(entry(0, [0]), directions=[[0]], path_components=[0]),
                 dict(entry(1, [1]), directions=[[1]], path_components=[0])]
        self.assertTrue(local_group(geometry, active_entries(heads, 0))['passed'])
        self.assertFalse(local_group(geometry, heads)['passed'])

    def test_shared_growth_updates_one_surface_used_by_both_tasks(self):
        geometry = SequentialGeometry.__new__(SequentialGeometry)
        native = dict(contact=dict(candidate_index=0, candidate_id='C001', radius_m=1.),
                      valid=True, reason='valid', area=.01, area_fraction=.01)
        geometry.count = 200
        geometry.problems = [SimpleNamespace(pose='pose_1'), SimpleNamespace(pose='pose_6')]
        larger = dict(native, contact=dict(native['contact'], radius_m=1.01), area=.0101, area_fraction=.0101)
        geometry.local = [SimpleNamespace(expand=Mock(return_value=larger))]
        geometry.view = Mock(return_value=dict(valid=True, reason='valid'))
        shared = geometry.wrap(native, 0, (0, 1))
        grown = geometry.expand(shared, 1.01)
        self.assertEqual(grown['active_tasks'], (0, 1))
        self.assertEqual(grown['contact']['candidate_id'], shared['contact']['candidate_id'])
        self.assertEqual(grown['native']['contact']['radius_m'], 1.01)
        geometry.view.assert_called_once_with(grown, 1)

    def test_shared_sweep_uses_owner_solid_not_other_task_head_thickness(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 1], [1, 0, 1], [0, 1, 1]], faces=[[0, 1, 2]], process=False)
        triangles = S.fan(mesh.triangles[0])
        contact = dict(candidate_id='A', candidate_index=0, radius_m=1., center_m=np.array([.3, .3, 1.]),
                       center_face=0, triangles_m=triangles, source_faces=np.zeros(3, int),
                       triangle_areas_m2=S.areas(triangles))
        owner = SimpleNamespace(mesh=mesh, depth=.1, clearance=SimpleNamespace(offsets=np.tile([0, 0, .1], (3, 1))))
        tester = Mock(return_value=dict(clear=True, reason='clear'))
        other = SimpleNamespace(mesh=mesh, scale=1., depth=.9,
            clearance=SimpleNamespace(offsets=np.tile([0, 0, .9], (3, 1))),
            catalogues=[dict(vectors=[[1, 0, 0]])], analyzers=[SimpleNamespace(test=tester)],
            paths=SimpleNamespace(ports=lambda point: [0]))
        geometry = SequentialGeometry.__new__(SequentialGeometry)
        geometry.local, geometry.cache = [owner, other], {}
        geometry.problems = [None, SimpleNamespace(domain=SimpleNamespace(work_ids=[]))]
        geometry.transforms = [[np.eye(4), np.eye(4)], [np.eye(4), np.eye(4)]]
        e = dict(contact=contact, owner_task=0, area=.5, native=dict(valid=True), active_tasks=(0, 1))
        view = geometry.view(e, 1)
        self.assertTrue(view['valid'])
        cells = tester.call_args.args[0]
        self.assertAlmostEqual(max(h.vertices[:, 2].max() for h in cells), 1.1)
        self.assertTrue(view['direction_records'][0]['same_owner_solid'])

    def test_pose2_adds_exactly_two_and_keeps_pose1_heads(self):
        search = SequentialSearch.__new__(SequentialSearch)
        search.poses = ('pose_1', 'pose_6')
        initial = [entry(0, [0]), entry(1, [0]), entry(2, [0, 1])]
        extra = [entry(3, [1]), entry(4, [1])]
        search.geometry = SimpleNamespace(pools=[[], extra],
            same_center=lambda c, es: c in es, task_check=lambda es, k: dict(passed=True, reason='passed'))
        search.task_score = lambda es, k: dict(mask=np.arange(100) < 30*len(active_entries(es, k)),
            covered_count=30*len(active_entries(es, k)), sample_count=100)
        with tempfile.TemporaryDirectory() as folder:
            result, rounds = search.extend(initial, 1, 2, np.random.default_rng(5), Path(folder))
        self.assertEqual(len(result), 5)
        self.assertEqual(result[:3], initial)
        self.assertEqual([len(active_entries(result, k)) for k in range(2)], [3, 3])
        self.assertEqual(len(rounds), 2)

    def test_pose_order_and_expansion_modes_have_separate_outputs(self):
        a = output_folder('B', ['pose_1', 'pose_6'], 'step3_scheculer')
        b = output_folder('B', ['pose_6', 'pose_1'], 'step3_scheculer')
        c = output_folder('B', ['pose_1', 'pose_6'], 'step3_scheculer', False)
        self.assertEqual(len({a, b, c}), 3)


if __name__ == '__main__':
    unittest.main()
