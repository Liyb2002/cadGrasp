"""Preserve force/base tradeoffs across decisions under one demand measure."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys
import unittest
import numpy as np

COOPT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'step4.2')]
import _bootstrap
from whole_search.model import Layout
from layout_frontier import Sample, LayoutFrontier, bounded_proposals, select_samples
from oracle_search import OracleSearch


def layout(index):
    q = Layout(np.eye(4)[None], np.array([[0., 0., 1.]]), np.array([0]), (0,))
    q.placements[0, 0, 3] = float(index)
    return q


def oracle(enclosed=True, connected=False, gap=0., feet=None):
    return dict(enclosure_passed=enclosed, connected_passed=connected,
                passed=connected, gap_m=gap, work_only_enclosure_passed=True,
                status='connected' if connected else 'enclosed' if enclosed else 'not_found_in_window',
                per_pose=[dict(legal_ground_xy_m=feet or [[-1., -1.], [1., 1.]],
                               demand_hull_xy_m=[[0., 0.]])])


def result(q, passed=False, volume=1.):
    return dict(layout=q, masks={0:np.array([passed], bool)}, counts={'0':int(passed)},
                serial=int(q.placements[0, 0, 3]), volume_cm3=volume)


class FrontierSelectionTests(unittest.TestCase):
    def test_force_improvement_cannot_erase_the_base_feasible_alternative(self):
        parent = Sample(layout(0), 10., oracle())
        cuts_critical_foot = Sample(layout(1), 1., oracle(False, gap=.1))
        selected = select_samples([cuts_critical_foot, parent], 2, 1.)
        self.assertIn(parent, selected)
        self.assertIn(cuts_critical_foot, selected)

    def test_feasible_incumbent_survives_a_lower_loss_unbuildable_child(self):
        parent = Sample(layout(0), .01, oracle(connected=True), result=result(layout(0), True))
        child = Sample(layout(1), 0., oracle(False, gap=.1))
        self.assertIs(select_samples([child, parent], 1, 1.)[0], parent)

    def test_duplicate_layout_keeps_classification_and_connection_witness_status(self):
        q = layout(0)
        descriptor = Sample(q, 0., oracle())
        classified = Sample(q.copy(), 0., oracle(connected=True), result=result(q, True))
        self.assertEqual(len(select_samples([descriptor, classified], 4, 1.)), 1)
        self.assertIs(select_samples([descriptor, classified], 4, 1.)[0], classified)

    def test_equal_scores_retain_different_extreme_foot_positions(self):
        samples = []
        for i in range(4):
            q = layout(0); q.directions[0] = [i*.001, 0., np.sqrt(1.-(i*.001)**2)]
            feet = [[-1., -1.], [1., 1.]] if i < 3 else [[-5., -1.], [1., 1.]]
            samples.append(Sample(q, 1., oracle(feet=feet)))
        chosen = select_samples(samples, 2, 1.)
        self.assertIn(samples[-1], chosen)

    def test_features_keep_fixed_size_across_different_host_counts(self):
        q = Layout(np.repeat(np.eye(4)[None], 2, axis=0), np.array([[0.,0.,1.]]*2),
                   np.array([0,1]), (0,1))
        other = q.copy(); other.hosts[1] = 0
        selected = select_samples([Sample(q,1.,oracle()), Sample(other,1.,oracle())], 2, .1)
        self.assertEqual(len(selected), 2)

    def test_sampling_budget_does_not_starve_base_repair_or_paired_jumps(self):
        rows = [('direction-gradient', layout(i), dict(pose=i)) for i in range(20)]
        rows += [('oracle-cohort-sample', layout(100), {}), ('juxtapose-pair', layout(101), {})]
        selected = bounded_proposals(rows, 6, np.random.default_rng(42))
        self.assertEqual(len(selected), 6)
        self.assertIn('oracle-cohort-sample', [row[0] for row in selected])
        self.assertIn('juxtapose-pair', [row[0] for row in selected])
        again = bounded_proposals(rows, 6, np.random.default_rng(42))
        self.assertEqual([row[1].key() for row in selected], [row[1].key() for row in again])


class FrontierSearchTests(unittest.TestCase):
    def setup_search(self, connection_passed=True):
        # Lower force loss takes the dead path 0->1. The retained alternative
        # 0->2 preserves the key foot and reaches full feasibility via 2->3.
        layouts = [layout(i) for i in range(4)]
        losses = [10., 1., 5., 0.]
        events = []; evaluations = []; holds = []; epoch = [0]
        def index(q): return int(q.placements[0, 0, 3])
        def query(q, connected=False):
            i = index(q); events.append(('oracle', i, connected))
            answer = oracle(i != 1, connected and i == 3 and connection_passed, .1 if i == 1 else 0.)
            if answer['connected_passed']:
                base.witness = q.key(), 'selected-witness'
            return answer
        base = SimpleNamespace(query=query)
        @contextmanager
        def hold(current):
            epoch[0] += 1; holds.append(index(current['layout']))
            yield
        def proxy(q):
            events.append(('proxy', index(q), epoch[0]))
            return dict(loss=losses[index(q)])
        def evaluate(q):
            i = index(q); evaluations.append(i); events.append(('classify', i, epoch[0]))
            value = result(q, i == 3); value['base_oracle'] = query(q)
            return value
        model = SimpleNamespace(base_oracle=base, proxy=proxy, evaluate=evaluate,
                                hold_measure=hold, commit=Mock(), extent=1.)
        proposed_from = []
        def proposals(current, *args):
            i = index(current['layout']); proposed_from.append(i)
            children = [1, 2] if i == 0 else [3] if i == 2 else []
            return [('direction-sample', layouts[j], dict(pose=0)) for j in children], {}
        search = SimpleNamespace(model=model, frontier_width=2, finalists=3, iterations=2,
            screen_budget=12, rng=np.random.default_rng(42), record=Mock(), process_snapshot=Mock(),
            targets=Mock(return_value=([], {}, [])), local_proposals=proposals,
            jump_guests=Mock(return_value=([], [], {})), juxtapose_proposals=Mock(return_value=[]),
            oracle_layout_proposals=Mock(return_value=[]))
        return search, result(layouts[0]), events, evaluations, holds, proposed_from

    def test_retained_alternative_is_expanded_in_the_next_round_and_solves(self):
        search, initial, events, evaluations, holds, parents = self.setup_search()
        with patch('layout_frontier.installation_proposals', return_value=[]):
            final = LayoutFrontier(search).solve(initial)
        self.assertTrue(final['masks'][0].all())
        self.assertTrue(final['base_oracle']['passed'])
        self.assertIn(1, parents)
        self.assertIn(2, parents)
        self.assertEqual(evaluations, [1, 2, 3])
        self.assertEqual(search.frontier_statistics['maximum_retained'], 2)
        self.assertEqual(search.model.base_oracle.witness[1], 'selected-witness')
        # Oracle screens every classified child before its full force check.
        for i in evaluations:
            query_at = next(j for j, e in enumerate(events) if e[:2] == ('oracle', i))
            classify_at = next(j for j, e in enumerate(events) if e[:2] == ('classify', i))
            self.assertLess(query_at, classify_at)
        self.assertEqual(len(holds), 2)
        for record in search.record.call_args_list:
            self.assertLessEqual(record.args[0]['screened'], search.screen_budget)
        # Search transitions reused the saved parent mask rather than changing it.
        self.assertFalse(initial['masks'][0][0])

    def test_force_and_enclosure_without_connection_never_become_overall_pass(self):
        search, initial, _, _, _, _ = self.setup_search(connection_passed=False)
        with patch('layout_frontier.installation_proposals', return_value=[]):
            final = LayoutFrontier(search).solve(initial)
        self.assertTrue(final['masks'][0].all())
        self.assertFalse(final['base_oracle']['passed'])

    def test_selected_witness_is_restored_after_other_connection_queries(self):
        search, initial, *_ = self.setup_search()
        controller = LayoutFrontier(search)
        selected = controller.sample(initial)
        controller.witnesses[selected.layout.key()] = 'correct-witness'
        search.model.base_oracle.witness = ('other-layout', 'wrong-witness')
        controller.finish([selected], selected)
        self.assertEqual(search.model.base_oracle.witness, (selected.layout.key(), 'correct-witness'))

    def test_previous_failed_connections_cannot_starve_a_new_finalist(self):
        search, _, *_ = self.setup_search()
        controller = LayoutFrontier(search)
        samples = [Sample(layout(i),0.,oracle(),result=result(layout(i),True,volume=i+1.))
                   for i in range(4)]
        controller.connected_attempts.update(item.layout.key() for item in samples[:3])
        self.assertIs(controller.connected(samples), samples[-1])
        self.assertEqual(controller.stats['connection_candidates'], 1)


class FrontierPublicationTests(unittest.TestCase):
    def test_material_publication_reuses_accepted_space_after_cache_eviction(self):
        accepted=oracle(connected=True);search=self.search(oracle(False))
        current=result(layout(0),True);witness=Mock()
        search.material_selected=SimpleNamespace(layout=current['layout'],oracle=accepted,
            witness=witness,nominal=None,solid=None,completed=False)
        with patch('oracle_search.StableGradientSearch.save',return_value=dict(force_passed=True)):
            with patch('oracle_search.save'):
                report=search.save(current,'test',0.,{})
        search.model.base_oracle.query.assert_not_called()
        witness.export.assert_called_once_with(search.out/'oracle_connected_witness.obj')
        self.assertTrue(report['base_oracle_passed'])
        self.assertFalse(report['passed'])  # A space witness alone is not a completed solid.

    def search(self, answer):
        base = SimpleNamespace(query=Mock(return_value=answer), seconds=0., queries=0,
                               component_queries=0, foot_width=0., work_mode='normal')
        search = OracleSearch.__new__(OracleSearch)
        search.model = SimpleNamespace(base_oracle=base, support_floor_clearance_m=0.)
        search.out = Path('/unused/frontier-test'); search.frontier_width = 4
        return search

    def test_force_failed_result_never_runs_or_claims_complete_connection(self):
        answer = oracle(); answer['passed'] = True  # Fast enclosure witness only.
        search = self.search(answer); current = result(layout(0))
        with patch('oracle_search.StableGradientSearch.save', return_value=dict(force_passed=False)):
            with patch('oracle_search.save'):
                report = search.save(current, 'test', 0., {})
        search.model.base_oracle.query.assert_called_once_with(current['layout'], connected=False)
        self.assertFalse(report['passed'])
        self.assertFalse(report['base_oracle_passed'])

    def test_publication_exports_the_selected_layout_witness_not_a_later_query(self):
        search = self.search(oracle(connected=True)); current = result(layout(0), True)
        mesh = Mock(); search.frontier_witnesses = {current['layout'].key():mesh}
        search.model.base_oracle.witness = ('other-layout', Mock())
        with patch('oracle_search.StableGradientSearch.save', return_value=dict(force_passed=True)):
            with patch('oracle_search.save'):
                report = search.save(current, 'test', 0., {})
        self.assertTrue(report['passed'])
        mesh.export.assert_called_once_with(search.out/'oracle_connected_witness.obj')


if __name__ == '__main__':
    unittest.main()
