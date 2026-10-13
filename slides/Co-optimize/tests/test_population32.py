"""Fixed population size, native rotations and the two mutation contracts."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys, tempfile, unittest
import numpy as np

COOPT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'step4.2')]
import _bootstrap
from whole_search.model import Layout
from fixture_state import installed_state
from material_evaluator import MaterialSample, membership, target_gap
from population32 import (registered_population, centered_seat, discrete_move,
                          differential_move, choose_branch, Population32Search)


def initial(n=4):
    return Layout(np.repeat(np.eye(4)[None], n, axis=0),
                  np.tile([0., 0., 1.], (n, 1)), np.arange(n), tuple(range(n)))


class SmallModel:
    def __init__(self, n=4):
        self.native = np.repeat(np.eye(4)[None], n, axis=0)
        self.native[:, 0, 3] = np.arange(n) * .2
        self.native[:, 2, 3] = .1
        self.mesh = SimpleNamespace(center_mass=np.array([.01, .02, .03]),
                                    vertices=np.array([[-.02, -.02, -.02], [.02, .02, .04]]))
        self.extent = .1
        self.allow_z_translation = True
        self.commit = Mock()
        self.hold_measure = lambda _: nullcontext()
        self.proxy = lambda _: dict(loss=1., residual_loss=[1.]*n)

    def floor_normal(self, q, k):
        return self.native[q.hosts[k], :3, :3].T @ np.array([0., 0., 1.])

    def workpiece_height(self, q, k):
        U = installed_state(self.native, q, q.hosts[k]) @ q.placements[k]
        return float((self.mesh.vertices @ U[:3, :3].T + U[:3, 3])[:, 2].min())

    def validate_placement(self, q, k):
        U = installed_state(self.native, q, q.hosts[k]) @ q.placements[k]
        np.testing.assert_allclose(U[:3, :3], self.native[k, :3, :3], atol=1e-10)
        if self.workpiece_height(q, k) < -1e-9:
            raise ValueError('floor crossing')

    def restore_registered(self, q, owner):
        from oracle_search import OracleModel
        return OracleModel.restore_registered(self, q, owner)


class PopulationContracts(unittest.TestCase):
    def test_initialization_has_32_registered_slots_without_predefined_roles(self):
        m=SmallModel(); q=initial()
        rows, common=registered_population(m, q, np.random.default_rng(42))
        self.assertTrue(common); self.assertEqual(len(rows), 32)
        for row in rows:
            self.assertEqual(membership(row), (4, 0))
            np.testing.assert_array_equal(row.directions, q.directions)
        rows[0].placements[0, 0, 3] = 1.
        self.assertEqual(rows[1].placements[0, 0, 3], 0.)
        self.assertEqual(q.placements[0, 0, 3], 0.)

    def test_nearby_exit_variations_never_change_registered_seats(self):
        m=SmallModel(); q=initial()
        q.directions[1]=[.1, 0., np.sqrt(.99)]
        rows, common=registered_population(m, q, np.random.default_rng(42))
        self.assertFalse(common); self.assertEqual(len(rows), 32)
        for row in rows:
            self.assertEqual(membership(row), (4, 0))
            for k in row.active:
                self.assertGreaterEqual(row.directions[k] @ m.floor_normal(row, k), 0.)

    def test_centered_birth_preserves_native_rotation_and_legal_world_height(self):
        m=SmallModel(); q=initial()
        m.native[1, :3, :3]=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        q.state_offsets[0]=[.3, -.2, .15]
        old=installed_state(m.native, q, 1) @ q.placements[1]
        moved=centered_seat(m, q, 1, 0)
        a=installed_state(m.native, moved, moved.hosts[0]) @ moved.placements[0]
        b=installed_state(m.native, moved, moved.hosts[1]) @ moved.placements[1]
        ca=a[:3, :3] @ m.mesh.center_mass + a[:3, 3]
        cb=b[:3, :3] @ m.mesh.center_mass + b[:3, 3]
        np.testing.assert_allclose(ca[:2], cb[:2])
        self.assertAlmostEqual(cb[2], (old[:3, :3] @ m.mesh.center_mass + old[:3, 3])[2])
        np.testing.assert_allclose(b[:3, :3], m.native[1, :3, :3])
        self.assertEqual(membership(moved), (3, 1))
        self.assertGreater(target_gap(moved), 0)  # Intermediate 3+1 is allowed.
        np.testing.assert_array_equal(moved.placements[2:], q.placements[2:])
        np.testing.assert_array_equal(q.hosts, np.arange(4))

    def test_all_registered_slots_can_create_the_first_juxtapose(self):
        m=SmallModel(); q=initial()
        rng=SimpleNamespace(choice=Mock(side_effect=[1, 0]), random=lambda: .8)
        with patch('reuse_proposals.port_matching_offsets', side_effect=AssertionError('port matching forbidden')):
            moved, detail=discrete_move(m, q, rng)
        self.assertEqual(detail['action'], 'create')
        self.assertFalse(detail['contact_matching_run'])
        self.assertEqual(membership(moved), (3, 1))

    def test_differential_reads_other_slots_without_mutating_donors_or_roles(self):
        m=SmallModel(); q=centered_seat(m, initial(), 1, 0)
        population=[q.copy() for _ in range(32)]
        for i,row in enumerate(population): row.placements[1,0,3] += i*.001
        keys=[row.key() for row in population]
        moved, detail=differential_move(m, q, population, 7, np.random.default_rng(42), 0)
        self.assertNotIn(7, detail['donor_slots'])
        self.assertEqual(len(set(detail['donor_slots'])), 2)
        self.assertEqual(membership(moved), membership(q))
        np.testing.assert_array_equal(moved.hosts, q.hosts)
        for k in [0,2,3]: np.testing.assert_array_equal(moved.placements[k], q.placements[k])
        self.assertEqual([row.key() for row in population], keys)

    def test_branch_is_forced_discrete_until_a_movable_guest_exists(self):
        q=initial()
        self.assertEqual(choose_branch(q, SimpleNamespace(random=lambda: .9)), 'discrete')
        q.hosts[1]=0
        self.assertEqual(choose_branch(q, SimpleNamespace(random=lambda: .1)), 'discrete')
        self.assertEqual(choose_branch(q, SimpleNamespace(random=lambda: .9)), 'differential')

    def test_two_rounds_keep_32_slots_and_exactly_one_move_and_adjustment_per_slot(self):
        m=SmallModel(); q=initial()
        def evidence(t):
            return dict(layout=t, masks={k:np.ones(1,bool) for k in t.active}, volume_cm3=1.)
        def preview(t, operation='parent', parent='', detail=None, result=None):
            return MaterialSample(t, 1., dict(enclosure_passed=False, passed=False, gap_m=1.),
                                  1., operation, parent, detail or {}, result=result)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'case/step4/step4.2/experiment'; out.mkdir(parents=True)
            search=SimpleNamespace(model=m, frontier_width=32, out=out, iterations=2,
                                   rng=np.random.default_rng(42), record=Mock(), process_snapshot=Mock())
            controller=Population32Search(search)
            controller.preview=preview
            controller.refine_preview=Mock(side_effect=lambda raw,iteration,maximum: raw)
            def check(*args, **kwargs):
                for item in controller.slots:
                    if item.result is None: item.result=evidence(item.layout)
                    controller.evaluated[item.layout.key()]=item
            controller.checked_slots=check
            with patch('population32.freeze_population_measure'):
                controller.solve(evidence(q))
            self.assertEqual(len(controller.slots), 32)
            self.assertEqual(controller.refine_preview.call_count, 64)
            self.assertEqual(sum(controller.branch_counts.values()), 64)
            for row in controller.audit:
                self.assertEqual(row['population_size'], 32)
                self.assertEqual(row['generated_trials'], 32)
                self.assertEqual(len(row['events']), 32)
                self.assertTrue(all(event['adjustment_passes']==1 for event in row['events']))


if __name__=='__main__':
    unittest.main()
