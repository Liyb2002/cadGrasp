"""Direction-set boundaries, eligibility, and optimized-geometry cache safety."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step2_local_support import insertion as D
from step3_scheculer import directions as T
from step2_local_support import withdrawal as W


class DirectionTests(unittest.TestCase):
    def test_closed_touching_intervals_preserve_single_direction(self):
        result = D.intersect(D.normalize([[10, 90]]), D.normalize([[90, 150]]))
        self.assertEqual(result, D.normalize(isolated=[90]))
        self.assertTrue(D.nonempty(result))
        self.assertEqual(D.summary(result)['interval_width_deg'], 0)

    def test_wrap_endpoints_are_identical(self):
        result = D.intersect(D.normalize([[350, 360]]), D.normalize([[0, 20]]))
        self.assertEqual(result, D.normalize(isolated=[0]))
        self.assertTrue(D.contains(D.normalize([[350, 360]]), 0))

    def test_gaps_are_never_filled(self):
        value = D.normalize([[0, 30], [30+1e-11, 60]])
        self.assertEqual(len(value['intervals_deg']), 2)
        self.assertFalse(D.contains(value, 30+5e-12))
        self.assertFalse(D.nonempty(D.intersect(value, D.normalize([[80, 90]]))))

    def test_isolated_tangent_survives_intersection(self):
        local = D.engine.local_angles(np.array([[1., 0, 0], [-1., 0, 0]]))
        self.assertEqual(D.intersect(local, D.normalize([[0, 180]])), D.normalize(isolated=[90]))
        self.assertEqual(D.intersect(local, D.full()), local)

    def test_unknown_angles_are_not_eligible(self):
        analysis = dict(clear_intervals_deg=[[10, 20]], unresolved_intervals_deg=[[20, 30]],
                        isolated_direction_checks=[dict(angle_deg=40, clear=False), dict(angle_deg=90, clear=True)])
        certified = D.from_analysis(analysis)
        self.assertFalse(D.contains(certified, 25))
        self.assertFalse(D.contains(certified, 40))
        self.assertTrue(D.contains(certified, 90))

    def test_opposing_individual_directions_cannot_destroy_common_direction(self):
        rows = [dict(candidate_index=i, candidate_id=f'C{i}', certified_directions=W.normalize(ids))
                for i, ids in enumerate([[0,1], [2,3], [1,2], [1]])]
        problem=SimpleNamespace(data=SimpleNamespace(valid=np.array([True,True,True,False])))
        original=problem.data.valid
        mask, records=T.candidate_filter(rows,[0],original,common_allowed=W.normalize([0,1]))
        np.testing.assert_array_equal(mask,[False,False,True,False])
        self.assertEqual(records[1]['reason'],'would_lock_last_common_direction')
        self.assertEqual(records[2]['remaining_if_selected'],W.normalize([1]))
        with self.assertRaisesRegex(RuntimeError,'test'):
            with T.scoring_mask(problem,mask,[0]):
                np.testing.assert_array_equal(problem.data.valid,[True,False,True,False])
                raise RuntimeError('test')
        self.assertIs(problem.data.valid,original)

    def test_optimized_size_uses_actual_geometry_and_can_expand_or_empty_set(self):
        initial = dict(candidate_id='C000', candidate_index=0, radius_m=1., center_m=np.zeros(3),
                       center_face=0, triangles_m=np.zeros((1, 3, 3)), source_faces=np.array([0]),
                       triangle_areas_m2=np.ones(1))
        initial_row = dict(candidate_id='C000', candidate_index=0, radius_m=1.,
                           geometry_signature=D.signature(initial, .1),
                           certified_directions=W.normalize([0]), analysis=dict(length_m=2.))
        for radius, directions in [(0.5, W.normalize([0,1])), (2., W.normalize())]:
            actual = dict(initial, radius_m=radius)
            tracker = T.Tracker.__new__(T.Tracker)
            tracker.problem = SimpleNamespace(name='test')
            tracker.catalogue = dict(normal_depth_m=.1, candidates=[initial_row])
            tracker.catalogue_path = I.ROOT/'catalogue.json'
            tracker.direction_catalogue=dict(vectors=[[1.,0,0],[0,1.,0]], preferred_withdrawal_direction=[1.,0,0],global_allowed_directions=W.normalize([0,1]))
            tracker.current = dict(mode=W.MODE, contacts=[],common_directions=W.normalize([0,1]))
            tracker.current_path = None
            calls = []
            def analyze(contact):
                calls.append(contact['radius_m'])
                return dict(initial_row, geometry_signature=D.signature(contact, .1),
                            radius_m=contact['radius_m'], certified_directions=directions)
            tracker.analyzer = SimpleNamespace(analyze=analyze)
            with patch.object(I, 'read_contacts', return_value=[actual]), \
                 patch.object(I, 'hashes', return_value={}), \
                 patch.object(I, 'save'), \
                 patch.object(I, 'check_report', side_effect=FileNotFoundError):
                result, _ = tracker.update(I.ROOT/'round/contacts.npz', I.ROOT/'round/state.json', 1)
            self.assertEqual(calls, [radius])
            self.assertEqual(result['contacts'][0]['certified_directions'], directions)
            self.assertEqual(result['all_contacts_have_certified_direction'],W.nonempty(directions))
            self.assertTrue(result['contacts'][0]['differs_from_step2_geometry'])

    def test_initial_area_filter_uses_strict_threshold_per_contact(self):
        rows = [dict(candidate_index=i, candidate_id=f'C{i}', certified_directions=W.normalize([0]))
                for i in range(3)]
        eligible, records = T.candidate_filter(rows, [], np.ones(3, bool), [.49,.5,.51], .5)
        np.testing.assert_array_equal(eligible, [False,False,True])
        self.assertEqual(records[1]['reason'], 'initial_area_not_above_minimum')
        self.assertEqual(records[2]['reason'], 'eligible')

    def test_old_score_without_current_filter_cannot_resume(self):
        with patch.object(I, 'hashes', return_value={'filter.json': 'new'}):
            self.assertFalse(T.inputs_match(dict(provenance=dict(inputs={})), ['filter.json']))
            self.assertFalse(T.inputs_match(dict(provenance=dict(inputs={'filter.json': 'old'})), ['filter.json']))
            self.assertTrue(T.inputs_match(dict(provenance=dict(inputs={'filter.json': 'new'})), ['filter.json']))


if __name__ == '__main__':
    unittest.main()
