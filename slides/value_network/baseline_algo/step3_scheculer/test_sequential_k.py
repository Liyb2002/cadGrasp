import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from step3_scheculer.run_sequential_k import SequentialKSearch, sample_groups
from step3_scheculer.sequential_geometry import active_entries
from step3_scheculer.sequential_withdrawal import WholeHeadWithdrawal


def entry(task, number, quality=0):
    return dict(contact=dict(candidate_id=f'pose_{task+1}_C{number:03d}', radius_m=.01),
                owner_task=task, active_tasks=(task,), valid=True, reason='valid', area=.01, quality=quality)


def toy(path, required):
    search = SequentialKSearch.__new__(SequentialKSearch)
    search.seed, search.out = 20260928, Path(path)
    search.poses = tuple(f'pose_{k+1}' for k in range(len(required)))
    search.problems = [None]*len(required)
    pools = [[entry(k, n) for n in range(8)] for k in range(len(required))]
    search.geometry = SimpleNamespace(pools=pools,
        task_check=lambda entries,k:dict(passed=True, reason='valid', common_direction_ids=[0,1]),
        same_center=lambda e,old:any(e['contact']['candidate_id']==a['contact']['candidate_id'] for a in old),
        group_check=lambda entries:dict(passed=True, per_pose=[dict(common_direction_ids=[0,1]) for _ in required]),
        contacts_by_pose=lambda entries:[[e['contact'] for e in active_entries(entries,k)] for k in range(len(required))])
    search.task_score = lambda entries,k,known=None:np.arange(required[k]) < len(active_entries(entries,k))
    def record(state):
        return dict(passed=all(state), per_pose=[dict(common_direction_ids=list(ids)) for ids in state])
    search.withdrawal = SimpleNamespace(initial=lambda:tuple((0,1) for _ in required),
        restrict=WholeHeadWithdrawal.restrict, record=record,
        append=lambda state,entry:(state,dict(passed=True, reason='passed')),
        verify=lambda entries,state:record(state))
    search.direction_state = search.withdrawal.initial()
    return search


class SequentialKTests(unittest.TestCase):
    def test_six_fixed_random_combinations_two_of_each_size(self):
        available = [f'pose_{i}' for i in range(1,11)]
        groups = sample_groups(available, 20260928)
        self.assertEqual([len(g) for g in groups], [2,2,3,3,4,4])
        self.assertEqual(len(set(groups)),6)
        self.assertEqual(groups,sample_groups(available[::-1],20260928))

    def test_three_head_success_stops_without_a_fourth(self):
        with TemporaryDirectory() as tmp:
            search=toy(tmp,[3,3])
            entries,result=search.search_particle(0)
        self.assertTrue(result['passed'])
        self.assertEqual([len(x) for x in result['active_ids_by_pose']],[3,3])
        self.assertEqual(len(entries),5)

    def test_four_pose_fallback_and_exactly_one_prior_head_per_stage(self):
        with TemporaryDirectory() as tmp:
            search=toy(tmp,[4,3,4,3])
            entries,result=search.search_particle(0)
        self.assertTrue(result['passed'])
        self.assertEqual([len(x) for x in result['active_ids_by_pose']],[4,3,4,3])
        previous=set()
        for k,ids in enumerate(result['active_ids_by_pose']):
            self.assertEqual(len(set(ids)&previous), int(k>0))
            previous.update(ids)
        self.assertEqual(len(entries),11)
        self.assertTrue(all(e['contact']['radius_m']==.01 for e in entries))

    def test_best_shared_considers_first_pose_heads_even_after_second(self):
        search=toy('/tmp/not_written',[3,3,3])
        entries=[entry(0,0,5),entry(1,0,2)]
        search.task_score=lambda entries,k,known=None:np.arange(6)<max([0]+[e['quality'] for e in entries])
        selected,decision=search.choose_shared_for(entries,2)
        self.assertEqual(decision['selected_id'],entries[0]['contact']['candidate_id'])
        self.assertEqual(entries[0]['active_tasks'],(0,))
        self.assertEqual(selected[0]['active_tasks'],(0,2))
        self.assertEqual(selected[1]['active_tasks'],(1,))

    def test_four_heads_incomplete_does_not_advance_to_next_pose(self):
        with TemporaryDirectory() as tmp:
            entries,result=toy(tmp,[5,3,3]).search_particle(0)
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['stages']),1)
        self.assertEqual([len(x) for x in result['active_ids_by_pose']],[4,0,0])

    def test_all_prior_heads_ineligible_is_explicit_failure(self):
        search=toy('/tmp/not_written',[3,3])
        search.geometry.task_check=lambda entries,k:dict(passed=False,reason='blocked')
        original=[entry(0,0)]
        selected,decision=search.choose_shared_for(original,1)
        self.assertIsNone(decision['selected_id'])
        self.assertEqual(selected,original)

    def test_new_head_cannot_destroy_an_inactive_pose_exit(self):
        with TemporaryDirectory() as tmp:
            search = toy(tmp, [3,3])
            # Every first-pose candidate blocks the second pose before it is solved.
            search.withdrawal.append = lambda state,e:(None, dict(passed=False,
                reason='no_common_whole_head_withdrawal', blocked_pose='pose_2'))
            entries, result = search.search_particle(0)
        self.assertFalse(result['passed'])
        self.assertEqual(entries, [])
        self.assertEqual(result['status'], 'no_eligible_head_with_inherited_withdrawal')

    def test_shared_head_must_inherit_previous_direction_set(self):
        search = toy('/tmp/not_written', [3,3])
        search.direction_state = ((0,), (0,))
        search.geometry.task_check = lambda entries,k:dict(passed=True, reason='valid', common_direction_ids=[1])
        selected, decision = search.choose_shared_for([entry(0,0)], 1)
        self.assertIsNone(decision['selected_id'])
        self.assertEqual(search.direction_state, ((0,), (0,)))
        self.assertEqual(selected[0]['active_tasks'], (0,))

    def test_inherited_sets_persist_across_pose_stages_and_reset_between_chains(self):
        with TemporaryDirectory() as tmp:
            search = toy(tmp, [3,3])
            seen = []
            def append(state, candidate):
                seen.append((candidate['owner_task'], state))
                return tuple((0,) for _ in state), dict(passed=True, reason='passed')
            search.withdrawal.append = append
            _, result = search.search_particle(0)
            self.assertTrue(result['passed'])
            self.assertTrue(all(state == ((0,), (0,)) for task,state in seen if task == 1))
            seen.clear()
            search.search_particle(1)
            self.assertEqual(seen[0][1], ((0,1), (0,1)))
