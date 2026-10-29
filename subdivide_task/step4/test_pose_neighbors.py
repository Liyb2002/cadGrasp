"""Neighbour ranking, genuine support turns and independent proof histories."""
import json
import tempfile
import unittest
import queue
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from subdivide_task.operations.state import Layout
from .pose_neighbors import closest_axial_angle
from .group_acceptance import random_pair, merged_groups
from .population_search import (clone_checkpoint, make_proposal, bounded_worker,
    planned_rounds,next_iteration,completed_rounds)
from subdivide_task.step1.common import sha256


class NeighbourTests(unittest.TestCase):
    def test_analytic_spin_minimizes_full_so3_distance(self):
        rng=np.random.default_rng(43)
        for _ in range(12):
            a,b=np.eye(4),np.eye(4)
            a[:3,:3]=Rotation.random(random_state=rng).as_matrix()
            b[:3,:3]=Rotation.random(random_state=rng).as_matrix()
            axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
            angle,distance=closest_axial_angle(a,b,axis)
            values=Rotation.from_rotvec(np.linspace(-np.pi,np.pi,1001)[:,None]*axis).as_matrix()
            brute=Rotation.from_matrix(values@a[:3,:3]@b[:3,:3].T).magnitude().min()
            self.assertLessEqual(distance,brute+1e-12)
            direct=Rotation.from_matrix(Rotation.from_rotvec(np.radians(angle)*axis).as_matrix()@a[:3,:3]@b[:3,:3].T).magnitude()
            self.assertAlmostEqual(distance,direct)

    def model(self):
        return SimpleNamespace(source=SimpleNamespace(center_mass=np.zeros(3),extents=np.ones(3),faces=np.zeros((3,3))),
            ids=['A','B','C'],records=[dict(working_face_ids=[i]) for i in range(3)],
            axis=np.array([0.,0.,1.]),head=np.zeros(3))

    def layout(self):
        q=Layout(np.tile(np.eye(4),(3,1,1)),np.tile(np.eye(4),(3,1,1)),np.tile([0.,0.,1.],(3,1)))
        q.material[2,0,3]=1.
        return q

    def test_existing_family_is_not_split_by_distance_or_seat_coincidence(self):
        model,q=self.model(),self.layout()
        groups=[[0,1],[2]]
        q.material[1,0,3]=100.
        for seed in range(20):
            pair=random_pair(model,q,groups,np.random.default_rng(seed))
            self.assertEqual(set(pair['owner_group']+pair['host_group']),{0,1,2})
            self.assertNotEqual(pair['owner_group'],pair['host_group'])
        self.assertEqual(merged_groups(groups,0,2),[[0,1,2]])

    def test_random_pair_does_not_depend_on_current_positions(self):
        model,q=self.model(),self.layout()
        groups=[[0],[1],[2]]
        pair=random_pair(model,q,groups,np.random.default_rng(19))
        q.material[:,:3,3]=np.random.default_rng(88).normal(size=(3,3))*1000
        self.assertEqual(pair,random_pair(model,q,groups,np.random.default_rng(19)))

    def test_equal_world_poses_still_propose_an_actual_fixture_turn(self):
        model,q=self.model(),self.layout()
        pair=dict(owner=2,host=0,axial_deg=0.)
        branch=SimpleNamespace(layout=q,groups=[[0],[1],[2]])
        with patch('subdivide_task.step4.population_search.floor_repair',side_effect=lambda m,q,g,a:q):
            moved,detail=make_proposal(model,branch,pair,'rotational_reuse',0,np.random.default_rng(1))
        relative=moved.material[2,:3,:3]@q.material[2,:3,:3].T
        self.assertGreater(Rotation.from_matrix(relative).magnitude(),np.radians(15.))
        np.testing.assert_allclose(moved.material[2]@moved.poses[2],moved.material[0]@moved.poses[0],atol=1e-12)

    def test_branch_checkpoint_writes_cannot_change_parent_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            source,target=Path(temp)/'parent',Path(temp)/'branch';source.mkdir()
            manifest=dict(accepted=0,state='state_000.npz',volume_cm3=10.)
            (source/'current.json').write_text(json.dumps(manifest))
            (source/'trace.json').write_text('[]')
            (source/'state_000.npz').write_bytes(b'immutable mesh')
            (source/'state_000.json').write_text('{}')
            clone_checkpoint(source,target)
            (target/'current.json').write_text(json.dumps(dict(manifest,volume_cm3=9.)))
            self.assertEqual(json.loads((source/'current.json').read_text())['volume_cm3'],10.)

    def test_async_round_plan_stays_balanced_and_reproducible_after_resume(self):
        rng=np.random.default_rng(71)
        first=planned_rounds({},rng,0,2,32)
        saved=dict(round_assignments={str(k):v for k,v in first.items()},rng_state=rng.bit_generator.state)
        restored=np.random.default_rng(3);restored.bit_generator.state=saved['rng_state']
        extended=planned_rounds(saved,restored,0,5,32)
        reference=planned_rounds({},np.random.default_rng(71),0,5,32)
        self.assertEqual(extended,reference)
        for values in extended.values():
            self.assertEqual(values.count('juxtaposition'),16)
            self.assertEqual(values.count('rotational_reuse'),16)

    def test_old_incomplete_round_resumes_without_redrawing_its_assignments(self):
        rng=np.random.default_rng(8)
        first=planned_rounds({},rng,0,1,32)[0]
        saved=dict(completed_rounds=0,assignments=first)
        plan=planned_rounds(saved,rng,0,3,32)
        self.assertEqual(plan,planned_rounds({},np.random.default_rng(8),0,3,32))

    def test_a_fast_sample_can_advance_without_waiting_for_other_samples(self):
        done={(1,0),(2,0)}
        self.assertEqual(next_iteration(done,0,4),2)
        self.assertEqual(next_iteration(done,1,4),0)
        self.assertEqual(completed_rounds(done,2),0)
        done.add((1,1))
        self.assertEqual(completed_rounds(done,2),1)

    def test_host_offset_is_independent_of_step3_separation(self):
        model,q=self.model(),self.layout()
        with patch('subdivide_task.step4.population_search.floor_repair',side_effect=lambda m,q,g,a:q):
            moved,detail=make_proposal(model,SimpleNamespace(layout=q,groups=[[0],[1],[2]]),dict(owner=2,host=0),
                'juxtaposition',1,np.random.default_rng(1))
        self.assertAlmostEqual(moved.material[2,0,3],.15)
        self.assertEqual(detail['host_local_offset_m'],[.15,0.,0.])

    def test_placing_a_family_preserves_its_internal_rigid_transform(self):
        model,q=self.model(),self.layout()
        with patch('subdivide_task.step4.population_search.floor_repair',side_effect=lambda m,q,g,a:q):
            moved,detail=make_proposal(model,SimpleNamespace(layout=q,groups=[[0,1],[2]]),dict(owner=0,host=2),
                'juxtaposition',1,np.random.default_rng(1))
        self.assertEqual(detail['moving_tasks'],['A','B'])
        np.testing.assert_allclose(moved.material[0],moved.material[1],atol=1e-12)
        np.testing.assert_allclose(moved.material[:2,0,3],[1.15,1.15])

    def test_slow_trial_retains_its_certified_parent_and_releases_worker_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            original=json.dumps(dict(volume_cm3=100.))
            (folder/'current.json').write_text(original)
            (folder/'trace.json').write_text('[]')
            slots=queue.Queue();slots.put(0)
            config=dict(folder=temp,iteration=0,branch=0,kind='juxtaposition',task_ids=['A'])
            with patch('subdivide_task.step4.population_search.subprocess.run',
                    side_effect=subprocess.TimeoutExpired('trial',1)):
                result=bounded_worker(config,slots)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['volume_cm3'],100.)
            self.assertEqual((folder/'current.json').read_text(),original)
            self.assertEqual(slots.get_nowait(),0)

    def test_timeout_recovery_cannot_promote_a_preflight_only_state(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'current.json').write_text(json.dumps(dict(volume_cm3=100.)))
            (folder/'trace.json').write_text('[]')
            def interrupted(*args,**kwargs):
                state=folder/'state_001.npz';state.write_bytes(b'partial proposal')
                state.with_suffix('.json').write_text(json.dumps(dict(audit=dict(passed=True,full_loads=True))))
                (folder/'current.json').write_text(json.dumps(dict(volume_cm3=90.,state=state.name,sha256=sha256(state))))
                raise subprocess.TimeoutExpired('trial',1)
            slots=queue.Queue();slots.put(0)
            config=dict(folder=temp,iteration=0,branch=0,kind='juxtaposition',task_ids=['A'])
            with patch('subdivide_task.step4.population_search.subprocess.run',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'incomplete checkpoint'):
                    bounded_worker(config,slots)
            self.assertEqual(slots.get_nowait(),0)


if __name__=='__main__': unittest.main()
