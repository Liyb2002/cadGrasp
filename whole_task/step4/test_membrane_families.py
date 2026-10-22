"""State and geometry regressions for the independent-fixture formulation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from whole_task.operations.state import Layout
from whole_task.step3.membranes import MembraneModel
from whole_task.step3 import independent_geometry as G
from whole_task.step4.membrane_search import saves_volume,partition,joined_layout,seed_layouts,complete_proof,checkpoint,improves_published,schedule_entry,complete_ground_proof,select_branch,save_group
from whole_task.step4.membrane_guidance import MembraneObjective
from whole_task.step4.world_spin import spin
from whole_task.step4.family_bank import register,best_cover,explore_cover,covers


class MembraneFamiliesTests(unittest.TestCase):
    def test_connected_baseline_cannot_subsidize_an_expensive_merge(self):
        self.assertFalse(saves_volume(201.,100.,100.))
        self.assertFalse(saves_volume(200.,100.,100.))
        self.assertFalse(saves_volume(199.,100.,100.))
        self.assertTrue(saves_volume(150.,100.,100.))
        self.assertFalse(saves_volume(150.00000001,100.,100.))
        self.assertTrue(saves_volume(149.,100.,100.))
        self.assertFalse(saves_volume(1.,-100.,200.))
        self.assertFalse(saves_volume(float('nan'),100.,100.))

    def test_legacy_volume_rule_must_be_requested_explicitly(self):
        self.assertFalse(saves_volume(199.,100.,100.))
        self.assertTrue(saves_volume(199.,100.,100.,ratio=1.))
        self.assertFalse(saves_volume(200.,100.,100.,ratio=1.))

    def test_exact_cover_changes_membership_without_assigning_untested_tasks(self):
        def group(ids,v):return dict(members=ids,volume_cm3=v,sha256=str(ids)+str(v))
        state={}
        groups=[group([i],100.) for i in range(4)]+[
            group([0,1],90.),group([1,2],80.),group([0,3],70.),group([2,3],120.)]
        for g in groups:register(state,g)
        best=best_cover(state,4)
        self.assertEqual({tuple(g['members']) for g in best},{(1,2),(0,3)})
        self.assertEqual(sum(g['volume_cm3'] for g in best),150.)
        before={g['sha256']:dict(g) for g in groups}
        for seed in range(20):
            selection=explore_cover(state,4,np.random.default_rng(seed))
            partition(selection,4)
            self.assertTrue(all(g==before[g['sha256']] for g in selection))
        self.assertEqual(covers(state['family_bank'],4,forced=groups[4])[2],
            [groups[4],groups[7]])
        register(state,group([0,1,2,3],160.))
        self.assertEqual(len(best_cover(state,4)),2)
        register(state,group([0,1,2,3],140.))
        self.assertEqual(len(best_cover(state,4)),1)

    def test_final_save_cannot_bypass_the_75_percent_gate(self):
        force=dict(passed=True,original_sample_count=32768,passed_original_sample_count=32768,
            continuous_outer_target_count=17,passed_continuous_outer_target_count=17)
        proof=dict(passed=True,full_loads=True,source_face_prisms_checked=True,
            tasks=[dict(id=name,force=force) for name in ('A','B')],
            continuous_paths=[dict(id=name,passed=True) for name in ('A','B')])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.npz'
            with self.assertRaisesRegex(ValueError,'75%'):
                save_group(path,[0,1],dict(audit=proof,volume_cm3=151.),'juxtaposition',
                    [dict(volume_cm3=100.),dict(volume_cm3=100.)])
            self.assertFalse(path.exists())

    def test_resume_cannot_overwrite_a_smaller_published_one_set_result(self):
        current=dict(method='independent_membrane_family_merges',group_count=1,volume_cm3=540.,
            ground_oracle_certified=True)
        self.assertFalse(improves_published(1,658.,current))
        self.assertFalse(improves_published(1,540.,current))
        self.assertTrue(improves_published(2,500.,current))
        self.assertFalse(improves_published(3,500.,current))
        self.assertTrue(improves_published(1,530.,current))

    def test_uncertified_old_small_result_does_not_block_a_ground_legal_result(self):
        old=dict(method='independent_membrane_family_merges',group_count=1,volume_cm3=540.)
        self.assertTrue(improves_published(2,800.,old))

    def test_old_loose_merge_policy_cannot_block_the_required_75_percent_policy(self):
        current=dict(method='independent_membrane_family_merges',group_count=2,volume_cm3=500.,
            ground_oracle_certified=True)
        self.assertTrue(improves_published(3,600.,current,required_merge_ratio=.75))
        current['merge_volume_ratio_limit']=.75
        self.assertFalse(improves_published(3,400.,current,required_merge_ratio=.75))

    def test_one_set_may_cost_more_than_a_valid_two_set_alternative(self):
        branches=[dict(groups=[dict(volume_cm3=951.)]),
            dict(groups=[dict(volume_cm3=350.),dict(volume_cm3=351.)]),
            dict(groups=[dict(volume_cm3=100.)]*3)]
        self.assertEqual(select_branch(branches),1)

    def test_async_sample_order_does_not_change_pairs_seeds_or_balanced_operations(self):
        groups=[dict(members=[i]) for i in range(13)]
        def state():return dict(branches=[dict(groups=groups) for _ in range(32)],
            counts=dict(juxtaposition=0,rotational_reuse=0))
        a,b=state(),state();forward={};backward={}
        for j in range(32):forward[j]=schedule_entry(a,j,17,Path('/source'),Path('/work'),1,42,32)
        for j in reversed(range(32)):backward[j]=schedule_entry(b,j,17,Path('/source'),Path('/work'),1,42,32)
        self.assertEqual(forward,backward)
        self.assertEqual(a['counts'],dict(juxtaposition=16,rotational_reuse=16))
        self.assertTrue(all(e['a']!=e['b'] for e in forward.values()))

    def test_family_work_axis_spin_preserves_every_seat_and_material_exit(self):
        rng=np.random.default_rng(15)
        poses=np.tile(np.eye(4),(3,1,1));material=poses.copy()
        poses[:,:3,:3]=Rotation.random(3,random_state=rng).as_matrix()
        material[:,:3,:3]=Rotation.random(3,random_state=rng).as_matrix()
        poses[:,:3,3]=rng.normal(size=(3,3));material[:,:3,3]=rng.normal(size=(3,3))
        axis=np.array([0.,-1.,-1.])/np.sqrt(2);head=np.array([0.,-.4,.37])
        q=Layout(poses,material,np.tile(-axis,(3,1)))
        changed=spin(q,90.,axis,head)
        np.testing.assert_allclose(changed.material@changed.poses,q.material@q.poses,atol=2e-15)
        np.testing.assert_allclose(np.einsum('nij,nj->ni',changed.material[:,:3,:3],changed.exits),
            np.einsum('nij,nj->ni',q.material[:,:3,:3],q.exits),atol=2e-15)
        for old,new in zip(q.poses,changed.poses):
            np.testing.assert_allclose(np.linalg.inv(new)@np.r_[head,1.],np.linalg.inv(old)@np.r_[head,1.],atol=2e-15)
            np.testing.assert_allclose(new[:3,:3].T@axis,old[:3,:3].T@axis,atol=2e-15)
        self.assertGreater(np.linalg.norm(changed.poses[0,:3,:3].T@[0,0,-1]-q.poses[0,:3,:3].T@[0,0,-1]),.5)

    def test_oracle_success_flag_without_real_patches_and_full_loads_is_incomplete(self):
        self.assertFalse(complete_ground_proof(dict(passed=True),1))
        self.assertFalse(complete_ground_proof(dict(passed=True,
            full_witness_audit=dict(passed=True,full_loads=True,source_face_prisms_checked=True)),1))

    def test_each_task_belongs_to_exactly_one_independent_fixture(self):
        partition([dict(members=[0,2]),dict(members=[1])],3)
        for groups in ([dict(members=[0,1]),dict(members=[1,2])],[dict(members=[0,2])]):
            with self.assertRaises(ValueError):partition(groups,3)

    def test_incomplete_force_screen_cannot_be_saved_as_a_merged_fixture(self):
        force=dict(passed=True,original_sample_count=32768,passed_original_sample_count=32768,
            continuous_outer_target_count=17,passed_continuous_outer_target_count=17)
        proof=dict(passed=True,full_loads=True,source_face_prisms_checked=True,
            tasks=[dict(id='A',force=force)],continuous_paths=[dict(id='A',passed=True)])
        self.assertTrue(complete_proof(proof,1))
        self.assertFalse(complete_proof(proof,2))
        force['preflight_only']=True
        self.assertFalse(complete_proof(proof,1))
        force.pop('preflight_only');proof['source_face_prisms_checked']=False
        self.assertFalse(complete_proof(proof,1))

    def test_interrupted_journal_write_preserves_the_previous_legal_population(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'population.json';path.write_text('{"legal": true}\n')
            original=path.read_bytes()
            def interrupted(destination,value):
                Path(destination).write_text('{"incomplete":');raise OSError('interrupted')
            with patch('whole_task.step4.membrane_search.write_json',side_effect=interrupted):
                with self.assertRaises(OSError):checkpoint(path,dict(legal=False))
            self.assertEqual(path.read_bytes(),original)

    def test_rotational_reuse_turns_whole_guest_family_at_fixed_world_poses(self):
        poses=np.tile(np.eye(4),(2,1,1));poses[1,:3,:3]=Rotation.from_euler('x',90,degrees=True).as_matrix()
        poses[1,:3,3]=[.1,.2,.3]
        a=Layout(poses[:1],np.eye(4)[None],np.array([[0.,0.,1.]]))
        b=Layout(poses.copy(),np.tile(np.eye(4),(2,1,1)),np.array([[0.,0.,1.],[0.,0.,1.]]))
        b.material[1,:3,3]=[.3,0.,0.]
        left=dict(members=[0]);right=dict(members=[1,2])
        class RNG:
            def integers(self,n):return n-1
        with patch('whole_task.step4.membrane_search.read_group',side_effect=[(a,None),(b,None)]):
            model=SimpleNamespace(source=SimpleNamespace(center_mass=np.zeros(3)))
            q,na,ia,ib,host=joined_layout(left,right,RNG(),'rotational_reuse',model)
        np.testing.assert_array_equal(q.poses,np.concatenate([a.poses,b.poses]))
        np.testing.assert_array_equal(q.exits,np.concatenate([a.exits,b.exits]))
        np.testing.assert_allclose(q.material[0]@q.poses[0],q.material[ib]@q.poses[ib],atol=1e-14)
        np.testing.assert_allclose(np.linalg.inv(q.material[1])@q.material[2],
            np.linalg.inv(b.material[0])@b.material[1],atol=1e-14)
        self.assertAlmostEqual(np.degrees(Rotation.from_matrix(q.material[1,:3,:3]).magnitude()),90.)

    def test_multiseat_proposal_includes_side_reuse_and_preserves_existing_families(self):
        poses=np.tile(np.eye(4),(4,1,1));material=poses.copy()
        poses[1,:3,:3]=Rotation.from_euler('x',90,degrees=True).as_matrix()
        poses[3,:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
        material[1,:3,3]=[.1,0.,0.];material[3,:3,3]=[0.,.1,0.]
        q=Layout(poses,material,np.tile([0.,0.,1.],(4,1)))
        a=Layout(poses[:2],material[:2],q.exits[:2]);b=Layout(poses[2:],material[2:],q.exits[2:])
        mesh=trimesh.creation.box([.1,.1,.1])
        with patch('whole_task.step4.membrane_search.read_group',side_effect=[(a,mesh),(b,mesh)]):
            seeds,_=seed_layouts(q,2,np.eye(4),np.random.default_rng(42),
                'rotational_reuse',.1,dict(members=[0,1]),dict(members=[2,3]))
        for i in range(2):
            for j in range(2):
                self.assertTrue(any(np.allclose(s.material[2+j]@s.poses[2+j],
                    a.material[i]@a.poses[i],atol=1e-14) for s in seeds))
        for s in seeds:
            np.testing.assert_array_equal(s.poses,q.poses)
            np.testing.assert_allclose(np.linalg.inv(s.material[2])@s.material[3],
                np.linalg.inv(b.material[0])@b.material[1],atol=1e-14)

    def test_thin_constructor_is_height_invariant_and_adds_no_ground_material(self):
        # Actual native CSG, using a box only to isolate the floor-free rule.
        # An upper half-space exit leaves a finite bottom/side membrane.
        volumes=[]
        for height in (-2.,3.):
            body=trimesh.creation.box([.1,.1,.1]);body.apply_translation([0.,0.,height])
            outer=trimesh.creation.box([.11,.11,.11]);outer.apply_translation([0.,0.,height])
            sweep=G.box([-.05,-.05,height-.05],[.05,.05,height+1.])
            task=dict(stock=G.solid(outer),sweep=sweep,work=G.box([2,2,2],[3,3,3]),body=body)
            model=object.__new__(MembraneModel);model.tolerance=1e-10
            model.prepare=lambda q:[task]
            q=Layout(np.eye(4)[None],np.eye(4)[None],np.array([[0.,0.,1.]]))
            candidate=model.build(q)
            volumes.append(candidate['volume_cm3'])
            self.assertGreater(candidate['support'].bounds[0,2],height-.056)
            self.assertLess(candidate['support'].bounds[1,2],height+.056)
            self.assertFalse(candidate['construction']['floor_constraints'])
        self.assertAlmostEqual(*volumes,places=7)

    def test_head_inside_body_rejects_even_with_a_valid_surface_sector(self):
        model=object.__new__(MembraneModel)
        model.source=trimesh.creation.box([.01,.01,.05])
        model.head=np.zeros(3);model.axis=np.array([0.,0.,1.])
        model.workspace=dict(maximum_reference_distance_m=.1,half_angle_deg=25.)
        ids=np.flatnonzero(model.source.face_normals[:,2]>.9)
        model.records=[dict(id='inside',working_face_ids=ids)]
        q=Layout(np.eye(4)[None],np.eye(4)[None],np.array([[0.,0.,1.]]))
        with self.assertRaisesRegex(ValueError,'head is inside'):model.prepare(q)

    def test_shared_volume_samples_enclose_every_boundary_placement_seed(self):
        shape=trimesh.creation.box([.1,.1,.1])
        model=SimpleNamespace(source=shape,outer=shape,world_work=G.box([2,2,2],[3,3,3]),ids=['A'])
        q=Layout(np.eye(4)[None],np.eye(4)[None],np.array([[0.,0.,1.]]))
        far=q.copy();far.material[0,:3,3]=[1.,2.,3.]
        objective=MembraneObjective(model,q,volume_power=4,envelope_layouts=[q,far])
        self.assertTrue((objective.low<shape.bounds[0]).all())
        self.assertTrue((objective.high>shape.bounds[1]+[1.,2.,3.]).all())

    def test_guidance_does_not_invent_material_previously_removed_from_a_parent(self):
        outer=trimesh.creation.box([.2,.2,.2])
        retained=G.mesh(G.box([-.1,-.1,-.1],[0.,.1,.1]))
        model=SimpleNamespace(source=outer,outer=outer,world_work=G.box([2,2,2],[3,3,3]),ids=['A'],
            guidance_parent_parts=[(0,np.eye(4),retained)])
        q=Layout(np.eye(4)[None],np.eye(4)[None],np.array([[0.,0.,1.]]))
        objective=MembraneObjective(model,q,volume_power=4)
        pts=np.array([[-.05,0.,0.],[.05,0.,0.]])
        np.testing.assert_array_equal(objective.parent_contains(pts,q),[True,False])
        moved=q.copy();moved.material[0,:3,3]=[.5,0.,0.]
        np.testing.assert_array_equal(objective.parent_contains(pts+[.5,0.,0.],moved),[True,False])


if __name__=='__main__':unittest.main()
