"""Explicit family unions and preservation of independent contact/path proofs."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import trimesh
from subdivide_task.operations.state import Layout
from subdivide_task.step3 import independent_geometry as G
from .group_acceptance import (random_pair, merged_groups, retained_certificates,
    combine_audit, validate_partition, place_merged_family, freeze_independent_material)
from .group_acceptance import same_contact_generators
from .optimize import Optimizer


class FamilyTests(unittest.TestCase):
    def test_abc_and_def_merge_as_six_tasks_and_leave_g_independent(self):
        groups=[[0,1,2],[3,4,5],[6]]
        self.assertEqual(merged_groups(groups,1,4),[[6],[0,1,2,3,4,5]])
        self.assertEqual(groups,[[0,1,2],[3,4,5],[6]])
        with self.assertRaises(ValueError): merged_groups(groups,0,1)

    def test_duplicate_missing_and_empty_families_rejected(self):
        for groups in ([[0],[0,1]],[[0]],[[0,1],[]]):
            with self.assertRaises(ValueError): validate_partition(groups,2)

    def test_old_step4_checkpoint_cannot_seed_a_new_family_population(self):
        from .population_search import solve
        with tempfile.TemporaryDirectory() as temp:
            old=SimpleNamespace(folder=Path(temp),accepted=18,groups=[[0,1]])
            with self.assertRaisesRegex(ValueError,'untouched Step3'):
                solve(old)

    def test_family_sampling_does_not_prefer_large_or_near_families(self):
        q=Layout(np.tile(np.eye(4),(6,1,1)),np.tile(np.eye(4),(6,1,1)),np.tile([0.,0.,1.],(6,1)))
        model=SimpleNamespace(ids=list('ABCDEF'),axis=np.array([0.,0.,1.]))
        groups=[[0,1,2,3],[4],[5]]
        counts={(0,1):0,(0,2):0,(1,2):0}
        rng=np.random.default_rng(13)
        for _ in range(3000):
            pair=random_pair(model,q,groups,rng)
            a=next(i for i,g in enumerate(groups) if pair['owner'] in g)
            b=next(i for i,g in enumerate(groups) if pair['host'] in g)
            counts[tuple(sorted((a,b)))]+=1
        self.assertTrue(all(850<v<1150 for v in counts.values()),counts)

    def test_rotational_reuse_preserves_existing_family_seats_during_world_spin(self):
        from .population_search import rotate_family_world_pose
        from scipy.spatial.transform import Rotation
        q=Layout(np.tile(np.eye(4),(3,1,1)),np.tile(np.eye(4),(3,1,1)),np.tile([0.,0.,1.],(3,1)))
        q.poses[1,:3,:3]=Rotation.from_euler('x',23,degrees=True).as_matrix()
        q.poses[1,:3,3]=[.1,.2,.3]
        q.material[1,:3,:3]=Rotation.from_euler('y',37,degrees=True).as_matrix()
        q.material[1,:3,3]=[.2,.3,.4]
        spun=rotate_family_world_pose(q,[0,1],70.,np.array([0.,0.,1.]),np.array([.3,.2,.5]))
        np.testing.assert_allclose(spun.material[:2]@spun.poses[:2],q.material[:2]@q.poses[:2],atol=1e-12)
        for key,array in q.arrays().items():
            np.testing.assert_array_equal(array[2],spun.arrays()[key][2])
        self.assertFalse(np.allclose(spun.poses[1],q.poses[1]))

    def test_unused_contact_corner_does_not_pull_a_proposed_shared_seat_apart(self):
        from .population_search import make_proposal
        from scipy.spatial.transform import Rotation
        source=trimesh.creation.box([1.,.2,.2]);source.apply_translation([0.,0.,.3])
        q=Layout(np.tile(np.eye(4),(2,1,1)),np.tile(np.eye(4),(2,1,1)),np.tile([0.,0.,1.],(2,1)))
        q.material[0,:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix();q.material[0,0,3]=-1.
        model=SimpleNamespace(ids=list('AB'),records=[dict(id=n) for n in 'AB'],source=source,
            current_contacts={'A':np.array([[.5,0.,.3]])},
            world_data=lambda *a:dict(endpoints=dict(endpoint_need_wrench=None),domain=SimpleNamespace(com=None)))
        branch=SimpleNamespace(layout=q,groups=[[0],[1]])
        pair=dict(owner=0,host=1,axial_deg=0.)
        gaps=[]
        with patch('subdivide_task.step4.spatial_search.G.pressure_points',return_value=np.array([[0.,0.],[.01,0.],[0.,.01]])):
            for policy in ('pressure_only','preserve_all'):
                model.placement_contact_policy=policy
                proposed,_=make_proposal(model,branch,pair,'juxtaposition',0,np.random.default_rng(1))
                center=source.center_mass
                a=proposed.material[0]@proposed.poses[0];b=proposed.material[1]@proposed.poses[1]
                gaps.append(np.linalg.norm((a[:3,:3]@center+a[:3,3])-(b[:3,:3]@center+b[:3,3])))
        self.assertLess(gaps[0],1e-12)
        self.assertGreater(gaps[1],.19)

    def test_member_translation_can_repair_spun_floor_offsets_without_splitting_family(self):
        from .population_search import rotate_family_world_pose
        from .spatial_search import floor_repair
        q=Layout(np.tile(np.eye(4),(3,1,1)),np.tile(np.eye(4),(3,1,1)),np.tile([0.,0.,1.],(3,1)))
        q.material[1,1,3]=.01
        spun=rotate_family_world_pose(q,[0,1],180.,np.array([0.,1.,1.])/np.sqrt(2),np.array([0.,0.,.5]))
        model=SimpleNamespace(records=[dict(id=n) for n in 'ABC'],current_contacts={},
            world_data=lambda *a:dict(endpoints=dict(endpoint_need_wrench=None),domain=SimpleNamespace(com=None)))
        families=[[0,1],[2]]
        pressure=np.array([[0.,0.],[.01,0.],[0.,.01]])
        with patch('subdivide_task.step4.spatial_search.G.pressure_points',return_value=pressure):
            self.assertIsNone(floor_repair(model,spun,families,2))
            projected=floor_repair(model,spun,[[0],[1],[2]],2)
        self.assertIsNotNone(projected)
        n=projected.material[0,:3,2]
        self.assertLess(abs(n@(projected.material[0,:3,3]-projected.material[1,:3,3])),1e-9)
        self.assertEqual(families,[[0,1],[2]])

    def fixture(self):
        q=Layout(np.tile(np.eye(4),(3,1,1)),np.tile(np.eye(4),(3,1,1)),np.tile([0.,0.,1.],(3,1)))
        support=trimesh.creation.box([2.,2.,.2]);support.apply_translation([0.,0.,.1])
        body=trimesh.creation.box([.2,.2,.2]);body.apply_translation([0.,0.,.3])
        tasks=[dict(id=name,body=body,body_world=body,material=np.eye(4),
            direction=np.array([0.,0.,1.]),length=1.,pressure_world=np.array([[0.,0.]]),
            patch_points=np.array([[x,y,z] for x in (-.05,.05) for y in (-.05,.05) for z in (0.,.008)]),
            work=G.box([4.,4.,0.],[5.,5.,1.]),stock=G.solid(support)) for name in 'ABC']
        previous=dict(passed=True,full_loads=True,source_face_prisms_checked=True,
            tasks=[dict(id='C',force=dict(passed=True),floor=dict(passed=True))],
            continuous_paths=[dict(id='C',passed=True)])
        model=SimpleNamespace(ids=list('ABC'),check_deadline=lambda:None)
        from types import MethodType
        from subdivide_task.operations.geometry import OperationModel
        model.path_cache={};model.tolerance=1e-10
        model.path_checks=MethodType(OperationModel.path_checks,model)
        candidate=dict(layout=q.copy(),support=support.copy(),tasks=tasks)
        return model,q,support,candidate,previous

    def test_untouched_family_reuses_its_certificate(self):
        model,q,support,candidate,previous=self.fixture()
        extra=G.box([2.,0.,0.],[2.1,.1,.1])
        candidate['support']=G.mesh(G.solid(support)+extra)
        rows,paths,guards=retained_certificates(model,q,support,candidate,[0,1],previous)
        self.assertEqual([r['id'] for r in rows],['C'])
        self.assertTrue(rows[0]['certificate_reused_from_parent'])
        self.assertTrue(paths[0]['passed'])
        self.assertEqual(guards[0]['changed_contact_neighbourhood_m3'],0.)

    def test_removing_an_independent_contact_layer_rejects_merge(self):
        model,q,support,candidate,previous=self.fixture()
        candidate['support']=G.mesh(G.solid(support)-G.box([-.04,-.04,.19],[.04,.04,.21]))
        with self.assertRaisesRegex(ValueError,'independent family'):
            retained_certificates(model,q,support,candidate,[0,1],previous)

    def test_new_material_in_an_independent_exit_path_rejects_merge(self):
        model,q,support,candidate,previous=self.fixture()
        candidate['support']=G.mesh(G.solid(support)+G.box([-.04,-.04,.6],[.04,.04,.7]))
        with self.assertRaisesRegex(ValueError,'independent family'):
            retained_certificates(model,q,support,candidate,[0,1],previous)

    def test_empty_corner_of_diagonal_path_box_does_not_obstruct_exit(self):
        model,q,support,candidate,previous=self.fixture()
        direction=np.array([1.,0.,1.])/np.sqrt(2.)
        candidate['tasks'][2]['direction']=direction
        previous['continuous_paths'][0]['leading_face_count']=12
        corner=G.box([.5,-.02,.25],[.55,.02,.3])
        candidate['support']=G.mesh(G.solid(support)+corner)
        rows,paths,guards=retained_certificates(model,q,support,candidate,[0,1],previous)
        self.assertIn(guards[0]['continuous_path_guard'],('identical_local_material','actual_new_original_face_prisms'))
        self.assertLessEqual(guards[0]['inherited_path_overlap_bound_m3'],1e-10)

    def test_path_bound_can_credit_removed_material_without_resolving_task_loads(self):
        from types import MethodType
        from subdivide_task.operations.geometry import OperationModel
        model,q,support,candidate,previous=self.fixture()
        old_cap=G.box([-.06,-.005,.2],[-.05,.005,.2000006])
        new_cap=G.box([.05,-.005,.2],[.06,.005,.2000006])
        before=G.mesh(G.solid(support)+old_cap)
        candidate['support']=G.mesh(G.solid(support)+new_cap)
        model.path_cache={};model.tolerance=1e-10
        model.path_checks=MethodType(OperationModel.path_checks,model)
        previous['continuous_paths'][0].update(continuous_overlap_upper_bound_m3=6e-11,leading_face_count=2)
        with patch('subdivide_task.step4.group_acceptance.same_contact_generators',return_value=(True,{})):
            rows,paths,guards=retained_certificates(model,q,before,candidate,[0,1],previous)
        self.assertEqual(guards[0]['continuous_path_guard'],'actual_new_original_face_prisms')
        self.assertTrue(paths[0]['geometry_refreshed_without_load_reclassification'])
        self.assertLessEqual(paths[0]['continuous_overlap_upper_bound_m3'],model.tolerance)

    def test_no_parameter_of_an_independent_family_can_change(self):
        model,q,support,candidate,previous=self.fixture()
        candidate['layout'].material[2,0,3]=.01
        with self.assertRaisesRegex(ValueError,'outside its two families'):
            retained_certificates(model,q,support,candidate,[0,1],previous)

    def test_side_floor_placement_moves_only_the_merged_family(self):
        model,q,support,candidate,previous=self.fixture()
        from scipy.spatial.transform import Rotation
        q.material[0,:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
        model.records=[{}]*3
        model.world_data=lambda *a:dict(endpoints=dict(endpoint_need_wrench=None),domain=SimpleNamespace(com=None))
        with patch('subdivide_task.step4.group_acceptance.G.pressure_points',return_value=np.array([[0.,0.]])):
            placed=place_merged_family(model,support,q,[0,1])
        np.testing.assert_array_equal(placed.material[2],q.material[2])
        np.testing.assert_allclose(placed.material[:2,:3,3],[[-1.,0.,0.],[-1.,0.,0.]],atol=1e-12)
        n=placed.material[0,:3,2]
        self.assertGreaterEqual(((support.vertices-placed.material[0,:3,3])@n).min(),-1e-12)

    def test_side_placement_reserves_the_entire_new_body_outside_fixed_seats(self):
        model,q,support,candidate,previous=self.fixture()
        from scipy.spatial.transform import Rotation
        from subdivide_task.operations.state import transform
        model.source=candidate['tasks'][0]['body_world']
        q.material[0,:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
        model.records=[{}]*3
        model.world_data=lambda *a:dict(endpoints=dict(endpoint_need_wrench=None),domain=SimpleNamespace(com=None))
        with patch('subdivide_task.step4.group_acceptance.G.pressure_points',return_value=np.array([[0.,0.]])):
            placed=place_merged_family(model,support,q,[0,1],clearance=.02)
        points=np.vstack([transform(model.source.vertices,placed.material[k]@placed.poses[k]) for k in [0,1]])
        normal=placed.material[0,:3,2]
        self.assertLessEqual((points@normal).max()+.02,(support.vertices@normal).min()+1e-12)
        np.testing.assert_array_equal(placed.material[2],q.material[2])

    def test_unchanged_installed_side_floor_does_not_drift_on_a_later_merge(self):
        model,q,support,candidate,previous=self.fixture()
        from scipy.spatial.transform import Rotation
        model.source=candidate['tasks'][0]['body_world']
        q.material[0,:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
        q.material[0,0,3]=-1.
        model.family_before_layout=q.copy()
        model.records=[{}]*3
        model.world_data=lambda *a:dict(endpoints=dict(endpoint_need_wrench=None),domain=SimpleNamespace(com=None))
        with patch('subdivide_task.step4.group_acceptance.G.pressure_points',return_value=np.array([[0.,0.]])):
            placed=place_merged_family(model,support,q,[0,1],clearance=.02)
        np.testing.assert_allclose(placed.material,q.material,atol=1e-12)

    def test_freezing_keeps_an_independent_contact_region_from_the_actual_parent(self):
        model,q,support,candidate,previous=self.fixture()
        model.prepare=lambda _:candidate['tasks']
        freeze_independent_material(model,q,support,[0,1])
        original=G.solid(support)
        delta=(original-model.family_before_support)+(model.family_before_support-original)
        self.assertLessEqual(G.volume(delta),1e-18)
        self.assertEqual(model.family_edit_ids,{'A','B'})

    def test_old_dense_pedestal_is_inside_edit_region_when_parent_already_has_thin_stock(self):
        model,q,support,candidate,previous=self.fixture()
        dense=G.box([-.5,-.5,0.],[.5,.5,.4])
        thin=G.box([-.1,-.1,0.],[.1,.1,.4])
        candidate['tasks'][0].update(stock=thin,dense_stock=dense)
        model.prepare=lambda _:candidate['tasks']
        freeze_independent_material(model,q,support,[0,1])
        self.assertLess(G.volume(dense-G.union(model.family_previous_regions)),1e-18)

    def test_free_material_pruning_retains_parent_contacts_and_pressure_foot(self):
        model,q,support,candidate,previous=self.fixture()
        model.family_refit_mode='free_material_box'
        model.prepare=lambda _:candidate['tasks']
        for task in candidate['tasks']:
            task['patch_points']=np.array([[x,y,z] for x in (-.05,.05)
                for y in (-.05,.05) for z in (0.,.01)])
        with patch('subdivide_task.operations.ribbed_stock.contact_skin',return_value=candidate['tasks'][2]['body']):
            freeze_independent_material(model,q,support,[0,1])
        retained=model.family_retained_material
        contact=G.box([-.05,-.05,.1995],[.05,.05,.2])
        foot=G.box([-.04,-.04,0.],[.04,.04,.009])
        free=G.box([.5,.5,.05],[.6,.6,.15])
        self.assertLess(G.volume(contact-retained),1e-18)
        self.assertLess(G.volume(foot-retained),1e-18)
        self.assertLess(G.volume(free^retained),1e-18)

    def test_generator_retention_checks_every_parent_reaction_without_resolving_loads(self):
        model,q,support,candidate,previous=self.fixture()
        task=candidate['tasks'][2]
        task['domain']=SimpleNamespace(work_ids=np.array([],int),com=task['body_world'].center_mass)
        parent=np.array([[1.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.]])
        for current, expected in ((parent.copy(),True),(parent[:1],False)):
            values=[(None,None,columns,None) for columns in (parent,current)]
            with patch('subdivide_task.step3.independent_physics.contacts',side_effect=values):
                passed,proof=same_contact_generators(task,support,support)
            self.assertEqual(passed,expected)
            self.assertEqual(proof['kernel_generator_deduplication_digits'],13)

    def test_retriangulated_contact_can_reuse_a_strictly_contained_parent_ray(self):
        model,q,support,candidate,previous=self.fixture()
        task=candidate['tasks'][2]
        task['domain']=SimpleNamespace(work_ids=np.array([],int),com=task['body_world'].center_mass)
        new=np.c_[np.eye(6),np.eye(6)[:,2]]
        for old,expected in ((np.ones((1,6)),True),(-np.ones((1,6)),False)):
            old=np.c_[old,old[:,2]]
            values=[(None,None,c,None) for c in (old,new)]
            with patch('subdivide_task.step3.independent_physics.contacts',side_effect=values):
                passed,proof=same_contact_generators(task,support,support)
            self.assertEqual(passed,expected)
            self.assertFalse(proof['original_loads_resolved'])

    def test_boundary_contact_ray_can_use_a_lower_rank_nonnegative_basis(self):
        model,q,support,candidate,previous=self.fixture()
        task=candidate['tasks'][2]
        task['domain']=SimpleNamespace(work_ids=np.array([],int),com=task['body_world'].center_mass)
        new=np.c_[np.eye(6),np.eye(6)[:,2]]
        for target,expected in (([1.,1.,0.,0.,0.,0.],True),
                                ([-1e-11,1.,0.,0.,0.,0.],False)):
            old=np.array([target]);old=np.c_[old,old[:,2]]
            values=[(None,None,c,None) for c in (old,new)]
            with patch('subdivide_task.step3.independent_physics.contacts',side_effect=values):
                passed,proof=same_contact_generators(task,support,support)
            self.assertEqual(passed,expected)
            self.assertFalse(proof['original_loads_resolved'])

    def test_shared_passive_auxiliary_ray_is_not_misclassified_as_a_head(self):
        model,q,support,candidate,previous=self.fixture()
        task=candidate['tasks'][2]
        task['domain']=SimpleNamespace(work_ids=np.array([],int),com=task['body_world'].center_mass)
        auxiliary=np.r_[np.zeros(6),-1.][None]
        heads=np.c_[np.eye(6),np.eye(6)[:,2]]
        target=np.array([[1.,1.,0.,0.,0.,0.,0.]])
        old=np.vstack([auxiliary,target]);new=np.vstack([auxiliary,heads])
        values=[(None,None,c,None) for c in (old,new)]
        with patch('subdivide_task.step3.independent_physics.contacts',side_effect=values):
            passed,proof=same_contact_generators(task,support,support)
        self.assertTrue(passed)
        self.assertEqual(proof['shared_auxiliary_generator_count'],1)

    def test_unchanged_family_retains_certified_basis_without_unused_parent_rays(self):
        model,q,support,candidate,previous=self.fixture()
        task=candidate['tasks'][2]
        task['domain']=SimpleNamespace(work_ids=np.array([],int),com=task['body_world'].center_mass)
        required=np.array([[1.,0.,0.,0.,0.,0.,0.]])
        with patch('subdivide_task.step3.independent_physics.contacts',return_value=(None,None,required,None)) as kernel:
            passed,proof=same_contact_generators(task,support,support,{'C':required})
        self.assertTrue(passed)
        self.assertEqual(kernel.call_count,1)
        self.assertFalse(proof['parent_certificate_replayed_for_witness_bootstrap'])

    def test_contact_cache_reuses_only_identical_local_boundary_and_load_frame(self):
        model,q,support,candidate,previous=self.fixture()
        task=candidate['tasks'][2]
        task['domain']=SimpleNamespace(work_ids=np.array([],int),com=task['body_world'].center_mass)
        required=np.array([[1.,0.,0.,0.,0.,0.,0.]])
        cache={}
        with patch('subdivide_task.step3.independent_physics.contacts',return_value=(None,None,required,None)) as kernel:
            for _ in range(2):
                self.assertTrue(same_contact_generators(task,support,support,{'C':required},cache)[0])
            self.assertEqual(kernel.call_count,1)
            task['domain'].com=task['domain'].com+np.array([0.,0.,.001])
            same_contact_generators(task,support,support,{'C':required},cache)
            self.assertEqual(kernel.call_count,2)

    def test_actual_planar_contact_patch_can_certify_an_interior_reaction(self):
        model,q,support,candidate,previous=self.fixture()
        task=candidate['tasks'][2]
        task['domain']=SimpleNamespace(work_ids=np.array([],int),com=task['body_world'].center_mass)
        required=np.array([[0.,0.,1.,0.,0.,0.,1.]])
        passed,proof=same_contact_generators(task,support,support,{'C':required},{})
        self.assertTrue(passed)
        self.assertTrue(proof['contact_subset_used'])
        self.assertLess(proof['checked_contact_source_face_count'],len(task['body_world'].faces))

    def test_contact_face_subset_falls_back_to_the_full_actual_cone(self):
        model,q,support,candidate,previous=self.fixture()
        task=candidate['tasks'][2]
        task['domain']=SimpleNamespace(work_ids=np.array([],int),com=task['body_world'].center_mass)
        required=np.array([[1.,0.,0.,0.,0.,.2,0.]])
        narrow=np.array([[1.,0.,0.,0.,0.,0.,0.]])
        wide=np.vstack([narrow,[0.,1.,0.,0.,0.,1.,0.],[0.,-1.,0.,0.,0.,1.,0.]])
        values=[(None,None,c,None) for c in (narrow,wide)]
        with patch('subdivide_task.step3.independent_physics.contacts',side_effect=values) as kernel:
            passed,proof=same_contact_generators(task,support,support,{'C':required},{})
        self.assertTrue(passed)
        self.assertTrue(proof['contact_subset_fallback_used'])
        self.assertEqual(kernel.call_count,2)

    def test_witness_cache_cannot_cross_a_parent_change_or_hash_failure(self):
        from .population_search import save_retention_witnesses,load_retention_witnesses
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);current=folder/'current.json'
            current.write_text(json.dumps(dict(sha256='parent-A')))
            m=SimpleNamespace(retention_witnesses={'A':np.ones((2,7))})
            save_retention_witnesses(m,folder)
            restored=SimpleNamespace()
            load_retention_witnesses(restored,folder)
            np.testing.assert_array_equal(restored.retention_witnesses['A'],m.retention_witnesses['A'])
            current.write_text(json.dumps(dict(sha256='parent-B')))
            restored=SimpleNamespace();load_retention_witnesses(restored,folder)
            self.assertFalse(hasattr(restored,'retention_witnesses'))
            current.write_text(json.dumps(dict(sha256='parent-A')))
            np.savez_compressed(folder/'retention_witnesses.npz',A=np.zeros((2,7)))
            restored=SimpleNamespace();load_retention_witnesses(restored,folder)
            self.assertFalse(hasattr(restored,'retention_witnesses'))

    def test_combined_proof_identifies_only_merged_family_as_freshly_checked(self):
        local=dict(passed=True,tasks=[dict(id=n) for n in 'AB'],
                   continuous_paths=[dict(id=n) for n in 'AB'])
        audit=combine_audit(local,[dict(id='C')],[dict(id='C')],[], 'parent-sha',[0,1],list('ABC'))
        self.assertEqual(audit['freshly_checked_tasks'],['A','B'])
        self.assertEqual(audit['retained_tasks'],['C'])
        self.assertEqual(audit['parent_state_sha256'],'parent-sha')

    def exercise_scoped_acceptance(self, fail=False, cleanup=False):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            names=list('ABCDEF')
            proof=dict(passed=True,original_sample_count=32768,passed_original_sample_count=32768,
                continuous_outer_target_count=200,passed_continuous_outer_target_count=200)
            def row(name):
                return dict(id=name,force=proof,floor=dict(passed=True,actual_contact_area_m2=1.,
                    all_pressure_centers_on_actual_bottom=True))
            parent=dict(passed=True,full_loads=True,source_face_prisms_checked=True,
                tasks=[row(n) for n in names],continuous_paths=[dict(id=n,passed=True) for n in names])
            (folder/'current.json').write_text(json.dumps(dict(state='state_000.npz',sha256='parent')))
            (folder/'state_000.json').write_text(json.dumps(dict(audit=parent)))
            q=Layout(np.tile(np.eye(4),(6,1,1)),np.tile(np.eye(4),(6,1,1)),np.tile([0.,0.,1.],(6,1)))
            support=trimesh.creation.box([.04,.04,.04])
            cleaned=support.copy();cleaned.apply_scale(.9)
            checked=[]
            def audit(candidate,full=True,continuous=True,**kwargs):
                checked.append([t['id'] for t in candidate['tasks']])
                if cleanup:np.testing.assert_array_equal(candidate['support'].vertices,cleaned.vertices)
                if fail and full: raise ValueError('merged-family load failure')
                result=dict(passed=True,full_loads=full,source_face_prisms_checked=continuous,
                    tasks=[row(t['id']) for t in candidate['tasks']],
                    continuous_paths=[dict(id=t['id'],passed=True) for t in candidate['tasks']])
                candidate['audit']=result
                return result
            model=SimpleNamespace(ids=names,audit=audit)
            opt=Optimizer.__new__(Optimizer)
            opt.model, opt.folder, opt.layout, opt.support=model,folder,q,support
            opt.volume, opt.initial_volume, opt.accepted=100.,100.,0
            opt.groups=[[0,1],[2],[3,4],[5]]
            opt.requirements={}
            opt.event=lambda *a,**k:None
            saved=[]
            opt.save=lambda *a,**k:saved.append(a)
            candidate=dict(layout=q.copy(),support=support,tasks=[dict(id=n) for n in names],
                construction={})
            model.build=lambda q:candidate
            inherited=[row('C'),row('F')]
            paths=[dict(id=n,passed=True) for n in 'CF']
            retention_calls=[];refined=[]
            def retention(*args):
                from .group_acceptance import RetainedExitError
                retention_calls.append(1)
                if cleanup and len(retention_calls)==1:raise RetainedExitError('C',1e-9,1e-10)
                return inherited,paths,[]
            def refine(model,candidate):
                refined.append([t['id'] for t in candidate['tasks']])
                return (cleaned if cleanup else support),dict(local_clearance_history=[])
            with patch('subdivide_task.step4.group_acceptance.retained_certificates',side_effect=retention), \
                 patch('subdivide_task.step4.optimize.refine_candidate',side_effect=refine):
                accepted=opt.try_operation(q,'juxtaposition',groups=[[2],[5],[0,1,3,4]],audit_owners=[0,1,3,4])
            self.assertEqual(checked,[list('ABDE'),list('ABDE')])
            if fail:
                self.assertFalse(accepted)
                self.assertEqual(opt.groups,[[0,1],[2],[3,4],[5]])
                self.assertEqual(opt.volume,100.)
                self.assertEqual(saved,[])
            else:
                self.assertTrue(accepted)
                self.assertEqual(opt.groups,[[2],[5],[0,1,3,4]])
                self.assertEqual(saved[0][0]['audit']['freshly_checked_tasks'],list('ABDE'))
                self.assertEqual(saved[0][0]['audit']['retained_tasks'],list('CF'))
                if cleanup:
                    self.assertEqual(refined[0],['C'])
                    self.assertEqual(refined[1],list('ABDE'))
                    np.testing.assert_array_equal(opt.support.vertices,cleaned.vertices)
                    np.testing.assert_array_equal(saved[0][0]['support'].vertices,opt.support.vertices)

    def test_acceptance_resolves_only_the_union_of_two_existing_families(self):
        self.exercise_scoped_acceptance()

    def test_failed_family_merge_retains_geometry_and_partition(self):
        self.exercise_scoped_acceptance(fail=True)

    def test_retained_exit_cleanup_precedes_load_checks_and_saved_geometry(self):
        self.exercise_scoped_acceptance(cleanup=True)


if __name__=='__main__': unittest.main()
