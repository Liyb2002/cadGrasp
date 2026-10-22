"""Sharing balance, pose freedoms and feasibility restoration before acceptance."""
import unittest
import numpy as np
import trimesh
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from whole_task_2.operations.geometry import OperationModel, contact_surface
from whole_task_2.step1.needle_geometry import geometric_mask
from whole_task_2.operations.state import transform
from .sharing_search import BalancedSchedule, propose, projection_model, KINDS
from .gradient_repair import block_step, global_pressure_feasible, PairObjective, guidance_contacts
from whole_task_2.operations.state import Layout
from .optimize import legal_improvement
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.exit_geometry import continuous_checks


class SharingTests(unittest.TestCase):
    def test_turned_remote_material_sets_full_exit_distance_and_longer_path_is_checked(self):
        from whole_task_2.operations.geometry import moved_mesh
        model=OperationModel.__new__(OperationModel);model.path_cache={};model.tolerance=1e-10
        M=np.eye(4);M[:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
        q=Layout(np.eye(4)[None],M[None],np.array([[0.,0.,1.]]))
        body=trimesh.creation.box([.02,.02,.02]);body.apply_translation([0.,0.,.2])
        support=G.mesh(G.box([0.,1.,1.],[2.,1.1,1.1]))
        world_support=moved_mesh(support,np.linalg.inv(M))
        task=dict(id='A',index=0,material=M,body_world=body,body=moved_mesh(body,M),
            direction=M[:3,:3]@q.exits[0],length=.5)
        self.assertLess(body.bounds[0,2]+task['length']-world_support.bounds[1,2],.015)
        candidate=dict(layout=q,support=support,tasks=[task])
        model.extend_exits_to_actual_support(candidate)
        self.assertGreaterEqual(body.bounds[0,2]+task['length']-world_support.bounds[1,2],.015-1e-12)
        paths,_=model.path_checks(G.solid(support),[('A',task['body'],task['direction'],task['length'])])
        self.assertTrue(paths[0]['passed'])
        self.assertGreater(paths[0]['leading_face_count'],0)

    def test_topology_cache_is_bound_to_actual_boundary_arrays(self):
        model=OperationModel.__new__(OperationModel)
        support=trimesh.creation.box([.1,.1,.1]);solid=G.solid(support)
        calls=[]
        def decompose():calls.append(True);return solid.decompose()
        wrapped=SimpleNamespace(decompose=decompose)
        self.assertEqual(model.material_component_count(support,wrapped),1)
        self.assertEqual(model.material_component_count(support.copy(),wrapped),1)
        self.assertEqual(len(calls),1)
        other=support.copy();other.apply_translation([.2,0.,0.])
        two=trimesh.util.concatenate([support,other])
        self.assertEqual(model.material_component_count(two,G.solid(two)),2)

    def test_guidance_choices_keep_legal_seed_fallback_and_use_actual_volume_order(self):
        from .population_search import certify_choices
        calls=[]
        def accept(q,kind,**kwargs):
            calls.append(q)
            return q=='seed'
        branch=SimpleNamespace(volume=100.,try_operation=accept)
        choices=[('seed',dict(volume_cm3=90.),{}),('polished',dict(volume_cm3=85.),{}),
            ('too_large',dict(volume_cm3=101.),{})]
        self.assertTrue(certify_choices(branch,choices,'rotational_reuse',audit_owners=[0,1],branch=7))
        self.assertEqual(calls,['polished','seed'])

    def test_every_schedule_prefix_is_balanced(self):
        for seed in range(12):
            schedule = BalancedSchedule(np.random.default_rng(seed))
            counts = dict.fromkeys(KINDS, 0)
            for _ in range(51):
                counts[schedule.next()] += 1
                self.assertLessEqual(abs(counts[KINDS[0]]-counts[KINDS[1]]), 1)

    def test_rng_and_pending_restore_the_same_schedule(self):
        rng = np.random.default_rng(42)
        schedule = BalancedSchedule(rng)
        schedule.next()
        state, pending = rng.bit_generator.state, schedule.pending.copy()
        expected = [schedule.next() for _ in range(15)]
        resumed_rng = np.random.default_rng(); resumed_rng.bit_generator.state = state
        resumed = BalancedSchedule(resumed_rng, pending)
        self.assertEqual(expected, [resumed.next() for _ in range(15)])

    def test_repair_may_increase_volume_while_restoring_coverage(self):
        def evaluate(x):
            return dict(x=float(x[0]), loss=float((x[0]-1)**2),
                covered=bool(abs(x[0]-1)<1e-8), volume_cm3=100.+10*x[0])
        current = evaluate(np.array([0.]))
        result, log = block_step(evaluate, current, lambda x:x, 1, .001, 1.)
        self.assertTrue(result['covered'])
        self.assertGreater(result['volume_cm3'], current['volume_cm3'])
        self.assertTrue(log['accepted'])
        self.assertEqual(log['phase'], 'repair')
        self.assertFalse(legal_improvement(200., result))

    def test_volume_descent_cannot_leave_feasibility(self):
        def evaluate(x):
            return dict(loss=max(0., -float(x[0])), covered=bool(x[0]>=0),
                        volume_cm3=10.+float(x[0]))
        current = evaluate(np.array([0.]))
        result, log = block_step(evaluate, current, lambda x:x, 1, .001, 1.)
        self.assertIs(result, current)
        self.assertFalse(log['accepted'])

    def test_one_sided_difference_at_a_geometric_boundary(self):
        def evaluate(x):
            if x[0]<0:return None
            return dict(loss=float((x[0]-1)**2), covered=bool(x[0]>=1), volume_cm3=10.)
        current = evaluate(np.array([0.]))
        result, log = block_step(evaluate, current, lambda x:x, 1, .001, 1.)
        self.assertTrue(log['accepted'])
        self.assertTrue(result['covered'])

    def test_sharing_proposals_have_distinct_freedoms_and_preserve_task_regions(self):
        model = OperationModel(task_ids=['W04', 'W06'])
        q = model.initial
        rng = np.random.default_rng(42)
        translated = propose(model, q, 'juxtaposition', 1, 0, rng, 0)
        np.testing.assert_array_equal(translated.poses, q.poses)
        np.testing.assert_array_equal(translated.material[:, :3, :3], q.material[:, :3, :3])
        seats = translated.material@translated.poses
        np.testing.assert_allclose(transform(model.source.center_mass, seats[0]),
            transform(model.source.center_mass, seats[1]), atol=1e-12)
        rotated = propose(model, q, 'rotational_reuse', 1, 0, rng, 0)
        np.testing.assert_allclose(rotated.material[0]@rotated.poses[0],
            rotated.material[1]@rotated.poses[1], atol=1e-12)
        self.assertFalse(np.allclose(rotated.poses[1], q.poses[1]))
        self.assertFalse(np.allclose(rotated.material[1, :3, :3], q.material[1, :3, :3]))
        pose = rotated.poses[1]
        head = transform(model.head, np.linalg.inv(pose))
        self.assertTrue(geometric_mask(model.source, head, pose[:3, :3].T@model.axis,
            model.workspace)[model.records[1]['working_face_ids']].all())
        np.testing.assert_array_equal(q.poses, model.initial.poses)

    def test_complete_path_crop_preserves_collision_and_clearance_results(self):
        model = OperationModel.__new__(OperationModel)
        model.path_cache = {}; model.tolerance = 1e-10
        body = trimesh.creation.box([.02, .02, .02])
        path = [('test', body, np.array([0.,0.,1.]), .2)]
        far = G.box([2.,2.,2.], [3.,3.,3.])
        obstruction = G.box([-.005,-.005,.08],[.005,.005,.12])
        for support in (far, far+obstruction):
            reference,_ = continuous_checks(support,path,model.tolerance)
            cropped,_ = model.path_checks(support,path)
            self.assertEqual(reference[0]['passed'],cropped[0]['passed'])
            self.assertEqual(reference[0]['leading_face_count'],cropped[0]['leading_face_count'])
            self.assertAlmostEqual(reference[0]['continuous_overlap_upper_bound_m3'],
                cropped[0]['continuous_overlap_upper_bound_m3'],places=15)

    def test_basic_pair_adjustment_cannot_put_a_third_task_below_a_floor(self):
        model = SimpleNamespace(records=[{}, {}, {}], world_data=lambda r,p:dict(
            endpoints=dict(endpoint_need_wrench=np.array([[0.,0.,1.,0.,0.,0.]])),
            domain=SimpleNamespace(com=np.zeros(3))))
        matrices = np.tile(np.eye(4),(3,1,1))
        matrices[0,0,3] = .2
        matrices[1,:3,:3] = Rotation.from_euler('y',90,degrees=True).as_matrix()
        matrices[2,0,3] = .05
        q = Layout(np.tile(np.eye(4),(3,1,1)),matrices,np.tile([0.,0.,1.],(3,1)))
        self.assertTrue(global_pressure_feasible(model,q))
        q.material[1,0,3] = .1
        self.assertFalse(global_pressure_feasible(model,q))

    def test_diagonal_exit_crop_matches_all_original_uncropped_prisms(self):
        model=OperationModel.__new__(OperationModel)
        model.path_cache={};model.tolerance=1e-10
        body=trimesh.creation.box([.02,.03,.015])
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[13.,21.,31.],degrees=True).as_matrix()
        T[:3,3]=[.1,.2,.3];body.apply_transform(T)
        direction=np.array([1.,2.,3.]);direction/=np.linalg.norm(direction)
        center=body.center_mass+.1*direction
        obstruction=G.box(center-.003,center+.003)
        far=G.box([2.,2.,2.],[3.,3.,3.])
        paths=[('diagonal',body,direction,.2)]
        for support in (far,far+obstruction):
            original,_=continuous_checks(support,paths,model.tolerance)
            cropped,_=model.path_checks(support,paths)
            self.assertEqual(original[0]['passed'],cropped[0]['passed'])
            self.assertEqual(original[0]['leading_face_count'],cropped[0]['leading_face_count'])
            self.assertAlmostEqual(original[0]['continuous_overlap_upper_bound_m3'],
                cropped[0]['continuous_overlap_upper_bound_m3'],places=15)

    def test_cached_contact_points_follow_the_new_task_pose(self):
        before = Layout(np.eye(4)[None],np.eye(4)[None],np.array([[0.,0.,1.]]))
        after = before.copy()
        after.poses[0,:3,:3] = Rotation.from_euler('z',90,degrees=True).as_matrix()
        after.poses[0,:3,3] = [.2,.1,0.]
        model = SimpleNamespace(ids=['A'],current_contacts={'A':np.array([[.1,0.,0.]])})
        projected = projection_model(model,before,after)
        np.testing.assert_allclose(projected.current_contacts['A'],[[.2,.2,0.]],atol=1e-12)
        np.testing.assert_array_equal(model.current_contacts['A'],[[.1,0.,0.]])

    def test_pair_guidance_keeps_other_tasks_cutouts_and_all_floors(self):
        objective=PairObjective.__new__(PairObjective)
        cut=G.box([-.01,-.01,-.01],[.01,.01,.04])
        objective.foreign=[cut];objective.foreign_bounds=[G.mesh(cut).bounds]
        matrices=np.tile(np.eye(4),(3,1,1))
        matrices[2,2,3]=.01
        q=Layout(matrices.copy(),matrices,np.tile([0.,0.,1.],(3,1)))
        support=trimesh.creation.box([.04,.04,.08]);support.apply_translation([0.,0.,.02])
        trimmed=objective.contextual_support(dict(support=support),q)
        self.assertGreaterEqual(trimmed.bounds[0,2],.01-1e-12)
        self.assertLess(G.volume(G.solid(trimmed)^cut),1e-15)
        self.assertLess(trimmed.volume,support.volume)

    def test_gradient_parent_crop_keeps_local_stock_and_floor_but_drops_constant_remote_cells(self):
        q=Layout(np.tile(np.eye(4),(3,1,1)),np.tile(np.eye(4),(3,1,1)),np.tile([0.,0.,1.],(3,1)))
        local=G.box([-.1,-.1,0.],[.1,.1,.3])
        far=G.box([3.,-.1,0.],[3.2,.1,.3])
        rows=[dict(id=n,stock=(local if k<2 else far),patch_points=np.array([
            [0. if k<2 else 3.,0.,0.],[0. if k<2 else 3.,0.,.008]]),
            sweep=far,work=far) for k,n in enumerate('ABC')]
        model=SimpleNamespace(ids=list('ABC'),records=[{}]*3,
            source=SimpleNamespace(extents=np.ones(3)),policy={'plate_thickness_m':.008},
            reference_material=lambda matrices:matrices[0],family_before_support=local+far,
            family_previous_regions=[local,far],family_protected_contact_regions=far)
        objective=PairObjective(model,q,[0,1],obstruction_context=rows)
        self.assertLess(G.volume(objective.pair.family_before_support^far),1e-18)
        self.assertAlmostEqual(G.volume(objective.pair.family_before_support^local),G.volume(local))
        self.assertAlmostEqual(G.volume(model.family_before_support),G.volume(local)+G.volume(far))
        self.assertEqual(len(objective.foreign),2)

    def test_contact_surface_pruning_preserves_actual_wrench_cone(self):
        from whole_task_2.step3.independent_physics import contacts
        from whole_task_2.step3.legacy import kernel as K
        body=trimesh.creation.box([.2,.2,.2])
        shell=G.box([-.11]*3,[.11]*3)-G.solid(body)
        full=G.mesh(shell+G.box([2.,2.,2.],[3.,3.,3.]))
        near=contact_surface(body,full)
        self.assertLess(len(near.faces),len(full.faces))
        a,sa,ra,scale=contacts(body,full,np.arange(6),body.center_mass)
        b,sb,rb,_=contacts(body,near,np.arange(6),body.center_mass)
        area=lambda tri,ids:np.bincount(ids,weights=trimesh.triangles.area(tri),minlength=len(body.faces))
        np.testing.assert_allclose(area(a,sa),area(b,sb),atol=1e-15,rtol=0)
        targets=np.random.default_rng(8).normal(size=(100,6))*scale
        ma,_=K.J.classify(ra,targets);mb,_=K.J.classify(rb,targets)
        np.testing.assert_array_equal(ma,mb)

    def test_gradient_contact_subset_cannot_invent_reaction_rays(self):
        from whole_task_2.step3.independent_physics import contacts
        from whole_task_2.step3.legacy import kernel as K
        body=trimesh.creation.icosphere(subdivisions=1,radius=.1)
        outer=body.copy();outer.vertices*=1.1
        support=G.mesh(G.solid(outer)-G.solid(body))
        work=np.arange(10)
        task=dict(id='A',body_world=body,domain=SimpleNamespace(work_ids=work,com=body.center_mass))
        _,_,full,_=contacts(body,support,work,body.center_mass)
        _,sources,sampled,_=guidance_contacts(task,support,SimpleNamespace(),limit=16)
        self.assertFalse(set(sources)&set(work))
        self.assertLess(len(set(sources)),len(body.faces)-len(work))
        passed,_=K.J.classify(full,sampled)
        self.assertTrue(passed.all())
        np.testing.assert_array_equal(work,np.arange(10))


if __name__ == '__main__':
    unittest.main()
