"""Original physical gates, mixed membership and static-cost invariants."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import sys, unittest
import numpy as np

COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/'helper_func'),str(COOPT/'step4.2')]
import _bootstrap
from whole_search.model import Layout
from material_evaluator import MaterialSample, MaterialEvaluator, membership, target_gap, mixed_target_applicable, align_world_xy, lower_box, floor_height_proposal
from fixture_state import installed_state
from registered_floor import solve_registered_heights
from material_search import MaterialSearch, select_pool,freeze_population_measure


def layout(n=8):
    return Layout(np.repeat(np.eye(4)[None],n,axis=0),np.tile([0.,0.,1.],(n,1)),
                  np.arange(n),(tuple(range(n))))


def sample(q, passed=True):
    result=dict(layout=q,masks={k:np.array([passed]) for k in q.active},volume_cm3=1.)
    return MaterialSample(q,0.,dict(passed=True,enclosure_passed=True,connected_passed=True),1.,
        result=result,construction=dict(passed=True),evaluation=dict(
            with_full_support=dict(volume_litres=1.),material_full_cm3=1.))


class MembershipTests(unittest.TestCase):
    def test_output_batch_sizes_outside_mixed_scope_have_no_impossible_target(self):
        for n in [2, 3, 9, 10]:
            q=layout(n)
            self.assertFalse(mixed_target_applicable(q))
            self.assertEqual(target_gap(q),0)
        q=layout(4)
        self.assertTrue(mixed_target_applicable(q))
        self.assertGreater(target_gap(q),0)
        q.hosts[2:]=0
        self.assertEqual(target_gap(q),0)

    def test_material_contact_box_includes_all_installations(self):
        q=layout(2);q.placements[1,0,3]=1.
        native=np.repeat(np.eye(4)[None],2,axis=0);native[:,2,3]=.05
        native[1,:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
        import trimesh
        mesh=trimesh.creation.box([.1]*3)
        rows=[dict(U=native[k]@q.placements[k],hull=np.array([[0.,float(k)]]))for k in q.active]
        m=SimpleNamespace(mesh=mesh,native=native,points=np.array([[.05,0.,0.]]),
            base_oracle=SimpleNamespace(rows=lambda t:rows),contact_delta=SimpleNamespace(
                state=lambda t:SimpleNamespace(available={0:np.array([False]),1:np.array([True])})))
        self.assertAlmostEqual(lower_box(m,q)['volume_litres'],121.)

    def test_eight_pose_goal_requires_exactly_four_plus_four(self):
        q=layout();q.hosts[4:]=0
        self.assertEqual(membership(q),(4,4));self.assertEqual(target_gap(q),0)
        q.hosts[3]=0
        self.assertEqual(membership(q),(3,5));self.assertEqual(target_gap(q),1)

    def test_installation_translation_keeps_registered_role(self):
        q=layout(5);q.state_offsets[:]=[1.,2.,3.]
        self.assertEqual(membership(q),(5,0))
        q.placements[0,0,3]=.01;q.placements[1,1,3]=-.01
        self.assertEqual(membership(q),(3,2));self.assertEqual(target_gap(q),0)


class CertificationTests(unittest.TestCase):
    def test_coherent_direction_event_crosses_overlapping_exit_locks(self):
        # Both exits must turn before either contact becomes available.
        q=layout(2);initial=sample(q,passed=False)
        def objective(t):return float(not np.all(t.directions[:,1]<-.1))
        m=SimpleNamespace(extent=.1,commit=Mock(),floor_normal=lambda t,k:np.array([0.,0.,1.]),
            proxy=lambda t:dict(loss=objective(t),residual_loss=[objective(t)]*2),
            useful_coordinates=lambda current:([0,1],{}))
        controller=MaterialSearch(SimpleNamespace(model=m,frontier_width=8))
        with patch('material_search.installation_gradient',return_value=np.zeros(3)),\
             patch('material_search.floor_height_proposal',return_value=None),\
             patch('material_search.project_floor_layout',return_value=None):
            rows=controller.continuation_rows(initial,1)
        coherent=[t for kind,t,detail in rows if kind=='direction-coherent-event']
        self.assertTrue(any(objective(t)==0. for t in coherent))
        np.testing.assert_array_equal(q.directions,np.tile([0.,0.,1.],(2,1)))
        self.assertLessEqual(len(rows),26)

    def test_shared_measure_has_a_fixed_budget_and_keeps_each_rare_failure(self):
        parents=[sample(layout(1)) for _ in range(4)];rules=[]
        for i,item in enumerate(parents):
            item.result['masks'][0]=np.r_[np.zeros(200,bool),np.ones(300,bool)]
            nodes=np.c_[np.arange(208)+i*1000,np.zeros(208)]
            rules.append((nodes,np.r_[np.full(8,.6/8),np.full(200,.4/200)]))
        m=SimpleNamespace(adaptive_demands={},quadrature_epoch=0)
        with patch('material_search.forward_strata',side_effect=rules):
            freeze_population_measure(m,parents)
        nodes,w,_=m.adaptive_demands[0]
        self.assertLessEqual(len(nodes),256);self.assertAlmostEqual(float(w.sum()),1.)
        for i in range(4):self.assertIn(1000*i+8,nodes[:,0])

    def test_shared_measure_retains_each_parents_rare_failure_nodes(self):
        q=layout(1);a=sample(q);b=sample(q.copy())
        m=SimpleNamespace(adaptive_demands={},quadrature_epoch=0)
        rules=[(np.array([[1.,0.],[0.,1.]]),np.array([.001,.999])),
               (np.array([[2.,0.],[0.,1.]]),np.array([.2,.8]))]
        with patch('material_search.forward_strata',side_effect=rules):
            freeze_population_measure(m,[a,b])
        nodes,weights,_=m.adaptive_demands[0]
        self.assertAlmostEqual(float(weights.sum()),1.)
        for node in [[1.,0.],[2.,0.]]:
            i=np.flatnonzero((nodes==node).all(1))[0]
            self.assertGreater(weights[i],0.)
        self.assertEqual(m.quadrature_epoch,1)

    def test_bad_force_or_space_or_actual_solid_never_becomes_completed(self):
        item=sample(layout(5));self.assertTrue(item.completed)
        item.result['masks'][0][0]=False;self.assertFalse(item.completed)
        item.result['masks'][0][0]=True;item.oracle['passed']=False
        self.assertFalse(item.completed)
        item.oracle['passed']=True;item.construction['passed']=False
        self.assertFalse(item.completed)

    def test_force_failed_candidate_never_spends_a_connection_query(self):
        item=sample(layout(5),False)
        query=Mock();s=SimpleNamespace(model=SimpleNamespace(base_oracle=SimpleNamespace(query=query)))
        evaluator=MaterialEvaluator(s);evaluator.connect(item)
        query.assert_not_called()

    def test_refinement_does_not_attach_parent_masks_to_a_new_layout(self):
        initial=sample(layout(5));child=initial.layout.copy();child.placements[0,0,3]=.01
        child.placements[1,1,3]=.01
        preview=MaterialSample(child,0.,dict(enclosure_passed=True,gap_m=0.),.5)
        controller=MaterialSearch(SimpleNamespace(model=SimpleNamespace(),frontier_width=8))
        controller.continuation_rows=Mock(return_value=[('seat-event',child,{})])
        controller.preview=Mock(return_value=preview)
        chosen=controller.refine_preview(initial,0,1)
        self.assertIs(chosen,preview);self.assertIsNone(chosen.result)
        self.assertTrue(initial.result['masks'][0].all())

    def test_completed_incumbent_survives_a_cheaper_unbuilt_preview(self):
        q=layout(5);q.hosts[3:]=0;parent=sample(q)
        r=q.copy();r.directions[0]=[.1,0.,np.sqrt(.99)]
        child=MaterialSample(r,0.,dict(enclosure_passed=True,gap_m=0.),.01)
        self.assertIs(select_pool([child,parent],1)[0],parent)


class WorldBoxTests(unittest.TestCase):
    def model(self):
        vertices=np.array([[-1.,-1.,0.],[1.,1.,1.]])
        native=np.repeat(np.eye(4)[None],2,axis=0)
        native[0,:3,3]=[10.,20.,0.];native[1,:3,3]=[-20.,-10.,0.]
        return SimpleNamespace(mesh=SimpleNamespace(vertices=vertices),native=native)

    def test_horizontal_alignment_preserves_rotations_seats_and_heights(self):
        q=layout(2);q.state_offsets[:,2]=[2.,3.];m=self.model()
        aligned=align_world_xy(m,q)
        np.testing.assert_array_equal(q.placements,aligned.placements)
        np.testing.assert_array_equal(q.state_offsets[:,2],aligned.state_offsets[:,2])
        np.testing.assert_allclose(m.native[:,:2,3]+aligned.state_offsets[:,:2],0.)

    def test_lower_bound_uses_actual_world_objects_once_and_required_floor(self):
        q=layout(2);m=self.model()
        rows=[]
        for k in range(2):
            T=m.native[k].copy();T[2,3]=2.+k
            rows.append(dict(U=T,hull=np.array([[-2.,0.],[2.,0.]])))
        m.base_oracle=SimpleNamespace(rows=Mock(return_value=rows))
        result=lower_box(m,q)
        np.testing.assert_allclose(result['min_m'],[-21.,-11.,0.])
        np.testing.assert_allclose(result['max_m'],[11.,21.,4.])


class FloorHeightTests(unittest.TestCase):
    def test_registered_floor_prior_can_certify_an_unbounded_height_conflict(self):
        fields={(0,1):(np.array([-1.]),np.array([-2.])),
                (1,0):(np.array([-1.]),np.array([-2.]))}
        r=solve_registered_heights(fields,(0,1))
        self.assertEqual(r['status'],'certified_infeasible')
        c=r['certificate'];self.assertTrue(c['verified_in_longdouble'])
        self.assertTrue(all(x>=0 for x in c['combined_coefficients']))
        self.assertLess(c['combined_bound_m'],-.9)
        self.assertFalse(c['height_upper_bound_imposed'])

    def test_registered_floor_prior_returns_a_height_proposal_not_full_acceptance(self):
        fields={(0,1):(np.array([-2.]),np.array([0.])),
                (1,0):(np.array([0.]),np.array([0.]))}
        r=solve_registered_heights(fields,(0,1))
        self.assertEqual(r['status'],'necessary_feasible')
        np.testing.assert_allclose(r['heights_m'],[0.,2.],atol=1e-7)
        self.assertNotIn('passed',r)

    def model(self):
        native=np.repeat(np.eye(4)[None],2,axis=0)
        native[1,:3,:3]=[[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]
        m=SimpleNamespace(native=native,mesh=SimpleNamespace(center_mass=np.zeros(3)))
        loads=[np.array([[0.,0.,1.,-2.,0.,0.]]),np.array([[0.,0.,1.,0.,0.,0.]])]
        def rows(q):
            return [dict(index=k,T=installed_state(native,q,k),U=installed_state(native,q,k))
                    for k in q.active]
        m.base_oracle=SimpleNamespace(rows=rows,loads=loads)
        m.workpiece_height=lambda q,k:float(q.state_offsets[k,2])
        def validate(q,k):
            if m.workpiece_height(q,k)<-1e-9:raise ValueError('floor crossing')
        m.validate_placement=validate
        return m

    def test_original_pressure_points_select_only_the_required_state_lift(self):
        q=layout(2);m=self.model();r=floor_height_proposal(m,q)
        self.assertIsNotNone(r)
        np.testing.assert_allclose(r.state_offsets[:,2],[0.,2.],atol=1e-9)
        np.testing.assert_array_equal(q.state_offsets,np.zeros((2,3)))
        np.testing.assert_array_equal(r.placements,q.placements)

    def test_failed_conservative_height_proposal_returns_unknown(self):
        self.assertIsNone(floor_height_proposal(self.model(),layout(2),fixed_hosts=(1,)))


if __name__=='__main__':unittest.main()
