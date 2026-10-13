"""Deletion cannot invent a connection, contact material or cavity fill."""
from pathlib import Path
import sys, unittest
import numpy as np
COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/p)for p in ['helper_func','step4.2','step5.2','step5.3']]
import _bootstrap
from connected_core import largest_material_component,legal_clip
from deterministic_base import box,union,volume,unpack,solid
from prune_support import Pruner,material_components
from test_support_pruning import Certificate
from types import SimpleNamespace
from unittest.mock import Mock,patch
from material_evaluator import MaterialSample


class ConnectedCoreTests(unittest.TestCase):
    def test_stage5_reads_core_policy_from_selected_step4_not_ground_report(self):
        import tempfile,json
        from stage5_shared import connected_core_policy
        with tempfile.TemporaryDirectory()as temporary:
            selected=Path(temporary);(selected/'data').mkdir()
            (selected/'data/report.json').write_text(json.dumps(dict(step4_contact_core_connected=True)))
            self.assertTrue(connected_core_policy(dict(selected=selected,record=dict(complete=True))))

    def test_removing_island_preserves_largest_component_without_addition(self):
        main=box(np.zeros(3),np.full(3,.03))
        island=box(np.array([.05,0.,0.]),np.array([.055,.005,.005]))
        kept,report=largest_material_component(union([main,island]))
        self.assertEqual(report['input_component_count'],2)
        self.assertEqual(len(material_components(kept)),1)
        self.assertAlmostEqual(volume(kept),volume(main))
        self.assertEqual(report['added_material_cm3'],0.)
        self.assertGreater(report['deleted_isolated_material_cm3'],0.)

    def test_largest_component_keeps_actual_cavity(self):
        main=box(np.zeros(3),np.full(3,.03))-box(np.full(3,.005),np.full(3,.025))
        kept,_=largest_material_component(main)
        self.assertLessEqual(volume(kept-main),1e-13)
        self.assertAlmostEqual(volume(kept),volume(main))

    def test_sampled_core_pass_does_not_replace_actual_contact_certificate(self):
        from test_material_search import sample,layout
        item=sample(layout(2))
        item.result.update(connected_contact_core=dict(connected=True,downward_resultant_passed=True),
            force_material_geometry_certified=False)
        self.assertFalse(item.completed)
        item.result['force_material_geometry_certified']=True
        self.assertTrue(item.completed)

    def test_leg_reference_cannot_rescue_uncertified_core_with_optional_override(self):
        from material_evaluator import MaterialEvaluator
        from test_material_search import sample,layout
        item=sample(layout(2));item.witness=object()
        item.result.update(connected_contact_core=dict(connected=True,downward_resultant_passed=True),
            force_material_geometry_certified=False)
        evaluator=MaterialEvaluator(SimpleNamespace(model=SimpleNamespace(require_connected_core=True)))
        evaluator.space_candidate=lambda value:True
        self.assertIs(evaluator.construct(item),item)
        self.assertEqual(evaluator.stats['constructions'],0)
        self.assertEqual(evaluator.build_attempts,set())

    def test_actual_polygon_measurement_keeps_core_and_original_failures(self):
        from material_evaluator import MaterialEvaluator
        from test_material_search import sample,layout
        item=sample(layout(1),passed=False)
        core=object();proof=dict(masks={0:np.array([False])})
        evaluator=MaterialEvaluator(SimpleNamespace(model=SimpleNamespace(require_connected_core=True)))
        with patch('connected_core.prepare',return_value=(core,{})),\
             patch('material_force.certify_material_contacts',return_value=proof):
            evaluator.certify_core(item)
        self.assertIs(item.nominal,core)
        self.assertFalse(item.result['force_material_geometry_certified'])
        self.assertEqual(item.result['grid_failed_demands'],1)
        self.assertFalse(item.completed)

    def test_sampled_force_screen_excludes_deleted_component_contacts(self):
        from oracle_search import OracleModel
        from test_material_search import layout
        model=object.__new__(OracleModel);q=layout(1)
        core=unpack(box(np.zeros(3),np.full(3,.01)))
        model.points=np.array([[0.,.005,.005],[.05,.005,.005]])
        model.point_normals=np.array([[1.,0.,0.],[1.,0.,0.]])
        model.core_force_context=(q.key(),core)
        with patch('whole_search.fast_search.FastModel.contact_flags',return_value={0:np.ones(2,bool)}):
            np.testing.assert_array_equal(model.contact_flags(q)[0],[True,False])

    def test_48_cold_slots_share_initial_layout_and_preserve_random_draws(self):
        from population32 import registered_population
        from test_population32 import SmallModel,initial
        model=SmallModel();q=initial();q.directions[0]=[1.,0.,0.]
        first=np.random.default_rng(42);second=np.random.default_rng(42)
        rows,_=registered_population(model,q,first,size=48,identical=True)
        registered_population(model,q,second,size=48,identical=False)
        self.assertEqual(len(rows),48)
        self.assertTrue(all(row.key()==q.key()for row in rows))
        self.assertEqual(first.random(),second.random())

    def test_failed_connection_evidence_survives_a_cheap_preview(self):
        from material_evaluator import MaterialEvaluator
        from test_material_search import layout
        q=layout(1)
        quick=dict(passed=True,enclosure_passed=True,status='enclosure_witness')
        failure=dict(passed=False,enclosure_passed=True,connected_passed=False,status='no_common_component_in_window')
        model=SimpleNamespace(validate_placement=Mock(),base_oracle=SimpleNamespace(query=Mock(return_value=quick)),
            proxy=Mock(return_value={'loss':0.}))
        evaluator=MaterialEvaluator(SimpleNamespace(model=model));evaluator.connection_results[q.key()]=failure
        with patch('material_evaluator.lower_box',return_value={'volume_litres':1.}):
            item=evaluator.preview(q)
        self.assertIs(item.oracle,failure)

    def test_exclusion_boundary_cleanup_only_deletes_and_preserves_tolerance(self):
        from test_material_search import layout
        q=layout(1);original=box(np.zeros(3),np.full(3,.03))
        legal=box(np.zeros(3),np.array([.03-1e-8,.03,.03]))
        model=SimpleNamespace(connected_core_repairs={q.key():(unpack(original),dict(component_count=1))})
        mesh,record=legal_clip(model,q,dict(core=unpack(legal),witness=unpack(legal)))
        self.assertLessEqual(volume(solid(mesh)-original),1e-10)
        self.assertLessEqual(volume(solid(mesh)-legal),1e-10)
        self.assertTrue(mesh.is_watertight)
        self.assertEqual(record['oracle_boundary_cleanup']['acceptance_tolerance_m3_unchanged'],1e-10)
        self.assertTrue(record['oracle_boundary_cleanup']['requires_actual_contact_recertification'])

    def test_leg_boundary_cleanup_cannot_fill_forbidden_cavity(self):
        from flat_tapered_base import Builder
        outer=box(np.zeros(3),np.full(3,.03))
        cavity=box(np.full(3,.005),np.full(3,.025))
        material=outer-cavity
        builder=object.__new__(Builder);builder.allowed=material
        with patch('material_builder_closed.regularize_external_boundaries',return_value=(outer,[{}])):
            kept=builder.regularize_material(material)
        self.assertLessEqual(volume(kept-material),1e-10)
        self.assertTrue(builder.boundary_cleanup_rejected)

    def test_bbx_cut_is_prioritized_and_preserves_required_port(self):
        body=box(np.zeros(3),np.array([.06,.02,.02]))
        port=box(np.array([0.,0.,0.]),np.array([.01,.02,.02]))
        objects=np.array([[0.,0.,0.],[.03,.02,.02]])
        row=dict(fixture_transform=np.eye(4),object_transform=np.eye(4))
        _,report,_=Pruner(unpack(body),unpack(body),Certificate([port]),grids=(),
            rows=[row],object_vertices=objects,require_connected=True).run()
        self.assertTrue(report['passed']);self.assertTrue(report['connected'])
        self.assertLess(report['retained_bbx_l'],report['input_bbx_l'])
        self.assertTrue(report['history'][0]['operation'].startswith('bbx_trim'))
        self.assertLessEqual(report['added_material_cm3'],1e-7)


if __name__=='__main__':unittest.main()
