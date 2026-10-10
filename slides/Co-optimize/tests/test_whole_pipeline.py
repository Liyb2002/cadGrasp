"""Whole progress and the real-volume guard against optimistic sampling."""
import sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from whole_pipeline import WholeSearch,prepare_case,run_case,initialization_trials,continue_final_validation
from whole_search.fast_search import FastReuseSearch
from whole_search.model import Layout,Model


def result(index,volume,masks):
    n=len(masks);q=Layout(np.repeat(np.eye(4)[None],n,axis=0),
        np.repeat([[0.,0.,1.]],n,axis=0),np.arange(n),tuple(range(n)))
    q.directions[0]=[np.sin(index*.01),0,np.cos(index*.01)]
    return dict(layout=q,serial=index,volume_cm3=volume,masks=masks,
        counts={str(k):int(v.sum()) for k,v in masks.items()},
        maximum_projected_footprint_m2=.01,
        actual_work_surface_checks=[dict(passed=True) for k in masks])


class WholePipelineTests(unittest.TestCase):
    def test_opposite_floor_normals_get_positive_margin_without_moving_poses(self):
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),
            np.array([[1.,0.,0.],[1.,0.,0.]]),np.arange(2),(0,1))
        normals=np.array([[0.,0.,1.],[0.,0.,-1.]])
        model=SimpleNamespace(floor_normal=lambda q,k:normals[k])
        upward=initialization_trials(model,layout)[1]
        self.assertTrue(np.all(np.einsum('ij,ij->i',upward.directions,normals)>0))
        np.testing.assert_allclose(np.linalg.norm(upward.directions,axis=1),1.)
        np.testing.assert_array_equal(upward.placements,layout.placements)
        np.testing.assert_array_equal(upward.hosts,layout.hosts)
        np.testing.assert_array_equal(layout.directions,[[1.,0.,0.],[1.,0.,0.]])

    def test_successful_child_exit_cannot_replace_a_failed_force_certificate(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);out=base/'step4/step4.2';out.mkdir(parents=True)
            (out/'sampled_layout.npz').touch()
            (out/'search_report.json').write_text(json.dumps(dict(sampled_force_passed=True,poses=['pose_1'])))
            with patch('whole_pipeline.subprocess.run',return_value=SimpleNamespace(returncode=0)),\
                 patch('whole_pipeline.I.check_report',return_value=dict(schema='whole_step4_v1',force_exit_work_passed=False)):
                with self.assertRaises(AssertionError):
                    continue_final_validation('B',dict(id='pose1',poses=['pose_1']),base,1)

    def test_stale_batch_cannot_overwrite_current_case(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir()
            (root/'data/whole_step4_active_run.json').write_text(json.dumps(dict(run_token='current')))
            stage=root/'pose1+2/step4';stage.mkdir(parents=True)
            sentinel=stage/'run.log';sentinel.write_text('current result\n')
            with patch('whole_pipeline.output_root',return_value=root):
                row=run_case(('B',dict(id='pose1+2',poses=['pose_1','pose_2']),dict(run_token='stale')))
            self.assertEqual(row['status'],'cancelled_previous_run')
            self.assertEqual(sentinel.read_text(),'current result\n')
            self.assertFalse((stage/'data/run_schema.json').exists())

    def test_partial_current_initialization_can_resume_before_report_exists(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stage=root/'pose1+2/step4'
            (stage/'data').mkdir(parents=True)
            (stage/'data/run_schema.json').write_text(json.dumps(dict(schema='whole_step4_v1')))
            (stage/'run.log').write_text('MODEL 2 poses; interrupted construction\n')
            with patch('whole_pipeline.output_root',return_value=root):
                returned=prepare_case('B',dict(id='pose1+2',poses=['pose_1','pose_2']))
            self.assertEqual(returned,root/'pose1+2')
            self.assertIn('interrupted',(stage/'run.log').read_text())
            self.assertTrue((stage/'step4.1/data').exists())
            self.assertFalse((root/'_history').exists())

    def test_same_failed_count_can_improve_hardest_gap(self):
        a=result(0,20.,{0:np.array([True,False])})
        b=result(1,21.,{0:np.array([True,False])})
        targets=[(0,1)];seen=[]
        def proxy(q,bank):
            seen.append(bank)
            loss=1. if q.key()==a['layout'].key() else .25
            return dict(loss=loss,sum_loss=loss)
        search=WholeSearch(SimpleNamespace(proxy=proxy),None)
        search.current_guidance_targets=targets
        self.assertLess(search.score(b),search.score(a))
        self.assertTrue(all(bank is targets for bank in seen))

    def test_whole_improvement_can_trade_an_individual_load(self):
        a=result(0,10.,{0:np.array([True,True]),1:np.array([False,False])})
        b=result(1,11.,{0:np.array([True,False]),1:np.array([True,True])})
        search=WholeSearch(SimpleNamespace(proxy=lambda q,t:dict(loss=0.,sum_loss=0.)),None)
        self.assertLess(search.score(b),search.score(a))

    def test_actual_growth_rejects_optimistic_candidate_and_keeps_baseline_slot(self):
        baseline=result(0,100.,{0:np.ones(2,bool)})
        optimistic=result(1,80.,{0:np.ones(2,bool)})
        checked_growth=dict(optimistic,volume_cm3=110.)
        calls=[]
        def exact(q):
            calls.append(q.key())
            return baseline if q.key()==baseline['layout'].key() else checked_growth
        model=SimpleNamespace(poses=['pose_1'],refresh_volume=lambda r:None,
            timing={},volume_delta=SimpleNamespace(seconds=0.),work_rays=[],
            sample_calls=2,exact_calls=0,proxy_calls=0,
            contact_delta=SimpleNamespace(pair_updates=0),complete_sampled_states={},exact=exact)
        with tempfile.TemporaryDirectory() as directory:
            search=FastReuseSearch(model,Path(directory));search.process_snapshot=lambda r,p:None
            search.initial_feasible_result=baseline;search.initial_feasible_volume_cm3=100.
            search.final_candidates=2
            with patch.object(Model,'save',side_effect=lambda m,r,o,e:dict(result=r,**e)):
                saved=search.save(optimistic,'whole',time.monotonic(),{})
        self.assertEqual(saved['result']['volume_cm3'],100.)
        self.assertEqual(calls,[optimistic['layout'].key(),baseline['layout'].key()])
        self.assertTrue(saved['final_validation_attempts'][0]['material_regression_against_feasible_step41'])
        self.assertTrue(saved['selected_earlier_sampled_state'])


if __name__=='__main__':unittest.main()
