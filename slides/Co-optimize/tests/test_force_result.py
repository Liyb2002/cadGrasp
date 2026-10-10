"""Publishing preserves force decisions without geometry recovery or replay."""
import sys,time,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from whole_search.fast_search import FastModel
from continuous_support.stable_gradient import StableGradientSearch
from whole_search.checked_solution import check_force_result
from co_common import save


class ForceResultTests(unittest.TestCase):
    def fixture(self,out,passed=True):
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[0.,0.,1.]]*2),np.array([0,1]),(0,1))
        mask=np.ones(32768,bool);mask[-1]=passed
        current=dict(layout=layout,serial=1,masks={0:np.ones(32768,bool),1:mask},
            supplies={0:np.zeros((1,7)),1:np.zeros((1,7))},counts={'0':32768,'1':int(mask.sum())},volume_cm3=10.)
        model=SimpleNamespace(name='B',poses=['a','b'],native=np.repeat(np.eye(4)[None],2,axis=0),
            allow_z_translation=True,workpiece_height=lambda q,k:0.,floor_signature=lambda q,k:True,
            volume_delta=SimpleNamespace(seconds=0.),refresh_volume=Mock(),work_rays=[],sample_calls=1,
            exact_calls=0,proxy_calls=0,timing={},contact_delta=SimpleNamespace(pair_updates=0),
            inputs=[],startup_sources={},evaluate=Mock(side_effect=AssertionError('No force replay')),
            exact=Mock(side_effect=AssertionError('No exact geometry validation')))
        return StableGradientSearch(model,out),current

    def test_save_reuses_existing_masks_and_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            search,current=self.fixture(Path(folder));before=current['layout'].key()
            with patch('continuous_support.force_result.export_nominal_mesh',return_value=9.) as export:
                report=search.save(current,'whole_gradient',time.monotonic(),{},passed=True)
            self.assertTrue(report['force_passed']);self.assertEqual(report['validation_seconds'],0.)
            self.assertFalse(report['full_demand_recheck_run']);self.assertFalse(report['geometry_recovery_run'])
            self.assertEqual(current['layout'].key(),before);export.assert_called_once()
            search.model.evaluate.assert_not_called();search.model.exact.assert_not_called()
            with np.load(Path(folder)/'b_force.npz') as saved:np.testing.assert_array_equal(saved['mask'],current['masks'][1])

    def test_mesh_failure_does_not_change_pass_or_trigger_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            search,current=self.fixture(Path(folder))
            with patch('continuous_support.force_result.export_nominal_mesh',side_effect=OSError('Export failed')) as export:
                report=search.save(current,'whole_gradient',time.monotonic(),{},passed=True)
            self.assertTrue(report['force_passed']);self.assertFalse(report['mesh_exported'])
            self.assertIn('Export failed',report['mesh_export_error']);export.assert_called_once()
            search.model.evaluate.assert_not_called();search.model.exact.assert_not_called()

    def test_uncovered_demand_cannot_be_published_as_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            search,current=self.fixture(Path(folder),False)
            with patch('continuous_support.force_result.export_nominal_mesh') as export:
                report=search.save(current,'whole_gradient',time.monotonic(),{},passed=True)
            self.assertFalse(report['force_passed']);export.assert_not_called()

    def test_real_two_pose_force_state_is_saved_without_new_evaluation(self):
        model=FastModel(['pose_1','pose_2'],'B',allow_z_translation=True)
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([r[:3,:3].T@np.array([0.,0.,1.]) for r in model.native]),np.arange(2),(0,1))
        current=model.evaluate(layout);model.commit(current);calls=model.sample_calls
        with tempfile.TemporaryDirectory() as folder:
            search=StableGradientSearch(model,Path(folder))
            with patch('continuous_support.force_result.export_nominal_mesh',return_value=10.), \
                 patch.object(model,'evaluate',side_effect=AssertionError('No replay')), \
                 patch.object(model,'exact',side_effect=AssertionError('No exact')):
                report=search.save(current,'whole_gradient',time.monotonic(),{})
            self.assertEqual(model.sample_calls,calls);self.assertEqual(model.exact_calls,0)
            self.assertEqual(report['counts'],{model.poses[k]:int(current['masks'][k].sum()) for k in layout.active})
            stored=dict(report,artifacts={'../'+k:v for k,v in report['artifacts'].items()})
            path=Path(folder)/'data/report.json';save(path,stored)
            checked=check_force_result(path)
            self.assertTrue(checked['force_passed']);self.assertFalse(checked['geometry_verified'])


if __name__=='__main__':unittest.main()
