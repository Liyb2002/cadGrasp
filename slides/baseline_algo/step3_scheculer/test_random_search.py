"""Stochastic selection, independent state, checkpoint compatibility and winner choice."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import scheduler as S, random_search as R, paths as PTH, run_all, contacts as I


class RandomSearchTests(unittest.TestCase):
    def rows(self, gains):
        return [dict(index=i, id=f'C{i:03d}', covered_count=100+gain,
                     status='scored_joint') for i,gain in enumerate(gains)]

    def test_probabilities_use_marginal_not_total_coverage(self):
        indices,p=S.M.top_k_distribution(self.rows([2,40,20,8,30,1]),100)
        self.assertEqual(indices,[1,4,2,3,0])
        np.testing.assert_allclose(p,[.4,.3,.2,.08,.02])

    def test_zero_gain_preserves_complementary_candidates_and_filters_ineligible(self):
        rows=self.rows([0]*6)
        rows[0]['status']='rejected_rest_equilibrium'
        rows[0]['covered_count']=None
        indices,p=S.M.top_k_distribution(rows,100)
        self.assertEqual(indices,[1,2,3,4,5])
        np.testing.assert_allclose(p,[.2]*5)
        self.assertEqual(S.M.top_k_distribution([],100)[0],[])

    def test_zero_gain_has_zero_probability_when_positive_gain_exists(self):
        _,p=S.M.top_k_distribution(self.rows([10,0]),100)
        np.testing.assert_array_equal(p,[1.,0.])

    def test_trajectory_paths_restore_after_exception(self):
        original=PTH.folder('test','step3.2_select_contact',1)
        with PTH.trajectory(0):
            first=PTH.folder('test','step3.2_select_contact',1)
            with self.assertRaises(ValueError),PTH.trajectory(1):
                self.assertNotEqual(first,PTH.folder('test','step3.2_select_contact',1))
                raise ValueError('exit')
            self.assertEqual(first,PTH.folder('test','step3.2_select_contact',1))
        self.assertEqual(original,PTH.folder('test','step3.2_select_contact',1))

    def test_independent_streams_reproduce_without_call_order_dependence(self):
        def draws(i):
            return np.random.default_rng(R.chain_seed(7,i)).random(3)
        np.testing.assert_array_equal(draws(3),draws(3))
        self.assertFalse(np.array_equal(draws(3),draws(4)))

    def test_verified_small_area_wins_over_full_sample_only(self):
        def result(passed,covered,area):
            return dict(continuous_coverage_proved=passed,random_covered_count=covered,
                        contact_count=2,rounds=[dict(total_area_m2=area)])
        results=[result(False,100,1),result(True,100,4),result(True,100,2)]
        self.assertEqual(min(range(3),key=lambda i:R.result_key(results[i])),2)
        self.assertLess(R.result_key(result(False,90,3)),R.result_key(result(False,80,1)))

    def test_resume_does_not_reuse_another_seed(self):
        checkpoint=dict(search_config=R.configuration(seed=12),status='candidates_exhausted',
                        continuous_coverage_proved=False)
        with patch.object(run_all,'completed_schedule',return_value=checkpoint), \
                patch.object(run_all.subprocess,'run') as invoke:
            invoke.return_value.returncode=0
            run_all.run(['test'],from_step=3,through_step=3,resume=True,search_seed=13)
            commands=[c.args[0] for c in invoke.call_args_list]
            first=commands[0]
            self.assertTrue(first[1].endswith('/scheduler.py'))
            self.assertEqual(first[first.index('--seed')+1],'13')

    def test_bad_budget_is_rejected_before_running_stages(self):
        with patch.object(run_all.subprocess,'run') as invoke:
            with self.assertRaises(ValueError):
                run_all.run(['test'],sizing_budget=2)
            invoke.assert_not_called()

    def test_ten_chains_publish_one_winner_and_resume_without_running_again(self):
        def chain(name,**kwargs):
            out=PTH.stage_folder(name,S.OUTPUT_NAME)
            out.mkdir(parents=True,exist_ok=True)
            I.save_contacts(out/'final_contacts.npz',[])
            I.save(out/'insertion_directions.json',dict(complete=True,contacts=[],
                provenance=dict(inputs={},code={}),artifacts={}))
            result=dict(object=name,complete=True,status='candidates_exhausted',rounds=[],
                continuous_coverage_proved=False,continuous_domain_status='not_verified',
                contact_count=0,random_covered_count=0,search_mode='top5',
                search_config={k:kwargs[k] for k in ('seed','top_k','sizing_sweeps','sizing_budget')},
                wall_timings=[],provenance=dict(inputs={},code={}),
                artifacts={f:I.sha256(out/f) for f in ['final_contacts.npz','insertion_directions.json']})
            I.save(out/'schedule.json',result)
            return result
        # Temporary results must be under ROOT for repository-relative provenance.
        parent=I.OUTPUTS/'test'/'pose_1'/'step3_scheculer'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp,patch.object(I,'OUTPUTS',Path(tmp)), \
                patch.object(S,'run_chain',side_effect=chain) as invoke:
            result=S.run('test',draw=False)
            self.assertEqual(invoke.call_count,10)
            self.assertEqual(len(result['trajectories']),10)
            self.assertEqual(len({r['seed'] for r in result['trajectories']}),10)
            self.assertEqual(result['selected_trajectory'],0)
            self.assertEqual(I.check_report(PTH.stage_folder('test',S.OUTPUT_NAME)/'schedule.json'),result)
            resumed=S.run('test',draw=False,resume=True)
            self.assertEqual(invoke.call_count,10)
            self.assertEqual(resumed['reused_trajectories'],list(range(10)))
            S.run('test',draw=False,resume=True,seed=1)
            self.assertEqual(invoke.call_count,20)


if __name__=='__main__': unittest.main()
