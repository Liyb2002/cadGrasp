"""Population fairness, actual one-step continuation, diversity and archive tests."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import scheduler as S, smc_search as SM, random_search as R, paths as PTH, contacts as I


def partial(covered,area=1.,**extra):
    return dict(random_sample_count=100,random_covered_count=covered,
        rounds=[dict(total_area_m2=area)],continuous_coverage_proved=False,**extra)


class SmcTests(unittest.TestCase):
    def test_weights_ignore_counterexamples_and_gain(self):
        rows=[partial(50,1,covered_count=500,sample_count=1000,gain=50),
              partial(90,2,covered_count=90,sample_count=100,gain=1)]
        weights=SM.resampling_weights(rows,1)
        self.assertGreater(weights[1],weights[0])
        rows[0].update(covered_count=0,sample_count=100000,gain=100000)
        np.testing.assert_array_equal(weights,SM.resampling_weights(rows,1))
        self.assertLessEqual(weights.max()/weights.min(),2)

    def test_area_only_breaks_equal_original_coverage(self):
        rows=[partial(80,1),partial(81,100),partial(81,2)]
        weights=SM.resampling_weights(rows,2)
        self.assertGreater(weights[1],weights[0])
        self.assertGreater(weights[2],weights[1])
        self.assertLessEqual(weights.max()/weights.min(),4)
        np.testing.assert_array_equal(SM.resampling_weights([partial(50),partial(50)],1),[.5,.5])
        with self.assertRaises(ValueError):
            SM.resampling_weights([partial(50),dict(partial(50),random_sample_count=99)],1)

    def test_elite_exploration_and_rng_replay(self):
        rows=[partial(50,contact_count=1),partial(90,contact_count=1)]
        a=SM.resample([3,7],rows,10,np.random.default_rng(12),1)
        b=SM.resample([3,7],rows,10,np.random.default_rng(12),1)
        self.assertEqual(a,b)
        self.assertEqual(a['selected_parent_ids'][0],7)
        self.assertEqual(a['slot_kinds'].count('uniform_exploration'),2)
        self.assertEqual(len(a['selected_parent_ids']),10)

    def test_sibling_exclusion_broadens_beyond_initial_top_five(self):
        rows=[dict(index=i,id=f'C{i}',covered_count=10-i,status='scored_joint') for i in range(8)]
        indices,p=S.M.top_k_distribution(rows,0,5,excluded_indices=range(5))
        self.assertEqual(indices,[5,6,7])
        self.assertAlmostEqual(float(p.sum()),1)

    def test_continue_evaluates_only_new_round_and_preserves_prefix(self):
        prefix=[dict(round=1,candidate_index=3,total_area_m2=1)]
        before=copy.deepcopy(prefix)
        calls=[]
        def score(n):calls.append(('score',n));return dict(elapsed_seconds=0)
        def choose(n):calls.append(('choose',n));return dict(winner=dict(id='C9',index=9,covered_percent=70))
        def optimize(n):
            calls.append(('optimize',n))
            return dict(covered_count=70,sample_count=100,adjusted=dict(area_m2=1,total_area_m2=2),
                selection=dict(converged=False),elapsed_seconds=0)
        rounds,status=S.iterate(score,choose,optimize,initial_rounds=prefix,stop_after_round=2)
        self.assertEqual(calls,[('score',2),('choose',2),('optimize',2)])
        self.assertEqual(status,'population_partial')
        self.assertEqual(rounds[0],prefix[0]);self.assertEqual(prefix,before)

    def test_lineage_paths_are_immutable_and_context_local(self):
        with PTH.trajectory(12),PTH.round_owners({'1':2,'2':8,'3':12}):
            self.assertEqual(PTH.folder('B','stage',1).parent.name,'trajectory_002')
            self.assertEqual(PTH.folder('B','stage',3).parent.name,'trajectory_012')
            self.assertEqual(PTH.stage_folder('B','stage').name,'trajectory_012')
        self.assertNotIn('trajectory_',str(PTH.folder('B','stage',1)))

    def test_population_expands_once_archives_and_skips_final_resampling(self):
        calls=[]
        def fake_chain(name,**kwargs):
            index=kwargs['particle_id'];depth=kwargs['stop_after_round'];prefix=kwargs['prefix']
            self.assertEqual(depth,1+len(prefix['rounds']) if prefix else 1)
            calls.append((index,depth,prefix['particle_id'] if prefix else None))
            excluded=kwargs['excluded_indices']
            candidate=next(i for i in range(40) if i not in excluded)
            rounds=copy.deepcopy(prefix['rounds']) if prefix else []
            rounds.append(dict(round=depth,candidate_index=candidate,total_area_m2=float(depth)))
            # One early success, other branches must still progress to layer 3.
            success=index==0
            if success:rounds[-1]['continuous_validation']='verified'
            status='continuous_contact_model_verified' if success else 'contact_limit_reached' if depth==3 else 'population_partial'
            out=PTH.stage_folder(name,S.OUTPUT_NAME);out.mkdir(parents=True,exist_ok=True)
            I.save_contacts(out/'final_contacts.npz',[])
            I.save(out/'insertion_directions.json',dict(complete=True,provenance=dict(inputs={},code={}),artifacts={}))
            result=dict(complete=True,particle_id=index,status=status,rounds=rounds,
                random_sample_count=100,random_covered_count=100 if success else depth*20+candidate,
                continuous_coverage_proved=success,continuous_domain_status='verified' if success else 'not_verified',
                contact_count=depth,elapsed_seconds=.01,wall_timings=[],round_trajectories=dict(prefix['round_trajectories']) if prefix else {},
                provenance=dict(inputs={},code={}),artifacts={f:I.sha256(out/f) for f in ['final_contacts.npz','insertion_directions.json']})
            result['round_trajectories'][str(depth)]=index
            I.save(out/'schedule.json',result)
            score=PTH.folder(name,S.C.OUTPUT_NAME,depth);score.mkdir(parents=True,exist_ok=True)
            I.save(score/'contributions.json',dict(complete=True,provenance=dict(inputs={},code={}),artifacts={},
                contributions=[dict(index=i,id=f'C{i}',covered_count=40-i,status='scored_joint') for i in range(40)]))
            return result
        parent=I.OUTPUTS/'test'/'pose_1'/'step3_scheculer';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp,patch.object(I,'OUTPUTS',Path(tmp)),\
                patch.object(S,'run_chain',side_effect=fake_chain),\
                patch.object(S.A,'read',return_value=dict(selection=dict(evaluated_sizes=7))):
            result=SM.run('test',R.configuration(mode='smc'),draw=False)
            self.assertEqual(len(calls),30)
            self.assertEqual(result['verified_archive'],[0])
            self.assertFalse(any(parent==0 for _,depth,parent in calls if depth>1))
            self.assertEqual(result['selected_trajectory'],0)
            self.assertEqual(result['evaluation_counts']['added_contact_optimizations'],30)
            self.assertEqual(result['evaluation_counts']['sizing_coverage_evaluations'],210)
            self.assertNotIn('resampling',result['layers'][-1])
            self.assertEqual(len({p['seed'] for p in result['particles']}),30)
            self.assertEqual(I.check_report(PTH.stage_folder('test',S.OUTPUT_NAME)/'schedule.json'),result)


if __name__=='__main__':unittest.main()
