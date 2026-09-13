"""Check phase order, post-optimization completion and termination."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import scheduler as S


class SchedulerTests(unittest.TestCase):
    def callbacks(self, counts):
        calls=[]
        def score(r):
            calls.append((r, '3.1'))
            return {'elapsed_seconds':0}
        def choose(r):
            calls.append((r, '3.2'))
            return {'winner':dict(id=f'C{r:03d}',index=r-1,covered_percent=100.)}
        def optimize(r):
            calls.append((r, '3.3'))
            return dict(covered_count=counts[r-1],sample_count=100,
                adjusted=dict(area_m2=1.,total_area_m2=float(r)),
                selection=dict(converged=True),elapsed_seconds=0)
        return calls,score,choose,optimize

    def test_selection_at_100_percent_does_not_stop_before_optimization(self):
        calls,score,choose,optimize=self.callbacks([70,100])
        rounds,status=S.iterate(score,choose,optimize)
        self.assertEqual(calls,[(1, '3.1'),(1, '3.2'),(1, '3.3'),(2, '3.1'),(2, '3.2'),(2, '3.3')])
        self.assertEqual(status,'sampled_complete')
        self.assertEqual(rounds[0]['after_optimization_covered_percent'],70)
        self.assertEqual(len(rounds),2)

    def test_third_contact_can_complete_but_no_fourth_is_scored(self):
        calls,score,choose,optimize=self.callbacks([50,60,100])
        rounds,status=S.iterate(score,choose,optimize,lambda n,r: {'status':'verified'})
        self.assertEqual(status,'continuous_contact_model_verified')
        self.assertEqual(len(rounds),3)
        self.assertEqual(calls[-3:],[(3,'3.1'),(3,'3.2'),(3,'3.3')])

    def test_three_contact_limit_is_a_completed_failure(self):
        calls,score,choose,optimize=self.callbacks([50,60,75,100])
        rounds,status=S.iterate(score,choose,optimize)
        self.assertEqual(status,'contact_limit_reached')
        self.assertEqual(len(rounds),3)
        self.assertNotIn((4,'3.1'),calls)

    def test_counterexample_on_third_contact_does_not_buy_a_fourth(self):
        calls,score,choose,optimize=self.callbacks([50,60,100])
        rounds,status=S.iterate(score,choose,optimize,lambda n,r:{'status':'counterexample'})
        self.assertEqual(status,'contact_limit_reached')
        self.assertEqual(rounds[-1]['continuous_validation'],'counterexample')
        self.assertNotIn((4,'3.1'),calls)

    def test_exhaustion_after_several_contacts_is_failure(self):
        calls,score,choose,optimize=self.callbacks([50,60,70,70])
        rounds,status=S.iterate(score,lambda r: choose(r) if r<=2 else {'winner':None},optimize)
        self.assertEqual(status,'candidates_exhausted')
        self.assertEqual(len(rounds),2)
        self.assertNotIn((3,'3.3'),calls)

    def test_duplicate_candidate_is_an_error_not_an_infinite_loop(self):
        _,score,choose,optimize=self.callbacks([50,60])
        with self.assertRaisesRegex(RuntimeError,'already fixed'):
            S.iterate(score,lambda r: choose(1),optimize)

    def test_no_candidate_terminates_without_optimizing(self):
        calls,score,choose,optimize=self.callbacks([50])
        rounds,status=S.iterate(score,lambda r: {'winner':None},optimize)
        self.assertEqual(status,'candidates_exhausted')
        self.assertEqual(rounds,[])
        self.assertEqual(calls,[(1, '3.1')])

    def test_zero_progress_does_not_prune_complementary_later_contacts(self):
        calls,score,choose,optimize=self.callbacks([0,0,100])
        rounds,status=S.iterate(score,choose,optimize)
        self.assertEqual(status,'sampled_complete')
        self.assertEqual(len(rounds),3)

    def test_continuous_counterexample_returns_to_scoring(self):
        calls,score,choose,optimize=self.callbacks([100,100])
        def validate(number,result):
            calls.append((number, '3'))
            return {'status':'counterexample' if number==1 else 'verified'}
        rounds,status=S.iterate(score,choose,optimize,validate)
        self.assertEqual(status,'continuous_contact_model_verified')
        self.assertEqual(calls,[(1, '3.1'),(1, '3.2'),(1, '3.3'),(1, '3'),(2, '3.1'),(2, '3.2'),(2, '3.3'),(2, '3')])
        self.assertEqual(len(rounds),2)

    def test_unresolved_continuous_check_is_not_success(self):
        _,score,choose,optimize=self.callbacks([100])
        _,status=S.iterate(score,choose,optimize,lambda n,r:{'status':'unresolved'})
        self.assertEqual(status,'continuous_validation_inconclusive')

    def test_lost_insertion_direction_stops_before_force_success(self):
        calls,score,choose,optimize=self.callbacks([100])
        def changed(number):
            return dict(optimize(number), insertion=dict(all_contacts_have_certified_direction=False))
        def validate(number,result):
            self.fail('Force completion must not override lost insertion directions')
        rounds,status=S.iterate(score,choose,changed,validate)
        self.assertEqual(status,'no_common_insertion_direction_after_optimization')
        self.assertEqual(len(rounds),1)
        self.assertFalse(rounds[0]['insertion']['all_contacts_have_certified_direction'])


if __name__ == '__main__':
    unittest.main()
