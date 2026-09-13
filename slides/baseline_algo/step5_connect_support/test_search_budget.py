"""A hard independent geometry problem cannot consume every sibling's budget."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step5_connect_support import test_fixed_feet as fixtures
from step5_connect_support import fixed_feet as H,routing as T
from step5_connect_support.bounded_connect import bounded_one


class SearchBudgetTests(unittest.TestCase):
    def test_exhaustion_is_inconclusive_and_next_support_has_its_own_budget(self):
        mesh,contact,direction,foot,work=fixtures.FixedFeetTests().setup_case()
        original=T.Router.clear
        contacts=[{**contact,'candidate_id':cid} for cid in ('limited','good')]
        feet=[{**foot,'candidate_id':c['candidate_id']} for c in contacts]
        def build(mesh,c,d,f,depth,w):
            return bounded_one(mesh,c,d,f,depth,w,max_edges=1 if c['candidate_id']=='limited' else 500)
        records,modules=H.search(mesh,contacts,[direction]*2,feet,.01,work,builder=build)
        self.assertEqual(records[0]['status'],'individual_search_budget_exhausted')
        self.assertFalse(records[0]['failure_is_global_impossibility_proof'])
        self.assertEqual(records[0]['search_budget']['new_edge_checks'],1)
        self.assertTrue(records[1]['passed'])
        self.assertEqual([m['plan']['candidate_id'] for m in modules],['good'])
        self.assertIs(T.Router.clear,original)


if __name__=='__main__':unittest.main()
