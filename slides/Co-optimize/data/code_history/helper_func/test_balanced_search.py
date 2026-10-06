"""Regression: retained covering bank and bounded multi-start fallback."""
from test_concurrent_search import *
from physics_guided_balanced_search import balanced_search

class BalancedSearchTests(unittest.TestCase):
    def test_complete_original_covering_bank(self):
        with tempfile.TemporaryDirectory() as folder:
            search=Probe(folder);report=balanced_search(search)
        self.assertEqual(len(search.axes),160)
        self.assertTrue(report['passed'])

    def test_sampling_budget_does_not_remove_final_gradient_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            search=Probe(folder);search.proposals=search.max_proposals
            search.beam=[search.best];calls=[]
            search.refine=lambda result,steps,label:(calls.append(steps) or search.winner)
            report=balanced_search(search,iterations=7)
        self.assertEqual(calls,[7]);self.assertTrue(report['passed'])

if __name__=='__main__':unittest.main()
