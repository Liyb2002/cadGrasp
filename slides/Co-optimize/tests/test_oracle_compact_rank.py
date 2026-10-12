"""Only numerical demand-loss ties use the compact-layout proposal priority."""
from pathlib import Path
import sys,unittest
COOPT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COOPT/'helper_func'))
import _bootstrap
from oracle_search import candidate_rank

class CompactRankTests(unittest.TestCase):
    def test_roundoff_cannot_prefer_a_much_larger_layout(self):
        self.assertLess(candidate_rank(2e-33,10.,.33,.02),candidate_rank(1e-34,100.,1.33,.02))

    def test_materially_uncovered_demand_still_precedes_compactness(self):
        self.assertLess(candidate_rank(1e-8,100.,1.33,.02),candidate_rank(1e-4,10.,.33,.02))

if __name__=='__main__':unittest.main()
