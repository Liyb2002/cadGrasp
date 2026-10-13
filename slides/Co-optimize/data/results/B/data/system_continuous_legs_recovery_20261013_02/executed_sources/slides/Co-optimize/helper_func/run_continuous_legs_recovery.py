"""Separate solid-leg recovery experiment; preserve the initial all52 ledger."""
import sys
from pathlib import Path
import run_continuous_legs as controller
sys.path[:0]=[str(controller.COOPT/x)for x in ['step5.3','step5.2','helper_func']]
import continuous_legs_case
from continuous_legs_recovery import Builder
continuous_legs_case.Builder=Builder
controller.__file__=__file__

if __name__=='__main__':controller.main()
