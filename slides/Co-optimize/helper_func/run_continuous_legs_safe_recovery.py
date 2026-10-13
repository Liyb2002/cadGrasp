"""Independent recovery with safe routing and alternative legal floor ends."""
import sys
import run_continuous_legs as controller
sys.path[:0]=[str(controller.COOPT/x)for x in ['step5.3','step5.2','helper_func']]
import continuous_legs
import continuous_legs_recovery
import continuous_legs_case
from continuous_legs_safe_recovery import Builder,RouteGuide
continuous_legs.RouteGuide=RouteGuide
continuous_legs_recovery.RouteGuide=RouteGuide
continuous_legs_case.Builder=Builder
controller.__file__=__file__

if __name__=='__main__':controller.main()
