"""Numerical deadlines must escape solver fallback exception handlers."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42 import *
import signal
class EvaluationDeadline(BaseException):pass

def deadline(signum,frame):raise EvaluationDeadline()
class TimeLimitedSearch(Search):
    def __init__(self,name,group):
        super().__init__(name,group);self.timeouts=0
    def candidate(self,directions,common=None,label='proposal',proxy=True):
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(30.,10.*len(self.states)))
        if label.startswith('independent'):proxy=False
        try:
            result=super().candidate(directions,common,label,proxy)
            if self.best is not None:np.savez(self.out/'data/best_paths.npz',directions=self.best['directions'],counts=self.best['counts'])
            return result
        except EvaluationDeadline:
            self.timeouts+=1;self.trace.append(dict(proposal=self.evaluations,label=label,status='candidate_numerically_unresolved_deadline'));return None
        finally:signal.setitimer(signal.ITIMER_REAL,0.)
