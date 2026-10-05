"""Bound near-degenerate candidate evaluations; refine independent exits."""
from step42 import *
import signal,shutil
class CandidateTimeout(RuntimeError):pass

def timed_out(signum,frame):raise CandidateTimeout('Candidate numerical classification time budget reached')
class RefineSearch(Search):
    def __init__(self,name,group):
        super().__init__(name,group);self.out=self.base/'step4/step4.2/data/refinement_run';(self.out/'data').mkdir(parents=True,exist_ok=True);self.timeouts=0
    def candidate(self,directions,common=None,label='proposal',proxy=True):
        signal.signal(signal.SIGALRM,timed_out);signal.setitimer(signal.ITIMER_REAL,5.)
        if label.startswith('independent'):proxy=False
        try:
            result=super().candidate(directions,common,label,proxy)
            if self.best is not None:
                np.savez(self.out/'data/best_paths.npz',directions=self.best['directions'],counts=self.best['counts'])
            return result
        except CandidateTimeout:
            self.timeouts+=1;self.trace.append(dict(proposal=self.evaluations,label=label,status='candidate_timeout_unresolved'));return None
        finally:signal.setitimer(signal.ITIMER_REAL,0.)

if __name__=='__main__':
    g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+4+7+12+21+27');began=time.monotonic();search=RefineSearch('B',g);result=search.search()
    if result is None:
        save(search.out/'data/unresolved.json',dict(complete=False,proposals=search.evaluations,timeouts=search.timeouts,best_counts=None if search.best is None else search.best['counts']));print('REFINEMENT UNRESOLVED',flush=True);raise SystemExit(2)
    row=finish(search,result,began);report_path=search.out/'data/report.json';report=json.loads(report_path.read_text());report['provenance']['code'].update(I.hashes([HERE/'step42_refine.py']));report['candidate_timeouts_unresolved']=search.timeouts;report['search_method']+='; numerical candidate time budget, with independent actual-cut refinement';save(report_path,report);I.check_report(report_path)
    target=search.base/'step4/step4.2'
    for file in search.out.iterdir():
        if file.is_dir():
            for entry in file.iterdir():shutil.copy2(entry,target/'data'/entry.name)
        else:shutil.copy2(file,target/file.name)
    I.check_report(target/'data/report.json');print('REFINEMENT PASS',row,'timeouts',search.timeouts,flush=True)
