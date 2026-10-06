"""Skip numerical candidate failures, never relabel them physical failures."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_connected import *
class SafeConnectedSearch(ConnectedSearch):
    def candidate(self,*args,**kwargs):
        try:return super().candidate(*args,**kwargs)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as error:
            self.connection_trace.append(dict(proposal=self.evaluations,label=kwargs.get('label','proposal'),status='numerically_unresolved',error=str(error)));return None

def safe_case(name,group):
    search=SafeConnectedSearch(name,group);began=time.monotonic();result=search.connected_search()
    if result is None:
        save(search.out/'data/unresolved.json',dict(complete=False,proposals=search.evaluations,trace=search.connection_trace));return dict(id=group['id'],passed=False,status='connected_search_unresolved')
    row=finish_connected(search,result,began);p=search.base/'step4/step4.2/data/report.json';r=json.loads(p.read_text());r['provenance']['code'].update(I.hashes([HERE/'step4.2/step42_connected_safe.py']));save(p,r);I.check_report(p);return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=2);args=parser.parse_args();groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'];groups=[g for g in groups if not args.sets or g['id'] in args.sets]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(safe_case,['B']*len(groups),groups))
    save(HERE/'output/B/data/connected_safe_batch.json',dict(complete=all(r['passed'] for r in rows),results=rows));print('SAFE CONNECTED TOTAL',sum(r['passed'] for r in rows),'/',len(rows),flush=True)
if __name__=='__main__':main()
