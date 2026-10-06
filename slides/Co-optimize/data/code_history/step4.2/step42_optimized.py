"""Resume difficult sets with exact sweep caching and saved best directions."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_value_guided import *
from step42_balanced import BalancedSearch
import step42_geometry_cache as cache

def optimized_case(name,group):
    cache.install();last=group['id']=='pose1+4+7+12+21+27';base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']/'step4/step4.2'
    checkpoint=base/'data/connected_run/data'/('balanced_best_paths.npz' if last else 'value_checkpoint.json')
    warm=None
    if checkpoint.exists():
        warm=np.load(checkpoint)['directions'].copy() if last else np.array(json.loads(checkpoint.read_text())['directions'])
    search=BalancedSearch(name,group) if last else ValueGuidedSearch(name,group)
    if warm is not None:search.warm=warm;np.savez(search.out/'data/warm_paths.npz',directions=warm)
    began=time.monotonic();result=search.connected_search()
    if result is None:
        counts=getattr(search,'balanced_best',{}).get('counts') if last else getattr(search,'value_best',{}).get('counts');save(search.out/'data/optimized_unresolved.json',dict(complete=False,proposals=search.evaluations,counts=counts,cache=cache.stats));print('OPTIMIZED UNRESOLVED',group['id'],counts,flush=True);return dict(id=group['id'],passed=False,proposals=search.evaluations,counts=counts)
    row=finish_connected(search,result,began);p=base/'data/report.json';r=json.loads(p.read_text());r['motion_consistent_contacts_required']=True;r['exact_sweep_cache']=cache.stats;r['provenance']['code'].update(I.hashes([HERE/f for f in ['step4.2/step42_connected_safe.py','step4.2/step42_connected_local.py','step4.2/step42_connected_independent.py','step4.2/step42_contact_consistency.py','step4.2/step42_topology.py','step4.2/step42_topology_common.py','step4.2/step42_value_guided.py','step4.2/step42_balanced.py','helper_func/step42_geometry_cache.py','step4.2/step42_optimized.py']]))
    if not last:r['artifacts']['component_values.json']=I.sha256(p.parent/'component_values.json')
    save(p,r);I.check_report(p);return row
if __name__=='__main__':
    groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'];groups=[g for g in groups if g['id'] in ['pose1+6+11+13+14+17','pose1+4+7+12+21+27']]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(optimized_case,['B']*len(groups),groups))
    save(HERE/'output/B/data/optimized_batch.json',dict(complete=all(r['passed'] for r in rows),results=rows));print('OPTIMIZED TOTAL',sum(r['passed'] for r in rows),len(rows),flush=True)
