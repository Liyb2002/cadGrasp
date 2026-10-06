"""Current B runner: certified cached results, then size-appropriate search."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_optimized import *
class SmallConnectedSearch(ConsistentSearch):
    def connected_search(self):return LocalConnectedSearch.connected_search(self)
def dispatch(name,group):
    if len(group['poses'])==6:return optimized_case(name,group)
    cache.install();search=SmallConnectedSearch(name,group);began=time.monotonic();result=search.connected_search()
    if result is None:return dict(id=group['id'],passed=False,status='connected_search_unresolved')
    row=finish_connected(search,result,began);p=search.base/'step4/step4.2/data/report.json';r=json.loads(p.read_text());r['motion_consistent_contacts_required']=True;r['provenance']['code'].update(I.hashes([HERE/f for f in ['step4.2/step42_connected_safe.py','step4.2/step42_connected_local.py','step4.2/step42_connected_independent.py','step4.2/step42_contact_consistency.py','helper_func/step42_geometry_cache.py','step4.2/step42_dispatch.py']]));save(p,r);I.check_report(p);return row
