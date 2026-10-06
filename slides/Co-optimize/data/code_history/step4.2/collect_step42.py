"""Aggregate current construction records; never restart or recheck solids."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
manifest=ROOT/'objects/B/pose_sets.json';groups=json.loads(manifest.read_text())['sets'];rows=[];inputs=[manifest]
for g in groups:
 p=HERE/'output/B'/g['id']/'step4/step4.2/data/report.json';r=I.check_report(p);inputs.append(p);ok=bool(r['passed'] and r.get('connectivity_required') and r['remaining_component_count']==1)
 rows.append(dict(id=g['id'],passed=ok,status=r['status'],poses=len(g['poses']),component_count=r['remaining_component_count'],proposals=r['proposal_count'],pruned_volume_cm3=r.get('pruned_unnecessary_volume_cm3'),seconds=r['seconds']))
batch=dict(complete=all(r['passed'] for r in rows),sets=len(rows),passed_sets=sum(r['passed'] for r in rows),pose_instances=sum(r['poses'] for r in rows),results=rows,connectivity_required=True,provenance=provenance(inputs,[HERE/'step4.2/collect_step42.py',HERE/'step4.2/step42_dispatch.py',HERE/'step4.2/step42_connected_all.py']))
save(HERE/'output/B/data/step42_batch.json',batch);print('Current Step4.2:',batch['passed_sets'],'/',len(rows))
