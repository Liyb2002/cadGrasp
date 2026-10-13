"""Actual serialized geometry checks; no force solving or artifact replacement."""
from pathlib import Path
import sys,json,hashlib,time,numpy as np
root=Path(__file__).resolve().parents[3];coopt=root/'slides/Co-optimize';base=coopt/'data/results/B'
sys.path[:0]=[str(coopt/n)for n in ['helper_func','step5.2','step5.3','step4.2']]
import _bootstrap
from deterministic_base import solid,volume,moved,union
from prune_support import material_components
import trimesh
import argparse
parser=argparse.ArgumentParser(description='Compare actual saved connected solids at original geometry tolerance')
parser.add_argument('--experiment',required=True);name=parser.parse_args().experiment;system=base/'data'/('system_'+name)
a=json.loads((system/'evaluation_results.json').read_text());body=trimesh.load(root/'objects/B/mesh.stl',force='mesh',process=False);body.merge_vertices(digits_vertex=15);object_solid=solid(body);rows=[];began=time.monotonic()
def load(path):return solid(trimesh.load(path,force='mesh',process=False))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for r in a['cases']:
 if not r['step53_passed']:continue
 label=r['id'];source=base/label/'step4/step4.2'/r['source_experiment'];prune=root/r['step52_source'];final=root/r['step53_source']
 pr=json.loads((prune/'data/report.json').read_text());fr=json.loads((final/'data/report.json').read_text());cp=source/'support.obj';pp=prune/pr['support'];fp=final/fr['support'];wp=source/'oracle_connected_witness.obj'
 core=load(cp);p=load(pp);f=load(fp);allowed=load(wp)
 with np.load(source/'layout.npz')as z:objects=union([moved(object_solid,z['placements'][k])for k in z['active']])
 row=dict(id=label,step4_source=str(source.relative_to(root)),step52_source=r['step52_source'],step53_source=r['step53_source'],core_components=len(material_components(core)),pruned_components=len(material_components(p)),final_components=len(material_components(f)),pruned_added_m3=volume(p-core),pruned_outside_m3=volume(p-allowed),pruned_object_overlap_m3=volume(p^objects),final_retained_pruned_loss_m3=volume(p-f),final_outside_m3=volume(f-allowed),final_object_overlap_m3=volume(f^objects),artifacts={str(x.relative_to(root)):sha(x)for x in [cp,pp,fp,wp]})
 source_report=json.loads((source/'report.json').read_text())
 if source_report['connected_contact_core'].get('added_material_cm3',0)>1e-4:
  # This diagnostic describes component extraction BEFORE Oracle cleanup.
  # Check the actual saved core against its original finite-wrap domain.
  from run_oracle import OracleModel,load_layout,S
  from whole_step4_render import NominalBuilder
  model=OracleModel(source_report['poses'],'B',demand_nodes=32,oracle_work_mode='normal')
  model.require_connected_core=True
  seed=base/label/'step3/step3.1/wrapped_support.obj'
  model.registered_seed=S.solid(trimesh.load(seed,force='mesh',process=False))
  domain=NominalBuilder(model).solid(load_layout(source/'layout.npz'))
  row['core_added_to_finite_wrap_m3']=volume(core-domain)
  row['finite_wrap_artifacts']={str(x.relative_to(root)):sha(x)for x in [source/'layout.npz',seed]}
 row['passed']=all(row[k]==1 for k in ['core_components','pruned_components','final_components'])and all(row[k]<=1e-10 for k in ['pruned_added_m3','pruned_outside_m3','pruned_object_overlap_m3','final_retained_pruned_loss_m3','final_outside_m3','final_object_overlap_m3']);rows.append(row)
 row['passed']=row['passed']and row.get('core_added_to_finite_wrap_m3',0)<=1e-10
 if not row['passed']:print('ACTUAL SOLID FAIL',json.dumps(row),flush=True)
report=dict(complete=a['complete'],passed=all(r['passed']for r in rows),groups=len(rows),original_tolerance_m3=1e-10,boolean_run=True,force_solver_run=False,cases=rows,seconds=time.monotonic()-began)
(system/'actual_solid_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('ACTUAL SOLID AUDIT',len(rows),'passed',sum(r['passed']for r in rows),flush=True)
raise SystemExit(0 if report['passed'] else 1)
