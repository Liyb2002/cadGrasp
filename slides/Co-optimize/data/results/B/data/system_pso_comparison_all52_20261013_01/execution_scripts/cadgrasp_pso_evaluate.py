from pathlib import Path
import importlib.util,json,csv,time,hashlib,sys,shutil
import numpy as np
import trimesh
ROOT=Path('/home/yli581/Desktop/cadGrasp');BASE=ROOT/'slides/Co-optimize/data/results/B';SYSTEM=BASE/'data/system_pso_comparison_all52_20261013_01';FROZEN=SYSTEM/'frozen_sources/slides/Co-optimize'
sys.path[:0]=[str(FROZEN/x)for x in ['helper_func','step5.2','step5.3','vis_func','evaluation']]
import _bootstrap
from metrics import static_bounds
from deterministic_base import solid,volume
from prune_support import material_components
import step51_render as V
spec=importlib.util.spec_from_file_location('pso_evaluation_renderer',FROZEN/'evaluation/run.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
policies={'differential':'differential_control_all52_20261013_01','pso':'pso_all52_20261013_01'}
while not all((BASE/'data'/(exp+'_sculpted')/'batch.json').exists()and json.loads((BASE/'data'/(exp+'_sculpted')/'batch.json').read_text())['complete']for exp in policies.values()):time.sleep(5)
body=trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False);renderer=V.Renderer();old=json.loads((SYSTEM/'previous_evaluation.json').read_text());oldmap={r['id']:r for r in old['cases']};evaluations={}
for policy,exp in policies.items():
 started=time.perf_counter();post=exp+'_sculpted';search=json.loads((BASE/'data'/exp/'batch.json').read_text());batch=json.loads((BASE/'data'/post/'batch.json').read_text());outcomes={r['id']:r for r in batch['results']};rows=[];audits=[]
 for case in sorted(search['results'],key=lambda r:(r['pose_count'],r['id'])):
  label=case['id'];previous=oldmap[label];out=outcomes[label];s4=BASE/label/'step4/step4.2'/exp;pr=BASE/label/'step5/step5.2'/post;fr=BASE/label/'step5/step5.3'/post
  row=dict(id=label,object_pose_count=case['pose_count'],step4_passed=case['passed'],passed=out['passed'],status=out.get('status','PASS'),no_support_bbx_l=previous['no_support_bbx_l'],step4_seconds=case['process_wall_seconds'],timings=out.get('timings',{}),step5_seconds=out.get('process_wall_seconds',0.),step52_source=str(pr.relative_to(ROOT)),step53_source=str(fr.relative_to(ROOT)),step51_source=out.get('floor',{}).get('output'))
  for key in ['support_pose_count','reuse','juxtaposed','step4_bbx_l','pruned_bbx_l','final_bbx_l','step4_material_cm3','pruned_material_cm3','final_material_cm3']:row[key]=None
  if not(s4/'layout.npz').exists():rows.append(row);continue
  report=json.loads((s4/'data/report.json').read_text())if(s4/'data/report.json').exists()else{}
  with np.load(s4/'layout.npz')as a:layout={k:a[k].copy()for k in a.files}
  nominal=trimesh.load(s4/'support.obj',force='mesh',process=False)if(s4/'support.obj').exists()else None
  p_report=json.loads((pr/'data/report.json').read_text())if(pr/'data/report.json').exists()else{}
  f_report=json.loads((fr/'data/report.json').read_text())if(fr/'data/report.json').exists()else{}
  pruned=trimesh.load(pr/p_report['support'],force='mesh',process=False)if p_report.get('passed')else None
  final=trimesh.load(fr/f_report['support'],force='mesh',process=False)if f_report.get('passed')else None
  names=report.get('poses',json.loads((BASE/label/'step4/step4.1/data/report.json').read_text())['poses']);metric=static_bounds(body.vertices,layout,nominal.vertices if nominal is not None else None,final.vertices if final is not None else None,names)
  assert abs(metric['object_static']['volume_litres']-previous['no_support_bbx_l'])<=1e-8
  row.update(support_pose_count=metric['support_pose_count'],reuse=metric['rotating_reuse_pose_count'],juxtaposed=metric['juxtaposed_pose_count'],members=metric['fixture_states'],rounds=report.get('frontier_search',{}).get('rounds'),branches=report.get('frontier_search',{}).get('branch_counts'))
  if nominal is not None:row.update(step4_bbx_l=metric['with_step4_support']['volume_litres'],step4_material_cm3=volume(solid(nominal))*1e6)
  if pruned is not None:
   pm=static_bounds(body.vertices,layout,nominal.vertices if nominal is not None else None,pruned.vertices,names);row.update(pruned_bbx_l=pm['with_full_support']['volume_litres'],pruned_material_cm3=volume(solid(pruned))*1e6)
  if final is not None:
   assert out['passed'];row.update(final_bbx_l=metric['with_full_support']['volume_litres'],final_material_cm3=volume(solid(final))*1e6)
   mask_count=0
   for folder,r,mesh in [(pr,p_report,pruned),(fr,f_report,final)]:
    assert r['passed']and mesh.is_watertight and len(material_components(solid(mesh)))==1
    for path,h in r['artifacts'].items():assert sha(folder/path)==h
    for path,h in r['provenance']['inputs'].items():assert sha(ROOT/path)==h
    with np.load(folder/'force_certificate.npz')as masks:
     for k in layout['active']:assert masks['mask_'+str(k)].shape==(32768,)and masks['mask_'+str(k)].all();mask_count+=32768
   subset=volume(solid(pruned)-solid(nominal));retained=volume(solid(pruned)-solid(final));witness=trimesh.load(s4/'oracle_connected_witness.obj',force='mesh',process=False);outside=volume(solid(final)-solid(witness))
   assert max(subset,retained,outside)<=1e-10
   assert p_report['pruning']['locked_bbx_unchanged']and f_report['construction']['locked_final_bbx_unchanged']
   audits.append(dict(id=label,passed=True,masks=mask_count,pruned_outside_step4_m3=subset,final_core_loss_m3=retained,outside_saved_space_m3=outside))
   visual=ROOT/'slides/Co-optimize/output/B'/label/'evaluation'/post;visual.mkdir(parents=True,exist_ok=True);E.render(renderer,body,nominal,final,layout,metric,visual)
   if policy=='pso':
    # Final active pictures belong to the PSO result rather than the timing control.
    for pic in (BASE/'data'/post/'workers/images'/label).rglob('*.png'):
     stage=pic.parent.name;target=ROOT/'slides/Co-optimize/output/B'/label/stage/pic.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pic,target)
   row['evaluation_image']=str((visual/'static_occupancy.png').relative_to(ROOT))
  rows.append(row);print('EVALUATION',policy,len(rows),label,row['passed'],flush=True)
 successful=[r for r in rows if r['passed']];summary=dict(complete=True,groups=52,passed=len(successful),cases=rows,evaluation_seconds=time.perf_counter()-started,geometry_audit=audits)
 save(SYSTEM/(policy+'_evaluation.json'),summary);evaluations[policy]=summary
protected=json.loads((SYSTEM/'protected_originals.json').read_text());changed=[p for p,h in protected.items()if sha(ROOT/p)!=h];assert not changed
save(SYSTEM/'evaluation_verification.json',dict(passed=True,protected_files=len(protected),changed=[]))
print('ALL EVALUATIONS COMPLETE',flush=True)
