from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,os,sys,time,resource,traceback
resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
payload=json.loads(Path(sys.argv[1]).read_text());ROOT=Path('/home/yli581/Desktop/cadGrasp');BASE=ROOT/'slides/Co-optimize/data/results/B';FROZEN=Path(payload['frozen']);CO=FROZEN/'slides/Co-optimize'
sys.path[:0]=[str(CO/'helper_func'),str(CO/'step5.2'),str(CO/'step5.3'),str(CO/'step4.2'),str(CO/'vis_func'),str(CO/'step5.1')]
import _bootstrap
import run_dynamic_postprocess as D
D.BASE=BASE
from stage5_batch import safe_worker
case=payload['case'];selection=Path(payload['selection']);initial=payload['initial'];aesthetic=payload['aesthetic'];final=payload['final'];floor_exp=payload['floor'];timings={};result=dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'],passed=False,timings=timings)
def persist():
 p=Path(payload['result']);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
def timed(label,fn):
 begin=time.perf_counter()
 try:return fn()
 finally:timings[label]=time.perf_counter()-begin;persist()
def run_stage(stage,args,label):
 row=timed(label,lambda:safe_worker(D.load_stage(stage).worker,case,args,BASE,stage))
 if not row['passed']:raise RuntimeError(label+': '+row.get('error','failed'))
 return row
try:
 stage=D.load_stage('step5.1');folder=BASE/case['id']/'step5/step5.1'/floor_exp;manifest=Path(payload['result']).parent/'floor_sources'/case['id'].replace('/','__');manifest.mkdir(parents=True,exist_ok=True)
 sources=stage.freeze_code(manifest);execution=manifest/'execution.json';D.save(execution,dict(case=case,executed_code=sources))
 body=stage.trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False)
 timed('step5.1',lambda:stage.compute_case('B',case,BASE,stage.NativeInputs('B',body),stage.V.Renderer(),0.,dict(manifest_path=execution,executed_code=sources),floor_exp))
 result['floor']=dict(passed=True,output=str(folder.relative_to(ROOT)))
 common=dict(object='B',selection_summary=str(selection),step51_experiment=floor_exp,reference_stage='step4',source_contact_material='auto',resume_pruning_experiment=None,clean_object_interiors=False,candidate_budget=64,leg_profile='flat',rounding_radius=.006)
 args=SimpleNamespace(**common,experiment=initial,reference_experiment=initial,prune_experiment=initial,pruning_style='legacy')
 p=run_stage('step5.2',args,'step5.2_bbx_pruning');f=run_stage('step5.3',args,'step5.3_leg_growth')
 legacy_prune,legacy_final=p['output'],f['output']
 args=SimpleNamespace(**common,experiment=aesthetic,reference_experiment=aesthetic,prune_experiment=aesthetic,pruning_style='rounded',shape_reference=str(ROOT/legacy_prune),fixed_leg_reference=str(ROOT/legacy_final))
 p=run_stage('step5.2',args,'step5.2_broad_rounding');f=run_stage('step5.3',args,'step5.3_fixed_leg_reconstruction')
 import sculpted_postprocess_case as SC
 SC.BASE=BASE
 previous=dict(step51_source=str(folder.relative_to(ROOT)),step52_source=p['output'],step53_source=f['output'],step53_passed=True)
 sculpted=timed('step5.2_and_5.3_surface_rounding',lambda:SC.run_case(case,SimpleNamespace(experiment=final,selection_summary=str(selection),step51_experiment=floor_exp),previous))
 result.update(sculpted);result['timings']=timings;result['postprocess_experiment']=final
 # Capture this policy's actual public images before the other policy publishes.
 import hashlib,shutil
 archive=Path(payload['result']).parent/'images'/case['id'];pictures=[]
 for stage_name in ['step5.1','step5.2','step5.3']:
  for pic in (ROOT/'slides/Co-optimize/output/B'/case['id']/stage_name).glob('*.png'):
   target=archive/stage_name/pic.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pic,target);pictures.append(dict(stage=stage_name,path=str(target.relative_to(ROOT)),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
 result['archived_images']=pictures
except BaseException as error:
 result.update(passed=False,status='POSTPROCESS_FAIL',error=str(error),traceback=traceback.format_exc());traceback.print_exc()
persist();print('POST RESULT',case['id'],result['passed'],timings,flush=True)
