"""Targeted all-pose gradient verification with immutable source snapshots."""
import sys,time,json,shutil,faulthandler
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from whole_search.fast_search import FastModel
from whole_pipeline import load_layout
from continuous_support.objective import CoverageObjective
from direction_choice import direction_choice
from translation import translation
from co_common import I,save

import argparse
parser=argparse.ArgumentParser(description='Check both gradients on large sets; no feasibility claim.')
parser.add_argument('--sets',nargs='+',default=['pose1+2+3+4+5+6+7+8','pose1+3+5+7+9+12+16+21+23+27'])
parser.add_argument('--rounds',type=int,default=1)
parser.add_argument('--translation-difference',type=float,default=1/512)
args=parser.parse_args();cases=args.sets
faulthandler.dump_traceback_later(60,repeat=True)
for case in cases:
 began=time.perf_counter();p=Path('slides/Co-optimize/output/B')/case
 out=p/'step4/step4.2/gradient_repair_v1'
 if out.exists() and any(out.iterdir()):shutil.copytree(out,p/'step4/_history'/('gradient_probe_'+time.strftime('%Y%m%d_%H%M%S')))
 out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
 frozen=out/'data/executed_sources';paths=[Path(__file__)]+list((Path(__file__).resolve().parents[1]/'helper_func/continuous_support').glob('*.py'))+[Path(__file__).resolve().parents[1]/'helper_func'/kind/'__init__.py' for kind in ('direction_choice','translation')]
 for source in paths:
  source=source.resolve();target=frozen/source.relative_to(Path.cwd());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
 save(out/'data/source_manifest.json',dict(code=I.hashes(paths)))
 poses=['pose_'+token for token in case[4:].split('+')]
 model=FastModel(poses,'B',initialization_report=p/'step3/step3.1/data/report.json')
 layout=load_layout(p/'step4/step4.1/layout.npz')
 objective=CoverageObjective(model,layout);before=objective.evaluate(layout)
 print(case,'initial','loss',before.residual_loss.tolist(),'coverage',before.coverage.tolist(),'seconds',time.perf_counter()-began,flush=True)
 with np.load(p/'step4/step4.1/data/initial_geometry.npz') as f:
  actual=[objective.distances[k].evaluate(f[f'supply_{k}']) for k in layout.active]
  print('ACTUAL_REFERENCE',case,[r['residual_loss'] for r in actual],flush=True)
 states=[];current=before
 save(out/'gradient_validation.json',dict(complete=False,scope='gradient verification, no feasibility claim',initial_loss=before.residual_loss.tolist(),actual_initial_loss=[r['residual_loss'] for r in actual],states=states))
 for iteration in range(args.rounds):
  start=time.perf_counter();old=current;current,record=direction_choice(objective,current)
  states.append(dict(phase='direction',iteration=iteration+1,seconds=time.perf_counter()-start,before_loss=objective.coverage_loss(old),after_loss=objective.coverage_loss(current),record=record))
  save(out/'data/direction_progress.json',states)
  print('DIRECTION',case,iteration+1,record['accepted'],objective.coverage_loss(current),'check',record['derivative_check'],'seconds',time.perf_counter()-start,flush=True)
 # Juxtapose is only the explicit setup of a Translation gradient test.
 guest=int(np.argmin(current.coverage));world=model.native[current.layout.hosts[guest],:3,:3]@current.layout.directions[guest]
 hosts=sorted((k for k in layout.active if k!=guest),key=lambda k:-float(world@(model.native[current.layout.hosts[k],:3,:3]@current.layout.directions[k])))
 jumped=model.juxtapose(current.layout,guest,hosts[0])
 center=np.array(model.mesh.center_mass)
 normal=model.floor_normal(jumped,guest)
 alignment=(current.layout.placements[hosts[0],:3,:3]@center+current.layout.placements[hosts[0],:3,3])-(jumped.placements[guest,:3,:3]@center+jumped.placements[guest,:3,3]);alignment-=normal*(alignment@normal)
 jumped.placements[guest,:3,3]+=alignment
 branch=objective.evaluate(jumped)
 print('TRANSLATION_START',case,poses[guest],poses[hosts[0]],objective.coverage_loss(branch),flush=True)
 for iteration in range(args.rounds):
  start=time.perf_counter();old=branch;branch,record=translation(objective,branch,difference_fraction=args.translation_difference)
  states.append(dict(phase='translation',iteration=iteration+1,seconds=time.perf_counter()-start,before_loss=objective.coverage_loss(old),after_loss=objective.coverage_loss(branch),record=record))
  save(out/'data/translation_progress.json',states)
  print('TRANSLATION',case,iteration+1,record['accepted'],objective.coverage_loss(branch),'check',record['derivative_check'],'seconds',time.perf_counter()-start,flush=True)
 result=dict(scope='gradient verification, not a feasible support result',complete=True,poses=poses,initial_loss=before.residual_loss.tolist(),actual_initial_loss=[r['residual_loss'] for r in actual],direction_loss=current.residual_loss.tolist(),translation_loss=branch.residual_loss.tolist(),states=states,seconds=time.perf_counter()-began,original_step41_unchanged=True,final_feasibility_claimed=False)
 save(out/'gradient_validation.json',result)
 np.savez_compressed(out/'gradient_layouts.npz',initial_placements=layout.placements,initial_directions=layout.directions,direction_placements=current.layout.placements,direction_directions=current.layout.directions,translation_placements=branch.layout.placements,translation_directions=branch.layout.directions,translation_hosts=branch.layout.hosts)
 print('GRADIENT_DONE',case,round(result['seconds'],2),flush=True)
