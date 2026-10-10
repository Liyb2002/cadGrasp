import os,sys,json,time,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path('/home/yli581/Desktop/cadGrasp/slides/Co-optimize/helper_func')))
import _bootstrap
from whole_pipeline import run_case
from run_all import saved_groups,output_root
from co_common import save,provenance

def main():
 names=sys.argv[1:];root=output_root('B')
 token=json.loads((root/'data/whole_step4_active_run.json').read_text())['run_token']
 os.environ['COOPT_WHOLE_RUN_TOKEN']=token
 options=dict(stage='both',iterations=8,finalists=3,branch_rounds=3,seed=42,volume_rounds=3,screen_budget=96,run_token=token)
 groups=[g for g in saved_groups('B') if g['id'] in names]
 assert len(groups)==len(names)
 rows=[];began=time.monotonic();out=root/'data/controller_chunks'/time.strftime('%Y%m%d_%H%M%S')
 out.mkdir(parents=True,exist_ok=True)
 (out/'runner.py').write_text(Path(__file__).read_text())
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
  futures={pool.submit(run_case,('B',g,options)):g for g in groups}
  for future in as_completed(futures):
   row=future.result();rows.append(row)
   save(out/'report.json',dict(complete=False,options=options,jobs=2,results=rows,seconds=time.monotonic()-began,provenance=provenance([],[out/'runner.py'])))
   print('JOINED CHUNK',row['id'],row['status'],row.get('error',''),flush=True)
 save(out/'report.json',dict(complete=True,options=options,jobs=2,results=rows,seconds=time.monotonic()-began,provenance=provenance([],[out/'runner.py'])))
 print('JOINED CHUNK COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
