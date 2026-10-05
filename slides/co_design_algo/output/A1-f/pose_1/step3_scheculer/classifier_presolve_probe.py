"""Isolated read-only classifier replay; no production module edits."""
import json,signal,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'slides/baseline_algo'))
from step3_scheculer.stage_imports import load_stage
from step3_scheculer import contacts as I
C=load_stage('score','contribution')
signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('80-second diagnostic cap')))
signal.alarm(80)
p=C.Problem('A1-f'); folder=ROOT/'slides/baseline_algo/output/A1-f/pose_1';fixed=I.read_contacts(folder/'step3.3_optimize_contact/trajectory_000/round_001/contacts.npz');masks=np.load(folder/'step3.1_score_candidate/trajectory_000/round_002/sample_coverage.npz');original=C.W.linprog;results=[];out=Path(__file__).with_suffix('.json')
try:
 for candidate in [30,35]:
  full=I.merge_columns(p.supply(fixed),C.columns(p.domain,p.data,candidate,p.floor,p.scale))
  for presolve in [True,False]:
   def configured(*args,**kwargs):
    kwargs['options']=dict(kwargs.get('options',{}),presolve=presolve,time_limit=12.)
    return original(*args,**kwargs)
   C.W.linprog=configured
   tick=time.monotonic();mask,info=C.J.classify(full,p.targets,known_covered=masks['base']);elapsed=time.monotonic()-tick
   verification=C.verify_classification(full,p.targets,mask)
   row=dict(candidate=candidate,presolve=presolve,seconds=elapsed,stored_mask_equal=bool(np.array_equal(mask,masks['covered'][candidate])),mismatch_count=int(np.count_nonzero(mask!=masks['covered'][candidate])),covered_count=int(mask.sum()),classifier=info,verification=verification)
   results.append(row);out.write_text(json.dumps(results,indent=2)+'\n');print({k:v for k,v in row.items() if k!='verification'},flush=True)
finally:
 C.W.linprog=original
