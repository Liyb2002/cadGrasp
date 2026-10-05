"""Read-only diagnostic of the frozen first trajectory round-two supplies."""
import json,signal,time,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'slides/baseline_algo'))
from step3_scheculer.stage_imports import load_stage
from step3_scheculer import contacts as I
C=load_stage('score','contribution')
signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('110-second diagnostic cap')))
signal.alarm(110)
out=Path(__file__).with_suffix('.json'); results=[]
def save():out.write_text(json.dumps(results,indent=2)+'\n')
try:
 p=C.Problem('A1-f'); folder=ROOT/'slides/baseline_algo/output/A1-f/pose_1'
 fixed=I.read_contacts(folder/'step3.3_optimize_contact/trajectory_000/round_001/contacts.npz')
 masks=np.load(folder/'step3.1_score_candidate/trajectory_000/round_002/sample_coverage.npz')
 for candidate in [30,35,17]:
  full=I.merge_columns(p.supply(fixed),C.columns(p.domain,p.data,candidate,p.floor,p.scale))
  _,singular,vt=np.linalg.svd(full,full_matrices=False);rank=int((singular>singular[0]*1e-10).sum());transform=vt[:rank].T/singular[:rank];matrix=(full@transform).T
  pending=np.flatnonzero(~masks['base'])
  for index in pending[[0,min(100,len(pending)-1)]]:
   target=C.U.target(p.targets[index],full.shape[1]);transformed=target@transform;scale=max(float(np.max(np.abs(transformed))),1.)
   for presolve in [True,False]:
    tick=time.monotonic();r=linprog(np.zeros(len(full)),A_eq=matrix,b_eq=transformed/scale,bounds=(0,None),method='highs',options=dict(presolve=presolve,time_limit=12.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9));elapsed=time.monotonic()-tick
    row=dict(candidate=candidate,sample=int(index),columns=len(full),rank=rank,presolve=presolve,seconds=elapsed,status=int(r.status),stored_feasible=bool(masks['covered'][candidate,index]))
    if r.success:row.update(original_max_residual=float(np.max(np.abs((r.x*scale)@full-target))),minimum_coefficient=float(r.x.min()))
    results.append(row);save();print(row,flush=True)
except BaseException as e:
 results.append(dict(error=repr(e)));save();raise
