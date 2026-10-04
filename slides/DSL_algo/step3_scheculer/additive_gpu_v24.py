"""CUDA certificate batching with explicit nonnegative primal proposals."""
import numpy as np
from scipy.optimize import linprog
from step3_scheculer import passive_support as U
from step3_scheculer.pair_scoring import C

class GPUClassifier:
    """Same primal/dual certificates as J.classify; GPU only batches target tests."""
    def __init__(self,targets,device='cuda'):
        import torch
        torch.set_num_threads(1)
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        self.torch=torch;self.device=device
        self.targets=U.target(targets,7)
        self.rhs=torch.as_tensor(self.targets,dtype=torch.float64,device=device)
        self.calls=0;self.lp_count=0
        self.info=dict(device=device,dtype='float64',gpu=torch.cuda.get_device_name() if device=='cuda' else None,
            method='Original LP with CUDA batched primal and dual certificates',load_subsampling=False)

    def classify(self,full,known=None):
        t=self.torch;b=self.targets
        accepted=np.zeros(len(b),bool) if known is None else known.copy()
        pending=~accepted;self.calls+=1;lp=0
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
        while pending.any():
            remaining=np.flatnonzero(pending);index=int(remaining[0])
            witness=C.W.solve(full,b[index]);lp+=1;pending[index]=False
            ids=t.as_tensor(remaining,device=self.device);rhs=self.rhs[ids]
            if witness is not None:
                accepted[index]=True;basis=full[witness['indices']]
                if not len(basis):continue
                pinv=np.linalg.pinv(basis)
                coefficients=rhs@t.as_tensor(pinv,dtype=t.float64,device=self.device)
                coefficients=t.clamp(coefficients,min=0)
                residual=(coefficients@t.as_tensor(basis,dtype=t.float64,device=self.device)-rhs).abs().amax(dim=1)
                # Verify the clamped nonnegative reaction, retaining the original residual tolerance.
                certified=((coefficients>=0).all(dim=1)&(residual<=C.FEASIBILITY_TOL-1e-12)).cpu().numpy()
                accepted[remaining[certified]]=True;pending[remaining[certified]]=False
            else:
                result=linprog(-b[index],A_ub=full,b_ub=np.zeros(len(full)),bounds=[(-1,1)]*full.shape[1],method='highs',options=options)
                lp+=1
                if not result.success or np.linalg.norm(result.x)<1e-12:continue
                normal=result.x/np.linalg.norm(result.x)
                if np.max(full@normal)>1e-12:continue
                certified=(rhs@t.as_tensor(normal,dtype=t.float64,device=self.device)>1e-8+1e-12).cpu().numpy()
                pending[remaining[certified]]=False
        self.lp_count+=lp
        return accepted,dict(equilibrium_and_separator_lps=lp,covered=int(accepted.sum()))

