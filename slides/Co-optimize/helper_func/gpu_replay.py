"""Optional bounded FP64 batches for the ORIGINAL reaction-basis replay.

The small inverse is the existing CPU long-double Gauss-Jordan inverse.
CUDA computes large batches, with conservative conversion/reduction bounds.
Ambiguous demands are replayed by the original long-double CPU implementation.
No LP, projection objective, reaction cap or acceptance tolerance changes.
"""
from collections import OrderedDict
import time
import numpy as np

_mode='off'
_engine=None
_initialization_error=None
MINIMUM_BATCH=4096


def configure(mode='off'):
    global _mode,_engine,_initialization_error
    if mode not in ('off','auto','required'):raise ValueError('Unknown GPU replay mode')
    _mode=mode;_engine=None;_initialization_error=None


class GPUReplay:
    def __init__(self):
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
        torch.set_num_threads(1)
        self.torch=torch
        self.device='cuda:0'
        self.stats=dict(calls=0,rows=0,certain_rows=0,ambiguous_rows=0,
                        seconds=0.,failures=0,device=torch.cuda.get_device_name(0),
                        dtype='float64',torch_version=torch.__version__,cuda_version=torch.version.cuda)

    def evaluate(self,basis,targets,coordinates,inverse):
        torch=self.torch;began=time.monotonic()
        b=torch.as_tensor(np.ascontiguousarray(basis,dtype=np.float64),device=self.device)
        t=torch.as_tensor(np.ascontiguousarray(targets,dtype=np.float64),device=self.device)
        inv=torch.as_tensor(np.ascontiguousarray(inverse,dtype=np.float64),device=self.device)
        selected=t[:,coordinates]
        coefficient=torch.clamp(selected@inv.T,min=0.)
        absolute_coeff=selected.abs()@inv.abs().T
        # This bound dominates inverse casting, both CPU long-double and
        # CUDA FP64 dot-product rounding, clipping, and final subtraction.
        # The inputs are finite binary64 and the basis has at most seven rays.
        eps=np.finfo(np.float64).eps
        factor=64.*eps*(len(basis)+1)
        coefficient_error=factor*(absolute_coeff+np.finfo(np.float64).tiny)
        absolute_replay=coefficient@b.abs()
        replay_error=(coefficient_error@b.abs()+
                      factor*(absolute_replay+t.abs()+np.finfo(np.float64).tiny))
        residual=(coefficient@b-t).abs().amax(dim=1)
        error=replay_error.amax(dim=1)
        # Original replay adds 32*longdouble-eps*max(coefficients @ |basis|).
        roundoff_upper=64.*float(np.finfo(np.longdouble).eps)*(
            absolute_replay+(coefficient_error@b.abs())+replay_error).amax(dim=1)
        upper=torch.nextafter(residual+error+roundoff_upper,
                              torch.full_like(residual,float('inf')))
        lower=torch.nextafter(residual-error,
                              torch.full_like(residual,float('-inf')))
        definite_pass=upper<=2e-9
        definite_fail=lower>2e-9
        finite=torch.isfinite(upper)&torch.isfinite(lower)
        certain=(definite_pass|definite_fail)&finite
        mask=(definite_pass&certain).cpu().numpy()
        ambiguous=(~certain).cpu().numpy()
        bounds=upper.cpu().numpy()
        self.stats['calls']+=1;self.stats['rows']+=len(targets)
        self.stats['certain_rows']+=int((~ambiguous).sum())
        self.stats['ambiguous_rows']+=int(ambiguous.sum())
        self.stats['seconds']+=time.monotonic()-began
        return dict(mask=mask,ambiguous=ambiguous,upper_bounds=bounds)


def evaluate(basis,targets,coordinates,inverse):
    global _engine,_initialization_error
    if _mode=='off' or len(targets)<MINIMUM_BATCH or coordinates is None:return None
    if _initialization_error is not None:
        if _mode=='required':raise RuntimeError(_initialization_error)
        return None
    try:
        if _engine is None:_engine=GPUReplay()
        return _engine.evaluate(basis,targets,coordinates,inverse)
    except Exception as error:
        if _engine is not None:_engine.stats['failures']+=1
        _initialization_error=repr(error)
        if _mode=='required':raise RuntimeError('Required GPU replay failed') from error
        return None


def report():
    return dict(mode=_mode,used=_engine is not None,error=_initialization_error,
                minimum_batch=MINIMUM_BATCH,statistics=dict(_engine.stats) if _engine else {},
                original_cpu_replays_ambiguous_demands=True,original_tolerance=2e-9)
