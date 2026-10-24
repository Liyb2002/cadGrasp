"""Record the reaction bases used by the unchanged complete-load classifier.

The wrapper observes successful primal solves, without changing their inputs
or outputs. Batch certificates use those same bases. Keeping their union is
sufficient to retain the parent load certificate; unused extreme contacts
need not survive a Boolean retriangulation.
"""
from threading import RLock
import numpy as np
from whole_task.step3.legacy import kernel as K

_lock=RLock()


def classify_with_witness(columns,targets):
    used=set()
    with _lock:
        original=K.J.C.W.solve
        def observe(full,target,*args,**kwargs):
            result=original(full,target,*args,**kwargs)
            if full is columns and result is not None:
                used.update(int(i) for i in result['indices'])
            return result
        K.J.C.W.solve=observe
        try:
            passed,report=K.J.classify(columns,targets)
        finally:
            K.J.C.W.solve=original
    if passed.all() and not used:
        raise ValueError('Complete load proof has no recorded reaction basis')
    return passed,report,columns[sorted(used)].copy()


def required_generators(task,columns):
    targets=np.vstack([task['endpoints']['endpoint_need_wrench'],task['samples']['need_wrench']])
    scale=np.r_[np.ones(3),np.ones(3)/task['body_world'].extents.max()]
    passed,report,basis=classify_with_witness(columns,targets*scale)
    if not passed.all():
        raise ValueError(task['id']+': cannot recover the complete parent load witness')
    return basis,dict(original_sample_count=len(task['samples']['need_wrench']),
        continuous_outer_target_count=len(task['endpoints']['endpoint_need_wrench']),
        witnessed_generator_count=len(basis),classifier=report,
        original_classifier_and_tolerance_unchanged=True)
