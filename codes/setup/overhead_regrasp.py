"""Find diverse grasps approached from above using the shared sequence search.

Restricting approach directions keeps the generic palm away from the floor.
All contact-location, grasp-direction, grounded-target and robot checks from
regrasp_sequence remain active. Exact IK memoization only avoids repeating
identical numerical queries on common search prefixes.
"""
import argparse
import hashlib

import regrasp_sequence as R
import kuka_transfer as K


def find(name, pairs=120, candidate_budget=120, downward_component=.65, resume=True):
    original_candidates,original_solve,original_export=R.G.candidates,K.solve,R.export
    cache={}

    def candidates(*args,**kwargs):
        for candidate in original_candidates(*args,**kwargs):
            if candidate['hand'][2,2] < -downward_component:
                yield candidate

    def solve(arm,target,previous=None):
        key=(arm.base.tobytes(),target.tobytes(),None if previous is None else previous.tobytes())
        if key not in cache:
            try:
                q,error=original_solve(arm,target,previous)
                cache[key]=(q.copy(),error)
            except ValueError as error:
                cache[key]=str(error)
        result=cache[key]
        if isinstance(result,str):
            raise ValueError(result)
        return result[0].copy(),result[1]

    def export(*args,**kwargs):
        kwargs['generator']='codes/setup/overhead_regrasp.py'
        metadata=dict(kwargs.get('metadata') or {})
        metadata['grasp_candidate_filter']=dict(maximum_world_approach_z=-downward_component,
            description='Restrict candidate approaches to the overhead cone; all diversity thresholds unchanged')
        metadata['implementation_sha256']={path:hashlib.sha256((R.G.ROOT/path).read_bytes()).hexdigest()
            for path in ('codes/setup/regrasp_sequence.py','codes/setup/grasp.py','codes/setup/kuka_transfer.py')}
        kwargs['metadata']=metadata
        return original_export(*args,**kwargs)

    R.G.candidates,K.solve,R.export=candidates,solve,export
    try:
        return R.find(name,pairs,candidate_budget,resume=resume)
    finally:
        R.G.candidates,K.solve,R.export=original_candidates,original_solve,original_export


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('--pairs',type=int,default=120)
    parser.add_argument('--candidate-budget',type=int,default=120)
    parser.add_argument('--downward-component',type=float,default=.65)
    parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args()
    if not 0 <= args.downward_component < 1:
        parser.error('--downward-component must be in [0, 1)')
    find(args.object,args.pairs,args.candidate_budget,args.downward_component,not args.fresh)
