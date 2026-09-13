"""Bound each independent connector search without weakening solid acceptance."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HERE))
from step5_connect_support import connect as C,fixed_feet as H,routing as T
from step1.cases import pose_name

MAX_NEW_EDGES=2000


class SearchBudgetExhausted(RuntimeError):
    pass


def bounded_one(mesh,contact,direction,foot,depth,work,max_edges=MAX_NEW_EDGES):
    """A deterministic budget across every angle, backing and pad of ONE part."""
    if max_edges<1:raise ValueError('The per-support edge budget must be positive')
    original=T.Router.clear;checked=0
    def clear(router,part):
        nonlocal checked
        key=np.asarray(part.vertices,np.float64).tobytes()
        if key not in router.cache:
            if checked>=max_edges:raise SearchBudgetExhausted()
            checked+=1
        return original(router,part)
    # H.search is sequential; the override is local to this process and restored
    # before the next support (also on interruption or unexpected exceptions).
    T.Router.clear=clear
    try:
        try:
            record,module=H.build_one(mesh,contact,direction,foot,depth,work)
        except SearchBudgetExhausted:
            record,module=dict(candidate_id=contact['candidate_id'],passed=False,
                status='individual_search_budget_exhausted',footprint_unchanged=True,
                failure_is_global_impossibility_proof=False,
                reason='Finite per-support edge budget exhausted; no statement that a connector is impossible.'),None
    finally:
        T.Router.clear=original
    record['search_budget']=dict(maximum_new_edge_checks=max_edges,new_edge_checks=checked,
        scope='All directions, backing thicknesses and pads of this support; cached checks are reused')
    return record,module


def build(name,max_edges=MAX_NEW_EDGES):
    original=H.search
    def search(*args,**kwargs):
        return original(*args,**kwargs,builder=lambda *parts:bounded_one(*parts,max_edges=max_edges))
    H.search=search
    try:
        result=C.build(name)
    finally:
        H.search=original
    result['search_settings']['maximum_new_edge_checks_per_support']=max_edges
    result['search_settings']['budget_exhaustion_is_inconclusive']=True
    result['provenance']['code'].update(C.I.hashes([Path(__file__)]))
    out=C.OUTPUTS/name/pose_name()/C.STAGE
    C.I.save(out/'connection.json',result)
    digest=C.sha256(out/'connection.json')
    C.I.save(out/'status.json',dict(complete=True,status=result['status'],connection_sha256=digest))
    C.I.save(out/'progress.json',dict(complete=True,processed_count=len(result['support_results']),
        contact_count=result['contact_count'],connection_sha256=digest))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',type=Path)
    parser.add_argument('objects',nargs='*')
    parser.add_argument('--edge-budget',type=int,default=MAX_NEW_EDGES)
    args=parser.parse_args()
    if args.stage.resolve()!=Path(C.__file__).resolve():parser.error('Expected the Step 5 connector entry point')
    if args.edge_budget<1:parser.error('Use a positive edge budget')
    for name in args.objects or C.OBJECTS:build(name,args.edge_budget)
