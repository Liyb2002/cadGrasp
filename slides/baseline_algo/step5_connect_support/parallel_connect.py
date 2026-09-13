"""Construct independent supports in isolated processes, publishing each finish."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import sys
import trimesh

HERE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HERE))
from step5_connect_support import bounded_connect as B
from step5_connect_support import connect as C, fixed_feet as H, solids as S
from step1.cases import pose_name

_state=None


def pack(module):
    if module is None:return None
    return dict(plan=module['plan'],solid=module['solid'],
        arrays=S.pack_parts(module['parts'],module['labels'],module['joined']))


def unpack(payload):
    if payload is None:return None
    arrays=payload['arrays']
    return dict(plan=payload['plan'],solid=payload['solid'],parts=S.unpack_parts(arrays),
        labels=arrays['part_labels'],
        joined=trimesh.Trimesh(arrays['union_vertices_m'],arrays['union_faces'],process=False))


def initialize(name,pose,max_edges,expected_hashes):
    global _state
    os.environ['CADGRASP_POSE']=pose
    domain,contacts,schedule,floor,directions,depth,inputs=C.read_inputs(name)
    if C.I.hashes(inputs)!=expected_hashes:
        raise RuntimeError('Worker inputs differ from the parent snapshot')
    work=C.W.WorkVolume.read(C.OUTPUTS/name/pose/C.W.STAGE/'work_volume.json')
    _state=(domain,contacts,directions['contacts'],
        {f['candidate_id']:f for f in floor['report']['ground_footprints']},depth,work,max_edges,expected_hashes)


def construct(index):
    domain,contacts,directions,feet,depth,work,max_edges,hashes=_state
    contact=contacts[index]
    record,module=B.bounded_one(domain.mesh,contact,directions[index],
        feet[contact['candidate_id']],depth,work,max_edges=max_edges)
    C.I.check_hashes(hashes)
    return record,pack(module)


def collect(contacts,futures,on_result=None):
    """Callbacks follow completion order; final records retain selected order."""
    found={}
    for future in as_completed(futures):
        index=futures[future];cid=contacts[index]['candidate_id']
        try:
            record,payload=future.result()
            if record['candidate_id']!=cid:
                raise ValueError('Worker returned a different contact ID')
            module=unpack(payload)
            if bool(module is not None)!=bool(record['passed']):
                raise ValueError('Worker status and saved geometry disagree')
        except Exception as error:
            record,module=dict(candidate_id=cid,passed=False,
                status='individual_worker_error',footprint_unchanged=True,
                failure_is_global_impossibility_proof=False,
                reason=f'{type(error).__name__}: {error}'),None
        found[index]=(record,module)
        if on_result is not None:on_result(record,module)
        print(cid,record['status'],flush=True)
    records=[found[i][0] for i in range(len(contacts))]
    modules=[found[i][1] for i in range(len(contacts)) if found[i][1] is not None]
    return records,modules


def build(name,max_edges=2000,workers=4):
    if max_edges<1 or workers<1:raise ValueError('Use positive edge and worker counts')
    expected=C.I.hashes(C.read_inputs(name)[-1]);pose=pose_name()
    original=H.search
    def search(mesh,contacts,directions,feet,depth,work,builder=None,on_result=None):
        if len(directions)!=len(contacts):raise ValueError('Missing contact directions')
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(name,pose,max_edges,expected)) as pool:
            futures={pool.submit(construct,i):i for i in range(len(contacts))}
            return collect(contacts,futures,on_result)
    H.search=search
    try:result=C.build(name)
    finally:H.search=original
    C.I.check_hashes(expected)
    result['search_settings'].update(maximum_new_edge_checks_per_support=max_edges,
        budget_exhaustion_is_inconclusive=True,parallel_support_workers=workers,
        individual_publication_order='completion',final_record_order='selected_contacts')
    result['provenance']['code'].update(C.I.hashes([Path(__file__),Path(B.__file__)]))
    out=C.OUTPUTS/name/pose/C.STAGE
    C.I.save(out/'connection.json',result);digest=C.sha256(out/'connection.json')
    C.I.save(out/'status.json',dict(complete=True,status=result['status'],connection_sha256=digest))
    C.I.save(out/'progress.json',dict(complete=True,processed_count=len(result['support_results']),
        contact_count=result['contact_count'],connection_sha256=digest))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',type=Path);parser.add_argument('objects',nargs='*')
    parser.add_argument('--edge-budget',type=int,default=2000)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage.resolve()!=Path(C.__file__).resolve():parser.error('Expected the Step 5 connector entry point')
    if min(args.edge_budget,args.workers)<1:parser.error('Use positive edge and worker counts')
    for name in args.objects or C.OBJECTS:build(name,args.edge_budget,args.workers)
