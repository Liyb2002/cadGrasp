"""Controlled reconstruction of one immutable, previously accepted proposal.

The parent, proposal, task families, loads and stock dimensions are held
fixed. Only the construction mode changes; all acceptance gates stay active.
Artifacts belong in data/, never in the visible output directory.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np

from whole_task.operations.construction import CompactConstructionModel
from whole_task.operations.state import Layout
from whole_task.step1.common import sha256
from .optimize import Optimizer, record
from .group_acceptance import freeze_independent_material


def parent_checkpoint(source, destination, index):
    source,destination=Path(source),Path(destination)
    if (destination/'current.json').exists():
        raise ValueError('Use a new data directory for a controlled comparison')
    destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'current.json').read_text())
    with np.load(source/'state_000.npz') as z:
        groups=[[i] for i in range(len(z['poses']))]
    for k in range(index+1):
        state=source/f'state_{k:03}.npz'
        metadata=json.loads(state.with_suffix('.json').read_text())
        if sha256(state)!=metadata['mesh_sha256']:
            raise ValueError('Source checkpoint changed')
        if k:
            with np.load(state) as z:count=len(z['poses'])
            # IDs in this pipeline are W01...W13. Read their ordered list
            # from the complete certificate, rather than inferring a number.
            ids=[r['id'] for r in metadata['audit']['tasks']]
            owners=sorted(ids.index(n) for n in metadata['audit']['freshly_checked_tasks'])
            inputs=[g for g in groups if set(g)&set(owners)]
            if len(inputs)!=2 or sorted(x for g in inputs for x in g)!=owners:
                raise ValueError('Source is not an explicit two-family merge chain')
            groups=[g for g in groups if g not in inputs]+[owners]
        for path in (state,state.with_suffix('.json')):
            try:os.link(path,destination/path.name)
            except OSError:shutil.copy2(path,destination/path.name)
    state=destination/f'state_{index:03}.npz'
    record(destination/'current.json',dict(state=state.name,sha256=sha256(state),
        groups=groups,accepted=index,volume_cm3=metadata['volume_cm3'],
        initial_report_sha256=manifest['initial_report_sha256']))
    record(destination/'trace.json',[])


def compare(source,parent,proposal,destination,mode,core_slot=0,reuse_candidate=None,proposal_layout=None,
            intermediate_components='all',connecting_stock=False,simplify_boundaries=False):
    if proposal!=parent+1:
        raise ValueError('Use one accepted transition from the same source chain')
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if 'output' in destination.parts:
        raise ValueError('Research artifacts must be stored under data/')
    available=sorted(os.sched_getaffinity(0))
    cores=np.array_split(available,max(1,len(available)//2))[core_slot%(max(1,len(available)//2))]
    os.sched_setaffinity(0,[int(k) for k in cores])
    parent_checkpoint(source,destination,parent)
    model=CompactConstructionModel()
    optimizer=Optimizer(model,destination,resume=True,audit_resume=False)
    from .population_search import load_retention_witnesses,save_retention_witnesses
    load_retention_witnesses(model,destination)
    if proposal_layout is None:
        proposal_path=source/f'state_{proposal:03}.npz'
        proposal_metadata=json.loads(proposal_path.with_suffix('.json').read_text())
        with np.load(proposal_path) as z:q=Layout.read(z)
    else:
        proposal_path=Path(proposal_layout)
        saved=json.loads(proposal_path.read_text())
        if saved['parent_state_sha256']!=sha256(source/f'state_{parent:03}.npz'):
            raise ValueError('Fixed proposal belongs to a different parent')
        q=Layout(*(np.asarray(saved['layout'][key]) for key in ('poses','material','exits')))
        proposal_metadata=dict(operation=saved['operation'],audit=dict(freshly_checked_tasks=saved['owners']))
    names=proposal_metadata['audit']['freshly_checked_tasks']
    owners=sorted(model.ids.index(n) for n in names)
    inputs=[g for g in optimizer.groups if set(g)&set(owners)]
    groups=[g for g in optimizer.groups if g not in inputs]+[owners]
    model.family_refit_mode=mode
    model.keep_intermediate_material_components=intermediate_components=='all'
    model.optimize_connecting_stock=connecting_stock
    model.simplify_construction_boundaries=simplify_boundaries
    wall,cpu=time.monotonic(),time.process_time()
    if reuse_candidate is None:
        freeze_independent_material(model,optimizer.layout,optimizer.support,owners)
    profile=json.loads(Path(__file__).with_name('policy.json').read_text())['candidate_stock']
    model.stock_grid=(profile['pitch_m'],profile['post_width_m'])
    model.proposal_deadline=time.monotonic()+900.
    if reuse_candidate is None:
        candidate=model.build(q)
    else:
        import trimesh
        from whole_task.step3 import independent_geometry as G
        from whole_task.operations.floors import clip,planes
        with np.load(reuse_candidate) as z:
            for name,array in q.arrays().items():np.testing.assert_array_equal(z[name],array)
            support=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        envelope=clip(G.box(support.bounds[0]-.001,support.bounds[1]+.001),planes(q.material))
        candidate=dict(layout=q.copy(),support=support,tasks=[],volume_cm3=float(support.volume*1e6),
            parent=envelope,construction=dict(ribbed_stock=dict(pitch_m=profile['pitch_m'],
                post_width_m=profile['post_width_m']),replayed_raw_candidate_sha256=sha256(reuse_candidate)))
    built=time.monotonic()
    np.savez_compressed(destination/'raw_candidate.npz',**q.arrays(),
        vertices=candidate['support'].vertices,faces=candidate['support'].faces)
    raw_volume=candidate['volume_cm3']
    accepted=optimizer.try_operation(q,proposal_metadata['operation'],groups=groups,
        audit_owners=owners,prepared_candidate=candidate,controlled_research=True,
        construction_mode=mode,fixed_proposal_sha256=sha256(proposal_path))
    save_retention_witnesses(model,destination)
    result=dict(mode=mode,accepted=accepted,intermediate_components=intermediate_components,
        optimize_connecting_stock=connecting_stock,simplify_boundaries=simplify_boundaries,
        parent_state_sha256=sha256(source/f'state_{parent:03}.npz'),
        proposal_state_sha256=sha256(proposal_path),selected_families=[[model.ids[k] for k in g] for g in inputs],
        freshly_checked_tasks=names,stock_profile=profile,parent_volume_cm3=json.loads(
            (source/f'state_{parent:03}.json').read_text())['volume_cm3'],
        raw_volume_cm3=raw_volume,accepted_volume_cm3=optimizer.volume,
        build_seconds=built-wall,total_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,
        reason=optimizer.last_trial_error,workdir=str(destination))
    result['replayed_raw_candidate_sha256']=sha256(reuse_candidate) if reuse_candidate else None
    record(destination/'comparison.json',result)
    print('CONTROLLED_COMPARISON',json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--parent',type=int,required=True)
    p.add_argument('--proposal',type=int,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--mode',choices=['local','free_material','free_material_box'],required=True)
    p.add_argument('--core-slot',type=int,default=0)
    p.add_argument('--reuse-candidate',type=Path)
    p.add_argument('--proposal-layout',type=Path)
    p.add_argument('--intermediate-components',choices=['all','largest'],default='all')
    p.add_argument('--connecting-stock',action='store_true')
    p.add_argument('--simplify-boundaries',action='store_true')
    a=p.parse_args()
    compare(a.source,a.parent,a.proposal,a.destination,a.mode,a.core_slot,a.reuse_candidate,a.proposal_layout,
        a.intermediate_components,a.connecting_stock,a.simplify_boundaries)


if __name__=='__main__':main()
