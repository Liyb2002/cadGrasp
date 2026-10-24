"""Construct and prune real sparse feet for the ground-certified membrane forest."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import trimesh

from whole_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task.operations.state import Layout
from whole_task.step3 import independent_geometry as G
from whole_task.step3.run_membranes import load_model
from whole_task.step4.ground_oracle import pressure_rows, ground_enclosure
from whole_task.step4.membrane_search import read_group, partition, complete_proof
from .geometry import SparseFeet, occupied_bounds, physical_component_count, save_mesh, ground_contact_metrics
from .prune import Deleter
from .ground_design import shared_plane_rows


def construction_rows(rows, layout):
    return [dict(pose=r['id'], fixture_transform=r['world_from_support'],
        object_transform=layout.poses[k], hull=np.c_[r['hull'], np.zeros(len(r['hull']))],
        object_ground=np.empty((0, 2))) for k, r in enumerate(rows)]


class Validator:
    """Recheck actual material, every installation and all original paired loads."""
    def __init__(self, model, layout, witness):
        self.model=model; self.layout=layout; self.witness=G.solid(witness)
        self.pressure=pressure_rows(model,layout)
        self.rows=construction_rows(self.pressure,layout)
        self.data=dict(body=model.source)

    def check(self, mesh, previous=None, bounds=None):
        value=G.solid(mesh)
        actual=occupied_bounds(self.model.source,mesh,self.rows)
        outside=G.volume(value-self.witness)
        added=0. if previous is None else G.volume(value-previous)
        expansion=0. if bounds is None else float(max(0.,np.max(bounds[0]-actual[0]),np.max(actual[1]-bounds[1])))
        cover=ground_enclosure(mesh,self.pressure)
        if outside>self.model.tolerance or added>self.model.tolerance or expansion>1e-8:
            raise ValueError(f'Actual material exceeds its certified witness, previous material or occupied bounds: '
                f'outside_m3={outside:.12g}, added_m3={added:.12g}, expansion_m={expansion:.12g}')
        if not all(r['passed'] for r in cover):
            raise ValueError('Actual ground patches do not enclose every pressure hull')
        candidate=dict(layout=self.layout,support=mesh,tasks=self.model.prepare(self.layout))
        audit=self.model.audit(candidate)
        if not complete_proof(audit,len(self.rows)):raise ValueError('Incomplete original force/exit proof')
        data=[t['contact_data'] for t in candidate['tasks']]
        proof=dict(passed=True,masks={k:d['sample_passed'] for k,d in enumerate(data)},
            endpoint_masks={k:d['endpoint_passed'] for k,d in enumerate(data)},
            supplies={k:d['supply_7d'] for k,d in enumerate(data)},audit=audit)
        checks=dict(passed=True,components=physical_component_count(value),
            material_cm3=float(mesh.volume*1e6),outside_m3=outside,added_m3=added,
            bbx_l=float(np.prod(actual[1]-actual[0])*1000),bounds_world_m=actual.tolist(),
            bbx_expansion_m=expansion,floor_passed=True,pressure_enclosure=cover,
            counts={t['id']:dict(passed=t['force']['passed_original_sample_count'],
                total=t['force']['original_sample_count']) for t in audit['tasks']},
            continuous_endpoint_counts={t['id']:dict(passed=t['force']['passed_continuous_outer_target_count'],
                total=t['force']['continuous_outer_target_count']) for t in audit['tasks']},
            watertight=True,winding_consistent=True,serialized_geometry_checked=True)
        return value,proof,checks


def solve_group(model, group, out, grids=(3,5), width=.005):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    q,core=read_group(group);part=model.subset(group['members'])
    foot=group['oracle_witness']
    if sha256(foot['file'])!=foot['sha256']:raise ValueError('Changed oracle witness')
    with np.load(foot['file']) as z:witness=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
    validator=Validator(part,q,witness)
    stock_path=out/'connected_stock.obj'
    if stock_path.exists() and (out/'construction.json').exists():
        construction=json.loads((out/'construction.json').read_text())
        if construction['core_sha256']!=group['sha256']:raise ValueError('Construction belongs to another group')
        stock=trimesh.load(stock_path,force='mesh',process=False)
    else:
        design_rows=shared_plane_rows(validator.rows)
        stock,construction=SparseFeet(core,witness,design_rows,width=width).run()
        stock=save_mesh(stock,stock_path)
        construction.update(core_sha256=group['sha256'],witness_sha256=foot['sha256'],
            ground_design='combined_demands_per_shared_material_ground_plane',
            ground_face_count=len({r['fixture_transform'].tobytes() for r in design_rows}),
            physical_task_poses_changed=False)
        write_json(out/'construction.json',construction)
    value,proof,checks=validator.check(stock)
    print('SPARSE FEET PASS',part.ids,round(checks['material_cm3'],3),flush=True)
    cutter=Deleter(stock,value,proof,checks,validator,out)
    cutter.run(grids)
    np.savez_compressed(out/'support.npz',**q.arrays(),vertices=cutter.mesh.vertices,
        faces=cutter.mesh.faces,members=np.asarray(group['members']))
    np.savez_compressed(out/'force_certificate.npz',
        **{f'sample_{k}':v for k,v in cutter.proof['masks'].items()},
        **{f'endpoint_{k}':v for k,v in cutter.proof['endpoint_masks'].items()},
        **{f'supply_{k}':v for k,v in cutter.proof['supplies'].items()})
    result=dict(file=str((out/'support.npz').resolve()),sha256=sha256(out/'support.npz'),
        members=group['members'],volume_cm3=cutter.checks['material_cm3'],
        oracle_witness=foot,core=group,construction=construction,
        initial_material_cm3=checks['material_cm3'],checks=cutter.checks,
        audit=cutter.proof['audit'],accepted_deletions=len(cutter.states)-1,
        stock=dict(file=str(stock_path.resolve()),sha256=sha256(stock_path)),
        force_certificate=dict(file=str((out/'force_certificate.npz').resolve()),sha256=sha256(out/'force_certificate.npz')),
        attempted_deletions=len(cutter.history),ground_contact_metrics=ground_contact_metrics(cutter.mesh,validator.rows))
    write_json(out/'result.json',result)
    return result


def solve_saved_group(source,group,out,grids):
    model,_,_=load_model(source)
    return solve_group(model,group,out,grids)


def validate(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    source=ROOT/'whole_task/data'/report['object']/'step3_membranes'
    model,_,_=load_model(source);groups=report['groups'];partition(groups,len(model.ids))
    for path,digest in report['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Changed Step5 input: '+path)
    proofs=[]
    for group in groups:
        q,mesh=read_group(group);foot=group['oracle_witness']
        core_layout,_=read_group(group['core'])
        for key,array in q.arrays().items():np.testing.assert_array_equal(array,core_layout.arrays()[key])
        if sha256(foot['file'])!=foot['sha256']:raise ValueError('Changed saved legal witness')
        with np.load(foot['file']) as z:witness=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        stock=group['stock'];cert=group['force_certificate']
        if sha256(stock['file'])!=stock['sha256'] or sha256(cert['file'])!=cert['sha256']:
            raise ValueError('Changed construction stock or force masks')
        original=trimesh.load(stock['file'],force='mesh',process=False)
        _,proof,checks=Validator(model.subset(group['members']),q,witness).check(mesh,G.solid(original))
        with np.load(cert['file']) as z:
            for k,mask in proof['masks'].items():
                np.testing.assert_array_equal(mask,z[f'sample_{k}'])
                np.testing.assert_array_equal(proof['endpoint_masks'][k],z[f'endpoint_{k}'])
        proofs.append(dict(members=group['members'],audit=proof['audit'],checks=checks))
    if report.get('video'):
        video=report['video']
        if sha256(video['file'])!=video['sha256']:raise ValueError('Changed final movie')
        if sha256(video['motion_file'])!=video['motion_sha256']:raise ValueError('Changed movie orientation diagram')
    if report.get('overview_sha256') and sha256(folder/'overview.png')!=report['overview_sha256']:
        raise ValueError('Changed final overview')
    result=dict(status='passed',fresh_disk_replay=True,group_count=len(groups),proofs=proofs,
        original_sample_count=sum(t['force']['original_sample_count'] for p in proofs for t in p['audit']['tasks']),
        continuous_outer_target_count=sum(t['force']['continuous_outer_target_count'] for p in proofs for t in p['audit']['tasks']))
    write_json(folder/'validation.json',result)
    return model,result


def run(step4=None,workdir=None,grids=(3,5),video=True,publish=True,output=None,workers=2):
    started=time.monotonic()
    if step4 is None:
        active=json.loads((ROOT/'whole_task/data/B/step4_active.json').read_text());step4=active['workdir']
    step4=Path(step4).resolve();old=json.loads((step4/'report.json').read_text())
    if not old.get('ground_oracle_certified',False):raise ValueError('Step4 lacks the actual ground-foot oracle')
    name='B';source=ROOT/'whole_task/data'/name/'step3_membranes'
    workdir=Path(workdir or ROOT/'whole_task/data'/name/'step5_ground_membranes').resolve()
    workdir.mkdir(parents=True,exist_ok=True);model,_,_=load_model(source)
    groups=[None]*len(old['groups']);pending=[]
    for k,group in enumerate(old['groups']):
        result_path=workdir/f'set_{k+1:02}'/'result.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if result['core']['sha256']!=group['sha256']:raise ValueError('Step5 checkpoint belongs to another Step4')
            read_group(result)
            groups[k]=result
        else:pending.append((k,group,result_path.parent))
    if workers>1 and len(pending)>1:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        import multiprocessing
        with ProcessPoolExecutor(max_workers=min(workers,len(pending)),mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs={pool.submit(solve_saved_group,source,group,out,grids):k for k,group,out in pending}
            for job in as_completed(jobs):groups[jobs[job]]=job.result()
    else:
        for k,group,out in pending:groups[k]=solve_group(model,group,out,grids)
    report=dict(status='passed',method='ground_membrane_sparse_feet',object=name,
        step4=str(step4),groups=groups,initial_group_count=old['initial_group_count'],
        final_group_count=len(groups),initial_volume_cm3=sum(g['initial_material_cm3'] for g in groups),
        final_volume_cm3=sum(g['volume_cm3'] for g in groups),
        step4_membrane_volume_cm3=old['final_volume_cm3'],ground_oracle_certified=True,
        all_original_paired_loads_and_continuous_exits=True,proposal_grids=list(grids),
        retained_volume_fraction_heuristic=.55,group_workers=workers,
        virtual_convex_hulls_filled_with_material=False,minimum_wall_or_strength_certified=False,
        input_sha256={str(p):sha256(p) for p in [step4/'report.json',step4/'validation.json',
            source/'initialization.npz',Path(__file__),Path(__file__).with_name('geometry.py'),
            Path(__file__).with_name('surface_route.py'),
            Path(__file__).with_name('ground_design.py'),
            Path(__file__).with_name('prune.py'),Path(__file__).with_name('membrane_movie.py'),
            ROOT/'whole_task/step4/render_membrane_families.py',ROOT/'whole_task/step4/ground_oracle.py']})
    write_json(workdir/'report.json',report)
    model,validation=validate(workdir)
    report['validation_sha256']=sha256(workdir/'validation.json')
    from whole_task.step4.render_membrane_families import render
    render(model,groups,report,workdir/'overview.png',step=5)
    report['overview_sha256']=sha256(workdir/'overview.png')
    if video:
        from .membrane_movie import render_movie
        report['video']=render_movie(model,groups,workdir/'result.mp4',workdir/'video_motion.npz')
    report['seconds']=time.monotonic()-started
    write_json(workdir/'report.json',report)
    validate(workdir)
    report['validation_sha256']=sha256(workdir/'validation.json')
    write_json(workdir/'report.json',report)
    if publish:
        out=Path(output or DEFAULT_OUTPUT/name/'step5');out.mkdir(parents=True,exist_ok=True)
        for filename in ['overview.png']+(['result.mp4'] if video else []):shutil.copy2(workdir/filename,out/filename)
        write_json(ROOT/'whole_task/data'/name/'step5_active.json',dict(workdir=str(workdir),
            report_sha256=sha256(workdir/'report.json'),group_count=len(groups),
            volume_cm3=report['final_volume_cm3']))
    print('MEMBRANE STEP5 PASS',report['final_volume_cm3'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step4',type=Path);parser.add_argument('--workdir',type=Path)
    parser.add_argument('--grids',type=int,nargs='*',default=[3,5])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--no-video',action='store_true');parser.add_argument('--no-publish',action='store_true')
    args=parser.parse_args();run(args.step4,args.workdir,tuple(args.grids),not args.no_video,not args.no_publish,workers=args.workers)
