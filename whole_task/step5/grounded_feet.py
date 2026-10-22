"""Grow and prune feet at the actual resting planes of each contact membrane."""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import trimesh

from whole_task.operations.state import Layout
from whole_task.step1.common import ROOT,DEFAULT_OUTPUT,sha256,write_json
from whole_task.step3 import independent_geometry as G
from whole_task.step3.run_membranes import load_model
from whole_task.step4.ground_oracle import GroundOracle
from whole_task.step4.membrane_search import read_group,partition
from .ground_contact import GroundFrameModel,FloorObjective,normalize_ground,resting_heights
from .membrane_feet import Validator as LegacyValidator
from .geometry import SparseFeet,save_mesh,ground_contact_metrics
from .ground_design import shared_plane_rows
from .prune import Deleter


class Validator(LegacyValidator):
    def __init__(self,model,layout,witness,core):
        super().__init__(model,layout,witness)
        self.core=G.solid(core)
        for row in self.rows:
            body=self.model.source.vertices@row['object_transform'][:3,:3].T+row['object_transform'][:3,3]
            if body[:,2].min()<-1e-9:raise ValueError('Object penetrates its resting plane')

    def check(self,mesh,previous=None,bounds=None):
        retained=G.mesh(G.solid(mesh)^self.core)
        if not len(retained.vertices):raise ValueError('Contact membrane entirely removed')
        heights=[float((retained.vertices@r['fixture_transform'][:3,:3].T+
                        r['fixture_transform'][:3,3])[:,2].min()) for r in self.rows]
        if max(abs(h) for h in heights)>1e-8:
            raise ValueError('Remaining membrane loses direct contact with a ground plane')
        value,proof,checks=super().check(mesh,previous,bounds)
        checks.update(membrane_direct_ground_contact=True,retained_membrane_ground_heights_m=heights,
                      object_ground_collision_checked=True)
        return value,proof,checks


def load_mesh(path):
    with np.load(path) as z:return trimesh.Trimesh(z['vertices'],z['faces'],process=False)


class ContactFeet(SparseFeet):
    """Choose small pads using finite-area ground faces, never point touches."""
    def ground(self):
        from scipy.spatial import ConvexHull,QhullError,cKDTree
        from .geometry import transform,box,union,unpack,volume,SCALE
        from whole_task.step4.ground_oracle import ground_enclosure
        core=unpack(self.body)
        def contact(mesh,T):
            world=transform(mesh.vertices,T);tri=world[mesh.faces]
            keep=np.max(np.abs(tri[:,:,2]),axis=1)<=1e-8
            tri=tri[keep]
            if not len(tri):return np.empty((0,2))
            a=tri[:,1,:2]-tri[:,0,:2];b=tri[:,2,:2]-tri[:,0,:2]
            tri=tri[np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])>2e-15]
            return tri[:,:,:2].reshape(-1,2)
        Ts=[r['fixture_transform'] for r in self.rows]
        base=[contact(core,T) for T in Ts]
        distinct=list({np.round(T,11).tobytes():T for T in Ts}.values())
        strict=[dict(id=r['pose'],world_from_support=T,hull=r['hull'][:,:2])
                for r,T in zip(self.rows,Ts)]
        for multiplier in (1,2,4,8,16,32,64,128):
            margin=self.width*multiplier
            domain=box(self.low-margin,self.high+margin)^self.allowed
            pads=[];locations=[]
            for T in distinct:
                slab=domain.trim_by_plane(tuple(-T[2,:3]),float((T[2,3]-self.width)/SCALE))
                required=np.vstack([r['hull'][:,:2] for r in self.rows
                                     if np.allclose(T,r['fixture_transform'],atol=1e-11,rtol=0)])
                low,high=required.min(0)-margin,required.max(0)+margin
                for axis in range(2):
                    slab=slab.trim_by_plane(tuple(T[axis,:3]),float((low[axis]-T[axis,3])/SCALE))
                    slab=slab.trim_by_plane(tuple(-T[axis,:3]),float((T[axis,3]-high[axis])/SCALE))
                xy=np.unique(np.round(contact(unpack(slab),T),12),axis=0)
                if len(xy)<3:continue
                try:corners=xy[ConvexHull(xy).vertices]
                except QhullError:continue
                matrix=np.linalg.inv(T)[:3].copy();matrix[:,3]/=SCALE
                for x,y in corners:
                    pad=box(np.array([x-self.width/2,y-self.width/2,0.]),
                            np.array([x+self.width/2,y+self.width/2,self.width]))
                    pad=pad.transform(matrix)^self.allowed
                    if volume(pad)>1e-13:
                        pads.append(pad);locations.append(dict(world_xy_m=[x,y],fixture_transform=T.tolist()))
            points=[[contact(unpack(p),T) for T in Ts] for p in pads]
            def covers(active):
                for i,row in enumerate(self.rows):
                    xy=np.vstack([base[i],*[points[j][i] for j in active]])
                    if len(xy)<3:return False
                    try:hull=ConvexHull(xy)
                    except QhullError:return False
                    if (row['hull'][:,:2]@hull.equations[:,:2].T+hull.equations[:,2]).max()>1e-8:return False
                return True
            active=set(range(len(pads)))
            if not covers(active):continue
            tree=cKDTree(core.vertices)
            order=sorted(active,key=lambda j:-float(tree.query(unpack(pads[j]).vertices)[0].min()))
            for j in order:
                if covers(active-{j}):active.remove(j)
            proposed=self.body+union([pads[j] for j in sorted(active)])
            if not all(r['passed'] for r in ground_enclosure(unpack(proposed),strict)):continue
            self.body=proposed
            self.history.append(dict(operation='finite_area_ground_patches',margin_m=margin,
                compact_feet_used=True,foot_count=len(active),foot_locations=[locations[j] for j in sorted(active)],
                volume_cm3=volume(union([pads[j] for j in sorted(active)]))*1e6,
                perimeter_ring_generated=False,full_slab_fallback_allowed=False,
                point_contacts_counted_as_ground_faces=False))
            return
        raise ValueError('Cannot construct finite-area pressure-enclosing sparse feet')


def constructive_witness(model,candidate,diagnostics=None):
    """Certify the actual foot stock rather than a fragile maximal free solid."""
    from whole_task.step4.ground_oracle import foot_regions,pressure_floor_gap
    gap=pressure_floor_gap(model,candidate['layout'])
    core=candidate['support'];q=candidate['layout']
    errors=[]
    for padding,allowed,rows in foot_regions(model,candidate):
        checker=Validator(model,q,allowed,core)
        for width in (.005,.008,.012):
            try:
                print('BUILD FEET',model.ids,padding,width,flush=True)
                builder=ContactFeet(core,allowed,shared_plane_rows(checker.rows),width=width)
                stock,construction=builder.run()
                print('BUILT FEET',len(stock.faces),construction['outside_saved_legal_component_cm3'],flush=True)
                if diagnostics is not None:
                    np.savez_compressed(Path(diagnostics)/f'foot_raw_{padding}_{width}.npz',
                        **q.arrays(),heights=model.ground_heights,
                        vertices=stock.vertices,faces=stock.faces,
                        core_vertices=core.vertices,core_faces=core.faces,
                        allowed_vertices=allowed.vertices,allowed_faces=allowed.faces)
                # This finite free solid only proposes a construction. Its
                # decomposed boundary is not a physical certificate: audit
                # the actual stock against every original exclusion/load.
                # The accepted stock then becomes the fixed pruning bound.
                outside=G.volume(G.solid(stock)-G.solid(allowed))
                construction.update(exported_outside_proposal_cm3=outside*1e6,
                                    proposal_space_is_not_an_acceptance_certificate=True)
                actual_checker=Validator(model,q,stock,core)
                try:
                    _,proof,checks=actual_checker.check(stock)
                except (ValueError,RuntimeError) as first_error:
                    print('REPAIR ACTUAL FOOT',str(first_error),flush=True)
                    if 'Continuous source-face exit failed' not in str(first_error):raise
                    test=dict(layout=q,support=stock,tasks=model.prepare(q),construction={})
                    model.refine_exits(test);stock=test['support']
                    actual_checker=Validator(model,q,stock,core)
                    _,proof,checks=actual_checker.check(stock)
                construction.update(passed=True,certified_actual_stock=True,
                                    material_cm3=checks['material_cm3'])
                print('CONSTRUCTIVE FOOT WITNESS PASS',model.ids,checks['material_cm3'],flush=True)
                return stock,dict(passed=True,status='actual_connected_sparse_feet_witness',
                    witness_is_constructed_stock=True,construction=construction,
                    full_witness_audit=proof['audit'],ground=checks['pressure_enclosure'],
                    finite_construction_padding_m=padding,foot_width_m=width,
                    pressure_floor_minimum_gap_m=gap,geometry_tolerance_m3=model.tolerance)
            except (ValueError,RuntimeError) as ex:
                errors.append(dict(padding=padding,width=width,reason=str(ex)))
                print('CONSTRUCTIVE FOOT RETRY',padding,width,str(ex),flush=True)
    raise ValueError('No actual sparse-foot witness: '+str(errors))


def prepare(base,group,out,number):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    original,source_core=read_group(group);part=base.subset(group['members'])
    saved=out/'preparation.json'
    if saved.exists():
        record=json.loads(saved.read_text())
        if record['source_step4_core']['sha256']!=group['sha256']:raise ValueError('Another Step4 family')
        for item in (record['core'],record['witness']):
            if sha256(item['file'])!=item['sha256']:raise ValueError('Prepared geometry changed')
        return record
    folder=out.parent;probe=folder/f'candidate_{number:02}_core.npz'
    witness_probe=folder/f'candidate_{number:02}_witness.npz'
    if probe.exists() and witness_probe.exists():
        with np.load(probe) as z:q_global=Layout.read(z);heights=z['heights'].copy()
        core=load_mesh(probe);witness=load_mesh(witness_probe)
        meta=json.loads((folder/f'candidate_{number:02}.json').read_text())
        oracle=json.loads((folder/f'candidate_{number:02}_oracle.json').read_text())
    else:
        rejected=[];objective=FloorObjective(part,original,source_core)
        prior=json.loads((out/'installation_rejections.json').read_text()) if (out/'installation_rejections.json').exists() else []
        if prior and prior[0].get('contact_screened_angle_ranges_rad'):
            objective.allowed_ranges=[[tuple(b)] for b in prior[0]['contact_screened_angle_ranges_rad']]
        else:objective.force_ranges()
        def cached_proposals():
            from .ground_contact import axial_installations
            # Step4 v2 already proved a tangent-ground installation; try it first.
            meta_path=Path(group['file']).with_suffix('.json')
            certified=json.loads(meta_path.read_text()).get('ground_oracle',{}).get('tangent_installation') if meta_path.exists() else None
            if certified and not prior:prior.append(dict(certified,reason='step4_tangent_installation'))
            for record in sorted(prior,key=lambda r: 'Oracle:' not in r['reason']):
                angles=objective.snap_angles(record['angles_rad']);gap=float(objective.slacks(angles).min())
                if gap<-1e-9:continue
                proposal=axial_installations(original,angles,part.axis,part.head)
                yield proposal,resting_heights(source_core,proposal),dict(record,
                    angles_rad=angles.tolist(),minimum_margin_m=gap,retried_with_exact_shared_floor_frames=True)
        from itertools import chain
        for q_global,heights,meta in chain(cached_proposals(),objective.proposals()):
            model=GroundFrameModel(part,heights);q=normalize_ground(q_global,heights)
            tasks=model.prepare(q);value=G.solid(source_core)
            value=value-G.union([x for t in tasks for x in (G.solid(t['body']),t['work'])])
            candidate=dict(layout=q,support=G.mesh(value.simplify(0.)),tasks=tasks,construction={})
            try:
                model.refine_exits(candidate);core=candidate['support']
                heights=resting_heights(core,q_global)
                model=GroundFrameModel(part,heights);q=normalize_ground(q_global,heights)
                candidate=dict(layout=q,support=core,tasks=model.prepare(q),construction={})
                model.audit(candidate)
                witness,oracle=constructive_witness(model,candidate,out)
            except (ValueError,RuntimeError) as ex:
                rejected.append(dict(meta,reason=str(ex)));print('INSTALLATION REJECT',part.ids,str(ex),flush=True)
                write_json(out/'installation_rejections.json',rejected)
                # Feed the actual force counterexample back: later proposals
                # avoid this task's failed angle instead of repeating it.
                failed=re.match(r'(W\d+): paired force/moment certificate failed',str(ex))
                if failed and failed.group(1) in part.ids:
                    i=part.ids.index(failed.group(1))
                    if objective.exclude(i,float(meta['angles_rad'][i])):
                        rejected[-1]['excluded_angle_window_deg']=[part.ids[i],float(np.degrees(meta['angles_rad'][i])),8.]
                continue
            break
        else:raise ValueError('No fully certified directly grounded installation proposal')
    np.testing.assert_allclose(q_global.material@q_global.poses,original.material@original.poses,atol=1e-13,rtol=0.)
    np.savez_compressed(out/'core.npz',**q_global.arrays(),vertices=core.vertices,faces=core.faces,
                        members=np.asarray(group['members']))
    np.savez_compressed(out/'witness.npz',vertices=witness.vertices,faces=witness.faces)
    write_json(out/'ground_oracle.json',oracle)
    meta=dict(meta);meta.pop('reason',None)
    record=dict(source_step4_core=group,
        core=dict(file=str((out/'core.npz').resolve()),sha256=sha256(out/'core.npz'),
                  members=group['members'],volume_cm3=float(core.volume*1e6)),
        witness=dict(file=str((out/'witness.npz').resolve()),sha256=sha256(out/'witness.npz')),
        grounding=dict(method='per_installation_tangent_ground_frame',global_floor_heights_m=heights.tolist(),
                       physical_head_world_m=base.head.tolist(),installation_repair=meta,
                       seats_preserved=True),ground_oracle_passed=True,
        witness_is_constructed_stock=oracle.get('witness_is_constructed_stock',False),
        witness_construction=oracle.get('construction'))
    write_json(saved,record)
    return record


def solve(source,group,out,number,grids):
    base,_,_=load_model(source);out=Path(out)
    prepared=prepare(base,group,out,number)
    q_global,core=read_group(prepared['core']);heights=prepared['grounding']['global_floor_heights_m']
    q=normalize_ground(q_global,np.asarray(heights))
    model=GroundFrameModel(base.subset(group['members']),heights)
    witness=load_mesh(prepared['witness']['file']);checker=Validator(model,q,witness,core)
    stockpath=out/'connected_stock.obj'
    if stockpath.exists():
        stock=trimesh.load(stockpath,force='mesh',process=False)
        construction=json.loads((out/'construction.json').read_text())
    elif prepared.get('witness_is_constructed_stock'):
        stock=save_mesh(witness,stockpath)
        construction=dict(prepared['witness_construction'],core_sha256=prepared['core']['sha256'],
                          witness_sha256=prepared['witness']['sha256'])
        write_json(out/'construction.json',construction)
    else:
        for width in (.005,.008,.012):
            try:
                stock,construction=SparseFeet(core,witness,shared_plane_rows(checker.rows),width=width).run()
                stock=save_mesh(stock,stockpath);checker.check(stock)
                construction.update(foot_width_m=width,core_sha256=prepared['core']['sha256'],
                                    witness_sha256=prepared['witness']['sha256'])
                write_json(out/'construction.json',construction);break
            except (ValueError,RuntimeError) as ex:
                print('FOOT BUILD RETRY',model.ids,width,str(ex),flush=True)
        else:raise ValueError('Could not construct sparse feet in the common legal space')
    value,proof,checks=checker.check(stock)
    print('GROUNDED FEET PASS',model.ids,checks['material_cm3'],flush=True)
    cutter=Deleter(stock,value,proof,checks,checker,out);cutter.run(grids)
    np.savez_compressed(out/'support.npz',**q.arrays(),vertices=cutter.mesh.vertices,faces=cutter.mesh.faces,
                        members=np.asarray(group['members']))
    np.savez_compressed(out/'force_certificate.npz',
        **{f'sample_{k}':v for k,v in cutter.proof['masks'].items()},
        **{f'endpoint_{k}':v for k,v in cutter.proof['endpoint_masks'].items()},
        **{f'supply_{k}':v for k,v in cutter.proof['supplies'].items()})
    result=dict(file=str((out/'support.npz').resolve()),sha256=sha256(out/'support.npz'),members=group['members'],
        volume_cm3=cutter.checks['material_cm3'],initial_material_cm3=checks['material_cm3'],
        core=prepared['core'],source_step4_core=group,oracle_witness=prepared['witness'],
        grounding=prepared['grounding'],display_final_support=True,construction=construction,
        checks=cutter.checks,audit=cutter.proof['audit'],
        stock=dict(file=str(stockpath.resolve()),sha256=sha256(stockpath)),
        force_certificate=dict(file=str((out/'force_certificate.npz').resolve()),sha256=sha256(out/'force_certificate.npz')),
        accepted_deletions=len(cutter.states)-1,attempted_deletions=len(cutter.history),
        ground_contact_metrics=ground_contact_metrics(cutter.mesh,checker.rows))
    write_json(out/'result.json',result);return result


def result_model(report):
    base,_,_=load_model(ROOT/'whole_task/data'/report['object']/'step3_membranes')
    heights=np.zeros(len(base.ids))
    for g in report['groups']:heights[g['members']]=g['grounding']['global_floor_heights_m']
    return GroundFrameModel(base,heights)


def validate(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text());model=result_model(report)
    partition(report['groups'],len(model.ids));proofs=[]
    for p,digest in report['input_sha256'].items():
        if sha256(p)!=digest:raise ValueError('Changed grounded Step5 input: '+p)
    for g in report['groups']:
        q,mesh=read_group(g);q_global,core=read_group(g['core'])
        expected=normalize_ground(q_global,np.asarray(g['grounding']['global_floor_heights_m']))
        for key,array in q.arrays().items():np.testing.assert_array_equal(array,expected.arrays()[key])
        old,_=read_group(g['source_step4_core'])
        np.testing.assert_allclose(q.material@q.poses,old.material@old.poses,atol=1e-13,rtol=0.)
        for item in (g['oracle_witness'],g['stock'],g['force_certificate']):
            if sha256(item['file'])!=item['sha256']:raise ValueError('Changed grounded construction/certificate')
        witness=load_mesh(g['oracle_witness']['file'])
        stock=trimesh.load(g['stock']['file'],force='mesh',process=False)
        _,proof,checks=Validator(model.subset(g['members']),q,witness,core).check(mesh,G.solid(stock))
        with np.load(g['force_certificate']['file']) as z:
            for k,mask in proof['masks'].items():
                np.testing.assert_array_equal(mask,z[f'sample_{k}'])
                np.testing.assert_array_equal(proof['endpoint_masks'][k],z[f'endpoint_{k}'])
        proofs.append(dict(members=g['members'],checks=checks,audit=proof['audit']))
    if report.get('video'):
        media=report['video']
        for key,digest in [('file','sha256'),('motion_file','motion_sha256'),('poster_file','poster_sha256')]:
            if sha256(media[key])!=media[digest]:raise ValueError('Changed final grounded-foot media')
        if not media['actual_final_geometry']:raise ValueError('Movie must show the saved final foot geometry')
    if report.get('robot_video') and sha256(report['robot_video']['file'])!=report['robot_video']['sha256']:
        raise ValueError('Changed two-robot movie')
    result=dict(status='passed',fresh_disk_replay=True,group_count=len(proofs),proofs=proofs,
        original_sample_count=sum(t['force']['original_sample_count'] for p in proofs for t in p['audit']['tasks']),
        continuous_outer_target_count=sum(t['force']['continuous_outer_target_count'] for p in proofs for t in p['audit']['tasks']),
        all_membranes_directly_touch_ground=True,actual_ground_patches_enclose_all_required_hulls=True,
        seats_preserved=True,physical_head_world_m=model.head.tolist())
    write_json(folder/'validation.json',result);return model,result


def run(step4=None,workdir=None,grids=(3,5),workers=2,video=True,publish=True,output=None):
    started=time.monotonic();root=ROOT/'whole_task/data/B'
    step4=Path(step4 or json.loads((root/'step4_active.json').read_text())['workdir'])
    old=json.loads((step4/'report.json').read_text());source=root/'step3_membranes'
    workdir=Path(workdir or root/'step5_ground_contact_final').resolve();workdir.mkdir(parents=True,exist_ok=True)
    groups=[None]*len(old['groups']);pending=[]
    for k,g in enumerate(old['groups']):
        out=workdir/f'set_{k+1:02}';p=out/'result.json'
        if p.exists():groups[k]=json.loads(p.read_text())
        else:pending.append((k,g,out))
    if workers>1 and len(pending)>1:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        import multiprocessing
        with ProcessPoolExecutor(max_workers=min(workers,len(pending)),mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs={pool.submit(solve,source,g,out,k+1,grids):k for k,g,out in pending}
            for job in as_completed(jobs):groups[jobs[job]]=job.result()
    else:
        for k,g,out in pending:groups[k]=solve(source,g,out,k+1,grids)
    inputs=[source/'initialization.npz',source/'report.json',step4/'report.json',Path(__file__),Path(__file__).with_name('ground_contact.py'),
            Path(__file__).with_name('geometry.py'),Path(__file__).with_name('membrane_feet.py'),
            Path(__file__).with_name('prune.py'),Path(__file__).with_name('ground_design.py'),
            Path(__file__).with_name('surface_route.py'),Path(__file__).with_name('hulls_presentation.py'),
            Path(__file__).with_name('grounded_overview.py'),
            ROOT/'whole_task/step3/membranes.py',ROOT/'whole_task/step3/independent_physics.py',
            ROOT/'whole_task/common/wrenches.py',ROOT/'whole_task/step4/ground_oracle.py']
    report=dict(status='passed',method='direct_contact_membrane_feet',object='B',step4=str(step4),groups=groups,
        initial_group_count=old['initial_group_count'],final_group_count=len(groups),
        initial_volume_cm3=sum(g['initial_material_cm3'] for g in groups),final_volume_cm3=sum(g['volume_cm3'] for g in groups),
        step4_membrane_volume_cm3=old['final_volume_cm3'],ground_definition='membrane_tangent_plane_per_installation',
        proposal_grids=list(grids),input_sha256={str(p):sha256(p) for p in inputs},
        ground_oracle_certified=True,all_original_paired_loads_and_continuous_exits=True)
    write_json(workdir/'report.json',report);model,proof=validate(workdir)
    report['validation_sha256']=sha256(workdir/'validation.json')
    from .hulls_presentation import render_movie
    if video:
        report['video']=render_movie(model,groups,workdir/'result.mp4',workdir/'video_motion.npz')
        from .video.sets import make as robot_movie
        report['robot_video']=robot_movie(workdir,workdir/'robot_use.mp4')
    from .grounded_overview import render
    render(model,groups,report,workdir/'all_task_installations.png')
    report['all_task_installations_sha256']=sha256(workdir/'all_task_installations.png');report['seconds']=time.monotonic()-started
    write_json(workdir/'report.json',report)
    validate(workdir)
    report['validation_sha256']=sha256(workdir/'validation.json');write_json(workdir/'report.json',report)
    if publish:
        out=Path(output or DEFAULT_OUTPUT/'B/step5');out.mkdir(parents=True,exist_ok=True)
        for p in list(out.iterdir()):
            if p.is_dir():shutil.rmtree(p)
            elif p.name not in ('result.mp4','robot_use.mp4','hulls_and_support.png'):p.unlink()
        for name in ('result.mp4','robot_use.mp4','hulls_and_support.png'):
            if (workdir/name).exists():shutil.copy2(workdir/name,out/name)
        write_json(root/'step5_active.json',dict(workdir=str(workdir),report_sha256=sha256(workdir/'report.json'),
                   group_count=len(groups),volume_cm3=report['final_volume_cm3']))
        write_json(root/'step5_visualization_active.json',dict(workdir=str(workdir),
            report_sha256=sha256(workdir/'report.json'),image=str(out/'hulls_and_support.png'),
            video=str(out/'result.mp4'),robot_video=str(out/'robot_use.mp4'),
            actual_final_geometry=True,ground_planes_checked=True))
    print('DIRECT CONTACT STEP5 PASS',report['final_volume_cm3'],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',type=Path)
    p.add_argument('--step4',type=Path,help='frozen Step4 folder; default data/B/step4_active.json')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--no-publish',action='store_true')
    p.add_argument('--no-video',action='store_true');args=p.parse_args()
    run(step4=args.step4,workdir=args.workdir,workers=args.workers,publish=not args.no_publish,video=not args.no_video)
