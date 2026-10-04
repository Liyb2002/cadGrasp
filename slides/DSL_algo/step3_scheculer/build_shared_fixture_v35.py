"""Construct current common-world-exit contact programs with free fixture seating.
Object geometry, native pose, original loads and contacts stay unchanged.
Old fixture transforms are proposal seeds only, never acceptance certificates.
"""
import argparse,json,time,traceback,shutil
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
from codes.precompute_objects import head_cache as HC
from step3_scheculer import contacts as I,operation_dsl as F,shared_direction_paths as P
from step3_scheculer.operation_growth_recovery import RecoveryGrow
from step3_scheculer.feasible_seating import restore
from step3_scheculer.initialize_current_v16 import current_task
from step3_scheculer.pair_scoring import J
from step3_scheculer.compare_shared_volume_v33 import GROUPS
from step5_current.evaluate import measure
from step5_current import render_bbox as BBOX
from step4_connect_support import clean_render as V,publish_compact as PUB

STAGE='shared_fixture_seating_v35'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def spread(tasks,state,gap):
    # A feasibility proposal, not a mandatory displacement or accepted bound.
    d=np.asarray(state.paths[0]['initial_object_exit_world']);axis=np.cross(d,[0.,0.,1.])
    if np.linalg.norm(axis)<1e-6:axis=np.array([1.,0.,0.])
    axis/=np.linalg.norm(axis);offsets=np.zeros((len(tasks),3));cursor=0.
    for i,(t,b) in enumerate(zip(tasks,state.bases)):
        cloud=t.domain.mesh.vertices@b;v=cloud@axis
        offsets[i]=axis*(cursor-v.min());cursor+=np.ptp(v)+gap
    return replace(state,offsets=offsets)


def proposals(group,tasks,seed):
    p=I.OUTPUTS/'B'/group/'step4/report.json'
    if p.exists():
        old=json.loads(p.read_text());placement=old.get('placement',{})
        if len(placement.get('bases',[]))==len(tasks):
            offsets=np.asarray(placement['offsets'],float);offsets[:,2]-=offsets[0,2]
            state=replace(seed,bases=np.asarray(placement['bases'],float),offsets=offsets)
            yield 'historical_transform_proposal',state,[]
            for attempt in (0,1,2):
                restored=restore(tasks,state,F.roots,attempt=attempt,iterations=40)
                if restored is not None:yield f'historical_seating_repair_{attempt}',restored[0],restored[1]
    for attempt in (0,1):
        restored=restore(tasks,seed,F.roots,attempt=attempt,iterations=40)
        if restored is not None:yield f'free_translation_repair_{attempt}',restored[0],restored[1]
    for gap in (.015,.04):yield f'separated_seating_{gap}',spread(tasks,seed,gap),[]


def piece(mesh,color,alpha=1.,bias=0):
    p=PUB.piece(mesh,color,bias);p['opacity']=alpha;return p


def pictures(out,step5,tasks,state,mesh,grow,metrics,group):
    renderer=V.Renderer();CPU=PUB.CPU
    colors=V.head_colors([c['candidate_id'] for row in state.groups for c in row])
    heads=[]
    for row,b,o in zip(state.groups,state.bases,state.offsets):
        for c in row:
            tri=c['triangles_m']@b+o
            heads.append((c['candidate_id'],trimesh.Trimesh(tri.reshape(-1,3),np.arange(tri.size//3).reshape(-1,3),process=False)))
    support=[piece(mesh,'#c2c9c8')]+[piece(h,colors[k],bias=3e-6) for k,h in heads]
    panels=[];size=700;titlefont=ImageFont.truetype(FONT,22)
    for t,b,o,path in zip(tasks,state.bases,state.offsets,state.paths):
        installed=(mesh.vertices-o)@b.T;d=np.asarray(path['initial_object_exit_world'])
        view=d+np.array([.5,-.8,.6]);camera=CPU.fit(np.vstack([installed,t.domain.mesh.vertices]),view,1.,padding=1.3)
        parts=[CPU.placed(p,b,-b@o) for p in support]
        parts.append(CPU.placed(piece(t.domain.mesh,'#91b5cd',.68)))
        im=renderer.render(parts,camera,size)
        ink=ImageDraw.Draw(im);ink.text((20,15),f'{t.pose} | complete shared support',font=titlefont,fill='#263642')
        panels.append(im)
    camera=CPU.fit(mesh.vertices,[1.,-1.3,.8],1.,padding=1.25)
    im=renderer.render([CPU.placed(p) for p in support],camera,size)
    ImageDraw.Draw(im).text((20,15),'One connected fixture',font=titlefont,fill='#263642');panels.append(im)
    columns=min(3,len(panels));rows=(len(panels)+columns-1)//columns
    canvas=Image.new('RGB',(columns*size,rows*size),'white')
    for i,im in enumerate(panels):canvas.paste(im,((i%columns)*size,(i//columns)*size))
    canvas.save(out/'overview.png')
    # All stages share a fixture-frame camera. Final panel has one object pose.
    cloud=np.vstack([mesh.vertices,tasks[0].domain.mesh.vertices@state.bases[0]+state.offsets[0]])
    camera=CPU.fit(cloud,[1.,-1.3,.8],1.,padding=1.3)
    stages=[('roots.obj','Actual contact heads'),('seed_targets.obj','Head starts'),('partial_growth.obj','Growing shared branches'),('shape.obj','Validated complete fixture')]
    sheet=Image.new('RGB',(4*size,size),'white')
    for i,(filename,label) in enumerate(stages):
        p=out/filename
        m=trimesh.load(p,force='mesh',process=False) if p.exists() else mesh
        parts=[CPU.placed(piece(m,'#c2c9c8'))]
        if i==3:
            obj=tasks[0].domain.mesh.copy();obj.vertices=obj.vertices@state.bases[0]+state.offsets[0]
            parts.append(CPU.placed(piece(obj,'#91b5cd',.68)))
        im=renderer.render(parts,camera,size);ImageDraw.Draw(im).text((20,15),label,font=titlefont,fill='#263642');sheet.paste(im,(i*size,0))
    sheet.save(out/'construction_steps.png')
    step5.mkdir(parents=True,exist_ok=True)
    _,_,supports=measure([t.domain.mesh.vertices for t in tasks],mesh.vertices,state.bases,state.offsets)
    BBOX.draw(step5/'overview.png',[t.domain.mesh for t in tasks],supports,mesh.faces,metrics)
    return ['overview.png','construction_steps.png']


def attempt(group,tasks,state,source,folder,label,cuts,proof):
    began=time.monotonic();folder.mkdir(parents=True,exist_ok=True)
    status=dict(complete=True,passed=False,label=label,stage='fixture_setup',placement=dict(bases=state.bases.tolist(),offsets=state.offsets.tolist()),separation_proposals=cuts)
    try:
        F.save_state(folder,tasks,state,dict(passed=True,scope='unchanged native heads with full source load proof'))
        points=np.vstack([t.domain.mesh.vertices@b+o for t,b,o in zip(tasks,state.bases,state.offsets)])
        lo=points.min(0)-.05;hi=points.max(0)+.05;lo[2]=min(state.offsets[:,2])
        guide=trimesh.creation.box(extents=hi-lo,transform=trimesh.transformations.translation_matrix((hi+lo)/2))
        F.SPACE.export_exact_obj(guide,folder/'navigation_guide.obj')
        ref=dict(complete=True,passed=False,scope='unaccepted navigation guide only',construction={},provenance=dict(inputs={},code={}))
        previous=F.E.PathAnalyzer;F.E.PathAnalyzer=P.PathAnalyzer
        previous_namespace=F.S.SimpleNamespace
        F.S.SimpleNamespace=lambda **kwargs:SimpleNamespace(**dict(metadata={},**kwargs))
        try:
            grow=RecoveryGrow(SimpleNamespace(name=group),tasks,state,folder,folder/'navigation_guide.obj',ref)
            if not proof:
                for task,row in zip(tasks,state.groups):
                    fresh=[{k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')} for c in row]
                    mask,_=J.classify(task.supply(fresh),task.targets)
                    if not mask.all():raise RuntimeError('Full original load proof failed')
            grow.case.paths.extend([source,folder/'state.json']);grow.recheck_export=False
            status['stage']='body_growth';grow.timings['setup']=time.monotonic()-began
            report=grow.run(.008)
        finally:
            F.E.PathAnalyzer=previous
            F.S.SimpleNamespace=previous_namespace
        report.pop('material_reduction_percent',None);report.pop('previous_material_volume_cm3',None)
        report.update(common_object_exit_world=state.paths[0]['initial_object_exit_world'],object_poses_changed=False,fixture_seating_fixed=False,load_count_per_pose=32768,original_loads_passed=True)
        report['construction']['reference_used_as']='navigation guide only; free fixture seating'
        report['provenance']['code'].update(I.hashes([Path(__file__),Path(F.__file__),Path(P.__file__),Path(__file__).with_name('operation_growth_recovery.py'),Path(__file__).with_name('feasible_seating.py')]))
        I.save(folder/'report.json',report)
        mesh=trimesh.load(folder/'shape.obj',force='mesh',process=False)
        status.update(passed=True,stage='complete_shared_fixture',material_cm3=report['volume_cm3'],seconds=time.monotonic()-began)
        I.save(folder/'attempt.json',status);return grow,report,mesh
    except Exception as error:
        status.update(error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc(),seconds=time.monotonic()-began)
        if hasattr(error,'access_report'):status['access_report']=error.access_report
        I.save(folder/'attempt.json',status);print('REJECT',group,label,status['stage'],str(error),flush=True);return None


def run(group):
    started=time.monotonic();base=I.OUTPUTS/'B'/group;source=base/'step3_scheculer/dsl_shared_direction_batch_v33/report.json';data=I.check_report(source)
    tasks=[current_task('B',p) for p in data['poses']];caches=[HC.load(HC.ROOT/'objects/B/poses'/p,200,strict=True) for p in data['poses']]
    trials=[dict(direction_world=data['shared_object_exit_world'],poses=data['per_pose'],passed=True,final_source=True)]
    trials += [dict(t,final_source=False) for t in data['history'] if t['passed'] and np.linalg.norm(np.asarray(t['direction_world'])-data['shared_object_exit_world'])>1e-8]
    # Prefer nearby upward exits if a horizontal local program cannot grow a body.
    trials[1:]=sorted(trials[1:],key=lambda t:-t['direction_world'][2])
    out=base/'step4'/STAGE;out.mkdir(parents=True,exist_ok=True);events=[];success=None
    for di,trial in enumerate(trials[:4]):
        contacts=[]
        for cache,row in zip(caches,trial['poses']):
            byid={e['contact']['candidate_id']:e['contact'] for pool in cache['pools'].values() for e in pool}
            contacts.append(tuple(byid[k] for k in row['selected_ids']))
        seed=F.State(tuple(contacts),np.repeat(np.eye(3)[None],len(tasks),axis=0),np.zeros((len(tasks),3)),tuple(F.ray(-np.asarray(trial['direction_world']),0) for _ in tasks))
        for pi,(label,state,cuts) in enumerate(proposals(group,tasks,seed)):
            folder=out/'attempts'/f'direction_{di:02d}_placement_{pi:02d}'
            proof=trial.get('final_source',False) and all(r.get('all_original_loads_passed') for r in trial['poses'])
            result=attempt(group,tasks,state,source,folder,label,cuts,proof)
            events.append(dict(direction=trial['direction_world'],label=label,folder=str(folder.relative_to(I.ROOT)),passed=result is not None))
            I.save(out/'progress.json',dict(complete=False,events=events))
            if result is not None:success=(state,folder,*result);break
        if success:break
    if not success:
        r=dict(complete=True,passed=False,constructed=False,group=group,events=events,seconds=time.monotonic()-started,provenance=dict(inputs=I.hashes([source]),code=I.hashes([Path(__file__)])))
        I.save(out/'report.json',r);return r
    state,folder,grow,report,mesh=success
    # Publish the new result inside its fresh stage, preserving historical public files.
    for name in ['shape.obj','geometry_certificate.npz','roots.obj','seed_targets.obj','partial_growth.obj','shared_tree.obj']:
        if (folder/name).exists():shutil.copy2(folder/name,out/name)
    F.save_state(out,tasks,state,dict(passed=True,all_original_loads_passed=True))
    aggregate,per_pose,_=measure([t.domain.mesh.vertices for t in tasks],mesh.vertices,state.bases,state.offsets)
    step5=base/'step5_evaluate'/STAGE
    pictures(out,step5,tasks,state,mesh,grow,aggregate,group)
    final=dict(report,events=events,seconds=time.monotonic()-started,group=group,source_witness=str((folder/'report.json').relative_to(I.ROOT)),artifacts={name:I.sha256(out/name) for name in ['shape.obj','geometry_certificate.npz','overview.png','construction_steps.png','state.json']})
    final['provenance']=dict(inputs=I.hashes([folder/'report.json',out/'state.json']),code=I.hashes([Path(__file__)]))
    I.save(out/'report.json',final);I.check_report(out/'report.json')
    step5report=dict(complete=True,passed=True,group=group,poses=data['poses'],aggregate=aggregate,per_pose=per_pose,material_volume_cm3=report['volume_cm3'],common_object_exit_world=state.paths[0]['initial_object_exit_world'],fixture_seating_fixed=False,object_poses_changed=False,provenance=dict(inputs=I.hashes([out/'report.json',out/'shape.obj']),code=I.hashes([Path(__file__),Path(BBOX.__file__)])),artifacts={'overview.png':I.sha256(step5/'overview.png')})
    I.save(step5/'report.json',step5report);I.check_report(step5/'report.json')
    print('PASS',group,'exit',state.paths[0]['initial_object_exit_world'],'material',report['volume_cm3'],'box',aggregate['object_and_support_poses']['box_volume_cm3'],'seconds',final['seconds'],flush=True)
    return dict(group=group,passed=True,seconds=final['seconds'],exit=state.paths[0]['initial_object_exit_world'],volume=aggregate['object_and_support_poses']['box_volume_cm3'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+',default=GROUPS);parser.add_argument('--jobs',type=int,default=1);args=parser.parse_args()
    began=time.monotonic()
    if args.jobs==1:results=[run(g) for g in args.sets]
    else:
        with ProcessPoolExecutor(args.jobs) as pool:
            futures={pool.submit(run,g):g for g in args.sets};results=[]
            for future in as_completed(futures):results.append(future.result())
    output=I.OUTPUTS/'B'/args.sets[-1]/'step5_evaluate'/STAGE;output.mkdir(parents=True,exist_ok=True)
    I.save(output/'batch.json',dict(complete=True,groups=results,wall_seconds=time.monotonic()-began))
