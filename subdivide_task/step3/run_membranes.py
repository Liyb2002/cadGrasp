"""Initialize thirteen independent membranes; render a separate connected baseline."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import trimesh

from subdivide_task.step1.common import ROOT,DEFAULT_OUTPUT,sha256,write_json
from subdivide_task.common.poses import spin_pose
from subdivide_task.operations.state import Layout
from subdivide_task.operations.geometry import moved_mesh
from .run_independent import load_inputs
from .membranes import MembraneModel,positive_components
from . import independent_geometry as G


def load_model(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    with np.load(folder/'initialization.npz') as z:
        source=trimesh.Trimesh(z['source_vertices'],z['source_faces'],process=False)
        outer=trimesh.Trimesh(z['outer_vertices'],z['outer_faces'],process=False)
        q=Layout.read(z)
    model=MembraneModel(source,outer,report['tasks'],report['workspace'],
        report['physical_properties'],report['policy'],report['object'])
    model.inputs=[folder/'report.json',folder/'initialization.npz']
    return model,q,report


def initialize_one(args):
    vertices,faces,ov,of,record,reference,workspace,physical,policy,name,folder=args
    source=trimesh.Trimesh(vertices,faces,process=False)
    model=MembraneModel(source,trimesh.Trimesh(ov,of,process=False),[record],workspace,physical,policy,name)
    path=Path(folder)/(record['id']+'.npz'); meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if saved.get('signature')==policy['initialization_signature'] and saved.get('sha256')==sha256(path):
            return saved
    trace=[]; direction=-model.axis
    for angle in policy['initial_axial_angles_deg']:
        P=spin_pose(reference,model.axis,model.head,angle)
        q=Layout(P[None],np.eye(4)[None],direction[None])
        try:
            candidate=model.build(q,facewise=True); model.audit(candidate)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as ex:
            trace.append(dict(angle_deg=angle,status='rejected',reason=str(ex)))
            print(record['id'],'REJECT',angle,str(ex)[:130],flush=True);continue
        np.savez_compressed(path,**q.arrays(),vertices=candidate['support'].vertices,faces=candidate['support'].faces)
        result=dict(**record,angle_deg=angle,volume_cm3=candidate['volume_cm3'],audit=candidate['audit'],
            sha256=sha256(path),signature=policy['initialization_signature'],trace=trace)
        write_json(meta,result)
        print(record['id'],'MEMBRANE PASS',round(candidate['volume_cm3'],3),flush=True)
        return result
    write_json(meta,dict(status='failed',trace=trace))
    raise RuntimeError(record['id']+': no legal thin-membrane initialization')


def baseline(model,q,shells):
    """Display comparison: separate thin shells connected by a ground frame.

    Lift each whole object/shell/tool installation to its own lowest point;
    join their existing lower boundary by narrow feet and a ground rail.
    These are actual additional solids and count in baseline volume only.
    """
    shells=[s.copy() for s in shells]; transforms=[];left=0.; feet=[];pieces=[]
    top=.008
    for i,shell in enumerate(shells):
        shift=np.array([left-shell.bounds[0,0],0.,top-shell.bounds[0,2]])
        shell.apply_translation(shift);left=shell.bounds[1,0]+.035
        transforms.append(shift);pieces.append(G.solid(shell))
        low=shell.bounds[0,2]; bottom=shell.vertices[np.argmin(shell.vertices[:,2])]
        feet.append(G.box([bottom[0]-.007,bottom[1]-.007,0.],
            [bottom[0]+.007,bottom[1]+.007,low+.004]))
    # The rail is an 8 mm high strip plus short lateral branches, rather
    # than the old filled casting/riser under each entire bunny silhouette.
    bounds=np.array([s.bounds for s in shells]); ymin=bounds[:,0,1].min()-.018
    pieces.extend(feet)
    pieces.append(G.box([bounds[:,0,0].min()-.01,ymin,0.],
        [bounds[:,1,0].max()+.01,ymin+.012,top]))
    for shell in shells:
        p=shell.vertices[np.argmin(shell.vertices[:,2])]
        pieces.append(G.box([p[0]-.007,ymin,0.],[p[0]+.007,p[1]+.007,top]))
    value=G.union(pieces)
    # The baseline uses the same physical cutouts in its translated stations.
    for i,shift in enumerate(transforms):
        single=model.subset([i]); qi=Layout(q.poses[i:i+1],np.eye(4)[None],q.exits[i:i+1])
        qi.material[0,:3,3]=shift
        t=single.prepare(qi)[0]
        value=value-t['sweep']-t['work']
    value=value.simplify(0.); actual=G.mesh(value)
    return actual,transforms,dict(volume_cm3=float(actual.volume*1e6),
        connected_component_count=len(positive_components(value)),
        method='same_membranes_on_thin_ground_rail',used_for_initialization=False)


def run(output_root=DEFAULT_OUTPUT,object_name='B',count=13,workers=4):
    import hashlib
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    began=time.monotonic(); output_root=Path(output_root).resolve()
    source,plan,arrays,config,physical,policy,inputs=load_inputs(output_root,object_name,count)
    membrane_policy=Path(__file__).with_name('membrane_policy.json')
    policy.update(json.loads(membrane_policy.read_text()))
    for p in [Path(__file__),Path(__file__).with_name('membranes.py'),membrane_policy]:inputs[str(p)]=sha256(p)
    folder=ROOT/'subdivide_task/data'/object_name/'step3_membranes';folder.mkdir(parents=True,exist_ok=True)
    outer_path=ROOT/'subdivide_task/data'/object_name/'step3_mold_plate/fitted_outer.npz'
    if not outer_path.exists():
        outer=G.mesh(G.fitted_outer(source,policy['mold_outward_displacement_m']))
        outer_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(outer_path,vertices=outer.vertices,faces=outer.faces)
    with np.load(outer_path) as z:outer=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
    inputs[str(outer_path)]=sha256(outer_path)
    signature=hashlib.sha256(json.dumps(dict(inputs=inputs,policy=policy),sort_keys=True).encode()).hexdigest()
    policy['initialization_signature']=signature
    args=[];records=[]
    for i in range(count):
        record=dict(id=str(arrays['working_area_ids'][i]),working_face_ids=np.flatnonzero(arrays['working_area_face_membership'][i]).tolist())
        records.append(record)
        args.append((source.vertices,source.faces,outer.vertices,outer.faces,record,
            arrays['screening_world_from_object'][i],plan['workspace'],physical,policy,object_name,folder))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(initialize_one,args))
    model=MembraneModel(source,outer,records,plan['workspace'],physical,policy,object_name)
    poses=[];exits=[];shells=[];saved={}
    for result in results:
        with np.load(folder/(result['id']+'.npz')) as z:
            poses.append(z['poses'][0]);exits.append(z['exits'][0])
            shell=trimesh.Trimesh(z['vertices'],z['faces'],process=False);shells.append(shell)
            saved[result['id']+'_vertices']=shell.vertices;saved[result['id']+'_faces']=shell.faces
    q=Layout(np.asarray(poses),np.tile(np.eye(4),(count,1,1)),np.asarray(exits))
    np.savez_compressed(folder/'initialization.npz',**q.arrays(),**saved,
        source_vertices=source.vertices,source_faces=source.faces,outer_vertices=outer.vertices,outer_faces=outer.faces)
    base,shifts,base_info=baseline(model,q,shells)
    np.savez_compressed(folder/'baseline.npz',vertices=base.vertices,faces=base.faces,display_shifts=shifts)
    base_q=q.copy();base_q.material[:,:3,3]=shifts
    base_candidate=dict(layout=base_q,support=base,tasks=model.prepare(base_q))
    base_proof=model.audit(base_candidate)
    write_json(folder/'baseline_validation.json',dict(status='passed',proof=base_proof,
        baseline_sha256=sha256(folder/'baseline.npz')))
    report=dict(status='passed',object=object_name,method='independent_contact_membranes',task_count=count,
        independent_fixture_count=count,tasks=records,task_results=results,workspace=plan['workspace'],policy=policy,
        physical_properties=physical,input_sha256=inputs,material_volume_cm3=sum(r['volume_cm3'] for r in results),
        connected_initialization_required=False,common_floor_or_head_height_required=False,
        original_sample_count=sum(r['audit']['tasks'][0]['force']['original_sample_count'] for r in results),
        continuous_outer_target_count=sum(r['audit']['tasks'][0]['force']['continuous_outer_target_count'] for r in results),
        baseline=base_info,seconds=time.monotonic()-began)
    write_json(folder/'report.json',report)
    # Independent reload: all geometry comes from the saved arrays, not the
    # workers' in-memory force classifications or constructive sweep unions.
    validate(folder)
    from .render_membranes import render
    render(model,q,shells,base,shifts,report,folder)
    target=output_root/object_name/'step3';target.mkdir(parents=True,exist_ok=True)
    for name in ('initialization.png','baseline_result.png'):shutil.copy2(folder/name,target/name)
    write_json(folder/'images.json',{name:sha256(folder/name) for name in ('initialization.png','baseline_result.png')})
    print('STEP3 MEMBRANES PASS',report['material_volume_cm3'],base_info,flush=True)
    return report


def validate(folder):
    folder=Path(folder);model,q,report=load_model(folder);proofs=[]
    for path,digest in report['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Changed initialization input: '+path)
    with np.load(folder/'initialization.npz') as z:
        for i,name in enumerate(model.ids):
            part=model.subset([i]);qi=Layout(q.poses[i:i+1],q.material[i:i+1],q.exits[i:i+1])
            support=trimesh.Trimesh(z[name+'_vertices'],z[name+'_faces'],process=False)
            candidate=dict(layout=qi,support=support,tasks=part.prepare(qi))
            proofs.append(part.audit(candidate));print(name,'DISK REPLAY PASS',flush=True)
    result=dict(status='passed',independent_fixture_count=len(proofs),tasks=proofs,
        initialization_sha256=sha256(folder/'initialization.npz'),report_sha256=sha256(folder/'report.json'))
    write_json(folder/'validation.json',result);return result


def refresh_saved(output_root=DEFAULT_OUTPUT,object_name='B'):
    """Re-certify unchanged saved membranes after implementation updates.

    Never rebuild or overwrite their arrays while an optimizer uses them.
    Both figures and both physical proofs are regenerated from the exact
    saved meshes; the new source binding is committed after successful checks.
    """
    folder=ROOT/'subdivide_task/data'/object_name/'step3_membranes'
    old,q,report=load_model(folder)
    source,plan,arrays,config,physical,policy,inputs=load_inputs(output_root,object_name,len(old.ids))
    np.testing.assert_array_equal(source.vertices,old.source.vertices)
    np.testing.assert_array_equal(source.faces,old.source.faces)
    for i,record in enumerate(old.records):
        np.testing.assert_array_equal(record['working_face_ids'],np.flatnonzero(arrays['working_area_face_membership'][i]))
    mp=Path(__file__).with_name('membrane_policy.json');policy.update(json.loads(mp.read_text()))
    for p in [Path(__file__),Path(__file__).with_name('membranes.py'),mp,
              Path(__file__).with_name('render_membranes.py'),
              ROOT/'subdivide_task/data'/object_name/'step3_mold_plate/fitted_outer.npz']:
        inputs[str(p)]=sha256(p)
    model=MembraneModel(source,old.outer,old.records,plan['workspace'],physical,policy,object_name)
    proofs=[];shells=[]
    with np.load(folder/'initialization.npz') as z:
        for i,name in enumerate(model.ids):
            support=trimesh.Trimesh(z[name+'_vertices'],z[name+'_faces'],process=False);shells.append(support)
            part=model.subset([i]);qi=Layout(q.poses[i:i+1],q.material[i:i+1],q.exits[i:i+1])
            proofs.append(part.audit(dict(layout=qi,support=support,tasks=part.prepare(qi))))
            print(name,'REFRESH REPLAY PASS',flush=True)
    with np.load(folder/'baseline.npz') as z:
        base=trimesh.Trimesh(z['vertices'],z['faces'],process=False);shifts=z['display_shifts']
    bq=q.copy();bq.material[:,:3,3]=shifts
    base_proof=model.audit(dict(layout=bq,support=base,tasks=model.prepare(bq)))
    report.update(input_sha256=inputs,policy=policy,physical_properties=physical,
        saved_arrays_unchanged_during_recertification=True)
    for record,proof in zip(report['task_results'],proofs):record['audit']=proof
    write_json(folder/'report.json',report)
    write_json(folder/'validation.json',dict(status='passed',independent_fixture_count=len(proofs),
        tasks=proofs,initialization_sha256=sha256(folder/'initialization.npz'),report_sha256=sha256(folder/'report.json')))
    write_json(folder/'baseline_validation.json',dict(status='passed',proof=base_proof,
        baseline_sha256=sha256(folder/'baseline.npz')))
    from .render_membranes import render
    render(model,q,shells,base,shifts,report,folder)
    target=Path(output_root)/object_name/'step3'
    for name in ('initialization.png','baseline_result.png'):shutil.copy2(folder/name,target/name)
    write_json(folder/'images.json',{name:sha256(folder/name) for name in ('initialization.png','baseline_result.png')})
    print('STEP3 REFRESH PASS',sha256(folder/'initialization.npz'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--refresh',action='store_true')
    a=p.parse_args()
    if a.refresh:refresh_saved()
    else:run(workers=a.workers)
