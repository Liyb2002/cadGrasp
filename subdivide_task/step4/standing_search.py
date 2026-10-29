"""Search a monotone branch whose final fixture is installed on a side face.

The previous flat incumbent is kept separately: a smaller flat solution cannot
replace a certified standing installation when standing is the requested goal.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh

from subdivide_task.operations import Layout, exit_direction_change,pose_change
from subdivide_task.operations.construction import CompactConstructionModel
from subdivide_task.operations.floors import planes
from subdivide_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from subdivide_task.step3.independent_physics import contacts
from subdivide_task.operations.geometry import moved_mesh
from subdivide_task.operations.installations import standing_pairs
from .optimize import Optimizer,subset
from .quarter_turns import propose, reassigned_groups
from .spatial_search import floor_repair,separate_free_axis
from . import aligned_side
from .placement import alternatives


def protect_exits(layout,owners=None,margin=.2):
    normals=[n for n,_ in planes(layout.material)]
    q=layout.copy()
    for k in range(len(q.poses))if owners is None else owners:
        direction=q.material[k,:3,:3]@q.exits[k]
        if min(n@direction for n in normals)<.05:
            for _ in range(20):
                for n in normals:
                    direction+=max(0.,margin-n@direction)*n
                direction/=np.linalg.norm(direction)
            world=q.material[k,:3,:3].T@direction
            if world[2]>=.05:
                q=exit_direction_change(q,k,world)
    return q


def seed_branch(source,target,index):
    """Fork a certified prefix without changing any previous accepted state."""
    source,target=Path(source),Path(target)
    if (target/'current.json').exists():return
    target.mkdir(parents=True,exist_ok=True)
    for k in range(index+1):
        for suffix in ('.json','.npz'):
            shutil.copy2(source/f'state_{k:03}{suffix}',target/f'state_{k:03}{suffix}')
    events=json.loads((source/'trace.json').read_text())
    end=next(i+1 for i,e in enumerate(events)if e['status']=='accepted' and
             abs(e['volume_cm3']-json.loads((target/f'state_{index:03}.json').read_text())['volume_cm3'])<1e-6)
    write_json(target/'trace.json',events[:end])
    manifest=json.loads((source/'current.json').read_text())
    state=target/f'state_{index:03}.npz'
    # The chosen prefix has the same task grouping as the later translation.
    if events[end-1].get('groups'):
        manifest['groups']=[[int(name[1:])-1 for name in group]for group in events[end-1]['groups']]
    manifest.update(state=state.name,sha256=sha256(state),accepted=index,
                    volume_cm3=json.loads(state.with_suffix('.json').read_text())['volume_cm3'])
    write_json(target/'current.json',manifest)


def run(workdir=None,seed_index=17,diverse=False,shared_exits=False,preserve_seats=False,ribbed=False,pack=False):
    data=ROOT/'subdivide_task/data/B'
    source=data/'step4_spatial_best'
    folder=Path(workdir or data/'step4_standing')
    seed_branch(source,folder,seed_index)
    write_json(folder/'requirements.json',dict(final_standing_installation_required=True,
        relative_floor_angle_deg=90.,tolerance_deg=1e-5,
        reason='The same complete fixture must actually be installed on its side for a task'))
    model=CompactConstructionModel(DEFAULT_OUTPUT,'B')
    if ribbed:model.stock_grid=(.03,.012)
    optimizer=Optimizer(model,folder,resume=True)
    if standing_pairs(optimizer.layout):
        print('ALREADY STANDING',optimizer.volume,flush=True)
        return optimizer
    # A proposed side turn can be combined with a translation; only its fully
    # audited endpoint is an accepted operation, so the prefix stays monotone.
    with np.load(source/'state_018.npz')as z:
        base=Layout.read(z)
        support=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
    model.current_contacts={}
    for t in model.prepare(base):
        tri,*_=contacts(t['body_world'],moved_mesh(support,np.linalg.inv(t['material'])),
                        t['domain'].work_ids,t['domain'].com)
        model.current_contacts[t['id']]=tri.reshape(-1,3)
    print('STANDING SEARCH READY',optimizer.volume,flush=True)
    tested=set()
    base_groups=json.loads((source/'current.json').read_text())['groups']
    if preserve_seats:
        for owner in range(len(model.ids)):
            raw=aligned_side.preserve_seat(model,base,owner)
            variants=[('floor_protected_exit_small_margin',protect_exits(raw,margin=.05)),
                      ('unchanged_material_exit',raw),('floor_protected_exit',protect_exits(raw))]
            for label,q in variants:
                for groups in(base_groups,[g for g in [[k for k in g if k!=owner]for g in base_groups]if g]+[[owner]]):
                    repaired=floor_repair(model,q,groups,owner)
                    if repaired is None:continue
                    assert standing_pairs(repaired)
                    if optimizer.try_operation(repaired,'support_turn',groups=groups,task=model.ids[owner],
                            final_standing_installation_required=True,preserve_original_object_to_fixture_seat=True,
                            axial_pose_angle_deg=180.,exit_variant=label):return optimizer
                    if 'paired force/moment certificate failed' in optimizer.last_trial_error:
                        failed=model.ids.index(optimizer.last_trial_error.split(':')[0])
                        for delta in(2.,-2.,5.,-5.,10.,-10.,20.,-20.):
                            adjusted=pose_change(repaired,failed,delta,model.axis,model.head)
                            adjusted=floor_repair(model,adjusted,groups,owner)
                            if adjusted is not None and optimizer.try_operation(adjusted,'support_turn',groups=groups,
                                    task=model.ids[owner],final_standing_installation_required=True,
                                    axial_pose_angle_deg=180.,exit_variant=label,
                                    repaired_task=model.ids[failed],repair_pose_angle_deg=delta):return optimizer
        raise RuntimeError('No certified seat-preserving side installation in this branch')
    ranked=[]
    for owner in range(len(model.ids)):
        for host in range(len(model.ids)):
            if owner==host:continue
            for angle in aligned_side.needle_angles(base,model.axis,owner,host):
                q=aligned_side.propose(model,base,owner,host,angle)
                if moved_mesh(model.source,q.poses[owner]).bounds[0,2]<-1e-12:continue
                ranked.append((abs(angle),owner,host,angle))
    print('EXACTLY ALIGNED SIDE CANDIDATES',len(ranked),flush=True)
    for _,owner,host,angle in sorted(ranked):
        print('SIDE PAIR',model.ids[owner],model.ids[host],angle,flush=True)
        offsets=((0.,0.,0.),)if diverse else((0.,0.,0.),(.015,0.,0.),(-.015,0.,0.),(0.,.015,0.),(0.,-.015,0.))
        for offset in offsets:
            raw=aligned_side.propose(model,base,owner,host,angle,offset)
            variants=[(True,protect_exits(raw)),(False,raw)]
            if shared_exits:
                protected=protect_exits(raw,margin=.05)
                variants=[(f'common_raw_{weight}',aligned_side.common_exit(raw,owner,host,weight))
                          for weight in(.5,.25,.75)]+[
                          (f'common_{weight}',aligned_side.common_exit(protected,owner,host,weight))
                          for weight in(.5,.25,.75)]+variants
            for joint,q in variants:
                if q is None:continue
                # Retain the previous cohorts where possible. Releasing just
                # these two tasks is also legal: cohorts are a search aid.
                cohesive=reassigned_groups(base_groups,[owner],host)
                released=[[k for k in g if k not in(owner,host)]for g in base_groups]
                released=[g for g in released if g]+[[owner,host]]
                for groups in(released,cohesive):
                    repaired=floor_repair(model,q,groups,host)
                    if repaired is None:continue
                    key=repaired.material.tobytes()+repaired.poses.tobytes()+repaired.exits.tobytes()
                    if key in tested:continue
                    tested.add(key)
                    assert standing_pairs(repaired)
                    detail=dict(tasks=[model.ids[owner],model.ids[host]],turn_angle_deg=90.,
                        axial_pose_angle_deg=angle,offset_m=list(offset),exact_object_orientation_alignment=True,
                        coupled_exit_direction_changes=joint,explicit_side_turn=True,
                        final_standing_installation_required=True,combined_with_translation=True)
                    if shared_exits:
                        local,lq=subset(model,repaired,[owner,host])
                        try:
                            small=local.build(lq)
                            local.audit(small,full=False,continuous=False)
                        except(ValueError,RuntimeError,AssertionError,np.linalg.LinAlgError)as error:
                            optimizer.event('aligned_side_local_rejected','support_turn',reason=str(error)[:250],**detail)
                            continue
                    if pack:
                        packed=separate_free_axis(model,repaired,groups,host)
                        if packed is not None and optimizer.try_operation(packed,'support_turn',groups=groups,
                                placement='separate other cohorts along the common floor tangent',**detail):return optimizer
                        continue
                    if optimizer.try_operation(repaired,'support_turn',groups=groups,**detail):return optimizer
                    # Small floor-tangent adjustments can protect an existing
                    # third seat without unpacking the complete fixture.
                    if not diverse and optimizer.last_trial_status=='rejected' and offset==(0.,0.,0.):
                        moving=next(g for g in groups if host in g)
                        for placed,delta in alternatives(model,repaired,moving):
                            if optimizer.try_operation(placed,'support_turn',groups=groups,
                                    placement_translation_m=delta.tolist(),**detail):return optimizer
    for owner,host in [(1,6),(6,12),(8,4),(6,4),(1,12),(8,12)]:
        for offset in ((.025,0.,0.),(0.,0.,0.),(.0125,0.,0.),(.0375,0.,0.),
                       (.025,.0125,0.),(.025,-.0125,0.),(.025,0.,.0125),(.025,0.,-.0125)):
            for joint in (True,False):
                q=propose(model,base,[owner],owner,host,'closest_side',90.,offset)
                if joint:q=protect_exits(q)
                groups=reassigned_groups(optimizer.groups,[owner],host)
                q=floor_repair(model,q,groups,host)
                if q is None:continue
                key=q.material.tobytes()+q.exits.tobytes()
                if key in tested:continue
                tested.add(key)
                assert standing_pairs(q)
                if optimizer.try_operation(q,'support_turn',groups=groups,
                        tasks=[model.ids[owner],model.ids[host]],turn_angle_deg=90.,
                        turn_axis_in_owner_world='closest_side',offset_m=list(offset),
                        coupled_exit_direction_changes=joint,explicit_side_turn=True,
                        task_level_turn=True,final_standing_installation_required=True,
                        combined_with_translation=True):
                    return optimizer
    raise RuntimeError('No certified standing improvement in this search; incumbent preserved')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir',type=Path)
    parser.add_argument('--seed-index',type=int,default=17)
    parser.add_argument('--diverse',action='store_true')
    parser.add_argument('--shared-exits',action='store_true')
    parser.add_argument('--preserve-seats',action='store_true')
    parser.add_argument('--ribbed',action='store_true')
    parser.add_argument('--pack',action='store_true')
    args=parser.parse_args()
    run(args.workdir,args.seed_index,args.diverse,args.shared_exits,args.preserve_seats,args.ribbed,args.pack)
