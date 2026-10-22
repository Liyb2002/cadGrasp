"""Coverage of six axis-aligned tabletop object poses at a fixed robot station.

Each named object side faces the robot horizontally; the object rests on one
table instead of tilting to match the slanted needle. Needle axes are searched
within the fixed world cone, with exact tool, object, table and arm checks.
Partial coverage is allowed by default. Small detached islands are removed and
enclosed pockets join their surrounding task. --require-coverage opts into the
configured coverage threshold; other unassigned faces are never forced to a task.
"""
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from robots_config.constraints import load_config
from .common import ROOT,DEFAULT_OUTPUT,load_shape,sha256,write_json
from . import reach as R
from . import arm as A
from .areas import SIDES,VIEWS,areas,task_regions,Surface,fill_enclosed,drop_small
from .workspace import tabletop_placements

HERE=Path(__file__).parent
POLICY=HERE/'reach_policy.json'
SOURCES=['reach.py','arm.py','areas.py','partition.py','workspace.py','plan_reach.py','reach_render.py','reach_video.py','reach_policy.json',
         'needle_workspace.json','common.py']

_screen=None;_mesh=None;_arm=None


def _init_screen(path,length,radius,needle_radius,pixel,samples,incidence):
    global _screen;_screen=R.Screen(load_shape(path),length,radius,needle_radius,pixel,samples,incidence)


def _screen_axis(d):return np.packbits(_screen(d))


def _init_mesh(path,setup=None):
    global _mesh,_arm;_mesh=load_shape(path);_arm=A.Check(_mesh,setup) if setup else None


def _arm_pairs(job):
    """One side's pairs in walking order, each solve starting from the previous joints."""
    j,W,pairs=job;out=[];previous=None
    for face,axis in pairs:
        code,Q=_arm.pair(face,axis,W,previous);out.append((face,j,code,Q))
        if code==A.OK:previous=Q[-1]
    return out


def _arm_replay(job):
    W,pairs=job;return sum(_arm.replay(face,axis,W,Q) for face,axis,Q in pairs)


def _certify(job):
    """First option (smallest tilt) that passes the exact test; record the ones that fail."""
    face,options,length,radius,needle_radius,eps,incidence=job;failed=[]
    for pose,axis,d in options:
        ok,needle,head=R.certify(_mesh,face,d,length,radius,needle_radius,eps,incidence)
        if ok:return face,pose,axis,needle,head,failed
        failed.append(axis)
    return face,None,None,None,None,failed


def setup(object_name='B',config_path=None,target_path=None):
    tool=json.loads((HERE/'needle_workspace.json').read_text())
    config=load_config(config_path or ROOT/'robots_config/04_kuka_lbr_med14_r820.json')
    shape=ROOT/'objects'/object_name/'mesh.stl';mesh=load_shape(shape)
    target=np.ones(len(mesh.faces),bool)
    if target_path:
        target[:]=False;target[json.loads(Path(target_path).read_text())['face_ids']]=True
    task=config['task_constraint'];w=np.asarray(task['working_direction_world'],float);window=task['world_position_constraint']
    base=config['feasibility']['base_pose_world']
    cube=dict(center_m=list(map(float,window['center_world_m'])))
    return mesh,shape,target,dict(needle_length_m=tool['needle_length_m'],head_radius_m=tool['head_proxy_radius_m'],
        needle_radius_m=tool['needle_outer_radius_m'],max_tilt_deg=float(task['cone_half_angle_deg']),
        max_incidence_deg=None,config_incidence_deg_not_applied=float(task['surface_incidence_half_angle_deg']),
        work_window_center_m=cube['center_m'],
        base_position_world_m=list(map(float,base['position_m'])),base_rotation_world=base['rotation_matrix'],
        working_direction_world=(w/np.linalg.norm(w)).tolist(),robot=config['id'],robot_name=config['name'],
        target='user_faces:'+str(target_path) if target_path else 'whole_surface',
        tool_source=tool['length_source']['title'])


def screen(mesh,shape,s,policy,work):
    axes=R.sphere_axes(policy['tool_axis_spacing_deg'])
    pixel=policy['screen_pixel_fraction_of_diagonal']*float(np.linalg.norm(mesh.extents))
    samples=int(policy['screen_samples_per_pixel']*mesh.area/pixel**2)
    key=json.dumps(dict(shape=sha256(shape),reach=sha256(HERE/'reach.py'),length=s['needle_length_m'],
        radius=s['head_radius_m'],needle_radius=s['needle_radius_m'],incidence=s['max_incidence_deg'],axes=len(axes),
        pixel=pixel,samples=samples),sort_keys=True)
    cache=work/'screen.npz'
    if cache.exists():
        with np.load(cache) as z:
            if str(z['key'])==key:return axes,z['reach'].copy(),pixel
    with ProcessPoolExecutor(policy['workers'],initializer=_init_screen,
            initargs=(str(shape),s['needle_length_m'],s['head_radius_m'],s['needle_radius_m'],pixel,samples,
                      s['max_incidence_deg'])) as pool:
        packed=np.array(list(pool.map(_screen_axis,axes,chunksize=8)))
    reach=np.unpackbits(packed,axis=1,count=len(mesh.faces)).astype(bool)
    np.savez_compressed(cache,key=key,reach=reach,axes=axes);return axes,reach,pixel


def pose_tilts(axes,reach,poses,beta):
    """tilt[f,a]: smallest needle tilt from pose axis a that reaches face f (inf: none)."""
    cosb=np.cos(np.radians(beta));out=np.full((reach.shape[1],len(poses)),np.inf)
    for j,a in enumerate(poses):
        c=axes@a;near=np.flatnonzero(c>=cosb-1e-12);angle=np.arccos(np.clip(c[near],-1,1))
        out[:,j]=np.where(reach[near],angle[:,None],np.inf).min(0)
    return out


def placed(mesh,directions,s):
    """Six named object sides face the robot horizontally and rest on one tabletop."""
    return tabletop_placements(mesh,directions,s)


def certify_directions(pool,mesh,ids,axes,reach,directions,beta,s,eps,placements):
    """Certified tilt (faces, directions) in radians (inf: no axis within beta passes) and the axis used."""
    cosb=np.cos(np.radians(beta));C=axes@np.asarray(directions).T;jobs=[]
    for f in ids:
        for j in range(len(directions)):
            k=np.flatnonzero((C[:,j]>=cosb-1e-12)&reach[:,f])
            if len(k):k=k[np.argsort(-C[k,j])];jobs.append((int(f),[(j,int(i),axes[i]) for i in k],
                s['needle_length_m'],s['head_radius_m'],s['needle_radius_m'],eps,s['max_incidence_deg']))
    tilt=np.full((len(mesh.faces),len(directions)),np.inf);witness=np.zeros((len(mesh.faces),len(directions),3))
    gaps=np.zeros((len(mesh.faces),len(directions),2));rejected=0
    for face,j,k,needle,head,failed in pool.map(_certify,jobs,chunksize=16):
        rejected+=len(failed)
        if j is not None:tilt[face,j]=np.arccos(min(C[k,j],1.));witness[face,j]=axes[k];gaps[face,j]=(needle,head)
    # The arm behind the head, for every pair the needle and head pass.
    joints=np.zeros((len(mesh.faces),len(directions),4,7));code=np.zeros((len(mesh.faces),len(directions)),np.int8)
    chunks=[]
    for j,pl in enumerate(placements):
        W=np.eye(4);W[:3,:3]=pl['rotation_world_object'];W[:3,3]=pl['translation_world_m']
        f=np.flatnonzero(np.isfinite(tilt[:,j]));f=f[A.order(mesh.triangles_center[f]@W[:3,:3].T)]
        chunks+=[(j,W,[(int(g),witness[g,j]) for g in f[i:i+60]]) for i in range(0,len(f),60)]
    for rows in pool.map(_arm_pairs,chunks):
        for face,j,c,Q in rows:code[face,j]=c;joints[face,j]=Q
    tilt[code!=A.OK]=np.inf
    return tilt,witness,gaps,joints,code,dict(pairs_tried=len(jobs),rejected_screened_axes=rejected,
        arm_pairs=int(sum(len(c[2]) for c in chunks)),arm_no_ik=int((code==A.NO_IK).sum()),
        arm_hits_object=int((code==A.ARM_HITS).sum()),tool_hits_table=int((code==A.TABLE_HITS).sum()))


def plan(object_name='B',config_path=None,target_path=None,output_root=DEFAULT_OUTPUT,render=True,allow_partial=True):
    start=time.monotonic();policy=json.loads(POLICY.read_text())
    mesh,shape,target,s=setup(object_name,config_path,target_path)
    from .reach_video import work_point
    # The configured window is an unverified estimate; the object sits where the arm reaches all of it.
    point,forward=work_point(s,.5*float(mesh.extents.max()))
    s.update(work_point_m=point.tolist(),work_point_forward_of_window_m=forward,
             work_point_rule='first point forward of the window centre, along base->window, where the KUKA reaches every tip within half the object size at any tilt in the cone')
    table_z=float(point[2]-.5*mesh.extents.max())
    s.update(table_height_m=table_z,table_bounds_world_m=[
        [point[0]-.25,point[1]-.25,table_z-.02],[point[0]+.25,point[1]+.25,table_z]],
        pose_rule='six object-coordinate sides toward the robot horizontally; original +Y up for side views, -Z/+Z up for top/bottom; lowest vertex on the common table')
    work=ROOT/'whole_task_2/data'/object_name/'step1_reach';work.mkdir(parents=True,exist_ok=True)
    axes,reach,pixel=screen(mesh,shape,s,policy,work);screened=time.monotonic()-start
    tilt_limit=s['max_tilt_deg'];eps=policy['needle_start_fraction_of_length']*s['needle_length_m'];required=policy['required_coverage']
    area=mesh.area_faces;ids=np.flatnonzero(target);total=float(area[ids].sum())
    placements=placed(mesh,VIEWS,s);names=list(SIDES)
    directions=np.array([np.asarray(p['rotation_world_object']).T@np.asarray(s['working_direction_world']) for p in placements])
    with ProcessPoolExecutor(policy['workers'],initializer=_init_mesh,initargs=(str(shape),s)) as pool:
        tilt,witness,gaps,joints,code,cert=certify_directions(pool,mesh,ids,axes,reach,directions,tilt_limit,s,eps,placements)
        print('STEP1 ARM',cert['arm_pairs'],'pairs, no IK',cert['arm_no_ik'],'arm hits object',cert['arm_hits_object'],flush=True)
        covered=np.isfinite(tilt).any(1)
        screened_reach=np.isfinite(pose_tilts(axes,reach,directions,tilt_limit))
        possible=float(area[ids][screened_reach.any(1)[ids]].sum()/total)
        goal=required
        print('STEP1 SIDES COVER',round(float(area[ids][covered[ids]].sum()/total),4),'possible',round(possible,4),'goal',round(goal,4),flush=True)
    directions=np.asarray(directions);covered_fraction=float(area[ids][covered[ids]].sum()/total)
    if covered_fraction<required and not allow_partial:
        raise ValueError(f'Certified coverage {covered_fraction:.4%} is below the required {required:.4%}; '
                         'remaining faces cannot be assigned as a completed plan')
    regions,enclosed=task_regions(mesh,ids,np.isfinite(tilt),policy['small_piece_fraction'])
    removed=np.isfinite(tilt)&~regions
    partition_tilt=np.where(regions,tilt,np.inf)
    label,split=areas(mesh,ids,partition_tilt,directions,tilt_limit,policy['assignment_tilt_weight'],policy['small_piece_fraction'])
    final_coverage=float(area[ids][(label>=0)[ids]].sum()/total)
    # Certified: the face's own side paints it.  The rest (no side paints it, or a small piece given
    # to a neighbour) has a task but no certified needle axis.
    hit=(label>=0)&np.isfinite(tilt[np.arange(len(label)),np.maximum(label,0)]);nearest=(label>=0)&~hit
    face_axis=np.zeros((len(mesh.faces),3));face_tilt=np.zeros(len(mesh.faces));face_gaps=np.zeros((len(mesh.faces),2))
    face_axis[hit]=witness[hit,label[hit]];face_tilt[hit]=np.degrees(tilt[hit,label[hit]]);face_gaps[hit]=gaps[hit,label[hit]]
    # An uncertified face gets its area's nominal direction.
    face_axis[nearest]=directions[label[nearest]]
    rows=[]
    for j,(name,a) in enumerate(zip(names,directions)):
        f=np.flatnonzero(label==j);can=np.isfinite(tilt[:,j])
        rows.append(dict(id=name,axis_object=a.tolist(),view_object=VIEWS[j].tolist(),placement=placements[j],
            reach_area_fraction_of_target=float(area[can].sum()/total),reach_face_count=int(can.sum()),
            task_area_fraction_of_target=float(area[regions[:,j]].sum()/total),
            enclosed_face_count=int(enclosed[:,j].sum()),
            removed_face_count=int(removed[:,j].sum()),removed_area_fraction_of_target=float(area[removed[:,j]].sum()/total),
            face_ids=f.tolist(),face_count=len(f),area_m2=float(area[f].sum()),area_fraction_of_target=float(area[f].sum()/total),
            pieces=split['pieces'][j],mean_tilt_deg=float(np.average(face_tilt[f],weights=area[f])) if len(f) else 0.,
            max_tilt_deg=float(face_tilt[f].max()) if len(f) else 0.))
    uncovered=ids[~np.isfinite(tilt[ids]).any(1)]
    for j,row in enumerate(rows):row['uncertified_face_count']=int((nearest&(label==j)).sum())
    face_task=label.copy()
    tasks=[dict(row,direction_index=j,direction_id=row['id']) for j,row in enumerate(rows)]
    report=dict(status='analyzed' if allow_partial else 'passed',
        model='fixed_robot_six_axis_aligned_tabletop_poses',object=object_name,setup=s,allow_partial_analysis=allow_partial,
        required_coverage=required,possible_coverage_screened=possible,coverage_goal=goal,
        requirement_met=bool(covered_fraction>=required),covered_fraction_of_target=final_coverage,
        coverage_definition='task assignment after removing small detached components and filling enclosed pockets; other faces stay unassigned',
        certified_fraction_before_completion=covered_fraction,
        task_regions_union_fraction=float(area[regions.any(1)&target].sum()/total),
        retained_certified_union_fraction=float(area[(regions&np.isfinite(tilt)).any(1)&target].sum()/total),
        overlapping=dict(sum_of_reach_fractions=float(sum(r['reach_area_fraction_of_target'] for r in rows)),
            faces_reached_by=[int(n) for n in np.bincount(np.isfinite(tilt[ids]).sum(1),minlength=len(names)+1)]),
        direction_count=len(rows),area_count=len(rows),areas=rows,split=split,extra_directions=len(names)-6,
        task_count=len(tasks),tasks=tasks,
        active_task_count=sum(bool(task['face_count']) for task in tasks),
        empty_task_ids=[task['id'] for task in tasks if not task['face_count']],
        task_definition='exclusive surface partition shared by Step1.2, Step1.3 and downstream tasks; empty input poses remain recorded',
        task_surface_rule='face_task == direction_index; cleaned task_reach contains overlapping candidate reach regions only',
        target_face_count=int(target.sum()),target_area_m2=total,
        unreachable_face_ids=uncovered.tolist(),unreachable_area_fraction_of_target=float(area[uncovered].sum()/total),
        unassigned_face_ids=np.flatnonzero(target&(label<0)).tolist(),unassigned_area_fraction_of_target=1-final_coverage,
        split_fraction_of_target=final_coverage,
        split_certified_fraction_of_target=float(area[hit].sum()/total),uncertified_face_ids=np.flatnonzero(nearest).tolist(),
        uncertified='small enclosed pockets belong to the surrounding task even if unpaintable; detached small islands are discarded, other gaps remain unassigned',
        reach_definition=dict(needle='line from the tip back to infinity misses the object beyond needle_start_m',
            needle_body='needle radius clears the object beyond the tip contact r(1+tan(incidence)), incidence capped at 80 deg',
            head='no object point within head_radius_m of the needle line at needle_length_m and beyond',
            facing='face normal faces the tool',face_test_points='three vertices and the centroid',
            arm_motion='fixed base (robots_config); needle within max_tilt_deg of the slanted nominal direction',
            arm='KUKA inverse kinematics at every test point along the certified axis (joint limits, above the floor), '
                'convex hulls of all link collision meshes clear of the object and tabletop; joints kept for replay',
            placement=s['pose_rule'],table='needle body and head above the table; arm collision hulls clear of the finite tabletop',
            needle_start_m=eps),
        grids=dict(tool_axes=len(axes),tool_axis_spacing_deg=policy['tool_axis_spacing_deg'],screen_pixel_m=pixel),
        certification=dict(exact=True,method='clipped-triangle distance to the needle line (reach.clearance)',
            faces_assigned=int(hit.sum()),**cert,
            min_needle_body_gap_minus_radius_m=float(face_gaps[hit,0].min()-s['needle_radius_m']),
            min_head_gap_minus_radius_m=float(face_gaps[hit,1].min()-s['head_radius_m'])),
        seconds=dict(screen=screened,total=time.monotonic()-start),
        input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [shape,*[HERE/n for n in SOURCES],
            Path(config_path or ROOT/'robots_config/04_kuka_lbr_med14_r820.json')]})
    np.savez_compressed(work/'plan.npz',face_area=label,face_task=face_task,face_certified=hit,face_axis=face_axis,face_tilt_deg=face_tilt,gaps=face_gaps,
        area_axes=directions,reach=np.isfinite(tilt),task_reach=regions,task_enclosed=enclosed,task_removed=removed,
        reach_axis=witness,target=target,arm_code=code,joints=joints)
    report['arrays_sha256']=sha256(work/'plan.npz');write_json(work/'plan.json',report)
    if render:
        from .reach_render import publish
        report['figures']=publish(mesh,report,Path(output_root)/object_name/'step1');write_json(work/'plan.json',report)
    print('STEP1 SIDES',len(rows),'TASKS',len(tasks),'final task coverage',final_coverage,'pieces',split['pieces'],flush=True)
    return report


def validate(object_name='B'):
    """Replay from disk: every split face and every overlapping (face, side) pair passes the exact test."""
    work=ROOT/'whole_task_2/data'/object_name/'step1_reach';report=json.loads((work/'plan.json').read_text())
    if sha256(work/'plan.npz')!=report['arrays_sha256']:raise ValueError('plan.npz changed')
    s=report['setup'];shape=ROOT/'objects'/object_name/'mesh.stl'
    with np.load(work/'plan.npz') as z:a={k:z[k] for k in z.files}
    eps=report['reach_definition']['needle_start_m'];L,H,N=s['needle_length_m'],s['head_radius_m'],s['needle_radius_m']
    pairs=np.argwhere(a['reach']);hit=a['face_area']>=0
    jobs=[(int(f),[(int(j),0,a['reach_axis'][f,j])],L,H,N,eps,s['max_incidence_deg']) for f,j in pairs]
    arm_jobs=[]
    for j,row in enumerate(report['areas']):
        W=np.eye(4);W[:3,:3]=row['placement']['rotation_world_object'];W[:3,3]=row['placement']['translation_world_m']
        f=np.flatnonzero(a['reach'][:,j]);arm_jobs+=[(W,[(int(g),a['reach_axis'][g,j],a['joints'][g,j]) for g in f[i:i+60]])
                                                      for i in range(0,len(f),60)]
    with ProcessPoolExecutor(20,initializer=_init_mesh,initargs=(str(shape),s)) as pool:
        passed=sum(r[1] is not None for r in pool.map(_certify,jobs,chunksize=32))
        held=sum(pool.map(_arm_replay,arm_jobs))
    D=a['area_axes'];tilt=np.degrees(np.arccos(np.clip(np.einsum('fjk,jk->fj',a['reach_axis'],D)[a['reach']],-1,1)))
    certified=a['face_certified'];split_ok=bool(np.all(a['reach'][np.flatnonzero(certified),a['face_area'][certified]]))
    split_axes=np.allclose(a['face_axis'][certified],a['reach_axis'][np.flatnonzero(certified),a['face_area'][certified]])
    mesh=load_shape(shape);area=mesh.area_faces
    certified_covered=float(area[a['reach'].any(1)&a['target']].sum()/area[a['target']].sum())
    covered=float(area[hit&a['target']].sum()/area[a['target']].sum())
    regions,enclosed=task_regions(mesh,np.flatnonzero(a['target']),a['reach'],report['split']['small_piece_fraction'])
    complete=certified_covered>=report['required_coverage']
    split_target=a['target']
    closed_labels=a['face_area'].copy()
    fill_enclosed(Surface(mesh),np.flatnonzero(split_target),closed_labels,area,
                  report['split']['small_piece_fraction']*area[split_target].sum())
    pruned_labels=closed_labels.copy()
    drop_small(Surface(mesh),np.flatnonzero(split_target),pruned_labels,area,
               report['split']['small_piece_fraction']*area[split_target].sum())
    surface=Surface(mesh);task_labels=np.full(len(mesh.faces),-1);disjoint=True;inherited=True;metadata=True
    expected=placed(mesh,VIEWS,s);poses_match=True;axes_match=True;on_table=True
    for k,task in enumerate(report['tasks']):
        f=np.asarray(task['face_ids'],int);j=task['direction_index'];parent=report['areas'][j]
        disjoint&=bool(np.all(task_labels[f]<0) and len(np.unique(f))==len(f));task_labels[f]=k
        inherited&=bool(j==k and np.all(a['face_area'][f]==j) and task['id']==task['direction_id']==parent['id'] and
            all(task[key]==parent[key] for key in ('placement','axis_object','view_object')))
        metadata&=bool(task['face_count']==len(f) and task['pieces']==len(surface.pieces(a['face_task']==k)) and
            abs(task['area_m2']-area[f].sum())<1e-12 and
            abs(task['area_fraction_of_target']-area[f].sum()/area[a['target']].sum())<1e-12)
        poses_match&=all(np.allclose(task['placement'][key],expected[k][key],atol=1e-12) for key in expected[k])
        rot=np.asarray(task['placement']['rotation_world_object']);shift=np.asarray(task['placement']['translation_world_m'])
        axes_match&=np.allclose(rot@a['area_axes'][k],s['working_direction_world'],atol=1e-12)
        on_table&=abs((mesh.vertices@rot.T+shift)[:,2].min()-s['table_height_m'])<1e-12
    own=hit&a['reach'][np.arange(len(hit)),np.maximum(a['face_area'],0)]
    checks=dict(exact_reach=passed==len(jobs),arm_replay=held==len(jobs),arm_code_ok=bool(np.all(a['arm_code'][a['reach']]==0)),tilt_within_limit=bool(tilt.max()<=s['max_tilt_deg']+1e-9),
        split_inside_reach=split_ok and bool(split_axes),certified_means_own_side_paints=bool(np.all(certified==own)),
        assigned_faces_inside_cleaned_task_regions=bool(np.all(regions[np.flatnonzero(hit),a['face_area'][hit]])),
        six_tabletop_poses_match=bool(poses_match and report['task_count']==6 and [t['id'] for t in report['tasks']]==list(SIDES)),
        fixed_world_needle_direction=bool(axes_match),objects_rest_on_common_table=bool(on_table),
        tasks_are_disjoint=bool(disjoint),
        task_labels_match_report=bool(np.array_equal(task_labels,a['face_task'])),
        tasks_cover_exactly_the_direction_partition=bool(np.array_equal(a['face_task'],a['face_area'])),
        tasks_inherit_direction_and_pose=bool(inherited),task_metadata_matches=bool(metadata),
        task_counts_match=bool(report['task_count']==len(report['tasks'])==report['direction_count']==len(report['areas']) and
            len({t['id'] for t in report['tasks']})==len(report['tasks'])),
        active_task_counts_match=bool(report['active_task_count']==sum(bool(t['face_count']) for t in report['tasks']) and
            report['empty_task_ids']==[t['id'] for t in report['tasks'] if not t['face_count']]),
        completion_status_matches_coverage=bool(report['requirement_met']==complete and
            report['status']==('analyzed' if report['allow_partial_analysis'] else 'passed')),
        partial_analysis_is_explicit=bool(complete or report.get('allow_partial_analysis',False)),
        goal_is_required=bool(report['coverage_goal']==report['required_coverage']),
        enclosed_task_regions_match=bool(np.array_equal(regions,a['task_reach']) and np.array_equal(enclosed,a['task_enclosed'])),
        removed_task_regions_match=bool(np.array_equal(a['reach']&~regions,a['task_removed'])),
        no_small_enclosed_split_holes=bool(np.array_equal(closed_labels,a['face_area'])),
        no_small_detached_split_islands=bool(np.array_equal(pruned_labels,a['face_area'])),
        coverage_matches_report=bool(abs(covered-report['covered_fraction_of_target'])<1e-12),
        certified_coverage_matches_report=bool(abs(certified_covered-report['certified_fraction_before_completion'])<1e-12),
        uncertified_listed=np.flatnonzero(hit&~certified).tolist()==report['uncertified_face_ids'],
        unassigned_listed=np.flatnonzero(a['target']&~hit).tolist()==report['unassigned_face_ids'],
        unreachable_listed=sorted(np.flatnonzero(a['target']&~a['reach'].any(1)).tolist())==sorted(report['unreachable_face_ids']))
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,pairs=len(jobs),
        pairs_passed=passed,arm_pairs_replayed=held,max_tilt_deg=float(tilt.max()),
        task_count=report['task_count'],direction_count=report['direction_count'],
        covered_fraction_of_target=covered,requirement_met=bool(complete),plan_sha256=sha256(work/'plan.json'))
    write_json(work/'validation.json',result)
    if result['status']!='passed':raise ValueError('Step1 reach validation failed: '+json.dumps(checks))
    print('STEP1 VALIDATION PASSED',passed,'pairs',flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',default='B');p.add_argument('--config',type=Path)
    p.add_argument('--target',type=Path,help='JSON with the user-drawn face_ids; default the whole surface')
    p.add_argument('--no-render',action='store_true');p.add_argument('--validate-only',action='store_true')
    coverage=p.add_mutually_exclusive_group()
    coverage.add_argument('--allow-partial',dest='allow_partial',action='store_true',help='Allow partial coverage (default)')
    coverage.add_argument('--require-coverage',dest='allow_partial',action='store_false',help='Require certified coverage to meet the configured target')
    p.set_defaults(allow_partial=True)
    a=p.parse_args()
    if not a.validate_only:plan(a.object,a.config,a.target,render=not a.no_render,allow_partial=a.allow_partial)
    validate(a.object)
