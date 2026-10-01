"""Reproduce the user's archived co-design shape and illustrate its construction.

This is an explicitly historical six-contact-surface replay. The source Step3
has five IDs; its repeated ID occupies two distinct physical patches in this
shape. Neither copying nor rendering certifies strict shared-head registration.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import manifold3d as md
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import ContinuousNeeds,demand
from step3_scheculer import contacts as I
from step2_local_support import geometry as G,circles
from step0_pose_selection.floor_points import pressure_centers
from step4_connect_support import build_coupled_saddle as S,outer_feet as O
from step4_connect_support import clean_render as V,publish_compact as P,working_surface as WS
from step4_connect_support import render_floor_demands as D
from step4_connect_support.render_pair_walkthrough import piece,grid

SOURCE=I.ROOT/'slides/co_design_algo/output/B/pose1+3'
ARCHIVE=SOURCE/'step5/data/history/before_fast_construction'
VARIANT=Path('sequential_3plus2/from_pose_1/terminal_expansion')
POSES=['pose_1','pose_3']
VIEW=np.array([np.cos(-2.35)*np.cos(.53),np.sin(-2.35)*np.cos(.53),np.sin(.53)])
GREY='#cbd2ce'
STEP_LABELS=[
    '1  选定头与物体摆放',
    '2  对齐物体，叠加退出方向与空间',
    '3  头延伸到最近地面',
    '4  补脚并裁剪，保留合法脚体',
    '5  添加短连接，得到最终支撑',
]
STEP_GROUPS=[[1],[2,3],[4],[5,6,7],[8,9]]
EXIT_COLORS=['#5298bd','#d89a59']
EXIT_DISPLAY_LENGTH=.10
PROCESS_DESCRIPTION='五步图从左到右、从上到下：①原头与固定摆放；②保留一个实体物体，把两个 pose 的退出方向与空间对齐到这个物体上；③头延伸到最近地面，彩色面为接地凸包需求；④补脚并裁剪后保留的合法脚体；⑤添加深灰短连接后的最终支撑。原步骤分组为 1 / 2–3 / 4 / 5–7 / 8–9。第②步中，Pose 3 的物体先对齐到 Pose 1，退出向量随同一旋转变换，蓝色为 Pose 1、橙色为 Pose 3。仅展示起始 100 mm。此图用于比较物体坐标中的退出方向；实际支撑禁区仍按原有各自摆放与完整 500 mm 扫掠计算，未改变构造几何。其余步骤以 Pose 1 的半透明物体作为参照。'


def steps_sheet(pictures,exit_view):
    """Five numbered stages with a single overlaid exit-space view."""
    size=850;header=60;footer=82
    canvas=Image.new('RGB',(3*size,header+2*(size+footer)),'white')
    draw=ImageDraw.Draw(canvas);font=P.font(32);small=P.font(25)
    draw.text((35,18),'第 2 步以同一个物体为参照 · 蓝色 Pose 1 / 橙色 Pose 3',font=small,fill='#657079')
    positions=[(0,header),(0,header+size+footer),(size,header+size+footer),(2*size,header+size+footer)]
    for picture,(x,y),label in zip(pictures,positions,[STEP_LABELS[i] for i in (0,2,3,4)]):
        canvas.paste(picture.resize((size,size),Image.Resampling.LANCZOS),(x,y))
        draw.text((x+size/2,y+size+12),label,font=font,fill='#303c43',anchor='mt')
    canvas.paste(exit_view.resize((2*size,size),Image.Resampling.LANCZOS),(size,header))
    draw.text((size+40,header+24),'Pose 1',font=font,fill=EXIT_COLORS[0],anchor='lt')
    draw.text((size+200,header+24),'Pose 3',font=font,fill=EXIT_COLORS[1],anchor='lt')
    draw.text((2*size,header+size+6),STEP_LABELS[1],font=font,fill='#303c43',anchor='mt')
    draw.text((2*size,header+size+47),'同一物体坐标 · 图示起始 100 mm',font=small,fill='#657079',anchor='mt')
    return canvas


def aligned_exit_data(case):
    """Register each object to Pose 1, applying the same rotation to its exit."""
    reference=case.tasks[0].domain
    frame0=np.asarray(reference.data['frame']['T_world_mesh'])
    basis0=case.bases[0];offset0=case.offsets[0]
    obj=trimesh.Trimesh(reference.mesh.vertices@basis0+offset0,reference.mesh.faces,process=False)
    directions=[];errors=[]
    for k,task in enumerate(case.tasks):
        frame=np.asarray(task.domain.data['frame']['T_world_mesh'])
        align=frame0@np.linalg.inv(frame)
        vertices=(task.domain.mesh.vertices@align[:3,:3].T+align[:3,3])@basis0+offset0
        error=float(np.linalg.norm(vertices-obj.vertices,axis=1).max())
        if error>1e-8:raise RuntimeError(f'Object registration failed for {task.pose}: {error}')
        # Fixture direction -> task world -> reference task world -> display.
        fixture_direction=-case.directions[k]@case.bases[k]
        direction=fixture_direction@case.bases[k].T@align[:3,:3].T@basis0
        directions.append(direction/np.linalg.norm(direction));errors.append(error)
    return obj,np.asarray(directions),errors


def exit_picture(case,renderer):
    """One solid object with both exits transformed into its reference frame."""
    obj,directions,_=aligned_exit_data(case)
    parts=[piece(obj,'#b5c1c7',1.,smooth=True)]
    bounds=[obj.vertices];arrows=[]
    view=VIEW/np.linalg.norm(VIEW)
    for k,direction in enumerate(directions):
        sweep=S.swept_solid(obj,EXIT_DISPLAY_LENGTH*direction)
        side=np.cross(view,direction);side/=np.linalg.norm(side)
        start=obj.bounds.mean(0)+side*.09
        end=start+direction*.085
        parts.append(piece(sweep,EXIT_COLORS[k],.14))
        bounds.extend([sweep.vertices,np.array([start,end])])
        arrows.append((start,end))
    focus,axes,span=P.CPU.fit(np.vstack(bounds),view,2.,padding=1.30)
    # Render and crop a square at twice the span to retain a 2:1 orthographic
    # viewport without stretching geometry or changing the shared camera.
    pic=renderer.render(parts,(focus,axes,2*span),1600).crop((0,400,1600,1200))
    ink=ImageDraw.Draw(pic)
    for k,(start,end) in enumerate(arrows):
        projected=(np.array([start,end])-focus)@axes.T
        xy=np.c_[800+projected[:,0]*800/span,400-projected[:,1]*800/span]
        vector=xy[1]-xy[0];unit=vector/np.linalg.norm(vector)
        normal=np.array([-unit[1],unit[0]])
        ink.line([tuple(xy[0]),tuple(xy[1]-20*unit)],fill=EXIT_COLORS[k],width=8)
        ink.polygon([tuple(xy[1]),tuple(xy[1]-34*unit+14*normal),
            tuple(xy[1]-34*unit-14*normal)],fill=EXIT_COLORS[k])
    return pic


def read_reference():
    report_path=ARCHIVE/'028c5be2286a_report.json'
    report=json.loads(report_path.read_text())
    lookup={I.sha256(p):p for p in ARCHIVE.iterdir() if p.is_file()}
    artifacts={name:lookup[digest] for name,digest in report['artifacts'].items()}
    source=SOURCE/'step3_scheculer'/VARIANT/'particle_009'
    schedule=json.loads((source/'schedule.json').read_text())
    domains=[];tasks=[];groups=[];paths=[report_path,source/'schedule.json'];directions=[]
    for pose,ident in zip(POSES,[60,68]):
        needs=SOURCE/'step_1_needs'/pose/'needs.json';samples=needs.with_name('samples.json')
        domain=ContinuousNeeds.read(needs);raw=json.loads(samples.read_text())
        assert I.sha256(needs)==raw['provenance']['physical_domain_sha256']
        snapshot=I.ROOT/'objects/B/history/before_compatible_poses_860a4233e74b/tasks'/pose/'setup.npz'
        assert I.sha256(snapshot)==domain.data['provenance']['setup_snapshot_sha256']
        with np.load(snapshot) as z:floor=z['floor_contact_m'].copy()
        targets=np.asarray(raw['need_wrench'])
        np.testing.assert_allclose(targets,demand(raw['pt_m'],raw['force_push_mg'],domain.com,domain.gravity),atol=1e-13,rtol=0)
        scale=np.r_[np.ones(3),np.ones(3)/domain.mesh.extents.max()]
        tasks.append(SimpleNamespace(pose=pose,domain=domain,targets=targets*scale,scale=scale,floor=floor))
        contact=source/f'contacts_{pose}.npz';groups.append(I.read_contacts(contact));domains.append(domain)
        catalogue=SOURCE/'step2_local_support'/VARIANT/f'candidates_{pose}.json'
        directions.append(json.loads(catalogue.read_text())['direction_catalogue']['vectors'][ident])
        paths.extend([needs,samples,snapshot,contact,catalogue])
    # Check historical input bytes, resolving only the known namespace/archive migration.
    for name,digest in report['provenance']['inputs'].items():
        if name.startswith('slides/baseline_algo/'):
            path=I.ROOT/name.replace('slides/baseline_algo/','slides/co_design_algo/',1)
        elif name.startswith('objects/B/tasks/'):
            path=I.ROOT/name.replace('objects/B/tasks/','objects/B/history/before_compatible_poses_860a4233e74b/tasks/',1)
        else:path=I.ROOT/name
        if I.sha256(path)!=digest:raise ValueError(f'Historical input mismatch: {path}')
        paths.append(path)
    with np.load(artifacts['geometry.npz']) as z:
        bases=z['rotations'].copy();offsets=z['local_offsets_m'].copy()
    owned={};heads=[];patches=[]
    frames=[np.asarray(d.data['frame']['T_world_mesh']) for d in domains]
    for k,(task,contacts) in enumerate(zip(tasks,groups)):
        vertex_offsets=G.vertex_offsets(task.domain.mesh,circles.DEPTH_FRACTION*task.domain.mesh.extents.max())[0]
        row=[]
        for contact in contacts:
            ident=contact['candidate_id']
            if ident not in owned:owned[ident]=(k,S.cells_for(contact,task.domain,vertex_offsets))
            owner,cells=owned[ident];transform=frames[k]@np.linalg.inv(frames[owner])
            world=[v@transform[:3,:3].T+transform[:3,3] for v in cells];row.append(world)
            local=[v@bases[k]+offsets[k] for v in world]
            v=np.vstack(local);v=v[ConvexHull(v).vertices]
            patches.append(dict(pose=k,id=ident,physical_id=f'{POSES[k]}:{ident}',v=v,
                root=v+.008*np.asarray(directions[k])@bases[k],
                original=O.union([S.solid(G.hull_mesh(v)) for v in local])))
        heads.append(row)
    return SimpleNamespace(report=report,artifacts=artifacts,source=source,schedule=schedule,
        tasks=tasks,groups=groups,heads=heads,patches=patches,bases=bases,offsets=offsets,
        directions=np.asarray(directions),paths=list(dict.fromkeys(paths)),
        design=json.loads(artifacts['design.json'].read_text()))


def reconstruct(case):
    b,o=case.bases,case.offsets;sweeps=[]
    box=md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
    for k,t in enumerate(case.tasks):
        obj=trimesh.Trimesh(t.domain.mesh.vertices@b[k]+o[k],t.domain.mesh.faces,process=False)
        sweeps.append(S.swept_solid(obj,-S.SWEEP_LENGTH*case.directions[k]@b[k]))
    forbidden=O.union([S.solid(m).minkowski_sum(box) for m in sweeps])
    forbidden=S.solid(S.unpack(forbidden))
    def carve(value):
        for basis,offset in zip(b,o):value=value.trim_by_plane(basis[2].tolist(),float(basis[2]@offset/S.SCALE))
        return value-forbidden
    bodies=[];initial=[];blanks=[]
    for patch,record in zip(case.patches,case.design['local_bodies']):
        k=record['initial_floor']['target_floor_index']
        xy=S.local_to_world(patch['v'],b[k],o[k])[:,:2];center=xy.mean(0)
        xy=center+.7*(xy-center)+record['initial_floor']['projection_shift_xy_m']
        pad=np.c_[xy,np.zeros(len(xy))]@b[k]+o[k]
        blank=S.solid(G.hull_mesh(np.vstack([patch['v'],patch['root'],pad])))
        body=O.component(carve(blank)+patch['original'],[patch['original']])
        if body is None:raise RuntimeError('Original local body could not be reconstructed')
        bodies.append(body);initial.append(body);blanks.append(blank)
    for row in case.design['attachments']:
        k=POSES.index(row['floor_pose']);h=row['head_body'];patch=case.patches[h]
        xy=np.asarray(row['polygon_xy_m']);pad=np.c_[xy,np.zeros(len(xy))]@b[k]+o[k]
        terminal=np.vstack([pad,pad+.003*b[k][2]])
        blank=S.solid(G.hull_mesh(np.vstack([patch['v'],patch['root'],terminal])))
        bodies[h]=O.component(bodies[h]+carve(blank),[bodies[h],carve(S.solid(G.hull_mesh(terminal)))])
        if bodies[h] is None:raise RuntimeError('Original attached foot could not be reconstructed')
        blanks.append(blank)
    differences=[]
    for k,body in enumerate(bodies):
        saved=S.solid(trimesh.load(case.artifacts[f'body{k}.obj'],force='mesh',process=False))
        diff=(abs(float((saved-body).volume()))+abs(float((body-saved).volume())))*S.SCALE**3
        differences.append(diff)
    bridges=[]
    for row in case.design['connections']:
        bead=trimesh.creation.icosphere(subdivisions=1,radius=row['radius_m']).vertices
        route=np.asarray(row['path_m'])
        bridges.append(carve(O.union([S.solid(G.hull_mesh(np.vstack([x+bead,y+bead]))) for x,y in zip(route[:-1],route[1:])])) )
    full=O.union(bodies+bridges)
    saved_mesh=trimesh.load(case.artifacts['../shape.obj'],force='mesh',process=False);saved=S.solid(saved_mesh)
    difference=(abs(float((full-saved).volume()))+abs(float((saved-full).volume())))*S.SCALE**3
    if max(differences)>8e-14:
        raise RuntimeError(f'Reference reconstruction differs: bodies={differences}, final={difference}')
    # The old mesher's bridge union may retain tiny boundary slivers. Validate
    # the replay against its recorded bridge surface before publishing it.
    old_bridges=trimesh.load(case.artifacts['bridges.obj'],force='mesh',process=False)
    new_bridges=S.unpack(O.union(bridges))
    bridge_surface_error=max(float(S.surface_distances(old_bridges,new_bridges.vertices).max()),
        float(S.surface_distances(new_bridges,old_bridges.vertices).max()))
    if bridge_surface_error>1e-8:raise RuntimeError(f'Bridge replay changed the saved surface: {bridge_surface_error}')
    return dict(sweeps=sweeps,forbidden=forbidden,initial=O.union(initial),legs=O.union(bodies),
        blanks=O.union(blanks),bridges=O.union(bridges),full=full,saved_mesh=saved_mesh,
        body_replay_errors_m3=differences,final_replay_error_m3=difference,
        bridge_vertex_surface_error_m=bridge_surface_error)


def install(case,state):
    group=I.OUTPUTS/'B/pose1+3';out=group/'step4'
    # Replace this turn's rejected presentation after the source comparisons.
    if out.exists():shutil.rmtree(out)
    for old in (group/'step0_pose_selection',group/'step3_scheculer/independent_poses_floor2mm'):
        if old.exists():shutil.rmtree(old)
    data=out/'data';data.mkdir(parents=True)
    imported=group/'step3_scheculer/co_design_reference';imported.mkdir(parents=True,exist_ok=True)
    for name in ('schedule.json','contacts_pose_1.npz','contacts_pose_3.npz','coverage.npz'):
        source=case.source/name
        if source.exists():shutil.copyfile(source,imported/name)
    copies=[]
    for k,pose in enumerate(POSES):
        inputs=data/'source_inputs'/pose;inputs.mkdir(parents=True)
        for name in ('needs.json','samples.json','examples.json'):
            src=SOURCE/'step_1_needs'/pose/name
            shutil.copyfile(src,inputs/name);copies.append(inputs/name)
        src=SOURCE/'step2_local_support'/VARIANT/f'candidates_{pose}.json'
        shutil.copyfile(src,inputs/'candidates.json');copies.append(inputs/'candidates.json')
    work=data/'current_build';work.mkdir()
    for name,src in case.artifacts.items():
        if name in ('../shape.obj','../overview.png'):continue
        if name.endswith('.obj') or name in ('geometry.npz','design.json'):
            shutil.copyfile(src,work/name)
    shutil.copyfile(case.artifacts['../shape.obj'],out/'shape.obj')
    shutil.copyfile(case.artifacts['../overview.png'],data/'reference_overview.png')
    I.save(data/'source_generation_report.json',case.report)
    shutil.copyfile(SOURCE/'step5/data/report.json',data/'source_registration_status.json')
    I.save(imported/'import.json',dict(complete=True,source=str(case.source.relative_to(I.ROOT)),
        copied_without_numerical_changes=True,search_rerun=False,source_pose_revision='archived co-design poses',
        source_step3_passed=case.schedule['passed'],covered_counts=case.schedule['covered_counts'],
        source_head_ids=case.schedule['selected_ids'],physical_patches_in_requested_shape=6,
        strict_shared_head_registration_passed=False,
        provenance=dict(inputs=I.hashes([case.source/'schedule.json']+[case.source/f'contacts_{p}.npz' for p in POSES]),
            code=I.hashes([Path(__file__)])),
        artifacts={p.name:I.sha256(p) for p in imported.iterdir() if p.is_file() and p.name!='import.json'}))
    return group,out,data


def render(case,state,group,out,data,steps_only=False):
    b,o=case.bases,case.offsets;mesh=state['saved_mesh'];renderer=V.Renderer()
    colors=V.head_colors([p['physical_id'] for p in case.patches])
    roots=[S.unpack(p['original']) for p in case.patches]
    heads=[piece(m,colors[p['physical_id']]) for p,m in zip(case.patches,roots)]
    for part,_,_ in heads:part['bias']=.000003
    patches=[]
    for k,contacts in enumerate(case.groups):
        for c in contacts:
            v=c['triangles_m'].reshape(-1,3)@b[k]+o[k]
            patches.append((f'{POSES[k]}:{c["candidate_id"]}',trimesh.Trimesh(v,np.arange(len(v)).reshape(-1,3),process=False)))
    support=[piece(mesh,GREY)]
    for ident,m in patches:
        patch=piece(m,colors[ident]);patch[0]['bias']=.000003;support.append(patch)
        collar=V.collar(mesh,m.vertices.min(0)-.004,m.vertices.max(0)+.004)
        if len(collar.faces):
            item=piece(collar,colors[ident]);item[0]['bias']=.000002;support.append(item)
    images=[]
    def draw(parts,camera,size=1200):
        return renderer.render(parts,camera,size)
    if not steps_only:
        overview=[]
        for k,task in enumerate(case.tasks):
            translation=-b[k]@o[k];points=np.vstack([mesh.vertices@b[k].T+translation,task.domain.mesh.vertices])
            cam=P.CPU.fit(points,VIEW,1.,padding=1.18)
            face_colors=np.tile(P.CPU.rgb('#acb9c0'),(len(task.domain.mesh.faces),3,1))
            face_colors[task.domain.work_ids]=P.CPU.rgb('#b4cba8')
            obj=P.CPU.piece(dict(v=task.domain.mesh.vertices.ravel(),f=task.domain.mesh.faces.ravel()),face_colors,smooth=True)
            obj['opacity']=.82
            lo,hi=points.min(0),points.max(0)
            floor=P.CPU.box(np.r_[hi[:2]-lo[:2]+.035,.001],np.r_[(lo[:2]+hi[:2])/2,-.0006],'#eff3f4');floor['unlit']=True
            parts=[P.CPU.placed(floor),P.CPU.placed(obj)]+[(part,b[k],translation) for part,_,_ in support]
            overview.append(draw(parts,cam))
        camera=P.CPU.fit(mesh.vertices,VIEW,1.,padding=1.2)
        overview.append(draw(support,camera))
        grid(overview,3,1200).save(out/'overview.png');images.append('overview.png')
    bound=mesh.bounds+np.array([[-.012]*3,[.012]*3]);box=trimesh.creation.box(bound[1]-bound[0]);box.apply_translation(bound.mean(0))
    raw=S.unpack(state['blanks']);initial=S.unpack(state['initial']);legs=S.unpack(state['legs'])
    removed=S.unpack(state['blanks']-state['legs']);connections=S.unpack(state['bridges'])
    demand_parts=[]
    for k,t in enumerate(case.tasks):
        xy=pressure_centers(t.targets/t.scale,t.domain.com)[0]
        hull=xy[ConvexHull(xy).vertices];pts=np.c_[hull,np.zeros(len(hull))]@b[k]+o[k]
        face=D.polygon_piece(pts,['#5298bd','#d89a59'][k]);face['opacity']=.22
        demand_parts.extend([P.CPU.placed(face),P.CPU.placed(D.boundary_piece(pts,['#5298bd','#d89a59'][k]))])
    stages=[heads,
        [piece(initial,GREY)]+heads+demand_parts,
        [piece(legs,GREY)]+heads,
        support+[piece(connections,'#7c8984')]]
    # Pose 1 stays fixed in the fixture frame throughout the construction.
    task=case.tasks[0]
    obj=trimesh.Trimesh(task.domain.mesh.vertices@b[0]+o[0],task.domain.mesh.faces,process=False)
    ghost=piece(obj,'#9aaeb9',.22,smooth=True)
    camera=P.CPU.fit(np.vstack([mesh.vertices,raw.vertices,box.vertices,obj.vertices]),VIEW,1.,padding=1.12)
    pics=[draw(parts+[ghost],camera) for parts in stages]
    steps_sheet(pics,exit_picture(case,renderer)).save(out/'construction_steps.png');images.append('construction_steps.png')
    if not steps_only:
        for name,m in [('initial_bodies',initial),('uncut_bodies',raw),('carved_bodies',legs),('removed_material',removed),('connections',connections),('forbidden',S.unpack(state['forbidden']))]:
            m.export(data/f'walkthrough_{name}.obj',digits=17,include_normals=False)
    return images


def render_saved_steps():
    """Update only the requested figure; reuse all existing construction meshes."""
    group=I.OUTPUTS/'B/pose1+3';out=group/'step4';data=out/'data'
    report_path=data/'report.json';report=json.loads(report_path.read_text())
    I.check_hashes(report['provenance']['inputs'])
    for name,digest in report['artifacts'].items():
        if I.sha256(data/name)!=digest:raise RuntimeError(f'Changed artifact: {name}')
    unchanged={name:I.sha256(out/name) for name in ('shape.obj','overview.png')}
    case=read_reference()
    state={'saved_mesh':trimesh.load(out/'shape.obj',force='mesh',process=False)}
    inputs=[out/'shape.obj']
    for key,name in [('initial','initial_bodies'),('blanks','uncut_bodies'),
        ('legs','carved_bodies'),('bridges','connections'),('forbidden','forbidden')]:
        path=data/f'walkthrough_{name}.obj';inputs.append(path)
        state[key]=S.solid(trimesh.load(path,force='mesh',process=False))
    state['sweeps']=[]
    for k,task in enumerate(case.tasks):
        obj=trimesh.Trimesh(task.domain.mesh.vertices@case.bases[k]+case.offsets[k],task.domain.mesh.faces,process=False)
        state['sweeps'].append(S.swept_solid(obj,-S.SWEEP_LENGTH*case.directions[k]@case.bases[k]))
    before=I.hashes(inputs)
    render(case,state,group,out,data,steps_only=True)
    assert unchanged=={name:I.sha256(out/name) for name in unchanged}
    assert before==I.hashes(inputs)
    report.setdefault('generation_provenance',copy.deepcopy(report['provenance']))
    report['provenance']['code'].update(I.hashes([Path(__file__)]))
    report['artifacts']['../construction_steps.png']=I.sha256(out/'construction_steps.png')
    report.update(text_in_images=True,text_by_image={'overview.png':False,'construction_steps.png':True},
        steps_object_pose='pose_1',steps_object_opacity=.22,arrows_in_images=True,
        steps_original_groups=STEP_GROUPS,exit_panel_object_opacity=1.,exit_space_opacity=.14,exit_panel_layout='one_object_aligned_exits',
        exit_display_length_m=EXIT_DISPLAY_LENGTH,exit_checked_length_m=S.SWEEP_LENGTH)
    report['presentation_update']=dict(geometry_changed=False,head_search_rerun=False,
        images_rendered=['construction_steps.png'],labels=STEP_LABELS,original_step_groups=STEP_GROUPS,
        exit_arrows='object-relative motion registered onto the single Pose 1 object',
        exit_directions_display=aligned_exit_data(case)[1].tolist(),
        object_registration_errors_m=aligned_exit_data(case)[2],exit_object_count=1,
        exit_directions_fixture=(-case.directions[:,None,:]@case.bases).reshape(-1,3).tolist(),
        exit_display_length_m=EXIT_DISPLAY_LENGTH,exit_checked_length_m=S.SWEEP_LENGTH,
        saved_geometry_inputs=before,object_pose='pose_1',object_opacity=.22,
        camera_frame='fixed fixture frame',exit_shared_view=VIEW.tolist(),exit_frame='object-aligned Pose 1 display frame')
    I.save(report_path,report)
    imported=group/'step3_scheculer/co_design_reference/import.json'
    manifest=json.loads(imported.read_text())
    manifest.setdefault('original_import_provenance',copy.deepcopy(manifest['provenance']))
    manifest['provenance']['code'].update(I.hashes([Path(__file__)]))
    manifest['presentation_update']=dict(import_rerun=False,code_change='Show one reference object and rotate each exit into its frame')
    I.save(imported,manifest)
    readme=out/'README.md';text=readme.read_text()
    paragraphs=text.split('\n\n')
    for i,paragraph in enumerate(paragraphs):
        if paragraph.startswith('五步图从左到右'):
            paragraphs[i]=PROCESS_DESCRIPTION
    readme.write_text('\n\n'.join(paragraphs))
    I.check_report(report_path);I.check_report(imported)
    print('PROCESS FIGURE UPDATED',out/'construction_steps.png',flush=True)


def main():
    case=read_reference();state=reconstruct(case)
    print('REFERENCE REPLAY',state['final_replay_error_m3'],flush=True)
    group,out,data=install(case,state)
    images=render(case,state,group,out,data)
    report=dict(complete=True,schema='co_design_reference_walkthrough_v1',constructed=True,passed=None,
        status='historical_shape_reconstructed_for_requested_walkthrough',poses=POSES,
        shape_matches_requested_reference=True,original_shape_bytes_preserved=True,
        old_and_current_pose_ids_are_not_interchangeable=True,
        step3_imported=True,step3_search_rerun=False,step3_passed=case.schedule['passed'],
        step3_covered_counts=case.schedule['covered_counts'],physical_contact_surface_count=6,
        source_candidate_id_count=5,strict_shared_head_registration_passed=False,
        force_torque_authority='step3',force_torque_recomputed=False,size_limit_enforced=False,
        volume_cm3=float(state['saved_mesh'].volume*1e6),dimensions_mm=(state['saved_mesh'].extents*1000).tolist(),
        placement=dict(bases=case.bases.tolist(),offsets=case.offsets.tolist(),directions=case.directions.tolist()),
        body_reconstruction_error_m3=state['body_replay_errors_m3'],
        final_reconstruction_error_m3=state['final_replay_error_m3'],
        bridge_vertex_surface_error_m=state['bridge_vertex_surface_error_m'],
        provenance=dict(inputs=I.hashes(case.paths+list(case.artifacts.values())),code=I.hashes([Path(__file__)])),
        artifacts={'../shape.obj':I.sha256(out/'shape.obj'),**{'../'+p:I.sha256(out/p) for p in images}},
        text_in_images=True,text_by_image={'overview.png':False,'construction_steps.png':True},
        steps_object_pose='pose_1',steps_object_opacity=.22,
        arrows_in_images=True,axes_in_images=False,html_generated=False,videos_generated=False,
        steps_original_groups=STEP_GROUPS,exit_panel_object_opacity=1.,exit_space_opacity=.14,exit_panel_layout='one_object_aligned_exits',
        exit_display_length_m=EXIT_DISPLAY_LENGTH,exit_checked_length_m=S.SWEEP_LENGTH)
    I.save(data/'report.json',report)
    (out/'README.md').write_text('''# co-design pose1+3 原 shape 的构造分解

按用户指定，沿用 co-design 图中的原始 shape、摆放、六块实际接触面和对应旧姿态。原 Step3 文件已复制到本组 `step3_scheculer/co_design_reference/`，每 pose 三个接触面、32,768/32,768 载荷通过。这次未重新选头。

仅保留两张大图（最终结果无文字，过程图含简短步骤说明与 Pose 1 的半透明物体）：[最终结果（两个 pose 与支撑）](overview.png) · [完整过程（五步合图）](construction_steps.png)。

五步图从左到右、从上到下：①原头与固定摆放；②保留一个实体物体，把两个 pose 的退出方向与空间对齐到这个物体上；③头延伸到最近地面，彩色面为接地凸包需求；④补脚并裁剪后保留的合法脚体；⑤添加深灰短连接后的最终支撑。原步骤分组为 1 / 2–3 / 4 / 5–7 / 8–9。第②步中，Pose 3 的物体先对齐到 Pose 1，退出向量随同一旋转变换，蓝色为 Pose 1、橙色为 Pose 3。仅展示起始 100 mm。此图用于比较物体坐标中的退出方向；实际支撑禁区仍按原有各自摆放与完整 500 mm 扫掠计算，未改变构造几何。其余步骤以 Pose 1 的半透明物体作为参照。

图由原配方逐块重建：初始脚体、原落脚面、头到脚的凸包、禁区裁剪、原短连接。每块身体及最终并集均与保存原模型比较；公开 shape.obj 保持原文件字节不变。原脚面和分配是已保存的设计参数，本次重放不是一次新的自动脚面搜索，也不声称新跑了 Step1–3。

来源限制：旧 Step3 有五个 ID，图中却有六块实际接触面，其中同名的两块并不重合。这次复现用户指定原形状，不把它当成严格五头共享的新成功解；原配准失败报告保留在 data/source_registration_status.json。旧 pose1/3 与当前二十姿态中的同名 pose 不同，未混用输入。其他八组与冻结的 co-design 文件没有修改。
''')
    print('CO-DESIGN WALKTHROUGH READY',out/'overview.png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps-only',action='store_true',help='Redraw the process sheet from saved geometry')
    args=parser.parse_args()
    if args.steps_only:render_saved_steps()
    else:main()
