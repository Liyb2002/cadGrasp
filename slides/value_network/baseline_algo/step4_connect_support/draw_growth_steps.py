"""Render head growth with adaptive ground coverage and a measured final box.

The first two panels align object poses for illustration only. The third
uses workstation coordinates; the last two use the actual fixture frame.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import MultiPoint
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import clean_render as V, publish_compact as P
from step4_connect_support import boxed_support as F, deterministic_space as D
from step4_connect_support import build_coupled_saddle as S, space_budget as B
from step5_evaluate import render_bbox as BB

COLORS=['#5298bd','#d89a59','#80a886','#a98db7','#c3a264']


def alignment(source,target):
    a=source.mean(axis=0);b=target.mean(axis=0)
    u,_,vh=np.linalg.svd((source-a).T@(target-b))
    rotation=u@vh
    if np.linalg.det(rotation)<0:
        u[:,-1]*=-1;rotation=u@vh
    offset=b-a@rotation
    error=float(np.max(np.linalg.norm(source@rotation+offset-target,axis=1)))
    if error>1e-8:raise RuntimeError('Object poses do not share rigid vertex correspondence')
    return rotation,offset,error


def part(mesh,color,opacity=1.,bias=0.):
    result=P.piece(mesh,color,bias=bias)
    result['opacity']=opacity
    return P.CPU.placed(result)


def domain_wire(pic,mesh,camera):
    xy=BB.project(mesh.vertices,camera,pic.width)
    edges=mesh.face_adjacency_edges[mesh.face_adjacency_angles>1e-6]
    ink=ImageDraw.Draw(pic)
    for a,b in edges:ink.line([tuple(xy[a]),tuple(xy[b])],fill='#bdcbd1',width=2)


def direction_arrow(pic,start,end,camera,color,label=None):
    xy=BB.project(np.asarray([start,end]),camera,pic.width)
    vector=xy[1]-xy[0];length=float(np.linalg.norm(vector))
    if length<5:return None
    unit=vector/length;side=np.array([-unit[1],unit[0]])
    head=min(16.,length*.3);ink=ImageDraw.Draw(pic)
    ink.line([tuple(xy[0]),tuple(xy[1]-head*.6*unit)],fill=color,width=4)
    ink.polygon([tuple(xy[1]),tuple(xy[1]-head*unit+head*.42*side),tuple(xy[1]-head*unit-head*.42*side)],fill=color)
    if label:
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
        ink.text(tuple(xy.mean(axis=0)+14*side),label,font=font,anchor='mm',fill=color,stroke_width=2,stroke_fill='white')
    return dict(start_m=np.asarray(start).tolist(),end_m=np.asarray(end).tolist(),label=label)


def exit_arrows(pic,objects,directions,camera,poses,colors):
    records=[];view=camera[1][2]
    for k,(obj,direction) in enumerate(zip(objects,directions)):
        side=np.cross(view,direction);norm=np.linalg.norm(side)
        if norm<1e-8:continue
        side/=norm
        start=obj.bounds.mean(axis=0)+.075*side
        record=direction_arrow(pic,start,start+.060*direction,camera,colors[k%len(colors)],poses[k].replace('pose_','P')+' exit')
        if record:records.append(record)
    return records


def draw(folder):
    group=folder.parents[2]
    manifest_path=folder/'stages.json'
    manifest=json.loads(manifest_path.read_text())
    growth=I.check_report(folder/'report.json')
    boxed_path=group/'step4/data/boxed_support/report.json'
    boxed=I.check_report(boxed_path)
    # Obtain the same original poses and continuous sweeps as construction.
    original_reader=F.read_case
    if group.name=='pose1+3copied':
        from step4_connect_support.run_copied_reference import historical_case
        F.read_case=historical_case
    try:
        search=D.Search(group);search.precompute()
    finally:
        F.read_case=original_reader
    case=search.case
    poses=case.poses
    placement=growth['placement']
    bases,offsets,directions=map(np.asarray,(placement['bases'],placement['offsets'],placement['directions']))
    objects=[t.domain.mesh for t in case.tasks]
    rotations=[alignment(m.vertices,objects[0].vertices) for m in objects]
    owner_roots=[S.unpack(F.union([S.solid(D.G.hull_mesh(v)) for cells in row for v in cells]))
                 for row in case.support_seeds]
    canonical_roots=[];canonical_sweeps=[];fixture_objects=[]
    inputs=[manifest_path,folder/'report.json',boxed_path]+case.paths
    for k,(obj,root,sweep,(rotation,offset,error),basis,translation) in enumerate(zip(objects,owner_roots,search.sweep_meshes,rotations,bases,offsets)):
        aligned=root.copy();aligned.vertices=root.vertices@rotation+offset
        canonical_roots.append(aligned)
        aligned_sweep=sweep.copy();aligned_sweep.vertices=sweep.vertices@rotation+offset
        canonical_sweeps.append(aligned_sweep)
        installed=obj.copy();installed.vertices=obj.vertices@basis+translation
        fixture_objects.append(installed)
    ideal=B.box([m.vertices for m in objects]+[np.c_[xy,np.zeros(len(xy))] for xy in case.demands])
    reference=growth['construction'].get('reference_envelope')
    if reference:ideal=B.box([np.asarray([reference['min_m'],reference['max_m']])])
    accepted=growth['space_budget']
    bounded=S.unpack(F.bounded_space(accepted,bases,offsets))
    meshes={}
    for key in ['roots','partial_growth','final']:
        name='shape.obj' if key=='final' else key+'.obj'
        meshes[key]=trimesh.load(folder/name,force='mesh',process=False)
        inputs.append(folder/name)
    # Save exact full sweeps plus a explicitly cropped display-only neighborhood.
    near=objects[0].bounds.copy();near[0]-=.075;near[1]+=.075
    crop=trimesh.creation.box(near[1]-near[0]);crop.apply_translation(near.mean(axis=0))
    display_sweeps=[]
    for pose,sweep in zip(poses,canonical_sweeps):
        path=folder/f'process_sweep_{pose}_full.obj'
        D.export_exact_obj(sweep,path);inputs.append(path)
        display=S.unpack(S.solid(sweep)^S.solid(crop))
        path=folder/f'process_sweep_{pose}_display.obj'
        D.export_exact_obj(display,path);inputs.append(path);display_sweeps.append(display)
    renderer=V.Renderer()
    size=660;top=170;bottom=205
    reference_image=I.OUTPUTS/'B/pose1+3/step4/construction_steps.png'
    reuse_reference=group.name=='pose1+3copied'
    canvas=Image.new('RGB',(size*(4 if reuse_reference else 5),top+(size+bottom)*(2 if reuse_reference else 1)),'white');ink=ImageDraw.Draw(canvas)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title=ImageFont.truetype(font,38);label=ImageFont.truetype(font,28);small=ImageFont.truetype(font,21)
    ink.text((canvas.width/2,24),group.name+' - Step4 construction',font=title,anchor='mt',fill='#283b43')
    ink.text((canvas.width/2,80),'Contacts > Exit sweeps > Greedy growth > Validation > Actual bounding box',font=label,anchor='mt',fill='#62727b')
    ink.text((canvas.width/2,125),'Original reference panels 1-2; growth: fixture frame; bounding box: workstation frame' if reuse_reference else 'Panels 1-2: object alignment; growth/support: fixture frame; box: workstation frame',font=small,anchor='mt',fill='#62727b')
    canonical_camera=P.CPU.fit(objects[0].vertices,[-.5,.866,.65],1.,padding=1.48)
    sweep_camera=P.CPU.fit(np.vstack([m.vertices for m in display_sweeps]),[-.5,.866,.65],1.,padding=1.15)
    world_camera=P.CPU.fit(BB.corners(accepted),BB.VIEW,1.,padding=1.25)
    view=(-directions[0]+[0,0,.65])@bases[0]
    fixture_camera=P.CPU.fit(np.vstack([bounded.vertices,meshes['partial_growth'].vertices]+[m.vertices for m in fixture_objects]),view,1.,padding=1.2)
    object_part=part(objects[0],'#91b5cd',.45)
    fixture_parts=[part(m,COLORS[k%len(COLORS)],.12) for k,m in enumerate(fixture_objects)]
    contacts=[part(m,COLORS[k%len(COLORS)]) for k,m in enumerate(canonical_roots)]
    stages=[
        ('contacts','Contact roots','Contacts aligned on one object','Object-aligned illustration', [object_part]+contacts,canonical_camera),
        ('sweeps','Exit sweep exclusions','Support must avoid exit sweeps','500 mm checked; local sweep shown',
         [part(m,COLORS[k%len(COLORS)],.24) for k,m in enumerate(display_sweeps)]+[object_part]+contacts,sweep_camera),
        ('partial_growth','Greedy support growth','Green: growth; blue: required coverage','Fixture frame; whole rods preferred',
         fixture_parts+[part(meshes['partial_growth'],'#70a092'),part(meshes['roots'],'#da9864',bias=4e-6)],fixture_camera),
        ('final','Final support','One object pose; full support validated','Fixture frame',
         [part(fixture_objects[0],COLORS[0],.45)]+[part(meshes['final'],'#c1c8c7'),part(meshes['roots'],'#da9864',bias=4e-6)],fixture_camera),
        ('box','Actual bounding box','Blue: fixed reference; orange: with support','Workstation frame',
         [part(m,'#91b5cd',.25) for m in objects],world_camera)]
    world_supports=[]
    for basis,offset in zip(bases,offsets):
        mesh=meshes['final'].copy();mesh.vertices=(mesh.vertices-offset)@basis.T
        world_supports.append(part(mesh,'#c1c8c7'))
    stages[-1]=(stages[-1][0],stages[-1][1],stages[-1][2],stages[-1][3],stages[-1][4]+world_supports,stages[-1][5])
    records=[]
    for index,(key,heading,description,frame,parts,camera) in enumerate(stages):
        width=size*2 if reuse_reference and index in (1,2) else size
        if reuse_reference and index<2:
            # Reuse exact native diagram panels, including both exit arrows.
            # No reconstructed or stylized substitute for the requested reference.
            with Image.open(reference_image) as original:
                crop_box=(0,60,850,910) if index==0 else (850,60,2550,910)
                pic=original.crop(crop_box).resize((width,size),Image.Resampling.LANCZOS)
            inputs.append(reference_image)
            heading=['Selected heads and object pose','Aligned object and exit sweeps'][index]
            description=['Same reference geometry as original pose1+3','Blue: Pose 1 / Orange: Pose 3'][index]
            frame=['Original contacts and placement','100 mm shown; 500 mm checked'][index]
        else:
            pic=renderer.render(parts,camera,size)
        arrows=[];foot_markers=[]
        if key=='box':
            BB.wire_box(pic,ideal,camera,'#467f9e');BB.wire_box(pic,accepted,camera,'#c38346')
            origin=np.asarray(accepted['min_m']);length=min(accepted['extents_mm'])*.00022
            for axis,(name,color) in enumerate(zip('XYZ',['#ad6056','#689365','#577eaf'])):
                endpoint=origin.copy();endpoint[axis]+=length
                record=direction_arrow(pic,origin,endpoint,camera,color,name)
                if record:arrows.append(record)
        elif key=='sweeps' and not reuse_reference:
            arrows=exit_arrows(pic,[objects[0]]*len(poses),[-d@r[0] for d,r in zip(directions,rotations)],camera,poses,COLORS)
        elif key=='partial_growth':
            marker_ink=ImageDraw.Draw(pic)
            for k,demands in enumerate(case.demands):
                hull=MultiPoint(demands).convex_hull
                xy=np.asarray(hull.exterior.coords) if hull.geom_type=='Polygon' else np.asarray(demands)
                region=np.c_[xy,np.zeros(len(xy))]@bases[k]+offsets[k]
                projected=BB.project(region,camera,pic.width)
                if len(projected)>1:marker_ink.line([tuple(p) for p in projected],fill='#5298bd',width=3)
                foot_markers.extend(region.tolist())
            journal=growth['construction']['growth_journal']
            count=manifest['partial_stage_step_count']
            chosen=journal[:count]
            # Mark actual construction flow, rather than backwards solver paths.
            selected=np.linspace(0,len(chosen)-1,min(3,len(chosen)),dtype=int) if chosen else []
            for arrow_index,j in enumerate(selected):
                path=np.asarray(chosen[j]['path_m'])
                if len(path)<2:continue
                lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);segment=int(np.argmax(lengths))
                a,b=path[segment:segment+2]
                record=direction_arrow(pic,a*.7+b*.3,a*.25+b*.75,camera,'#287b55','Growth' if arrow_index==0 else None)
                if record:arrows.append(record)
        elif key=='final':
            arrows=exit_arrows(pic,fixture_objects[:1],[-directions[0]@bases[0]],camera,poses[:1],COLORS[:1])
        if pic.width!=width:
            padded=Image.new('RGB',(width,size),'white')
            padded.paste(pic,((width-pic.width)//2,0));pic=padded
        col,row=([(0,0),(1,0),(0,1),(2,1),(3,1)][index] if reuse_reference else (index,0))
        y=top+row*(size+bottom);x=col*size+width/2
        canvas.paste(pic,(col*size,y))
        ink.text((x,y+size+12),f'{index+1}  {heading}',font=label,anchor='mt',fill='#283b43')
        ink.text((x,y+size+57),description,font=small,anchor='mt',fill='#62727b')
        ink.text((x,y+size+91),frame,font=small,anchor='mt',fill='#62727b')
        if key=='box':
            dims=lambda box:' × '.join(f'{v:.0f}' for v in box['extents_mm'])+' mm'
            ink.text((x,y+size+127),('Fixed reference ' if reference else 'Lower bound ')+dims(ideal),font=small,anchor='mt',fill='#467f9e')
            ink.text((x,y+size+161),'Actual '+dims(accepted),font=small,anchor='mt',fill='#c38346')
        reused=reuse_reference and index<2
        records.append(dict(key=key,frame=frame,reused_reference_panel=reused,
            reference_crop_px=list(crop_box) if reused else None,
            direction_arrows=arrows,ground_coverage_outlines_m=foot_markers,displayed_object_poses=poses[:1] if key=='final' else poses,
            projection_square_size_px=None if reused else size,panel_padding_left_px=0 if reused else (width-size)//2,
            camera=None if reused else dict(focus=camera[0].tolist(),basis=camera[1].tolist(),span_m=float(camera[2]))))
    if reuse_reference:
        x=3*size+35;y=top+160
        lines=['How is the box obtained?','1. Fix the initial object/demand envelope.','2. Prefer less added box volume, then length.','3. Grow joins and repair coverage gaps.','4. Bound all objects and installed supports','   using workstation XYZ min/max.','','Validate once; measure the occupied box.','No box expansion or global optimum claim.']
        for i,line in enumerate(lines):ink.text((x,y+41*i),line,font=label if i==0 else small,fill='#283b43' if i==0 else '#62727b')
    output=group/'step4/construction_steps.png';canvas.save(output)
    code=[Path(__file__),Path(V.__file__),Path(P.__file__),Path(P.CPU.__file__),Path(BB.__file__),Path(D.__file__),Path(F.__file__),Path(S.swept_solid.__code__.co_filename)]
    I.save(folder/'construction_steps_render.json',dict(complete=True,actual_geometry=True,
        stages=records,object_alignment_errors_m=[v[2] for v in rotations],
        ideal_box=ideal,box_is_postconstruction_measurement=True,actual_final_box=growth['space_budget'],
        sweep_length_m=S.SWEEP_LENGTH,sweeps_include_full_geometry=True,
        saved_sweep_frame='Object-aligned illustration; actual construction uses each unaligned pose sweep transformed to fixture',
        reference_panels_source=str(reference_image.relative_to(I.ROOT)) if reuse_reference else None,
        reference_sweep_display_length_m=.1 if reuse_reference else None,
        display_only_crop=dict(min_m=near[0].tolist(),max_m=near[1].tolist()),
        source_acceptance_unchanged=True,provenance=dict(inputs=I.hashes(list(dict.fromkeys(inputs))),code=I.hashes(code)),
        artifacts={'../../construction_steps.png':I.sha256(output)}))
    I.check_report(folder/'construction_steps_render.json')
    print('FIVE-STAGE DIRECT HEAD GROWTH',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',default='pose1+3copied')
    args=p.parse_args();draw(I.OUTPUTS/'B'/args.group/'step4/data/growing_support')
