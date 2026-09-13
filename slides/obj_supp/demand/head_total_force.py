"""Schematic workpiece reactions on a connected support, without a floor base.

Actual B pose_2 contact C139 and the saved side frame are reused. Candidate
C023 sits on the crown with an overhead arm; C151 sits below the chin. The
ground plane is visible, without the ground-base ring.
Orange/blue arrows point outward: force FROM the workpiece ON the support.
A red arrow presses downward on the green work region. Magnitudes are
illustrative; no load-case, bearing, or trajectory claim is made.
"""
from pathlib import Path
import sys
import json
from types import SimpleNamespace
import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'tools'))
import slide_scene as SC
BASE=HERE.parents[1]/'baseline_algo'
sys.path.insert(0,str(BASE))
from step1.cases import selected_pose
from step3_scheculer import contacts as I
from step3_scheculer.stage_imports import load_stage
from step2_local_support import render as R, insertion_directions as ID
from step5_connect_support import solids as S
C=load_stage('score','contribution')
PAPER=SC.PAPER
INK='#151515'
HIGH=SC.ORANGE
LOW=SC.BLUE
FRAME=SC.FRAME
APPLIED=SC.RED


def arrow(draw,a,b,color,width=9):
    a,b=np.asarray(a,float),np.asarray(b,float)
    v=b-a; length=np.linalg.norm(v); u=v/length
    draw.line([tuple(a),tuple(b-u*14)],fill=color,width=width)
    side=np.array([-u[1],u[0]])
    draw.polygon([tuple(b),tuple(b-u*24+side*12),tuple(b-u*24-side*12)],fill=color)


def centered(draw,xy,text,size):
    draw.text(xy,text,font=R.font(size),fill=INK,anchor='mm')


def flared_contact(center, normal, radius):
    """A shallow, broad contact lip tapering back to the existing neck."""
    profile=np.array([[0.,.0005],[radius*.94,.0005],
                      [radius,.0015],[radius*.95,.0030],
                      [.005,.0070],[0.,.0070]])
    mesh=trimesh.creation.revolve(profile,sections=48)
    mesh.apply_transform(trimesh.geometry.align_vectors([0.,0.,1.],normal))
    mesh.apply_translation(center)
    return mesh


def run():
    R.PAPER=PAPER
    case=I.OUTPUTS/'B/pose_2'
    with selected_pose('pose_2'):domain,_,_=C.P.read('B')
    contacts={h['candidate_id']:h for h in I.read_contacts(case/'step3_scheculer/final_contacts.npz')}
    z=I.load_npz(case/'step5_connect_support/geometry.npz')
    parts=S.unpack_parts(z);labels=z['part_labels'].tolist()
    head_labels=[s for s in labels if s.startswith('contact_head_')]
    selected=[]
    for i,label in enumerate(labels):
        if label.startswith('contact_head_'):
            keep=any(k in label for k in ['C139'])
        elif label.startswith('neck_'):
            keep=any(k in head_labels[int(label.split('_')[-1])] for k in ['C139'])
        else:keep=not label.startswith(('ground_strip_','connector_'))
        if keep:selected.append(i)
    # Add an actual upward-bearing contact below the chin, visibly between the
    # existing high and low heads. Its broad neck is a conceptual connection.
    circles_path=case/'step2_local_support/circles.json'
    data_path=case/'step2_local_support/circles.npz'
    directions_path=case/'step2_local_support/insertion_directions.json'
    circles=json.loads(circles_path.read_text())
    data=SimpleNamespace(**dict(np.load(data_path)))
    directions=json.loads(directions_path.read_text())
    depth=directions['normal_depth_m']
    top_row=next(r for r in directions['candidates'] if r['candidate_id']=='C023')
    top=ID.candidate(data,circles,top_row['candidate_index'])
    contacts['C023']=top
    assert not np.intersect1d(top['source_faces'],domain.work_ids).size
    top_heads,_=S.joined_heads(domain.mesh,[top],depth)
    top_n=domain.mesh.face_normals[top['center_face']]
    top_a=top['center_m']+.5*depth*top_n
    top_b=top_a+np.array([0.,0.,.015])
    top_c=np.array([.107,.020,top_b[2]])
    top_d=np.array([.107,.020,.156])
    top_parts=top_heads+[
        trimesh.creation.cylinder(radius=.005,segment=[top_a,top_b],sections=20),
        trimesh.creation.cylinder(radius=.005,segment=[top_b,top_c],sections=20),
        trimesh.creation.icosphere(subdivisions=2,radius=.005).apply_translation(top_b)]
    for j,part in enumerate(top_parts):
        selected.append(len(parts));parts.append(part);labels.append(f'illustrative_C023_{j:03d}')
    for j,part in enumerate([
        trimesh.creation.cylinder(radius=.005,segment=[top_c,top_d],sections=20),
        trimesh.creation.icosphere(subdivisions=2,radius=.005).apply_translation(top_c)]):
        selected.append(len(parts));parts.append(part);labels.append(f'overhead_frame_{j:03d}')
    upper_only=[]
    for i in selected:
        label=labels[i]
        if label.startswith('contact_head_C139_'):continue
        if label.startswith('neck_') and 'C139' in head_labels[int(label.split('_')[-1])]:continue
        upper_only.append(i)
    row=next(r for r in directions['candidates'] if r['candidate_id']=='C151')
    added=ID.candidate(data,circles,row['candidate_index'])
    contacts['C151']=added
    assert not np.intersect1d(added['source_faces'],domain.work_ids).size
    depth=directions['normal_depth_m']
    added_heads,_=S.joined_heads(domain.mesh,[added],depth)
    n=domain.mesh.face_normals[added['center_face']]
    a=added['center_m']+.5*depth*n
    b=np.array([.109,-.011,.121])
    c=np.array([.106,.039,.121])
    extra=added_heads+[
        trimesh.creation.cylinder(radius=.005,segment=[a,b],sections=20),
        trimesh.creation.cylinder(radius=.005,segment=[b,c],sections=20),
        trimesh.creation.icosphere(subdivisions=2,radius=.005).apply_translation(b)]
    for j,part in enumerate(extra):
        selected.append(len(parts));parts.append(part);labels.append(f'illustrative_C151_{j:03d}')
    blue_lips=[]
    for key,radius in [('C139',.0180),('C151',.0140)]:
        contact=contacts[key]
        normal=domain.mesh.face_normals[contact['center_face']]
        lip=flared_contact(contact['center_m'],normal,radius)
        selected.append(len(parts));parts.append(lip);labels.append(f'flared_lip_{key}')
        blue_lips.append({'candidate_id':key,'radius_m':radius,'depth_m':.007,
                          'scope':'Illustrative widened contact lip, not a recomputed contact patch'})
    extent=np.vstack([domain.mesh.vertices,SC.floor(domain).reshape(-1,3)]+[parts[i].vertices for i in selected])
    # Preserve the original view after migrating the world coordinates to Z-up.
    view_vector=SC.VIEW.tolist()
    basis=R.axes(view_vector)
    bounds=np.vstack([(extent@basis.T).min(0),(extent@basis.T).max(0)])
    focus=bounds.mean(0)@basis
    width=SC.CAMERA_MARGIN*max(bounds[1,:2]-bounds[0,:2])
    size=1100
    image=Image.new('RGB',(2400,1400),PAPER);draw=ImageDraw.Draw(image)
    centered(draw,(1200,66),"The heads' total force",54)
    centered(draw,(1200,122),'B / pose 2     |     Schematic unanchored support; support weight neglected',28)
    draw.line((1200,190,1200,1280),fill='#dbdbd7',width=2)
    metadata={}
    load_face=SC.LOAD_FACE
    assert load_face in domain.work_ids
    load_point=domain.mesh.triangles_center[load_face]
    load_direction=np.array([0.,0.,-1.])
    load_arrow_length=.045
    for panel,ids in enumerate([upper_only,selected]):
        x=40+1200*panel;y=182
        tri=[SC.floor(domain),domain.mesh.triangles]
        palette=[np.tile(SC.FLOOR,(2,1)),np.tile(SC.GREY,(len(domain.mesh.faces),1))]
        palette[-1][domain.work_ids]=SC.GREEN
        for i in ids:
            tri.append(parts[i].triangles)
            label=labels[i]
            color=HIGH if 'C023' in label else LOW if any(k in label for k in ['C139','C151']) else FRAME
            if label.startswith('neck_'):
                owner=head_labels[int(label.split('_')[-1])]
                color=HIGH if 'C023' in owner else LOW
            palette.append(np.tile(color,(len(parts[i].faces),1)))
        picture,_=R.raster(np.concatenate(tri),np.concatenate(palette),focus,basis,width,size,unlit=[0,1])
        image.paste(picture,(x,y))
        centered(draw,(x+550,204),'(a) One upper contact' if panel==0 else '(b) One upper and two lower contacts',31)
        # The arrowhead lands on the actual green work face; the applied force
        # acts ON the workpiece, unlike the contact reaction arrows below.
        load_end=R.project(load_point,focus,basis,width,size)[:2]+[x,y]
        load_start=R.project(load_point-load_arrow_length*load_direction,focus,basis,width,size)[:2]+[x,y]
        arrow(draw,load_start,load_end,APPLIED,10)
        draw.text((load_start[0]-22,load_start[1]+24),'Applied force',
                  font=R.font(25),fill=APPLIED,anchor='rm')
        for key,color,force_length in [('C023',HIGH,.036)]+([('C139',LOW,.026),('C151',LOW,.034)] if panel else []):
            h=contacts[key];p=h['center_m'];n=domain.mesh.face_normals[h['center_face']]
            assert n[2]>0 if key=='C023' else n[2]<0
            # Workpiece force ON the support is along the outward surface normal.
            start=R.project(p,focus,basis,width,size)[:2]+[x,y]
            end=R.project(p+force_length*n,focus,basis,width,size)[:2]+[x,y]
            arrow(draw,start,end,color,10)
            draw.ellipse((start[0]-6,start[1]-6,start[0]+6,start[1]+6),fill=INK)
            metadata[key]={'center_m':p.tolist(),'outward_normal':n.tolist(),'force_on_support_direction':n.tolist(),'illustrative_arrow_length_m':force_length}
        centered(draw,(x+550,1230),
            'The workpiece lifts the upper contact.' if panel==0
            else 'The workpiece also pushes the lower contacts down.',26)
    arrow(draw,[350,1310],[430,1310],HIGH,7)
    arrow(draw,[445,1310],[525,1310],LOW,7)
    draw.text((555,1310),'Workpiece force on support',font=R.font(27),fill=INK,anchor='lm')
    draw.rectangle((1570,1297,1611,1323),fill=FRAME)
    draw.text((1641,1310),'Rigid connection',font=R.font(27),fill=INK,anchor='lm')
    centered(draw,(1200,1366),'Illustrative forces and arrow lengths.',24)
    image.save(HERE/'head_total_force.png')
    report={
        'case':'B/pose_2','depicted_heads':['C023','C139','C151'],
        'arrow_magnitudes':'illustrative, not calculated reactions',
        'arrow_convention':'Red: downward applied force ON the workpiece. Orange/blue: workpiece forces ON the support, along actual outward contact normals.',
        'arrow_counts_by_panel':[2,4],'contact_reaction_arrow_counts_by_panel':[1,3],'forces':metadata,
        'applied_force':{'source_face':load_face,'point_m':load_point.tolist(),
            'direction':load_direction.tolist(),'on_body':'workpiece','region':'green work region',
            'illustrative_arrow_length_m':load_arrow_length,'shown_in_both_panels':True},
        'ground_plane_depicted':True,'floor_base_depicted':False,'floor_force_depicted':False,
        'camera_view_vector':view_vector,'camera_basis':basis.tolist(),
        'focus_m':focus.tolist(),'width_m':width,
        'rendered_parts_panel_a':[labels[i] for i in upper_only],
        'rendered_parts_panel_b':[labels[i] for i in selected],
        'upper_contact':{'candidate_id':'C023','location':'Crown of the head',
            'connector_centerline_m':[top_a.tolist(),top_b.tolist(),top_c.tolist(),top_d.tolist()],
            'connector_scope':'Illustrative overhead arm'},
        'blue_contact_lips':blue_lips,
        'added_blue_contact':{'candidate_id':'C151','location':'Below chin, above existing lower contact',
            'source':'Actual Step2 non-work-region contact patch',
            'connector_centerline_m':[a.tolist(),b.tolist(),c.tolist()],
            'connector_radius_m':.005,'connector_scope':'Illustrative neck, not a Step5 result'},
        'inputs':I.hashes([case/'step3_scheculer/final_contacts.npz',case/'step5_connect_support/geometry.npz',circles_path,data_path,directions_path]),
        'scope':'Conceptual connected-support force illustration without a floor base. Ground plane shown without a base ring. Actual C139 head and saved side frame reused; C023 crown contact and overhead arm replace C024; actual C151 candidate added with an illustrative blue neck. No solved load, balance, bearing, or insertion claim.'}
    (HERE/'head_total_force.json').write_text(json.dumps(report,indent=2)+'\n')
    print(HERE/'head_total_force.png')

if __name__=='__main__':run()
