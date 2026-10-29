"""Port of Co-optimize evaluation: identical BBX metrics and three-column image.

Use the actual centered execution transforms, because the support can translate
between tasks even when its installation orientation is unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from subdivide_task.step1.common import ROOT, write_json, sha256
from subdivide_task.step3.legacy import kernel as K, COOPT
from .metrics import static_bounds,BOX_KEYS
sys.path.insert(0,str(COOPT/'vis_func'))
import step51_render as V

VERSION='whole_task_centered_static_occupancy_v1'
FIGURE_SIZE=(1950,1210)

def normalize(result):
    reference=result['object_static']['volume_litres']
    if not np.isfinite(reference)or reference<=0:raise ValueError('Invalid object-only BBX')
    result['relative_bbx_volume']={k:None if result[k]is None else result[k]['volume_litres']/reference for k in BOX_KEYS}
    result['relative_bbx_volume']['object_static']=1.
    result['bbx_increase_percent']={k:None if v is None else 100*(v-1)for k,v in result['relative_bbx_volume'].items()}
    result['bbx_increase_litres']={k:None if result[k]is None else result[k]['volume_litres']-reference for k in BOX_KEYS}
    result['evaluation_output_version']=VERSION
    return result


def corners(record):
    lo,hi=record['min_m'],record['max_m']
    return np.array([[x,y,z]for x in [lo[0],hi[0]]for y in [lo[1],hi[1]]for z in [lo[2],hi[2]]])


def presentation(layout,result):
    """Group actual installation transforms without superimposing their meshes."""
    poses={p['layout_index']:p for p in result['per_pose']};groups={}
    for value in layout['active']:
        k=int(value);T=np.asarray(result['execution'][k]['world_from_support'])
        group=groups.setdefault(np.round(T,11).tobytes(),dict(indices=[],fixture_transform=T.tolist()))
        group['indices'].append(k)
    groups=list(groups.values())
    if not result['passed']:groups=[dict(indices=[int(k)for k in layout['active']],fixture_transform=None)]
    largest=max(groups,key=lambda group:len(group['indices']))
    representative=max(largest['indices'],key=lambda k:poses[k]['objects_in_final_layout']['dimensions_m'][2])
    for index,group in enumerate(groups):
        group.update(placement=index+1 if result['passed']else None,poses=[poses[k]['pose']for k in group['indices']])
    return dict(representative_layout_index=representative,representative_pose=poses[representative]['pose'],
        representative_selection='largest placement group; greatest object vertical extent',
        groups=groups,main_instances=[],thumbnail_poses=[],bbox_includes_all_task_poses=True,
        column_percent_labels=['']+[''if result[k]is None else f'{result["bbx_increase_percent"][k]:+.0f}%'for k in BOX_KEYS[1:]],
        visible_text=[],annotation_policy='percent_and_placement_only',
        bbox_style='complete_wireframe_box',bbox_edges_per_main_column=[],
        volume_ratios_printed=False,litres_printed=False,figure_size=list(FIGURE_SIZE))


def box_piece(record,camera,size):
    """Render all twelve exact BBX edges with ordinary scene depth occlusion."""
    _,basis,span=camera;box=corners(record)
    pairs=[(index,index^bit)for index in range(8)for bit in [1,2,4]if index<(index^bit)]
    starts=box[[a for a,b in pairs]];ends=box[[b for a,b in pairs]]
    projected=(ends-starts)@basis[:2].T
    lengths=np.linalg.norm(projected,axis=1)
    perpendicular=np.c_[-projected[:,1],projected[:,0]]/lengths[:,None]
    offsets=perpendicular@basis[:2]*(.9*span/size)
    vertices=np.stack([starts+offsets,ends+offsets,ends-offsets,starts-offsets],axis=1)
    triangles=vertices[:,[[0,1,2],[0,2,3]],:].reshape(-1,3,3)
    return dict(triangles=triangles,normals=np.broadcast_to(basis[2],triangles.shape),
        colors=np.broadcast_to(V.CPU.rgb('#9FB1BD'),triangles.shape),bias=4e-6,unlit=True),len(pairs)


def posed_piece(mesh,transform,color,opacity=1.):
    posed=mesh.copy();posed.apply_transform(transform)
    return V.piece(posed,color,opacity)


def render(renderer,body,nominal,full,layout,result,out):
    """One representative scene, full-task BBX, and one shared grouped pose strip."""
    normalize(result);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    size=650;header=145;display=presentation(layout,result);result['presentation']=display
    cameras={k:V.camera_for(corners(result[k]))for k in BOX_KEYS if result[k]is not None}
    span=max(c[2]for c in cameras.values())
    baseline={p['layout_index']:np.array(p['unsupported_world_transform'])for p in result['per_pose']}
    k=display['representative_layout_index'];T=np.asarray(result['execution'][k]['world_from_support'])
    board=Image.new('RGB',FIGURE_SIZE,'white');ink=ImageDraw.Draw(board)
    font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
    for column,(key,fixture)in enumerate(zip(BOX_KEYS,[None,nominal,full])):
        center=size*(column+.5)
        label=display['column_percent_labels'][column]
        if label:
            ink.text((center,106),label,font=font(36),fill='#267C9F',anchor='mt')
            display['visible_text'].append(label)
        record=result[key]
        if record is None:
            display['main_instances'].append(dict(objects=0,supports=0))
            display['bbox_edges_per_main_column'].append(0)
            continue
        focus,basis,_=cameras[key];camera=(focus,basis,span)
        U=baseline[k]if column==0 else T@layout['placements'][k]
        pieces=[posed_piece(body,U,V.GRAY)]
        if fixture is not None:pieces.append(posed_piece(fixture,T,V.BLUE))
        display['main_instances'].append(dict(objects=1,supports=int(fixture is not None)))
        box,edge_count=box_piece(record,camera,size);pieces.append(box)
        display['bbox_edges_per_main_column'].append(edge_count)
        tile=renderer.render([V.CPU.placed(p)for p in pieces],camera,size)
        board.paste(tile,(column*size,header))
    ink.line([(60,853),(1890,853)],fill='#E3E9ED',width=2)
    groups=display['groups'];gap=34;count=sum(len(g['indices'])for g in groups)
    cell=min(210.,(1830-gap*(len(groups)-1))/count);thumb=min(176,int(cell)-8)
    group_cameras=[]
    for group in groups:
        box=result['fixture_states'][group['placement']-1]['with_full_support']if result['passed']else result['object_static']
        group_cameras.append(V.camera_for(corners(box)))
    thumb_span=max(c[2]for c in group_cameras)
    x=(1950-(cell*count+gap*(len(groups)-1)))/2
    for group,(focus,basis,_)in zip(groups,group_cameras):
        width=cell*len(group['indices']);center=x+width/2;camera=(focus,basis,thumb_span)
        group['thumbnail_camera']=dict(focus_m=focus.tolist(),basis=basis.tolist(),span_m=thumb_span)
        if result['passed']:
            label=('P' if len(groups)>7 else 'Placement ')+str(group['placement'])
            ink.text((center,926),label,font=font(21),fill='#5E7D8E',anchor='mt')
            display['visible_text'].append(label)
            ink.line([(x+10,951),(x+10,958),(x+width-10,958),(x+width-10,951)],fill='#B3C7D2',width=2)
        for position,index in enumerate(group['indices']):
            cx=x+cell*(position+.5);S=np.asarray(result['execution'][index]['world_from_support'])
            U=S@layout['placements'][index]if result['passed']else baseline[index]
            pieces=[posed_piece(body,U,V.GRAY)]
            if full is not None:pieces.append(posed_piece(full,S,V.BLUE,.8))
            tile=renderer.render([V.CPU.placed(p)for p in pieces],camera,thumb)
            board.paste(tile,(round(cx-thumb/2),round(975+(176-thumb)/2)))
            pose=next(p['pose']for p in result['per_pose']if p['layout_index']==index)
            display['thumbnail_poses'].append(pose)
        x+=width+gap
    info=PngImagePlugin.PngInfo()
    info.add_text('evaluation_output_version',VERSION)
    info.add_text('relative_bbx_volume',json.dumps(result['relative_bbx_volume']))
    info.add_text('bbx_increase_percent',json.dumps(result['bbx_increase_percent']))
    info.add_text('presentation',json.dumps(display))
    pending=out.with_name('.'+out.name+'.pending')
    board.save(pending,format='PNG',pnginfo=info);pending.replace(out)



def evaluate(out,model,layout):
    out=Path(out);report=json.loads((out/'report.json').read_text())
    step4=Path(report['step4'])/'step4.2'
    nominal=K.trimesh.load(step4/'support.obj',force='mesh',process=False)
    complete4=K.trimesh.load(step4/'complete_support.obj',force='mesh',process=False)
    full=K.trimesh.load(out/'support.obj',force='mesh',process=False)
    with np.load(out/'layout.npz') as z:arrays={k:z[k].copy() for k in z.files}
    execution=[model.execution(layout,k) for k in range(len(model.poses))]
    result=static_bounds(model.mesh.vertices,arrays,nominal.vertices,full.vertices,model.poses,execution=execution)
    result.update(passed=True,status='PASS',execution=execution,
        material_step4_core_cm3=abs(float(nominal.volume))*1e6,
        material_step4_complete_cm3=abs(float(complete4.volume))*1e6,
        material_step5_cm3=abs(float(full.volume))*1e6,
        columns=['object_only','object_plus_step4_contact_core','object_plus_step5_complete_fixture'])
    prior=static_bounds(model.mesh.vertices,arrays,full_vertices=complete4.vertices,pose_names=model.poses,execution=execution)
    from subdivide_task.step5.geometry import ground_contact_metrics,execution_rows
    result['ground_contacts']=ground_contact_metrics(full,execution_rows(model,layout))
    result['step4_complete_bbx']=prior['with_full_support']
    result['material_reduction_vs_step4_complete_percent']=100*(1-result['material_step5_cm3']/result['material_step4_complete_cm3'])
    result['bbx_reduction_vs_step4_complete_percent']=100*(1-result['with_full_support']['volume_litres']/prior['with_full_support']['volume_litres'])
    render(V.Renderer(),model.mesh,nominal,full,arrays,result,out/'evaluation.png')
    result['image_sha256']=sha256(out/'evaluation.png')
    write_json(out/'evaluation.json',result)
    return {k:result[k] for k in ['material_step4_core_cm3','material_step4_complete_cm3','material_step5_cm3',
        'material_reduction_vs_step4_complete_percent','bbx_reduction_vs_step4_complete_percent',
        'relative_bbx_volume','bbx_increase_percent','support_pose_count','object_pose_counts_per_support_pose','with_full_support','step4_complete_bbx','ground_contacts']}


def main():
    from subdivide_task.step4.mobile import MobileTaskModel
    from whole_search.model import Layout
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    r=json.loads((a.folder/'report.json').read_text())
    model=MobileTaskModel(r['poses'],r['object'],r['output_root'])
    with np.load(a.folder/'layout.npz') as z:layout=Layout.from_arrays(z)
    print(json.dumps(evaluate(a.folder,model,layout),indent=2))


if __name__=='__main__':main()
