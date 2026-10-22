"""Actual common support and each task's own certified straight extraction."""
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
from whole_task.step1.needle_render import sector_lines,rod
from whole_task.step1.common import ROOT,sha256
from .render_independent import Renderer,head_context,BLUE
from whole_task.step5.video import raster


def arrow(start,end):
    delta=end-start;axis=delta/np.linalg.norm(delta)
    shaft=rod(start,end-.012*axis,.0016)
    tip=trimesh.creation.cone(radius=.006,height=.012,sections=16)
    T=trimesh.geometry.align_vectors([0,0,1],axis);T[:3,3]=end-.012*axis;tip.apply_transform(T)
    return trimesh.util.concatenate([shaft,tip])


def render(source,support,bedding,records,plate_bounds,folder,report):
    choices=[np.array([0.,1.,.4]),np.array([0.,1.,2.])]
    choices=[v/np.linalg.norm(v)for v in choices]
    directions=np.asarray([r['exit_direction_world']for r in records])
    view=min(choices,key=lambda v:np.abs(directions@v).max())
    objects=[];exited=[];contexts=[];lines=[];arrows=[];centers=[]
    for record in records:
        direction=np.asarray(record['exit_direction_world']);length=record['exit_length_m']
        body=source.copy();body.apply_transform(record['world_from_object'])
        mask=np.zeros(len(source.faces),bool);mask[record['working_face_ids']]=True
        lifted=body.copy();lifted.apply_translation(length*direction)
        objects.append((body,raster.GRAY,mask));exited.append((lifted,raster.GRAY,mask))
        contexts.extend(head_context(record['workspace']));lines.append(sector_lines(record['workspace']))
        arrows.append((arrow(body.center_mass,lifted.center_mass),(.78,.55,.16),None));centers.append(body.bounds.mean(0))
    lo,hi=plate_bounds;floor=trimesh.creation.box([hi[0]-lo[0]+.7,hi[1]-lo[1]+.8,.005])
    floor.apply_translation([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,-.0025]);ground=(floor,(.96,.97,.98),None)
    lines=np.vstack(lines)
    bounds=np.vstack([support.vertices,lines,*[m.vertices for m,_,_ in objects+exited+contexts]])
    width,height=5600,1000;renderer=Renderer(width,height);canvas=Image.new('RGB',(width,3*height),'white')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
    projected=bounds@np.array([right,up,view]).T;low,high=projected.min(0),projected.max(0)
    scale=max(high[0]-low[0],(high[1]-low[1])*width/height)*1.16
    scenes=[[(support,BLUE,None),*objects,*contexts],[(support,BLUE,None),*arrows],[(support,BLUE,None),*exited,*arrows]]
    try:
        for row,items in enumerate(scenes):
            picture=renderer.draw([ground,*items],bounds,view,lines if row==0 else None)
            draw=ImageDraw.Draw(picture)
            for record,center in zip(records,centers):
                x=width/2+(center@right-(low[0]+high[0])/2)/scale*width
                draw.text((x-26,height-36),record['id'],font=font,fill='#344d60')
            canvas.paste(picture,(0,row*height))
        canvas.save(folder/'overview.png')
    finally:renderer.close()
    report['visualization']=dict(engine='OpenGL rasterization',overview_size=list(canvas.size),
        row_order=['seated_with_working_heads','empty_support_with_per_task_exit_arrows','whole_objects_fully_extracted'],
        actual_saved_support_geometry=True,working_area_color='pink',exit_arrows_from_per_task_saved_paths=True,
        working_heads_retracted_in_exit_rows=True,same_camera_and_same_source_poses=True,
        camera_direction_world=view.tolist(),projected_exit_motion_fraction_per_task=np.sqrt(np.maximum(0,1-(directions@view)**2)).tolist(),
        source_sha256={str(p.relative_to(ROOT)):sha256(p)for p in [Path(__file__),Path(raster.__file__)]})
