"""One image: actual empty fixture and thirteen individually installed uses."""
import math
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
from whole_task_2.step1.needle_render import sector_lines
from whole_task_2.step3.render_independent import Renderer, head_context, BLUE
from whole_task_2.step3.render_exit import arrow
from whole_task_2.step5.video import raster
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.operations.state import transform
from whole_task_2.operations.installations import floor_angles,representative_views

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def render(model,candidate,validation,path):
    support,q=candidate['support'],candidate['layout']
    width,tile_width,tile_height=3500,700,660
    header=1040
    height=header+math.ceil(len(model.ids)/5)*tile_height+70
    canvas=Image.new('RGB',(width,height),'white')
    draw=ImageDraw.Draw(canvas)
    font=lambda size:ImageFont.truetype(FONT,size)
    draw.text((55,25),'STEP 4  |  Optimized shared support',font=font(45),fill='#28485e')
    before,after=validation['initial_volume_cm3'],validation['final_volume_cm3']
    draw.text((55,92),f'{before:,.0f}  →  {after:,.0f} cm³     ({100*(1-after/before):.1f}% less material)',
              font=font(32),fill='#38657e')
    draw.text((55,145),f"{len(model.ids)} / {len(model.ids)} tasks   ·   {validation['original_sample_count']:,} paired loads   ·   Continuous whole-object exits",
              font=font(26),fill='#687b88')
    empty=Renderer(width//2,800)
    reference,side=representative_views(q,model.source.center_mass)
    tilt=float(floor_angles(q)[reference,side])
    turn=float(np.degrees(Rotation.from_matrix(
        q.material[side,:3,:3]@q.material[reference,:3,:3].T).magnitude()))
    rotated=tilt>1e-5
    if rotated:
        a=q.material[reference]@q.poses[reference]
        b=q.material[side]@q.poses[side]
        if np.allclose(a,b,atol=1e-8,rtol=0):
            draw.text((55,180),f'Same seat: {model.ids[reference]}  ↔  {model.ids[side]} with the COMPLETE fixture turned {turn:.1f}°',
                      font=font(21),fill='#38657e')
    try:
        for col,view in enumerate(([1.1,1.4,1.1],[-1.,-.85,1.3])):
            if rotated:view=[1.1,1.4,1.1]
            owner=(reference,side)[col]
            installed=moved_mesh(support,np.linalg.inv(q.material[owner]))
            lo,hi=installed.bounds
            floor=trimesh.creation.box([*(hi[:2]-lo[:2]+.25),.004])
            floor.apply_translation([*((lo[:2]+hi[:2])/2),-.002])
            items=[(floor,(.96,.97,.98),None),(installed,BLUE,None)]
            fit=installed.vertices
            if rotated and col==1:
                body=moved_mesh(model.source,q.poses[owner])
                mask=np.zeros(len(model.source.faces),bool)
                mask[model.records[owner]['working_face_ids']]=True
                items.append((body,raster.GRAY,mask))
                fit=np.vstack([fit,body.vertices])
            picture=empty.draw(items,fit,view)
            canvas.paste(picture,(col*width//2,205))
    finally:
        empty.close()
    if rotated:
        draw.text((55,1000),'Actual fixture, empty  |  Reference installation',font=font(25),fill='#486273')
        draw.text((1805,1000),f'{model.ids[side]}  |  SAME complete fixture, turned {turn:.1f}°',font=font(25),fill='#486273')
    else:
        draw.text((55,1000),'Actual fixture, empty  /  Each task below uses this same material body',font=font(25),fill='#486273')
    renderer=Renderer(tile_width,tile_height-68)
    contexts=head_context(model.workspace)
    lines=sector_lines(model.workspace)
    try:
        for i,name in enumerate(model.ids):
            install=np.linalg.inv(q.material[i])
            world_support=moved_mesh(support,install)
            body=moved_mesh(model.source,q.poses[i])
            mask=np.zeros(len(model.source.faces),bool)
            mask[model.records[i]['working_face_ids']]=True
            view=np.array([.35,1.3,1.0]);view/=np.linalg.norm(view)
            if floor_angles(q)[reference,i]>1e-5:
                # Look into the installed fixture from above every floor;
                # the old bottom plate is a vertical wall after a side turn.
                for normal in q.material[:,:3,2]:
                    world_normal=install[:3,:3]@normal
                    view+=max(0.,.6-world_normal@view)*world_normal
                view/=np.linalg.norm(view)
            right=np.cross([0.,0.,1.],view);right/=np.linalg.norm(right)
            start=body.center_mass+.65*np.ptp(body.vertices@right)*right+.035*view
            exit_arrow=arrow(start,start+.08*q.exits[i])
            # The large view exposes the real seat; the inset retains the
            # entire installed fixture, so a close-up cannot hide duplication.
            close=np.vstack([body.vertices,lines,*[m.vertices for m,_,_ in contexts]])
            extra=np.array([[body.bounds[0,0]-.02,body.bounds[0,1]-.02,0.],
                            [body.bounds[1,0]+.02,body.bounds[1,1]+.02,.008]])
            close=np.vstack([close,extra,exit_arrow.vertices])
            world_floor=trimesh.creation.box([*(world_support.extents[:2]+.25),.004])
            world_floor.apply_translation([*world_support.bounds.mean(0)[:2],-.002])
            items=[(world_floor,(.96,.97,.98),None),(world_support,BLUE,None),(body,raster.GRAY,mask),*contexts]
            image=renderer.draw([*items,(exit_arrow,(.16,.64,.40),None)],close,view,lines)
            inset=renderer.draw(items,np.vstack([world_support.vertices,body.vertices]),[1.1,1.4,1.1]).resize((240,172),Image.Resampling.LANCZOS)
            ImageDraw.Draw(inset).rectangle((0,0,239,171),outline='#dce4e9',width=2)
            image.paste(inset,(tile_width-244,tile_height-68-176))
            row=i//5
            count_in_row=min(5,len(model.ids)-row*5)
            x=int((5-count_in_row)*tile_width/2)+(i%5)*tile_width
            y=header+row*tile_height
            canvas.paste(image,(x,y+48))
            draw.text((x+25,y+10),name,font=font(30),fill='#28485e')
            yaw=np.degrees(np.arctan2(install[1,0],install[0,0]))
            tilt=floor_angles(q)[reference,i]
            draw.text((x+125,y+17),f'Fixture tilt {tilt:.0f}°  /  yaw {yaw:+.0f}°',font=font(19),fill='#74858f')
    finally:
        renderer.close()
    draw.text((55,height-52),'Blue: actual support   ·   Pink: task surface   ·   Gold: 100 mm / 25° sector   ·   Green: exit direction (head retracts first)',
              font=font(24),fill='#687b88')
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path)
    return path
