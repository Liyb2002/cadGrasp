"""Final sparse-foot fixtures and all certified ground-frame installations."""
import math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import trimesh

from whole_task.operations.geometry import moved_mesh
from whole_task.step4.membrane_search import read_group
from whole_task.step3.render_independent import Renderer,head_context,BLUE
from whole_task.step1.needle_render import sector_lines
from .video import raster


def render(model,groups,report,path):
    width=3500;tilew=700;tileh=620;header=180;family_height=850;usage_top=header+family_height+70
    rows=[]
    for family,g in enumerate(groups):
        q,mesh=read_group(g)
        rows.extend((owner,family,k,q,mesh) for k,owner in enumerate(g['members']))
    rows.sort();height=usage_top+math.ceil(len(rows)/5)*tileh+50
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
    draw.text((35,20),'STEP 5 | Ground-contact fixtures with sparse feet',font=font(43),fill='#29485c')
    draw.text((35,82),f"{len(groups)} shared fixtures | {report['initial_volume_cm3']:.1f} -> {report['final_volume_cm3']:.1f} cm3",
              font=font(31),fill='#587386')
    draw.text((35,128),'Membrane rests on each ground plane; real feet enclose its required pressure hull',font=font(25),fill='#6c8390')
    familyw=width//len(groups);renderer=Renderer(familyw,family_height-80)
    try:
        for i,g in enumerate(groups):
            q,mesh=read_group(g);world=moved_mesh(mesh,np.linalg.inv(q.material[0]))
            image=renderer.draw([(world,BLUE,None)],world.vertices,[1.,-1.,.9])
            canvas.paste(image,(i*familyw,header+50))
            draw.text((i*familyw+28,header+4),f"Set {i+1} | {g['volume_cm3']:.1f} cm3",font=font(28),fill='#29485c')
            draw.text((i*familyw+28,header+family_height-20),' / '.join(model.ids[j] for j in g['members']),font=font(23),fill='#6c8390')
    finally:renderer.close()
    draw.text((35,usage_top-48),'Every task uses its complete set fixture',font=font(29),fill='#29485c')
    renderer=Renderer(tilew,tileh-80)
    try:
        for j,(owner,family,k,q,mesh) in enumerate(rows):
            installed=moved_mesh(mesh,np.linalg.inv(q.material[k]));body=moved_mesh(model.source,q.poses[k])
            mask=np.zeros(len(model.source.faces),bool);mask[model.records[owner]['working_face_ids']]=True
            workspace=model.children[owner].workspace;sector=sector_lines(workspace);contexts=head_context(workspace)
            fit=np.vstack([installed.vertices,body.vertices,sector,*[m.vertices for m,_,_ in contexts]])
            low,high=fit[:,:2].min(0)-.02,fit[:,:2].max(0)+.02
            floor=trimesh.creation.box([*(high-low),.0004]);floor.apply_translation([*(low+high)/2,-.0002])
            image=renderer.draw([(floor,(.96,.97,.98),None),(installed,BLUE,None),(body,raster.GRAY,mask),*contexts],
                                fit,[1.,-1.,.9],sector)
            row=j//5;count=min(5,len(rows)-row*5);x=int((5-count)*tilew/2)+(j%5)*tilew;y=usage_top+row*tileh
            canvas.paste(image,(x,y+60));draw.text((x+24,y+8),f'{model.ids[owner]} | Set {family+1}',font=font(28),fill='#29485c')
    finally:renderer.close()
    canvas.save(path);return path
