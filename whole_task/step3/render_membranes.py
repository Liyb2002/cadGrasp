"""Two geometry-backed figures: independent initialization and connected baseline."""
import math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from whole_task.operations.geometry import moved_mesh
from whole_task.step1.needle_render import sector_lines
from whole_task.step5.video import raster
from .render_independent import Renderer,head_context,BLUE
from .render_exit import arrow

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)


def render(model,q,shells,base,shifts,report,folder):
    width,tilew,tileh=3900,780,660
    canvas=Image.new('RGB',(width,160+math.ceil(len(shells)/5)*tileh+70),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((45,22),'STEP 3  |  Initialization',font=font(45),fill='#28485e')
    draw.text((45,85),f"{len(shells)} independent 5 mm membranes   ·   {report['material_volume_cm3']:,.1f} cm³   ·   Work areas, paired loads and exits verified",font=font(27),fill='#597586')
    renderer=Renderer(tilew//2,tileh-120); lines=sector_lines(model.workspace)
    contexts=head_context(model.workspace)
    try:
        for i,shell in enumerate(shells):
            body=moved_mesh(model.source,q.poses[i]);mask=np.zeros(len(model.source.faces),bool)
            mask[model.records[i]['working_face_ids']]=True
            # Same camera and fit for empty and occupied views of this fixture.
            fit=np.vstack([body.vertices,lines,*[m.vertices for m,_,_ in contexts]])
            view=[.45,1.4,1.1]
            start=body.bounds.mean(0)+np.array([.095,0.,0.])
            exit_arrow=arrow(start,start+.065*q.exits[i])
            fit=np.vstack([fit,exit_arrow.vertices])
            empty=renderer.draw([(shell,BLUE,None),(exit_arrow,(.16,.64,.4),None)],fit,view)
            occupied=renderer.draw([(shell,BLUE,None),(body,raster.GRAY,mask),*contexts],fit,view,lines)
            row=i//5;count=min(5,len(shells)-row*5)
            x=int((5-count)*tilew/2)+(i%5)*tilew;y=160+row*tileh
            canvas.paste(empty,(x,y+60));canvas.paste(occupied,(x+tilew//2,y+60))
            draw.text((x+24,y+8),model.ids[i],font=font(30),fill='#28485e')
            draw.text((x+140,y+16),f"{report['task_results'][i]['volume_cm3']:.1f} cm³",font=font(23),fill='#688391')
            draw.text((x+64,y+tileh-42),'Empty membrane',font=font(20),fill='#688391')
            draw.text((x+tilew//2+56,y+tileh-42),'Working pose',font=font(20),fill='#688391')
    finally:renderer.close()
    draw.text((45,canvas.height-50),'Blue: actual membrane   ·   Pink: task surface   ·   Gold: needle sector   ·   Green: exit direction',font=font(25),fill='#688391')
    canvas.save(folder/'initialization.png')

    # All display translations are saved alongside the actual baseline mesh.
    width,rowh=5600,720
    canvas=Image.new('RGB',(width,2*rowh+180),'white');draw=ImageDraw.Draw(canvas)
    draw.text((45,20),'STEP 3  |  Baseline result',font=font(46),fill='#28485e')
    draw.text((45,83),f"Same 13 membranes joined by a ground frame   ·   {report['baseline']['volume_cm3']:,.1f} cm³",font=font(30),fill='#597586')
    objects=[];context=[];lineparts=[];centers=[]
    for i,shift in enumerate(shifts):
        P=q.poses[i].copy();P[:3,3]+=shift;body=moved_mesh(model.source,P)
        mask=np.zeros(len(model.source.faces),bool);mask[model.records[i]['working_face_ids']]=True
        objects.append((body,raster.GRAY,mask));centers.append(body.bounds.mean(0))
        ws=dict(model.workspace);ws['reference_head_front_world_m']=(model.head+shift).tolist()
        context.extend(head_context(ws));lineparts.append(sector_lines(ws))
    lines=np.vstack(lineparts);fit=np.vstack([base.vertices,lines,*[m.vertices for m,_,_ in objects+context]])
    renderer=Renderer(width,rowh)
    view=np.array([0.,1.,1.5]);view/=np.linalg.norm(view)
    right=np.cross([0.,0.,1.],view);right/=np.linalg.norm(right)
    up=np.cross(view,right);projected=fit@np.array([right,up,view]).T
    low,high=projected.min(0),projected.max(0)
    scale=max(high[0]-low[0],(high[1]-low[1])*width/rowh)*1.16
    try:
        for row,items in enumerate(([(base,BLUE,None),*objects,*context],[(base,BLUE,None)])):
            image=renderer.draw(items,fit,[0.,1.,1.5],lines if row==0 else None)
            for i,center in enumerate(centers):
                x=width/2+(center@right-(low[0]+high[0])/2)/scale*width
                ImageDraw.Draw(image).text((x-30,rowh-48),model.ids[i],font=font(28),fill='#597586')
            canvas.paste(image,(0,155+row*rowh))
    finally:renderer.close()
    canvas.save(folder/'baseline_result.png')
