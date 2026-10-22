"""One figure showing separate final fixtures and every member task's use."""
import math
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.step3.render_independent import Renderer,head_context,BLUE
from whole_task_2.step1.needle_render import sector_lines
from whole_task_2.step5.video import raster
from .membrane_search import read_group

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)


def render(model,groups,report,path,step=4):
    width,tilew,tileh=3500,700,620
    header=200;group_columns=min(3,len(groups));group_width=width//group_columns
    group_height=800 if group_columns<=2 else 600
    group_rows=math.ceil(len(groups)/group_columns)
    usages=[]
    for family,g in enumerate(groups):
        q,support=read_group(g)
        for local,owner in enumerate(g['members']):usages.append((owner,family,local,q,support))
    usages.sort(key=lambda row:row[0])
    usage_top=header+group_rows*(group_height+90)+90
    height=usage_top+math.ceil(len(usages)/5)*tileh+65
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    title='Shared membrane fixtures' if step==4 else 'Grounded shared fixtures'
    draw.text((40,20),f'STEP {step}  |  {title}',font=font(44),fill='#28485e')
    if step==4:
        headline=f"13 independent membranes → {len(groups)} fixtures   ·   {report['initial_volume_cm3']:,.1f} → {report['final_volume_cm3']:,.1f} cm³"
        note=('Every merge saves at least 25% of parent material; certified ground extension for every task'
              if report.get('merge_volume_ratio_limit',1.)<=.75 else
              'Every merge saves membrane material; connected ground-foot extension certified for every task')
    else:
        headline=f"{len(groups)} fixtures for 13 tasks   ·   Actual material including sparse feet: {report['final_volume_cm3']:,.1f} cm³"
        note='Same complete fixture in every installation; real ground contacts enclose each task pressure hull'
    draw.text((40,82),headline,font=font(31),fill='#597586')
    draw.text((40,133),note,font=font(25),fill='#688391')
    renderer=Renderer(group_width,group_height-50)
    try:
        for k,g in enumerate(groups):
            q,support=read_group(g);row=k//group_columns
            count=min(group_columns,len(groups)-row*group_columns)
            x=int((group_columns-count)*group_width/2)+(k%group_columns)*group_width
            y=header+row*(group_height+90)
            # Empty fixture in its first member's working installation.
            world=moved_mesh(support,np.linalg.inv(q.material[0]))
            image=renderer.draw([(world,BLUE,None)],world.vertices,[.6,1.4,1.1])
            canvas.paste(image,(x,y+50))
            draw.text((x+25,y+4),f"Set {k+1}  ·  {g['volume_cm3']:.1f} cm³",font=font(27),fill='#28485e')
            names=' / '.join(model.ids[i] for i in g['members'])
            draw.text((x+25,y+group_height+12),names,font=font(23),fill='#688391')
    finally:renderer.close()
    draw.text((40,usage_top-57),'Each task below uses its complete set fixture; the needle stays fixed',font=font(29),fill='#28485e')
    renderer=Renderer(tilew,tileh-80);lines=sector_lines(model.workspace);contexts=head_context(model.workspace)
    try:
        for j,(owner,family,k,q,support) in enumerate(usages):
            installed=moved_mesh(support,np.linalg.inv(q.material[k]));body=moved_mesh(model.source,q.poses[k])
            mask=np.zeros(len(model.source.faces),bool);mask[model.records[owner]['working_face_ids']]=True
            fit=np.vstack([installed.vertices,body.vertices,lines,*[m.vertices for m,_,_ in contexts]])
            ground=trimesh.creation.box([1.6,1.6,.004]);ground.apply_translation([0,-.4,-.002])
            image=renderer.draw([(ground,(.95,.96,.97),None),(installed,BLUE,None),
                (body,raster.GRAY,mask),*contexts],fit,[.45,1.4,1.1],lines)
            row=j//5;count=min(5,len(usages)-row*5)
            x=int((5-count)*tilew/2)+(j%5)*tilew;y=usage_top+row*tileh
            canvas.paste(image,(x,y+60))
            angle=np.degrees(Rotation.from_matrix(q.material[k,:3,:3]@q.material[0,:3,:3].T).magnitude())
            draw.text((x+25,y+8),f'{model.ids[owner]}  ·  Set {family+1}',font=font(28),fill='#28485e')
            draw.text((x+375,y+15),f'Turn {angle:.0f}°',font=font(21),fill='#688391')
    finally:renderer.close()
    draw.text((40,height-47),'Blue: actual set fixture   ·   Pink: task surface   ·   Gold: fixed needle sector',font=font(25),fill='#688391')
    canvas.save(path);return path
