"""OpenGL rasterization of the two-robot fixture-set motion, streamed to ffmpeg."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import moderngl
from .raster import (VERTEX,FRAGMENT,SHADOW_VERTEX,SHADOW_FRAGMENT,LINE_VERTEX,LINE_FRAGMENT,
    WHITE,BLACK,GRAY,PINK,INK,BLUE,context,Shape,Studio,orthographic,view_matrix,world_from_bounds,tool_mesh,trace_prefix)

FIXTURE_COLORS=[BLUE,(.20,.62,.48),(.62,.42,.74)]


def render(data,output,frames=None,width=1600,height=900):
    start=time.perf_counter();data=Path(data);output=Path(output)
    plan=json.loads((data/'plan.json').read_text())
    def arrays(name):
        with np.load(data/name) as z:return {key:z[key] for key in z.files}
    t=arrays('trajectory.npz');g=arrays('scene_meshes.npz');r=arrays('robot_meshes.npz')
    ctx=context();ctx.enable(moderngl.DEPTH_TEST);studio=Studio(ctx,width,height)
    shader=ctx.program(vertex_shader=VERTEX,fragment_shader=FRAGMENT)
    shadow_shader=ctx.program(vertex_shader=SHADOW_VERTEX,fragment_shader=SHADOW_FRAGMENT)
    key=np.array([-.65,-.9,1.45]);key/=np.linalg.norm(key)
    light_vp=orthographic(2.7,2.7)@view_matrix(key,np.array([.55,0,.65]))
    for program in (shader,shadow_shader):program['light_projection'].write(np.asarray(light_vp,dtype='f4').T.tobytes())
    shader['shadow_map'].value=3
    line_shader=ctx.program(vertex_shader=LINE_VERTEX,fragment_shader=LINE_FRAGMENT);line_shader['ink'].value=INK
    body=Shape(ctx,shader,g['object_v'],g['object_f'],GRAY,dynamic=True,metallic=.05,roughness=.62)
    fixtures=[Shape(ctx,shader,g[f'fixture{s}_v'],g[f'fixture{s}_f'],FIXTURE_COLORS[s%len(FIXTURE_COLORS)],
                    smooth=False,metallic=.12,roughness=.38) for s in range(plan['fixture_count'])]
    view=np.array([-.35,-1.75,.85]);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
    parts=[];bounds=[]
    for part in plan['robot_parts']:
        color=part['color'];color=(WHITE if color=='white' else BLACK) if isinstance(color,str) else color
        v=r[part['name']+'_v'];f=r[part['name']+'_f']
        index=part['body'] if part['robot']=='kuka' else plan['panda_bodies'].index(part['body'])
        name='kuka_links' if part['robot']=='kuka' else 'panda_links'
        parts.append((Shape(ctx,shader,v,f,color),name,index))
        lo,hi=v.min(0),v.max(0);corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        matrices=t[name][:,index];bounds.append((np.einsum('fij,nj->fni',matrices[:,:3,:3],corners)+matrices[:,None,:3,3]).reshape(-1,3))
    for s in range(plan['fixture_count']):
        v=g[f'fixture{s}_v'][::max(1,len(g[f'fixture{s}_v'])//200)]
        bounds.append((np.einsum('fij,nj->fni',t['fixture_T'][:,s,:3,:3],v)+t['fixture_T'][:,s,None,:3,3]).reshape(-1,3))
    points=np.concatenate(bounds);x=points@right;y=points@up
    wide=right*((x.min()+x.max())/2)+up*((y.min()+y.max())/2)+view*.1
    wide_width=max(x.max()-x.min(),(y.max()-y.min())*width/height)*1.10
    z=plan['table_height_m'];base=plan['work_robot_base_world_m']
    bench=plan['bench']
    tables=[Shape(ctx,shader,*world_from_bounds((*bench['center'],z/2),(*bench['size'],z)),(.91,.92,.93),smooth=False),
            Shape(ctx,shader,*world_from_bounds((base[0],base[1],base[2]/2),(.30,.34,base[2])),(.91,.92,.93),smooth=False)]
    rod=plan.get('loader_cup_rod_m',0.);hand=plan['panda_bodies'].index('hand')
    cup=[Shape(ctx,shader,*tool_mesh(a,b,r0,r1),c,metallic=m,roughness=.35) for a,b,r0,r1,c,m in
         [(.055,.091+rod,.007,.007,(.60,.66,.72),.6),(.091+rod,.103+rod,.006,.013,BLACK,.1)]] if rod else []
    tools=[Shape(ctx,shader,*tool_mesh(a,b,r0,r1),c,metallic=.65,roughness=.3) for a,b,r0,r1,c in
           [(.030,.050,.018,.018,BLACK),(.050,.105,.010,.010,(.60,.66,.72)),(.105,.123,.008,.003,(.60,.66,.72)),(.123,.155,.0012,.0006,(.60,.66,.72))]]
    line_buffer=ctx.buffer(reserve=1024*1024,dynamic=True);line_vao=ctx.vertex_array(line_shader,[(line_buffer,'3f','position')]);ctx.line_width=2
    kernel=np.exp(-.5*(np.arange(-18,19)/6.)**2);kernel/=kernel.sum();push=np.convolve((t['work_task']>=0).astype(float),kernel,mode='same')
    # The needle head moves with each installation: zoom towards the nearest task's head.
    heads=np.asarray(plan['head_world_m']).reshape(-1,3);working=np.flatnonzero(t['work_task']>=0)
    nearest=t['work_task'][working[np.abs(np.arange(len(push))[:,None]-working[None]).argmin(1)]] if len(working) else np.zeros(len(push),int)
    close=heads[nearest]+np.array([0,0,-.08])
    previous_color=None;selected=list(range(plan['frames'])) if frames is None else list(frames)
    output.parent.mkdir(parents=True,exist_ok=True)
    ffmpeg=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(plan['fps']),'-i','-',
        '-vf','vflip','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(output)],stdin=subprocess.PIPE)
    try:
        for number,f in enumerate(selected):
            scale=wide_width+(.75-wide_width)*push[f];target=wide+(close[f]-wide)*push[f]
            vp=orthographic(scale,scale*height/width)@view_matrix(view,target)
            vp_bytes=np.asarray(vp,dtype='f4').T.tobytes()
            shader['view_projection'].write(vp_bytes);line_shader['view_projection'].write(vp_bytes)
            shader['eye'].value=tuple(target+view*4)
            colors=np.tile(GRAY,(len(g['object_f']),1))
            for k in range(len(plan['object_poses'])):
                if t['paint_progress'][f,k]>0 or int(t['work_task'][f])==k:colors[g[f'patch_{k}_ids']]=PINK
            if previous_color is None or not np.array_equal(colors,previous_color):body.paint(colors);previous_color=colors
            station=[(body,t['object_T'][f]),
                     *((shape,t['fixture_T'][f,s]) for s,shape in enumerate(fixtures))]
            draws=[*((shape,np.eye(4)) for shape in tables),*station,
                   *((shape,t[name][f,index]) for shape,name,index in parts),
                   *((shape,t['kuka_links'][f,-1]) for shape in tools),
                   *((shape,t['panda_links'][f,hand]) for shape in cup)]
            ctx.enable(moderngl.DEPTH_TEST)
            studio.shadow_buffer.use();ctx.viewport=(0,0,1536,1536);studio.shadow_buffer.clear(depth=1)
            for shape,world in draws[:len(tables)+len(station)]:shape.shadow(ctx,shadow_shader,world)
            studio.gbuffer.use();ctx.viewport=(0,0,*studio.size);studio.gbuffer.clear(1,1,1,1,depth=1);studio.shadow.use(3)
            for shape,world in draws:shape.draw(shader,world)
            line_shader['world'].write(np.asarray(t['object_T'][f],dtype='f4').T.tobytes())
            for k in range(len(plan['object_poses'])):
                p=float(t['paint_progress'][f,k])
                if p<=0:continue
                axis=np.asarray(plan['work_axis_world'])@np.asarray(plan['object_poses'][k])[:3,:3]
                for name,u in [('outline',min(p/(.47/.8),1.)),('interior',max(0.,(p-.47/.8)/(1-.47/.8)))]:
                    if u<=0:continue
                    points=trace_prefix(g,k,name,u)-axis*.00015
                    line_buffer.write(points.astype('f4').tobytes());line_vao.render(moderngl.LINE_STRIP,vertices=len(points))
            ffmpeg.stdin.write(studio.finish(vp,view))
            if number%240==0:print('RASTER FRAME',number,'/',len(selected),flush=True)
    finally:
        ffmpeg.stdin.close();code=ffmpeg.wait()
    if code:raise RuntimeError('Raw raster video encoding failed')
    result=dict(engine='OpenGL rasterization',device=ctx.info['GL_RENDERER'],frames=len(selected),width=width,height=height,
        elapsed_s=time.perf_counter()-start,no_text=True,fixture_count=plan['fixture_count'],fixed_floor=True,lift_table_drawn=False)
    (data/'raster.json').write_text(json.dumps(result,indent=2));print('RASTER COMPLETE',result,flush=True)
    ctx.release();return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='*');a=p.parse_args();render(a.data,a.output,a.frames)
