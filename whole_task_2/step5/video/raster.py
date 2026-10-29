"""Direct OpenGL rasterization of saved motion, streamed to ffmpeg without PNGs."""
import os
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('EGL_PLATFORM','surfaceless')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES','/usr/share/glvnd/egl_vendor.d/50_mesa.json')
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import moderngl
import trimesh

WHITE=(.88,.90,.92);BLACK=(.16,.19,.22);GRAY=(.66,.69,.72)
PINK=(1.,.52,.74);INK=(.72,.19,.43);BLUE=(.16,.50,.74)

VERTEX='''#version 330
in vec3 position; in vec3 normal; in vec3 color;
uniform mat4 world; uniform mat4 view_projection; uniform mat4 light_projection;
out vec3 N; out vec3 C; out vec3 P; out vec4 shadow_coord;
void main(){vec4 p=world*vec4(position,1);gl_Position=view_projection*p;P=p.xyz;N=mat3(world)*normal;C=color;shadow_coord=light_projection*p;}
'''
FRAGMENT='''#version 330
in vec3 N;in vec3 C;in vec3 P;in vec4 shadow_coord;
uniform sampler2DShadow shadow_map;uniform vec3 eye;uniform float roughness;uniform float metallic;
layout(location=0)out vec4 pixel;layout(location=1)out vec4 surface_normal;
float shadow(vec3 n,vec3 l){
 vec3 q=shadow_coord.xyz/shadow_coord.w*.5+.5;
 if(any(lessThan(q,vec3(0)))||any(greaterThan(q,vec3(1))))return 1.;
 vec3 dx=dFdx(q),dy=dFdy(q);mat2 J=mat2(dx.x,dy.x,dx.y,dy.y);
 vec2 gradient=abs(determinant(J))>1e-12?inverse(J)*vec2(dx.z,dy.z):vec2(0);
 float bias=.00013;float result=0.;vec2 size=1./vec2(textureSize(shadow_map,0));
 for(int i=0;i<16;i++){
  float radius=sqrt((float(i)+.5)/16.)*5.5;float a=float(i)*2.39996323;vec2 offset=vec2(cos(a),sin(a))*radius*size;
  result+=texture(shadow_map,vec3(q.xy+offset,q.z+dot(gradient,offset)-bias));
 }
 return result/16.;
}
vec3 light(vec3 base,vec3 n,vec3 v,vec3 l,vec3 energy){
 float nl=max(dot(n,l),0.);vec3 h=normalize(l+v);float nh=max(dot(n,h),0.);
 float exponent=mix(100.,14.,roughness);vec3 f0=mix(vec3(.035),base,metallic);
 vec3 spec=f0*pow(nh,exponent)*(exponent+2.)/8.;
 return energy*nl*(base*(1.-metallic*.55)+spec);
}
void main(){
 vec3 n=normalize(N);if(!gl_FrontFacing)n=-n;vec3 v=normalize(eye-P);vec3 base=pow(C,vec3(2.2));
 vec3 key=normalize(vec3(-.65,-.9,1.45));float s=shadow(n,key);
 vec3 ambient=base*mix(vec3(.17,.19,.24),vec3(.35,.37,.41),n.z*.5+.5);
 vec3 c=ambient+light(base,n,v,key,vec3(.72,.70,.67))*s;
 c+=light(base,n,v,normalize(vec3(1.3,.7,1.1)),vec3(.30,.33,.38));
 c+=light(base,n,v,normalize(vec3(-.3,1.4,1.2)),vec3(.22,.25,.30));
 pixel=vec4(c,1);surface_normal=vec4(n*.5+.5,1);
}
'''
SHADOW_VERTEX='''#version 330
in vec3 position;uniform mat4 world;uniform mat4 light_projection;
void main(){gl_Position=light_projection*world*vec4(position,1);}
'''
SHADOW_FRAGMENT='''#version 330
void main(){}
'''
SCREEN_VERTEX='''#version 330
out vec2 uv;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(p*2.-1.,0,1);}
'''
AO_FRAGMENT='''#version 330
in vec2 uv;out float occlusion;
uniform sampler2D depth_map;uniform sampler2D normal_map;
uniform mat4 inverse_projection;uniform mat4 view_projection;
uniform vec3 view_direction;uniform vec3 samples[16];
vec3 position(vec2 p,float z){vec4 q=inverse_projection*vec4(p*2.-1.,z*2.-1.,1);return q.xyz/q.w;}
void main(){
 float z=texture(depth_map,uv).r;if(z>.99999){occlusion=1.;return;}
 vec3 p=position(uv,z);vec3 n=normalize(texture(normal_map,uv).xyz*2.-1.);
 vec3 t=normalize(cross(n,abs(n.y)<.9?vec3(0,1,0):vec3(1,0,0)));vec3 b=cross(n,t);
 vec2 cell=mod(floor(gl_FragCoord.xy),vec2(4));float a=fract(sin(dot(cell,vec2(12.9898,78.233)))*43758.5453)*6.2831853;
 mat3 basis=mat3(t*cos(a)+b*sin(a),b*cos(a)-t*sin(a),n);float total=0.;
 for(int i=0;i<16;i++){
  vec3 q=p+basis*samples[i]*.027;vec4 projected=view_projection*vec4(q,1);vec2 q_uv=projected.xy/projected.w*.5+.5;
  if(any(lessThan(q_uv,vec2(0)))||any(greaterThan(q_uv,vec2(1))))continue;
  float d=texture(depth_map,q_uv).r;if(d>.99999)continue;vec3 hit=position(q_uv,d);
  float dz=dot(hit-q,view_direction);float range=1.-smoothstep(.018,.055,abs(dot(hit-p,view_direction)));
  total+=(dz>.0006?1.:0.)*range;
 }
 occlusion=clamp(1.-total/16.*1.35,.24,1.);
}
'''
COMPOSITE_FRAGMENT='''#version 330
in vec2 uv;out vec4 pixel;
uniform sampler2D color_map;uniform sampler2D depth_map;uniform sampler2D ao_map;
void main(){
 float z=texture(depth_map,uv).r;if(z>.99999){pixel=vec4(pow(texture(color_map,uv).rgb,vec3(1./2.2)),1);return;}
 vec2 step_uv=1./vec2(textureSize(ao_map,0));float ao=0.;float weights=0.;
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
  vec2 p=uv+vec2(x,y)*step_uv;float w=exp(-float(x*x+y*y)*.55)*exp(-abs(texture(depth_map,p).r-z)*2400.);
  ao+=texture(ao_map,p).r*w;weights+=w;
 }
 ao/=max(weights,1e-6);vec3 c=texture(color_map,uv).rgb*mix(.35,1.,ao);
 c=pow(max(c,vec3(0)),vec3(1./2.2));c=(c-.5)*1.055+.5;pixel=vec4(clamp(c,0.,1.),1);
}
'''
FXAA_FRAGMENT='''#version 330
in vec2 uv;out vec4 pixel;uniform sampler2D image;
void main(){
 vec2 p=1./vec2(textureSize(image,0));vec3 nw=texture(image,uv+vec2(-1,1)*p).rgb;
 vec3 ne=texture(image,uv+vec2(1,1)*p).rgb;vec3 sw=texture(image,uv+vec2(-1,-1)*p).rgb;
 vec3 se=texture(image,uv+vec2(1,-1)*p).rgb;vec3 c=texture(image,uv).rgb;vec3 l=vec3(.299,.587,.114);
 float a=dot(nw,l),b=dot(ne,l),d=dot(sw,l),e=dot(se,l),m=dot(c,l);
 float lo=min(m,min(min(a,b),min(d,e))),hi=max(m,max(max(a,b),max(d,e)));
 vec2 dir=vec2(-((a+b)-(d+e)),(a+d)-(b+e));float reduce=max((a+b+d+e)*(.25*.125),1./128.);
 dir=clamp(dir/(min(abs(dir.x),abs(dir.y))+reduce),vec2(-8),vec2(8))*p;
 vec3 r=.5*(texture(image,uv+dir*(-1./6.)).rgb+texture(image,uv+dir*(1./6.)).rgb);
 vec3 s=r*.5+.25*(texture(image,uv+dir*(-.5)).rgb+texture(image,uv+dir*.5).rgb);
 float value=dot(s,l);pixel=vec4(value<lo||value>hi?r:s,1);
}
'''
LINE_VERTEX='''#version 330
in vec3 position;uniform mat4 world;uniform mat4 view_projection;
void main(){gl_Position=view_projection*world*vec4(position,1);}
'''
LINE_FRAGMENT='''#version 330
uniform vec3 ink;layout(location=0)out vec4 pixel;layout(location=1)out vec4 surface_normal;
void main(){pixel=vec4(pow(ink,vec3(2.2)),1);surface_normal=vec4(.5,.5,1,1);}
'''


def context():
    # EGL_DEFAULT_DISPLAY + Mesa surfaceless avoids selecting the broken NVIDIA
    # device exposed by eglQueryDevices on this workstation.
    from OpenGL import EGL
    from ctypes import c_int,byref
    display=EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY);major=c_int();minor=c_int()
    EGL.eglInitialize(display,byref(major),byref(minor))
    attributes=(c_int*13)(EGL.EGL_SURFACE_TYPE,EGL.EGL_PBUFFER_BIT,EGL.EGL_RENDERABLE_TYPE,EGL.EGL_OPENGL_BIT,
        EGL.EGL_RED_SIZE,8,EGL.EGL_GREEN_SIZE,8,EGL.EGL_BLUE_SIZE,8,EGL.EGL_DEPTH_SIZE,24,EGL.EGL_NONE)
    config=(EGL.EGLConfig*1)();count=c_int();EGL.eglChooseConfig(display,attributes,config,1,byref(count))
    EGL.eglBindAPI(EGL.EGL_OPENGL_API)
    size=(c_int*5)(EGL.EGL_WIDTH,1,EGL.EGL_HEIGHT,1,EGL.EGL_NONE)
    surface=EGL.eglCreatePbufferSurface(display,config[0],size)
    native=EGL.eglCreateContext(display,config[0],EGL.EGL_NO_CONTEXT,None)
    EGL.eglMakeCurrent(display,surface,surface,native)
    ctx=moderngl.create_context();ctx.extra=(display,surface,native)
    return ctx


def normals(v,f):
    # Area-weighted: a long thin face (a flat plate's) is not bent by the many small ones at its rounded rim.
    tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    ids=f.ravel();result=np.column_stack([np.bincount(ids,weights=np.repeat(n[:,j],3),minlength=len(v)) for j in range(3)])
    return result/np.maximum(np.linalg.norm(result,axis=1)[:,None],1e-20)


class Shape:
    def __init__(self,ctx,shader,v,f,color,smooth=True,dynamic=False,metallic=.08,roughness=.42):
        v=np.asarray(v);f=np.asarray(f);self.faces=f
        if smooth:n=normals(v,f)[f].reshape(-1,3)
        else:
            tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-20)
            n=np.repeat(n,3,axis=0)
        self.vertices=len(f)*3
        self.buffer=ctx.buffer(np.c_[v[f].reshape(-1,3),n].astype('f4').tobytes())
        self.colors=ctx.buffer(np.tile(np.asarray(color,dtype='f4'),(self.vertices,1)).tobytes(),dynamic=dynamic)
        self.vao=ctx.vertex_array(shader,[(self.buffer,'3f 3f','position','normal'),(self.colors,'3f','color')])
        self.world=np.eye(4);self.metallic=metallic;self.roughness=roughness;self.shadow_vao=None

    def paint(self,colors):self.colors.write(np.repeat(colors,3,axis=0).astype('f4').tobytes())
    def draw(self,shader,world):
        shader['world'].write(np.asarray(world,dtype='f4').T.tobytes())
        shader['metallic'].value=self.metallic;shader['roughness'].value=self.roughness
        self.vao.render(moderngl.TRIANGLES)

    def shadow(self,ctx,shader,world):
        if self.shadow_vao is None:self.shadow_vao=ctx.vertex_array(shader,[(self.buffer,'3f 12x','position')])
        shader['world'].write(np.asarray(world,dtype='f4').T.tobytes());self.shadow_vao.render(moderngl.TRIANGLES)


def world_from_bounds(center,size):
    m=trimesh.creation.box(extents=size);m.apply_translation(center);return m.vertices,m.faces


def tool_mesh(start,end,r0,r1):
    angles=np.linspace(0,2*np.pi,17)[:-1];v=[]
    for z,r in [(start,r0),(end,r1)]:v.extend(np.c_[np.cos(angles)*r,np.sin(angles)*r,np.full(16,z)])
    f=[]
    for i in range(16):j=(i+1)%16;f.extend([(i,j,j+16),(i,j+16,i+16)])
    for i in range(1,15):f.extend([(0,i+1,i),(16,16+i,17+i)])
    return np.asarray(v),np.asarray(f)


def trace_prefix(geometry,k,name,u):
    path=geometry[f'path_{k}_{name}'];arc=geometry[f'path_{k}_{name}_s']
    j=min(int(np.searchsorted(arc,u,side='right')),len(arc)-1);i=max(j-1,0)
    a=(u-arc[i])/max(arc[j]-arc[i],1e-15)
    return np.concatenate([path[:j],((1-a)*path[i]+a*path[j])[None]])



class Studio:
    def __init__(self,ctx,width,height):
        self.ctx=ctx;self.size=(width*2,height*2);self.output_size=(width,height);self.small=(width//2,height//2)
        self.color=ctx.texture(self.size,4,dtype='f2');self.normal=ctx.texture(self.size,4,dtype='f2')
        self.depth=ctx.depth_texture(self.size);self.depth.compare_func='';self.depth.filter=(moderngl.NEAREST,moderngl.NEAREST)
        self.gbuffer=ctx.framebuffer([self.color,self.normal],self.depth)
        self.shadow=ctx.depth_texture((1536,1536));self.shadow.compare_func='<=';self.shadow.filter=(moderngl.LINEAR,moderngl.LINEAR)
        self.shadow.repeat_x=False;self.shadow.repeat_y=False;self.shadow_buffer=ctx.framebuffer(depth_attachment=self.shadow)
        self.ao=ctx.texture(self.small,1);self.ao.filter=(moderngl.LINEAR,moderngl.LINEAR);self.ao_buffer=ctx.framebuffer([self.ao])
        self.composite=ctx.texture(self.output_size,4);self.composite.filter=(moderngl.LINEAR,moderngl.LINEAR);self.composite_buffer=ctx.framebuffer([self.composite])
        self.output=ctx.simple_framebuffer(self.output_size,components=3)
        self.ao_shader=ctx.program(vertex_shader=SCREEN_VERTEX,fragment_shader=AO_FRAGMENT)
        self.composite_shader=ctx.program(vertex_shader=SCREEN_VERTEX,fragment_shader=COMPOSITE_FRAGMENT)
        self.fxaa_shader=ctx.program(vertex_shader=SCREEN_VERTEX,fragment_shader=FXAA_FRAGMENT)
        self.ao_vao=ctx.vertex_array(self.ao_shader,[]);self.composite_vao=ctx.vertex_array(self.composite_shader,[]);self.fxaa_vao=ctx.vertex_array(self.fxaa_shader,[])
        self.ao_shader['depth_map'].value=0;self.ao_shader['normal_map'].value=1
        self.composite_shader['color_map'].value=0;self.composite_shader['depth_map'].value=1;self.composite_shader['ao_map'].value=2
        self.fxaa_shader['image'].value=0
        samples=[]
        for i in range(16):
            z=(i+.5)/16.;phi=i*2.39996323;r=np.sqrt(1-z*z);length=.18+.82*((i+.5)/16.)**2
            samples.append(np.array([r*np.cos(phi),r*np.sin(phi),z])*length)
        self.ao_shader['samples'].write(np.asarray(samples,dtype='f4').tobytes())

    def finish(self,vp,direction):
        ctx=self.ctx;ctx.disable(moderngl.DEPTH_TEST)
        self.ao_buffer.use();ctx.viewport=(0,0,*self.small);self.depth.use(0);self.normal.use(1)
        self.ao_shader['inverse_projection'].write(np.asarray(np.linalg.inv(vp),dtype='f4').T.tobytes())
        self.ao_shader['view_projection'].write(np.asarray(vp,dtype='f4').T.tobytes());self.ao_shader['view_direction'].value=tuple(direction)
        self.ao_vao.render(moderngl.TRIANGLES,vertices=3)
        self.composite_buffer.use();ctx.viewport=(0,0,*self.output_size);self.color.use(0);self.depth.use(1);self.ao.use(2)
        self.composite_vao.render(moderngl.TRIANGLES,vertices=3)
        self.output.use();self.composite.use(0);self.fxaa_vao.render(moderngl.TRIANGLES,vertices=3)
        return self.output.read(components=3,alignment=1)


def orthographic(width,height,near=.1,far=10.):
    p=np.zeros((4,4));p[0,0]=2/width;p[1,1]=2/height;p[2,2]=-2/(far-near);p[2,3]=-(far+near)/(far-near);p[3,3]=1
    return p


def view_matrix(direction,target,distance=4.):
    right=np.cross([0,0,1],direction);right/=np.linalg.norm(right);up=np.cross(direction,right)
    m=np.eye(4);m[:3,:3]=np.stack([right,up,direction]);m[:3,3]=-m[:3,:3]@(target+direction*distance)
    return m


def render(data,output,frames=None,width=1600,height=900):
    start=time.perf_counter();data=Path(data);output=Path(output)
    plan=json.loads((data/'plan.json').read_text())
    def arrays(name):
        with np.load(data/name) as z:return {key:z[key] for key in z.files}
    t=arrays('trajectory.npz');g=arrays('scene_meshes.npz');r=arrays('robot_meshes.npz')
    ctx=context();ctx.enable(moderngl.DEPTH_TEST)
    studio=Studio(ctx,width,height)
    shader=ctx.program(vertex_shader=VERTEX,fragment_shader=FRAGMENT)
    shadow_shader=ctx.program(vertex_shader=SHADOW_VERTEX,fragment_shader=SHADOW_FRAGMENT)
    key=np.array([-.65,-.9,1.45]);key/=np.linalg.norm(key)
    light_vp=orthographic(2.7,2.7)@view_matrix(key,np.array([.55,0,.65]))
    for program in (shader,shadow_shader):program['light_projection'].write(np.asarray(light_vp,dtype='f4').T.tobytes())
    shader['shadow_map'].value=3
    line_shader=ctx.program(vertex_shader=LINE_VERTEX,fragment_shader=LINE_FRAGMENT);line_shader['ink'].value=INK
    body=Shape(ctx,shader,g['object_v'],g['object_f'],GRAY,dynamic=True,metallic=.05,roughness=.62)
    fixture=Shape(ctx,shader,g['fixture_v'],g['fixture_f'],BLUE,smooth=False,metallic=.12,roughness=.38)
    parts=[];bounds=[]
    view=np.array([-.35,-1.75,.85]);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
    for part in plan['robot_parts']:
        color=part['color'];color=(WHITE if color=='white' else BLACK) if isinstance(color,str) else color
        v=r[part['name']+'_v'];f=r[part['name']+'_f']
        index=part['body'] if part['robot']=='kuka' else plan['panda_bodies'].index(part['body'])
        key='kuka_links' if part['robot']=='kuka' else 'panda_links'
        parts.append((Shape(ctx,shader,v,f,color),key,index))
        lo,hi=v.min(0),v.max(0);corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        matrices=t[key][:,index];bounds.append((np.einsum('fij,nj->fni',matrices[:,:3,:3],corners)+matrices[:,None,:3,3]).reshape(-1,3))
    points=np.concatenate(bounds);x=points@right;y=points@up
    wide=right*((x.min()+x.max())/2)+up*((y.min()+y.max())/2)+view*.1
    wide_width=max(x.max()-x.min(),(y.max()-y.min())*width/height)*1.12
    z=plan['table_height_m'];base=plan['work_robot_base_world_m']
    tables=[Shape(ctx,shader,*world_from_bounds((.79,.03,z/2),(1.10,1.,z)),(.91,.92,.93),smooth=False),
            Shape(ctx,shader,*world_from_bounds((base[0],base[1],base[2]/2),(.30,.34,base[2])),(.91,.92,.93),smooth=False)]
    tools=[Shape(ctx,shader,*tool_mesh(a,b,r0,r1),c,metallic=.65,roughness=.3) for a,b,r0,r1,c in [(.030,.050,.018,.018,BLACK),(.050,.105,.010,.010,(.60,.66,.72)),(.105,.123,.008,.003,(.60,.66,.72)),(.123,.155,.0012,.0006,(.60,.66,.72))]]
    line_buffer=ctx.buffer(reserve=1024*1024,dynamic=True);line_vao=ctx.vertex_array(line_shader,[(line_buffer,'3f','position')]);ctx.line_width=2
    kernel=np.exp(-.5*(np.arange(-18,19)/6.)**2);kernel/=kernel.sum();push=np.convolve((t['work_task']>=0).astype(float),kernel,mode='same')
    eye_rotation=np.stack([right,up,view]);previous_color=None
    selected=list(range(plan['frames'])) if frames is None else list(frames)
    output.parent.mkdir(parents=True,exist_ok=True)
    ffmpeg=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(plan['fps']),'-i','-',
        '-vf','vflip','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(output)],stdin=subprocess.PIPE)
    try:
        for number,f in enumerate(selected):
            scale=wide_width+(.56-wide_width)*push[f];target=wide+(np.array([.50,0,.74])-wide)*push[f]
            camera=view_matrix(view,target);vp=orthographic(scale,scale*height/width)@camera
            vp_bytes=np.asarray(vp,dtype='f4').T.tobytes()
            shader['view_projection'].write(vp_bytes);line_shader['view_projection'].write(vp_bytes)
            shader['eye'].value=tuple(target+view*4)
            colors=np.tile(GRAY,(len(g['object_f']),1))
            for k in range(len(plan['object_poses'])):
                if t['paint_progress'][f,k]>0 or int(t['work_task'][f])==k:colors[g[f'patch_{k}_ids']]=PINK
            if previous_color is None or not np.array_equal(colors,previous_color):body.paint(colors);previous_color=colors
            draws=[*( (shape,np.eye(4)) for shape in tables),
                (body,t['object_T'][f]),(fixture,t['fixture_T'][f]),
                *((shape,t[key][f,index]) for shape,key,index in parts),
                *((shape,t['kuka_links'][f,-1]) for shape in tools)]
            ctx.enable(moderngl.DEPTH_TEST)
            studio.shadow_buffer.use();ctx.viewport=(0,0,1536,1536);studio.shadow_buffer.clear(depth=1)
            # Keep the foreground arm from casting a large distracting shadow
            # across the small painting area. Station parts retain real shadows.
            for shape,world in draws[:len(tables)+2]:shape.shadow(ctx,shadow_shader,world)
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
            if number%120==0:print('RASTER FRAME',number,'/',len(selected),flush=True)
    finally:
        ffmpeg.stdin.close();code=ffmpeg.wait()
    if code:raise RuntimeError('Raw raster video encoding failed')
    result=dict(engine='OpenGL rasterization with studio lighting',device=ctx.info['GL_RENDERER'],frames=len(selected),width=width,height=height,
        elapsed_s=time.perf_counter()-start,path_tracing=False,intermediate_image_files=False,
        soft_shadow_map_px=1536,screen_space_ambient_occlusion_samples=16,ambient_occlusion_resolution='half',
        material_specular=True,robot_cast_shadows=False,anti_aliasing='2x supersampling + FXAA',closeup_width_m=.56,
        style_source='slides/idea/code/render_blender.py')
    (data/'raster.json').write_text(json.dumps(result,indent=2));print('RASTER COMPLETE',result,flush=True)
    ctx.release();return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,nargs='*')
    a=p.parse_args();render(a.data,a.output,a.frames)
