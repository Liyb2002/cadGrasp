"""How a given head works: the Step1 directions painted in turn at the robot station (idea video).

The KUKA LBR Med 14 stands at the station's fixed base (robots_config) and carries
the head (needle and head cylinder at their real sizes, flange behind the head).
For each direction the object sits at the work point with that side facing the
slanted nominal needle direction; the arm moves the head over the side's faces
and its wrist tilts the needle to each face's certified axis, inside the gold
max-tilt cone.  Light pink is what the current side still has to paint; deep
pink is painted.  No text.  A kinematic presentation: the arm body behind the
head is not part of the Step1 reach model; sampled overlaps are reported.
The exported video starts six seconds into the full motion by default.
"""
import argparse,json,subprocess,time
from pathlib import Path

import numpy as np

from .common import ROOT,load_shape

GOLD=(.86,.66,.14);LIGHT=(1.,.84,.90);HEAD=(.20,.23,.27);STEEL=(.62,.67,.72)
GHOST_HEAD=(.64,.68,.72);GHOST_STEEL=(.82,.85,.88)
STATION_VIEW=np.array([1.,-.30,.38])/np.linalg.norm([1.,-.30,.38])


def tool_frame(tip,d):
    """Tool coordinates: +z runs from the tip back along the needle, away from the surface."""
    z=-np.asarray(d,float);x=np.cross(z,[1.,0,0] if abs(z[0])<.9 else [0,1.,0]);x/=np.linalg.norm(x)
    T=np.eye(4);T[:3,0]=x;T[:3,1]=np.cross(z,x);T[:3,2]=z;T[:3,3]=tip;return T


def slerp(a,b,u):
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);w=np.arccos(np.clip(a@b,-1,1))
    if w<1e-9:return a
    return (np.sin((1-u)*w)*a+np.sin(u*w)*b)/np.sin(w)


def ease(u):return u*u*(3-2*u)


def stops(V,faces,ids,axes,cell):
    """Group a pose's faces into cells; visit the cells row by row, alternating direction."""
    C=V[faces[ids]].mean(1);key=np.floor(C/cell).astype(int);out=[]
    for k in np.unique(key,axis=0):
        members=ids[(key==k).all(1)];c=V[faces[members]].mean(1)
        rep=members[np.argmin(np.linalg.norm(c-c.mean(0),axis=1))]
        out.append(dict(key=k,faces=members,tip=V[faces[rep]].mean(0),axis=axes[rep]))
    rows=sorted({int(s['key'][1]) for s in out});ordered=[]
    for i,row in enumerate(rows):
        line=sorted([s for s in out if s['key'][1]==row],key=lambda s:(s['key'][0],-s['key'][2]))
        ordered+=line if i%2==0 else line[::-1]
    return ordered


def timeline(mesh,report,arrays,cell,fps):
    """Per frame: object transform, tool tip and axis, painted mask, current pose."""
    center=mesh.bounds.mean(0);frames=[];painted=np.zeros(len(mesh.faces),bool)
    L=report['setup']['needle_length_m'];down=np.asarray(report['setup']['working_direction_world']);hover=.6*L
    Ws=[]
    for pose in report['areas']:
        W=np.eye(4);W[:3,:3]=pose['placement']['rotation_world_object'];W[:3,3]=pose['placement']['translation_world_m'];Ws.append(W)
    def add(W,tip,axis,pose,n=1):
        for _ in range(n):frames.append(dict(W=W,tip=np.asarray(tip,float),axis=np.asarray(axis,float),pose=pose,painted=painted.copy()))
    for j,(pose,W) in enumerate(zip(report['areas'],Ws)):
        V=mesh.vertices@W[:3,:3].T+W[:3,3];axes=arrays['face_axis']@W[:3,:3].T
        path=stops(V,mesh.faces,np.flatnonzero(arrays['face_area']==j),axes,cell)
        top=path[0]['tip']-hover*down
        if j==0:
            add(W,top,down,j,int(.6*fps))
            for u in np.linspace(0,1,int(.8*fps)):add(W,(1-ease(u))*top+ease(u)*path[0]['tip'],down,j)
            # The needle sweeps its whole allowed cone around one contact point.
            tilt=np.radians(report['setup']['max_tilt_deg']);x=np.cross(down,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(down,x)
            for u in np.linspace(0,1,int(.5*fps)):add(W,path[0]['tip'],slerp(down,np.cos(tilt)*down+np.sin(tilt)*x,ease(u)),j)
            for phi in np.linspace(0,2*np.pi,int(2.4*fps)):
                add(W,path[0]['tip'],np.cos(tilt)*down+np.sin(tilt)*(np.cos(phi)*x+np.sin(phi)*y),j)
            for u in np.linspace(0,1,int(.5*fps)):add(W,path[0]['tip'],slerp(np.cos(tilt)*down+np.sin(tilt)*x,path[0]['axis'],ease(u)),j)
        else:
            for u in np.linspace(0,1,int(.6*fps)):add(W,(1-ease(u))*top+ease(u)*path[0]['tip'],path[0]['axis'],j)
        for a,b in zip(path,path[1:]+[None]):
            painted[a['faces']]=True;add(W,a['tip'],a['axis'],j)
            if b is None:break
            gap=np.linalg.norm(b['tip']-a['tip']);lift=min(.03,.5*gap+.004)
            for u in np.linspace(0,1,3)[1:-1]:
                add(W,(1-u)*a['tip']+u*b['tip']-down*lift*np.sin(np.pi*u),slerp(a['axis'],b['axis'],u),j)
        last=path[-1]
        up=last['tip']-hover*down
        for u in np.linspace(0,1,int(.6*fps)):add(W,(1-ease(u))*last['tip']+ease(u)*up,slerp(last['axis'],down,ease(u)),j)
        if j+1<len(Ws):
            from scipy.spatial.transform import Rotation,Slerp
            spin=Slerp([0,1],Rotation.from_matrix([W[:3,:3],Ws[j+1][:3,:3]]))
            for u in np.linspace(0,1,int(1.4*fps)):
                M=np.eye(4);M[:3,:3]=spin(ease(u)).as_matrix();M[:3,3]=Ws[j][:3,3]+Ws[j][:3,:3]@center-M[:3,:3]@center
                add(M,up,down,j+1)
            V2=mesh.vertices@Ws[j+1][:3,:3].T+Ws[j+1][:3,3]
            nxt=stops(V2,mesh.faces,np.flatnonzero(arrays['face_area']==j+1),arrays['face_axis']@Ws[j+1][:3,:3].T,cell)[0]
            top=nxt['tip']-hover*down
            for u in np.linspace(0,1,int(.5*fps)):add(Ws[j+1],(1-ease(u))*up+ease(u)*top,down,j+1)
    add(Ws[-1],frames[-1]['tip'],down,len(Ws)-1,int(1.5*fps))
    return frames


class Scene:
    """The object, the head at its real size and the gold max-tilt cone in one studio."""
    def __init__(self,mesh,setup,width,height,arm=False):
        from subdivide_task.step5.video import raster as r
        import moderngl
        self.r=r;self.gl=moderngl;self.mesh=mesh;self.size=(width,height);self.setup=setup;self.workspace_wireframe=None
        L,H,N=setup['needle_length_m'],setup['head_radius_m'],setup['needle_radius_m'];self.L=L
        self.ctx=ctx=r.context();ctx.enable(moderngl.DEPTH_TEST);self.studio=r.Studio(ctx,width,height)
        self.shader=ctx.program(vertex_shader=r.VERTEX,fragment_shader=r.FRAGMENT)
        self.shadow_shader=ctx.program(vertex_shader=r.SHADOW_VERTEX,fragment_shader=r.SHADOW_FRAGMENT)
        self.line_shader=ctx.program(vertex_shader=r.LINE_VERTEX,fragment_shader=r.LINE_FRAGMENT);self.line_shader['ink'].value=GOLD
        span=1.6*float(np.linalg.norm(mesh.extents))+L
        key=np.array([-.65,-.9,1.45]);key/=np.linalg.norm(key)
        light_vp=r.orthographic(span,span)@r.view_matrix(key,np.array([0,0,.3*L]))
        for program in (self.shader,self.shadow_shader):program['light_projection'].write(np.asarray(light_vp,dtype='f4').T.tobytes())
        self.shader['shadow_map'].value=3
        self.body=r.Shape(ctx,self.shader,mesh.vertices,mesh.faces,r.GRAY,dynamic=True,metallic=.03,roughness=.63)
        self.tools={False:(r.Shape(ctx,self.shader,*r.tool_mesh(0.,L,N,N),STEEL,metallic=.6,roughness=.3),
                           r.Shape(ctx,self.shader,*r.tool_mesh(L,L+.07,H,H),HEAD,metallic=.2,roughness=.45)),
                    True:(r.Shape(ctx,self.shader,*r.tool_mesh(0.,L,N,N),GHOST_STEEL,metallic=.2,roughness=.5),
                          r.Shape(ctx,self.shader,*r.tool_mesh(L,L+.07,H,H),GHOST_HEAD,metallic=.1,roughness=.6))}
        angles=np.linspace(0,2*np.pi,97);tilt=np.radians(setup['max_tilt_deg'])
        self.rim=np.c_[np.cos(angles)*np.sin(tilt),np.sin(angles)*np.sin(tilt),np.full(97,np.cos(tilt))]*L
        self.sides=np.vstack([[np.zeros(3),self.rim[k]] for k in range(0,96,12)])
        self.lines=ctx.buffer(reserve=4096*12,dynamic=True)
        self.line_vao=ctx.vertex_array(self.line_shader,[(self.lines,'3f','position')]);ctx.line_width=3
        self.view=np.array([.40,-1.,.62]);self.view/=np.linalg.norm(self.view);self.colors=None
        # The cone is the max tilt about the nominal needle direction.
        self.nominal=np.asarray(setup.get('working_direction_world',[0.,0.,-1.]),float);self.nominal/=np.linalg.norm(self.nominal)
        self.arm=[]
        if arm:
            from codes.simulation.kuka import description,visual_parts
            links,_,_=description()
            for i,link in enumerate(links):
                offset=np.fromstring(link.find('visual/origin').get('xyz'),sep=' ')
                for v,f,rgba in visual_parts(i):
                    color=rgba[:3] if not (i in (2,4) and np.allclose(rgba[:3],.2)) else np.full(3,.6)
                    self.arm.append((i,r.Shape(ctx,self.shader,v+offset,f,tuple(color),metallic=.25,roughness=.4)))

    def points(self,W,tools):
        """Everything a frame shows: the placed object, each tool to its head top, the cone."""
        out=[self.mesh.vertices[::7]@W[:3,:3].T+W[:3,3]]
        R=tool_frame(np.zeros(3),self.nominal)[:3,:3]
        for tip,axis,*_ in tools:out.append(tip+np.vstack([np.zeros(3),-axis*(self.L+.07),self.rim[::12]@R.T]))
        return np.vstack(out)

    def fit(self,points,margin=1.08,target=None):
        width,height=self.size;right=np.cross([0,0,1],self.view);right/=np.linalg.norm(right);up=np.cross(self.view,right)
        x=points@right;y=points@up;self.target=right*(x.min()+x.max())/2+up*(y.min()+y.max())/2
        if target is not None:self.target=np.asarray(target,float)
        scale=max(x.max()-x.min(),(y.max()-y.min())*width/height)*margin
        self.vp=self.r.orthographic(scale,scale*height/width)@self.r.view_matrix(self.view,self.target)
        for program in (self.shader,self.line_shader):program['view_projection'].write(np.asarray(self.vp,dtype='f4').T.tobytes())
        self.shader['eye'].value=tuple(self.target+self.view*4)

    def draw(self,W,colors,tools,links=None,segments=(),shapes=()):
        """tools: (tip, needle axis, ghost) triples; the cone is drawn at the first tip, if any;
        links: the arm's link transforms; segments: (world line pairs, ink); shapes: more (Shape, world)
        pairs. Returns RGB bytes."""
        ctx,studio,gl=self.ctx,self.studio,self.gl
        if self.colors is None or not np.array_equal(colors,self.colors):self.body.paint(colors);self.colors=colors
        draws=[(self.body,W)]+[(shape,tool_frame(tip,axis)) for tip,axis,ghost in tools for shape in self.tools[ghost]]
        if links is not None:draws+=[(shape,links[i]) for i,shape in self.arm]
        draws+=list(shapes)
        studio.shadow_buffer.use();ctx.viewport=(0,0,1536,1536);studio.shadow_buffer.clear(depth=1)
        for shape,world in draws:shape.shadow(ctx,self.shadow_shader,world)
        ctx.enable(gl.DEPTH_TEST)
        studio.gbuffer.use();ctx.viewport=(0,0,*studio.size);studio.gbuffer.clear(1,1,1,1,depth=1);studio.shadow.use(3)
        for shape,world in draws:shape.draw(self.shader,world)
        if tools:
            cone=tool_frame(tools[0][0],self.nominal);self.line_shader['world'].write(np.asarray(cone,dtype='f4').T.tobytes())
            self.line_shader['ink'].value=GOLD
            for strip,mode in ((self.rim,gl.LINE_STRIP),(self.sides,gl.LINES)):
                self.lines.write(strip.astype('f4').tobytes());self.line_vao.render(mode,vertices=len(strip))
        for points,ink in segments:
            self.line_shader['world'].write(np.eye(4,dtype='f4').tobytes());self.line_shader['ink'].value=ink
            self.lines.write(np.asarray(points,'f4').tobytes());self.line_vao.render(gl.LINES,vertices=len(points))
        if self.workspace_wireframe is not None:
            points,ink=self.workspace_wireframe
            # These are world-space 3D edges: the object must hide the edges behind it.
            ctx.enable(gl.DEPTH_TEST)
            self.line_shader['world'].write(np.eye(4,dtype='f4').tobytes());self.line_shader['ink'].value=ink
            self.lines.write(np.asarray(points,'f4').tobytes());self.line_vao.render(gl.LINES,vertices=len(points))
        return studio.finish(self.vp,self.view)

    def close(self):self.ctx.release()


def pose_colors(mesh,face_area,pose,painted,gray,pink):
    colors=np.tile(gray,(len(mesh.faces),1));colors[(face_area==pose)&~painted]=LIGHT;colors[painted]=pink
    return colors


def kuka(setup):
    """The work arm at the station's base pose, carrying the head: flange 70 mm behind the needle root."""
    from subdivide_task.step5.video.kuka import Arm
    return Arm(np.asarray(setup['base_position_world_m'],float),np.asarray(setup['base_rotation_world'],float),
               tool=setup['needle_length_m']+.07)


def work_point(setup,radius,step=.01,samples=60,seed=3):
    """Where the object sits: from the configured window centre forward, away from the base, the first
    point where the arm reaches every tip within `radius` with the needle anywhere in its cone."""
    arm=kuka(setup);a=np.asarray(setup['working_direction_world']);c=np.asarray(setup['work_window_center_m'])
    base=np.asarray(setup['base_position_world_m']);out=c-base;out[2]=0;out/=np.linalg.norm(out)
    rng=np.random.default_rng(seed);tilt=np.radians(setup['max_tilt_deg']);trials=[]
    for _ in range(samples):
        u=rng.normal(size=3);u-=u@a*a;u/=np.linalg.norm(u);t=tilt*np.sqrt(rng.random())
        trials.append((rng.uniform(-radius,radius,3),np.cos(t)*a+np.sin(t)*u))
    for k in range(200):
        point=c+k*step*out
        try:
            q=arm.solve(point,-a)
            for offset,d in trials:arm_links(arm,point+offset,d,q)
            return point,k*step
        except ValueError:continue
    raise ValueError('No reachable work point within 2 m forward of the window')


def arm_links(arm,tip,axis,previous=None):
    """Link transforms (8, 4, 4) with the needle tip at `tip` along `axis`, and the joints."""
    from subdivide_task.step5.video.prepare import matrices
    try:q=arm.solve(tip,-np.asarray(axis),previous)
    except ValueError:q=arm.solve(tip,-np.asarray(axis))
    return matrices(arm.forward(q,geometry=True)[2]),q


def arm_motion(mesh,frames,setup):
    """Joint path of the KUKA carrying the head: flange behind the head, needle tip on the path.

    Returns per-frame link transforms and a sampled check of the arm body against the object."""
    from subdivide_task.step5.video.prepare import matrices
    from scipy.spatial import ConvexHull
    arm=kuka(setup);q=None;links=[];fallbacks=0
    for f in frames:
        try:q=arm.solve(f['tip'],-f['axis'],q)
        except ValueError:q=arm.solve(f['tip'],-f['axis']);fallbacks+=1
        links.append(matrices(arm.forward(q,geometry=True)[2]))
    hulls=[ConvexHull(v) for v in arm.collisions];sample=mesh.vertices[::3];hits=0;checked=0
    for f,L in list(zip(frames,links))[::4]:
        P=sample@f['W'][:3,:3].T+f['W'][:3,3];checked+=1
        for i in range(2,8):
            local=(P-L[i][:3,3])@L[i][:3,:3]
            if np.any(np.all(local@hulls[i].equations[:,:3].T+hulls[i].equations[:,3]<=0,axis=1)):hits+=1;break
    return links,dict(base_m=setup['base_position_world_m'],tool_from_flange_m=setup['needle_length_m']+.07,ik_restarts=fallbacks,
        sampled_frames=checked,sampled_frames_arm_body_in_object=hits,
        method='object vertices inside the convex hulls of KUKA links 2-7, every fourth frame')


def render(object_name='B',output=None,fps=24,width=1400,height=1200,cell=.015,start_s=6.):
    start=time.perf_counter();work=ROOT/'subdivide_task/data'/object_name/'step1_reach'
    report=json.loads((work/'plan.json').read_text())
    with np.load(work/'plan.npz') as z:arrays={k:z[k] for k in z.files}
    mesh=load_shape(ROOT/'objects'/object_name/'mesh.stl')
    frames=timeline(mesh,report,arrays,cell,fps)
    if not np.isfinite(start_s) or not 0<=start_s<len(frames)/fps:raise ValueError('Video start must be within the full motion')
    first_frame=round(start_s*fps)
    if first_frame>=len(frames):raise ValueError('Video start rounds past the last frame')
    links,arm=arm_motion(mesh,frames,report['setup'])
    output=Path(output or ROOT/'subdivide_task/idea/head_reach.mp4');output.parent.mkdir(parents=True,exist_ok=True)
    scene=Scene(mesh,report['setup'],width,height,arm=True)
    # The whole station: base, arm and object, seen from the side so the slant shows.
    scene.view=STATION_VIEW
    scene.fit(np.vstack([np.vstack([scene.points(f['W'],[(f['tip'],f['axis'])]),L[:,:3,3]]) for f,L in zip(frames[::6],links[::6])]),1.06)
    # Keep the original camera fit, painted state and continuous IK solution when trimming.
    source_frames=len(frames);frames=frames[first_frame:];links=links[first_frame:]
    ffmpeg=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
        '-vf','vflip','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(output)],
        stdin=subprocess.PIPE)
    try:
        for number,(f,L) in enumerate(zip(frames,links)):
            colors=pose_colors(mesh,arrays['face_area'],f['pose'],f['painted'],scene.r.GRAY,scene.r.PINK)
            ffmpeg.stdin.write(scene.draw(f['W'],colors,[(f['tip'],f['axis'],False)],L))
            if number%240==0:print('HEAD VIDEO FRAME',number,'/',len(frames),flush=True)
    finally:
        ffmpeg.stdin.close();code=ffmpeg.wait();scene.close()
    if code:raise RuntimeError('Video encoding failed')
    result=dict(file=str(output),frames=len(frames),fps=fps,duration_s=len(frames)/fps,areas=len(report['areas']),
        source_start_s=first_frame/fps,source_frames=source_frames,arm=arm,elapsed_s=time.perf_counter()-start)
    print('HEAD VIDEO',result,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--object',default='B');p.add_argument('--output',type=Path)
    p.add_argument('--start',type=float,default=6.,help='Start time in the full motion, in seconds (default: 6)')
    a=p.parse_args();render(a.object,a.output,start_s=a.start)
