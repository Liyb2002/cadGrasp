from pathlib import Path
import argparse,json,shutil,subprocess,time
from scipy.spatial.transform import Rotation,Slerp
from geometry import ROOT,PROOF,TEMP,C,G,D,np,trimesh,Model,smooth
from renderer import Renderer

def placed(p,T):return C.V.CPU.placed(p,rotation=T[:3,:3],point=T[:3,3])

def basis(angle):
    c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    view=R@(np.array([1.,-1.,1.])/np.sqrt(3));right=R@(np.array([1.,1.,0.])/np.sqrt(2))
    return np.array([right,np.cross(view,right),view])

def flat(points,color,alpha):
    mesh=trimesh.Trimesh(points,[[0,i,i+1] for i in range(1,len(points)-1)],process=False)
    p=C.V.piece(mesh,color,alpha);p['unlit']=True;return p

class Movie:
    def __init__(self):
        self.model=Model();m=self.model;self.renderer=Renderer()
        self.slerp=Slerp([0,1],Rotation.from_matrix(np.stack([m.T0[:3,:3],m.T1[:3,:3]])))
        points=[]
        mesh=G.unpack(m.core+m.maximum)
        for t in np.linspace(0,8,81):
            T=self.state(t);points.extend([C.F.transform_points(mesh.vertices,T),C.F.transform_points(m.start['hull'],T@m.inverse)])
        points.extend([C.F.transform_points(m.data['body'].vertices,row['object_transform']) for row in m.rows])
        points=np.vstack(points);low=points.min(0);high=points.max(0)
        self.focus=(low+high)/2
        # Face the real open edge in the starting pose.
        edge=m.xy[(m.gap_edge+1)%len(m.xy)]-m.xy[m.gap_edge]
        normal=np.array([edge[1],-edge[0]]);normal/=np.linalg.norm(normal)
        angle=np.arctan2(normal[1],normal[0])+np.pi/4
        self.B=basis(angle)
        projected=(points-self.focus)@self.B[:2].T
        self.span=2*max(np.abs(projected[:,1]).max(),np.abs(projected[:,0]).max()*900/1600)*1.15
        self.camera=(self.focus,self.B,self.span)
        floor_low=low[:2]-.015;floor_high=high[:2]+.015
        floor=np.array([[*floor_low,0],[floor_high[0],floor_low[1],0],[*floor_high,0],[floor_low[0],floor_high[1],0]])
        self.floor=C.V.CPU.placed(flat(floor,'#D1D6DB',.4))
        floor_outline=C.H.outline_piece(floor,'#AFB8C1',self.camera,900,half_width_pixels=.7)
        floor_outline['opacity']=.48;floor_outline['unlit']=True
        self.floor_border=C.V.CPU.placed(floor_outline)
        self.core=C.V.piece(m.core_mesh,C.V.BLUE,.78)
        self.body=C.V.piece(m.data['body'],C.V.GRAY,.18)
        self.hull0=flat(m.start['hull'],m.start['color'],.10);self.hull0['bias']=.5
        self.hull1=flat(m.end['hull'],m.end['color'],.08);self.hull1['bias']=.5
        outer=G.unpack(m.maximum)
        in_floor=C.F.transform_points(outer.vertices,m.T0)[:,:2]
        lo=in_floor.min(0)-.012;hi=in_floor.max(0)+.012
        rectangle=np.array([[*lo],[hi[0],lo[1]],[*hi],[lo[0],hi[1]]])
        legal=D.growth.clip_polygon(rectangle,m.a,m.b)
        forbidden=D.growth.clip_polygon(rectangle,-m.a,-m.b)
        self.legal=flat(np.c_[legal,np.zeros(len(legal))],'#31A184',.14)
        self.forbidden=flat(np.c_[forbidden,np.zeros(len(forbidden))],'#CF5667',.045)
        crossings=[]
        for a,b in zip(rectangle,np.roll(rectangle,-1,axis=0)):
            x=a@m.a+m.b;y=b@m.a+m.b
            if x*y<0:crossings.append(np.r_[a+(b-a)*x/(x-y),0.])
        assert len(crossings)==2
        self.separator=C.V.piece(__import__('geometry').rod(*crossings,radius=.00032),'#5D7775',.6)
        self.cache={}

    def state(self,t):
        m=self.model;f=smooth((t-1.4)/2.)
        T=np.eye(4);T[:3,:3]=self.slerp([f]).as_matrix()[0];T[:3,3]=(1-f)*m.T0[:3,3]+f*m.T1[:3,3]
        return T

    def frame(self,t):
        m=self.model;T=self.state(t);padding=.029*smooth((t-4.3)/2.2)
        key=round(padding,9)
        if key not in self.cache:
            base=m.expanded_solid(padding);self.cache[key]=(base,C.V.piece(G.unpack(base),C.V.BLUE,.82))
        base,p=self.cache[key];parts=[self.floor,self.floor_border,placed(self.core,T),placed(p,T)]
        if t<1.4:
            opacity=.18*(1-smooth((t-1.)/.35))
            if opacity>1e-5:
                body=dict(self.body,opacity=opacity);parts.append(placed(body,m.start['object_transform']))
        elif t>=3.4:
            opacity=.18*smooth((t-3.4)/.5)
            if opacity>1e-5:parts.append(placed(dict(self.body,opacity=opacity),m.end['object_transform']))
        H=T@m.inverse
        parts.append(placed(self.hull0,H))
        # A translucent platform and continuous contour, with no scattered points.
        world_hull=C.F.transform_points(m.start['hull'],H)
        outline=C.H.outline_piece(world_hull,m.start['color'],self.camera,900,half_width_pixels=1.)
        outline['opacity']=.8;outline['bias']=1.5;outline['unlit']=True;parts.append(C.V.CPU.placed(outline))
        if t>=3.4:
            parts.append(C.V.CPU.placed(self.hull1))
            contour=C.H.outline_piece(m.end['hull'],m.end['color'],self.camera,900,half_width_pixels=1.)
            contour['opacity']=.65;contour['bias']=.75;contour['unlit']=True;parts.append(C.V.CPU.placed(contour))
        if t>=1.4:
            below=base.trim_by_plane(tuple(-T[2,:3]),float(T[2,3]*1000))
            if G.volume(below)>1e-14:
                red=C.V.piece(G.unpack(below),'#D95260',.55);red['unlit']=True;red['bias']=1.
                parts.append(placed(red,T))
        if t>=5.8:
            alpha=smooth((t-5.8)/.7)
            parts.extend([placed(dict(self.legal,opacity=.14*alpha),H),placed(dict(self.forbidden,opacity=.045*alpha),H),placed(dict(self.separator,opacity=.6*alpha),H)])
        return self.renderer.render(parts,self.camera)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--encode',action='store_true');args=parser.parse_args()
    movie=Movie();directory=TEMP/'execution';sources=C.freeze(directory)
    paths=[TEMP/n for n in ['geometry.py','renderer.py','transparency.cpp','movie.py']]
    own={p.name:C.digest(p) for p in paths}
    for path in paths:shutil.copy2(path,directory/path.name)
    original=json.loads((TEMP/'original_videos.json').read_text())
    def verify():
        movie.model.verify()
        for p,h in sources.items():assert C.digest(ROOT/p)==C.digest(directory/'executed_sources'/p)==h
        for name,h in own.items():assert C.digest(TEMP/name)==C.digest(directory/name)==h
        assert {p.name:C.digest(p) for p in (PROOF/'videos').iterdir()}==original
        C.save(TEMP/'verification.json',dict(inputs=movie.model.inputs.protected,sources=sources,temp_sources=own,complete=True))
        print('Complete connected support retained; one pose change; open base; unbounded halfplane certificate verified.',flush=True)
    if not args.encode:
        for t,name in [(0.,'start'),(2.4,'turn'),(3.8,'collision'),(5.4,'expand'),(7.75,'proof')]:movie.frame(t).save(TEMP/(name+'.png'))
        verify();return
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r','24','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-profile:v','main','-pix_fmt','yuv420p','-movflags','+faststart',str(TEMP/'02.mp4')],stdin=subprocess.PIPE)
    started=time.monotonic()
    try:
        for i in range(192):
            encoder.stdin.write(movie.frame(i/24).tobytes())
            if i%24==0:print(json.dumps(dict(frame=i,total=192,elapsed_s=round(time.monotonic()-started,1))),flush=True)
        encoder.stdin.close();assert encoder.wait()==0
    except BaseException:encoder.kill();encoder.wait();raise
    verify()

if __name__=='__main__':main()
