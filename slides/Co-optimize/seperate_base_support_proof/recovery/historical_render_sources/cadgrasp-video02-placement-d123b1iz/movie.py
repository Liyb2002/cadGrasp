from pathlib import Path
import argparse,json,shutil,subprocess,time
from geometry import ROOT,PROOF,TEMP,C,G,np,trimesh,Model,smooth
from renderer import Renderer

def placed(p,T):return C.V.CPU.placed(p,rotation=T[:3,:3],point=T[:3,3])

def camera_basis(angle):
    view=np.array([np.cos(angle),np.sin(angle),.48]);view/=np.linalg.norm(view)
    right=np.array([-np.sin(angle),np.cos(angle),0.])
    return np.array([right,np.cross(view,right),view])

def platform(points,color):
    mesh=trimesh.Trimesh(points,[[0,i,i+1] for i in range(1,len(points)-1)],process=False)
    p=C.V.piece(mesh,color,.10);p['unlit']=True;p['bias']=.5;return p

class Movie:
    def __init__(self):
        self.model=Model();m=self.model;self.renderer=Renderer()
        minimum=min(float(C.F.transform_points(m.maximum_mesh.vertices,T)[:,2].min()) for T in [m.T0,m.T1])
        self.lift_height=max(.085,-minimum+.020)
        points=[]
        for T in [m.T0,m.T1]:
            base=C.F.transform_points(m.maximum_mesh.vertices,T)
            points.extend([base,base+[0,0,self.lift_height]])
        points.extend(C.F.transform_points(m.data['body'].vertices,r['object_transform']) for r in m.rows)
        points=np.vstack(points)
        edge=m.xy[(m.gap_edge+1)%len(m.xy)]-m.xy[m.gap_edge]
        normal=np.array([edge[1],-edge[0]]);normal/=np.linalg.norm(normal)
        self.B=camera_basis(np.arctan2(normal[1],normal[0]))
        u=self.B[0];v=np.r_[-self.B[2,:2]/np.linalg.norm(self.B[2,:2]),0.]
        rotation=np.array([u,v,[0.,0.,1.]]).T
        angle=np.radians(25.);c,s=np.cos(angle),np.sin(angle)
        rotation=rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        planar=points@rotation
        low=planar[:,:2].min(0)-.015;high=planar[:,:2].max(0)+.015
        depth=max(.085,-minimum+.018)
        transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=rotation@np.r_[(low+high)/2,-depth/2]
        floor=trimesh.creation.box(extents=np.r_[high-low,depth],transform=transform)
        self.ground=C.V.CPU.placed(C.V.piece(floor,'#C5CDD3',1.))
        all_points=np.vstack([points,floor.vertices])
        projected=all_points@self.B.T;lo=projected.min(0);hi=projected.max(0)
        focus=((lo+hi)/2)@self.B
        span=max(hi[1]-lo[1],(hi[0]-lo[0])*900/1600)*1.08
        self.camera=(focus,self.B,span)
        self.core=C.V.piece(m.core_mesh,C.V.BLUE,.78);self.body=C.V.piece(m.data['body'],C.V.GRAY,.18)
        self.platforms=[platform(r['hull'],r['color']) for r in [m.start,m.end]]
        self.contours=[]
        for r in [m.start,m.end]:
            p=C.H.outline_piece(r['hull'],r['color'],self.camera,900,half_width_pixels=1.)
            p['opacity']=.8;p['bias']=1.5;p['unlit']=True;self.contours.append(p)
        self.cache={};self.cut_cache={}
        self.assert_motion()
        C.save(TEMP/'motion.json',dict(lift_height_m=self.lift_height,duration_seconds=8,
            stages=['initial installed pose19','vertical pickup','instant pose switch while airborne','vertical set-down','installed pose11 ground overlap','open base enlargement'],
            rotation_interpolated=False,horizontal_movement_during_lift_and_drop=False,
            ground_solid=True,ground_opacity=1.,collision_highlight_only_after_setdown=True,
            demand_platforms_stay_in_task_floor=True,full_fixture_retained=True,
            text=False,scatter_points=False,right_panels=False,grid=False))

    def state(self,t):
        m=self.model
        if t<2.25:
            T=m.T0.copy();height=self.lift_height*smooth((t-1.)/1.)
        else:
            T=m.T1.copy();height=self.lift_height*(1-smooth((t-2.5)/1.3))
        T[2,3]+=height
        return T

    def assert_motion(self):
        m=self.model
        for t in np.linspace(0,2.24,80):
            T=self.state(t);assert np.allclose(T[:3,:3],m.T0[:3,:3]) and np.allclose(T[:2,3],m.T0[:2,3])
        for t in np.linspace(2.25,8.,80):
            T=self.state(t);assert np.allclose(T[:3,:3],m.T1[:3,:3]) and np.allclose(T[:2,3],m.T1[:2,3])
        np.testing.assert_allclose(self.state(0),m.T0,atol=1e-12)
        np.testing.assert_allclose(self.state(3.8),m.T1,atol=1e-12)
        for t in [2.,2.24,2.25,2.5]:
            assert C.F.transform_points(m.maximum_mesh.vertices,self.state(t))[:,2].min()>.019
        for i in range(116,165):
            base=m.expanded_solid(.029*smooth((i/24-4.8)/2.))
            assert G.volume(m.full_min-base)<1e-14

    def frame(self,t):
        m=self.model;T=self.state(t);padding=.029*smooth((t-4.8)/2.)
        key=round(padding,9)
        if key not in self.cache:
            base=m.expanded_solid(padding)
            self.cache[key]=(base,C.V.piece(G.unpack(base),C.V.BLUE,.82))
        base,p=self.cache[key];parts=[self.ground,placed(self.core,T),placed(p,T)]
        if t<1.:
            opacity=.18*(1-smooth((t-.8)/.2))
            if opacity>1e-5:parts.append(placed(dict(self.body,opacity=opacity),m.start['object_transform']))
            parts.extend([C.V.CPU.placed(self.platforms[0]),C.V.CPU.placed(self.contours[0])])
        if t>=3.8:
            parts.append(placed(self.body,m.end['object_transform']))
            parts.extend([C.V.CPU.placed(self.platforms[1]),C.V.CPU.placed(self.contours[1])])
            if key not in self.cut_cache:
                full=m.core+base
                below=full.trim_by_plane(tuple(-T[2,:3]),float(T[2,3]*1000))
                red=C.V.piece(G.unpack(below),'#D94C5D',.92);red['bias']=1.
                self.cut_cache[key]=red
            parts.append(placed(self.cut_cache[key],T))
        return self.renderer.render(parts,self.camera)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--encode',action='store_true');args=parser.parse_args()
    movie=Movie();directory=TEMP/'execution';sources=C.freeze(directory)
    paths=[TEMP/n for n in ['geometry.py','saved_inputs.py','renderer.py','transparency.cpp','movie.py']]
    own={p.name:C.digest(p) for p in paths}
    for path in paths:shutil.copy2(path,directory/path.name)
    original=json.loads((TEMP/'original_videos.json').read_text())
    def verify():
        movie.model.inputs.verify()
        for p,h in sources.items():assert C.digest(ROOT/p)==C.digest(directory/'executed_sources'/p)==h
        for name,h in own.items():assert C.digest(TEMP/name)==C.digest(directory/name)==h
        assert {p.name:C.digest(p) for p in (PROOF/'videos').iterdir()}==original
        C.save(TEMP/'verification.json',dict(complete=True,inputs=movie.model.inputs.protected,sources=sources,temp_sources=own))
        print('Verified vertical pickup/set-down, exact installed states, airborne-only instant pose change, and protected sources.',flush=True)
    if not args.encode:
        for t,name in [(0.,'start'),(2.,'lift'),(2.4,'air_pose11'),(3.4,'drop'),(4.3,'landed'),(7.75,'expanded')]:movie.frame(t).save(TEMP/(name+'.png'))
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
