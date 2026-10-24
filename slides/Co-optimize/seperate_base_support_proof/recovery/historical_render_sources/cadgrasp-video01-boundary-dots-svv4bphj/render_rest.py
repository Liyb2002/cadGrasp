"""Videos 02--04 always carry the exact completed video-01 material mesh."""
from pathlib import Path
import argparse,json,shutil,subprocess,time
import scene as S
from scene import C,np,G,Image,MAIN,SMALL,FPS,SECONDS,TEMP,PROOF,ROOT,placed,compose,smooth,basis
from scipy.spatial.transform import Rotation,Slerp
from work_access import tangent_frame

def shifted(T,z):
    result=np.array(T,copy=True);result[2,3]+=z;return result

def arrow(start,direction,length,large=False):
    direction=np.asarray(direction)/np.linalg.norm(direction)
    head=min(length*.23,.005 if large else .003)
    radius=.0007 if large else .00016
    shaft=S.rod(start,start+direction*(length-head),radius=radius)
    rot=S.trimesh.geometry.align_vectors([0,0,1],direction)
    rot[:3,3]=start+direction*(length-head)
    tip=S.trimesh.creation.cone(radius=.0025 if large else .0008,height=head,sections=10 if large else 6,transform=rot)
    return S.trimesh.util.concatenate([shaft,tip])

def polygon_piece(points,color,alpha):
    mesh=S.trimesh.Trimesh(points,[[0,1,2],[0,2,3]],process=False)
    p=C.V.piece(mesh,color,alpha);p['unlit']=True;return C.V.CPU.placed(p)

def floor_for(points):
    xy=np.asarray(points)[:,:2];low=xy.min(0)-.018;high=xy.max(0)+.018
    return np.array([[*low,0.],[high[0],low[1],0.],[*high,0.],[low[0],high[1],0.]])

class GroundVideos:
    def __init__(self,scene,geometry,mode):
        self.scene,self.mode,self.geometry=scene,mode,geometry
        Ts=[row['fixture_transform'] for row in scene.rows]
        self.pairs=[(a,b,Slerp([0,1],Rotation.from_matrix(np.stack([a[:3,:3],b[:3,:3]])))) for a,b in zip(Ts,Ts[1:]+Ts[:1])]
        samples=[]
        for t in np.linspace(0,8,81):
            T=self.transform(t)
            samples.append(C.F.transform_points(scene.final.vertices,T))
        samples.extend(scene.body_world[0].vertices+[0,0,z] for z in [0,geometry['safe_rotation_lift_m']])
        all_points=np.vstack(samples)
        self.floor=floor_for(all_points)
        self.camera=C.V.camera_for(np.vstack([all_points,self.floor]))
        self.floor_piece=polygon_piece(self.floor,'#DFE3E6',.25)
        self.right_cameras=[];self.right_floors=[]
        for row,body in zip(scene.rows,scene.body_world):
            support=C.F.transform_points(scene.final.vertices,row['fixture_transform'])
            floor=floor_for(np.vstack([support,body.vertices]))
            maximum=geometry['safe_rotation_lift_m'] if mode==3 else 0.
            points=np.vstack([support,support+[0,0,maximum],body.vertices,body.vertices+[0,0,maximum],floor])
            self.right_cameras.append(C.V.camera_for(points));self.right_floors.append(floor)
        span=max(cam[2] for cam in self.right_cameras)
        self.right_cameras=[(f,b,span) for f,b,_ in self.right_cameras]
        self.panel_cache={}

    def path(self,t):
        u=float(np.clip((t-.8)/1.5,0,4))
        if u>=4:return self.scene.rows[0]['fixture_transform'].copy()
        index=int(u);fraction=smooth(u-index)
        a,b,slerp=self.pairs[index]
        T=np.eye(4);T[:3,:3]=slerp([fraction]).as_matrix()[0]
        T[:3,3]=(1-fraction)*a[:3,3]+fraction*b[:3,3]
        return T

    def lift(self,t,installed_index=0):
        if self.mode==2:return 0.
        safe=self.geometry['safe_rotation_lift_m'];final=self.geometry['installed_lift_to_floor_m'][installed_index]
        if t<.7:return safe*smooth(t/.7)
        if t<6.8:return safe
        return safe+(final-safe)*smooth((t-6.8)/.8)

    def transform(self,t):return shifted(self.path(t),self.lift(t))

    def collision(self,T):
        below=self.scene.solid.trim_by_plane(tuple(-T[2,:3]),float(T[2,3]*1000))
        if G.volume(below)<1e-14:return []
        mesh=G.unpack(below);p=C.V.piece(mesh,'#D95260',.55)
        # X-ray tint exposes the actual below-floor material through the fixture.
        p['bias']=1.;p['unlit']=True
        return [placed(p,T)]

    def frame(self,t):
        scene=self.scene;T=self.transform(t)
        parts=[self.floor_piece,*scene.fixture_parts(T),*self.collision(T)]
        # Remove the workpiece before the empty fixture changes installed pose.
        if t<.6 or t>=6.8:
            placement=np.linalg.inv(scene.rows[0]['fixture_transform'])@scene.rows[0]['object_transform']
            parts.append(placed(C.V.piece(scene.data['body'],C.V.GRAY,.18),T@placement))
        main=scene.renderer.render(parts,self.camera,MAIN)
        key='static' if self.mode==2 else 'raised' if .7<=t<6.8 else 'end' if t>=7.6 else None
        if key in self.panel_cache:panel=self.panel_cache[key]
        else:
            panel=Image.new('RGB',(700,900),'white')
            for i,(row,floor,cam) in enumerate(zip(scene.rows,self.right_floors,self.right_cameras)):
                z=self.lift(t,i);state=shifted(row['fixture_transform'],z)
                values=[polygon_piece(floor,'#DFE3E6',.25),*scene.fixture_parts(state),*self.collision(state),
                        placed(C.V.piece(scene.data['body'],C.V.GRAY,.18),shifted(row['object_transform'],z))]
                panel.paste(scene.renderer.render(values,cam,SMALL),(7+(i%2)*346,55+(i//2)*450))
            if key is not None:self.panel_cache[key]=panel
        return compose(main,panel)

class WorkVideo:
    def __init__(self,scene,geometry):
        self.scene,self.geometry=scene,geometry
        self.patches=[];self.families=[];self.selected=[];self.collisions=[];self.family_vertices=[]
        placements=[];body_parts=[]
        for i,(row,work) in enumerate(zip(scene.rows,geometry['work'])):
            with np.load(TEMP/f'work_{i}.npz',allow_pickle=False) as arrays:
                triangles=arrays['triangles'].copy();normals=arrays['normals'].copy();P=arrays['placement'].copy()
            if not any(np.allclose(P,other,atol=1e-12) for other in placements):
                placements.append(P);body_parts.append((C.V.piece(scene.data['body'],C.V.GRAY,.18),P))
            vertices=C.F.transform_points(triangles.reshape(-1,3),P)
            mesh=S.trimesh.Trimesh(vertices,np.arange(len(vertices)).reshape(-1,3),process=False)
            self.patches.append(C.V.piece(mesh,row['color'],.55))
            rays=[]
            for j,(triangle,normal) in enumerate(zip(triangles,normals)):
                root=triangle.mean(0);directions=[normal]
                if j%3==0:
                    tangent=tangent_frame(normal)
                    for phase in [0,2*np.pi/3,4*np.pi/3]:
                        direction=normal+np.tan(np.radians(work['half_angle_deg']*.85))*(np.cos(phase)*tangent[:,0]+np.sin(phase)*tangent[:,1])
                        directions.append(direction/np.linalg.norm(direction))
                for direction in directions:
                    start=C.F.transform_points(root[None],P)[0]
                    rays.append(arrow(start,P[:3,:3]@direction,.045))
            family=S.trimesh.util.concatenate(rays)
            self.families.append(C.V.piece(family,row['color'],.31));self.family_vertices.append(family.vertices)
            if work['selected_face'] is None:self.selected.append(None);self.collisions.append(None)
            else:
                selected=arrow(np.array(work['selected_ray_root_m']),np.array(work['selected_ray_direction']),
                               work['selected_ray_material_distance_m']+.012,large=True)
                self.selected.append(C.V.piece(selected,'#C74452',.96))
                collision=S.trimesh.load(TEMP/f'work_collision_{i}.obj',force='mesh',process=False)
                highlight=C.V.piece(collision,'#CC5360',.9);highlight['bias']=8e-6;self.collisions.append(highlight)
        self.body_parts=body_parts
        self.chosen=next(i for i,p in enumerate(self.selected) if p is not None)
        all_points=np.vstack([scene.final.vertices,*self.family_vertices,
                             *[C.F.transform_points(scene.data['body'].vertices,P) for P in placements]])
        displayed=C.F.transform_points(all_points,scene.ref)
        self.focus=(displayed.min(0)+displayed.max(0))/2
        self.span=2*max(float(np.abs((displayed-self.focus)@basis(a)[:2].T).max()) for a in np.linspace(0,2*np.pi,361))*1.12
        cameras=[]
        for i,(row,body) in enumerate(zip(scene.rows,scene.body_world)):
            points=np.vstack([C.F.transform_points(np.vstack([scene.final.vertices,self.family_vertices[i]]),row['fixture_transform']),body.vertices])
            cameras.append(C.V.camera_for(points))
        span=max(cam[2] for cam in cameras);self.cameras=[(f,b,span) for f,b,_ in cameras]
        self.right_cache={}

    def frame(self,t):
        scene=self.scene;camera=(self.focus,basis(2*np.pi*min(t/7.75,1)),self.span)
        T=scene.ref;parts=scene.fixture_parts(T)
        parts.extend(placed(p,T@P) for p,P in self.body_parts)
        parts.extend(placed(p,T) for p in self.patches)
        if t>=1:
            parts.extend([placed(self.selected[self.chosen],T),placed(self.collisions[self.chosen],T)])
        if t>=3:parts.extend(placed(p,T) for p in self.families)
        main=scene.renderer.render(parts,camera,MAIN)
        key=0 if t<1 else 1 if t<3 else 2
        if key in self.right_cache:panel=self.right_cache[key]
        else:
            panel=Image.new('RGB',(700,900),'white')
            for i,(row,cam) in enumerate(zip(scene.rows,self.cameras)):
                state=row['fixture_transform'];values=scene.fixture_parts(state)
                values.extend([placed(C.V.piece(scene.data['body'],C.V.GRAY,.18),row['object_transform']),placed(self.patches[i],state)])
                if t>=1 and self.selected[i] is not None:values.extend([placed(self.selected[i],state),placed(self.collisions[i],state)])
                if t>=3:values.append(placed(self.families[i],state))
                panel.paste(scene.renderer.render(values,cam,SMALL),(7+(i%2)*346,55+(i//2)*450))
            self.right_cache[key]=panel
        return compose(main,panel)

def prepare(mode):
    scene=S.Scene();geometry=json.loads((TEMP/'geometry.json').read_text())
    assert geometry['support_geometry_sha256']==scene.support_hash
    model=WorkVideo(scene,geometry) if mode==4 else GroundVideos(scene,geometry,mode)
    directory=TEMP/f'execution{mode:02d}';sources=C.freeze(directory)
    own={}
    for path in [Path(__file__),Path(S.__file__),TEMP/'transparency.py',TEMP/'transparency.cpp']:shutil.copy2(path,directory/path.name);own[path.name]=C.digest(path)
    hashes=json.loads((TEMP/'original_videos.json').read_text())
    def verify():
        scene.verify()
        for path,h in sources.items():assert C.digest(ROOT/path)==C.digest(directory/'executed_sources'/path)==h
        for name,h in own.items():assert C.digest(TEMP/name)==C.digest(directory/name)==h
        assert {p.name:C.digest(p) for p in (PROOF/'videos').iterdir()}==hashes
        print(f'Video {mode:02d}: identical fixture, no demand dots, original sources unchanged.',flush=True)
    C.save(TEMP/f'metadata{mode:02d}.json',dict(support_geometry_sha256=scene.support_hash,
        duration_seconds=8,frames=192,demand_points_rendered=False,text=False,grid=False,
        mode=['','', 'original installed states and collision during empty-fixture reorientation',
              'coherent component lift and safe empty-fixture reorientation','true work rays into the same completed material'][mode]))
    return model.frame,verify

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',type=int,choices=[2,3,4]);parser.add_argument('--encode',action='store_true');args=parser.parse_args()
    frame,verify=prepare(args.mode)
    if not args.encode:
        for t,name in [(0,'start'),(2,'action'),(4,'middle'),(7.75,'end')]:frame(t).save(TEMP/f'{name}{args.mode:02d}.png')
        verify();return
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900',
        '-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-profile:v','main',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(TEMP/f'{args.mode:02d}.mp4')],stdin=subprocess.PIPE)
    started=time.monotonic()
    try:
        for i in range(FPS*SECONDS):
            encoder.stdin.write(frame(i/FPS).tobytes())
            if i%24==0:print(json.dumps(dict(video=args.mode,frame=i,total=192,seconds=round(time.monotonic()-started,1))),flush=True)
        encoder.stdin.close();assert encoder.wait()==0
    except BaseException:encoder.kill();encoder.wait();raise
    verify()
if __name__=='__main__':main()
