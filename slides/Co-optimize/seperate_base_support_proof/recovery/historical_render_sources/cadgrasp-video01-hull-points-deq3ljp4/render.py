"""See-through demand platforms with real demand dots; blue hollow rim growth."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys,time
ROOT=Path('/home/yli581/Desktop/cadGrasp')
PROOF=ROOT/'slides/Co-optimize/seperate_base_support_proof'
TEMP=Path(__file__).resolve().parent
sys.path.insert(0,str(PROOF))
import diagnose as D
import base_geometry as G
import numpy as np
import trimesh
from shapely.geometry import Polygon
from PIL import Image,ImageDraw
C=D.C
FPS,SECONDS,MAIN,SMALL=24,8,900,340
WIDTH,HEIGHT,RADIUS=.004,.005,.003
GROW_START=.75
PLATFORM_ALPHA=.06
DEST=PROOF/'videos/01_hulls_and_support.mp4'
EXPECTED={'01_hulls_and_support.mp4','02_ground_collision.mp4','03_lift_and_layout.mp4','04_work_cones.mp4'}

def smooth(x):
    x=float(np.clip(x,0,1));return x*x*(3-2*x)

def basis(angle):
    angle+=np.deg2rad(75.)
    c,s=np.cos(angle),np.sin(angle)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    view=R@(np.array([1.,-1.,1.])/np.sqrt(3));right=R@(np.array([1.,1.,0.])/np.sqrt(2))
    return np.array([right,np.cross(view,right),view])

def prism(xy):
    n=len(xy);v=np.c_[xy,np.zeros(n)]
    faces=[[0,i+1,i] for i in range(1,n-1)]+[[n,n+i,n+i+1] for i in range(1,n-1)]
    for i in range(n):
        j=(i+1)%n;faces.extend([[i,j,n+j],[i,n+j,n+i]])
    mesh=trimesh.Trimesh(np.vstack([v,v+[0,0,HEIGHT]]),np.asarray(faces),process=False)
    if mesh.volume<0:mesh.invert()
    return G.solid(mesh)

def rod(start,end):
    vector=end-start;length=float(np.linalg.norm(vector))
    transform=trimesh.geometry.align_vectors([0,0,1],vector/length);transform[:3,3]=(start+end)/2
    return trimesh.creation.cylinder(radius=RADIUS,height=length,sections=20,transform=transform)

def placed(p,T):return C.V.CPU.placed(p,rotation=T[:3,:3],point=T[:3,3])

class Ring:
    def __init__(self,row,support):
        self.inverse=np.linalg.inv(row['fixture_transform']);xy=row['hull'][:,:2]
        edges=np.roll(xy,-1,axis=0)-xy
        out=np.c_[edges[:,1],-edges[:,0]]/np.linalg.norm(edges,axis=1)[:,None]
        outer=np.array([np.linalg.solve(np.array([out[i-1],out[i]]),
                         np.array([out[i-1]@xy[i],out[i]@xy[i]])+WIDTH) for i in range(len(xy))])
        inside,outside=Polygon(xy),Polygon(outer)
        assert outside.covers(inside)
        band=outside.difference(inside)
        assert len(band.interiors)==1 and band.intersection(inside).area<1e-16
        middle=(xy+outer)/2;candidates=[];indices=[]
        for i in range(len(xy)):
            for f in (0.,.25,.5,.75):
                candidates.append([*(middle[i]*(1-f)+middle[(i+1)%len(xy)]*f),HEIGHT/2]);indices.append((i,f))
        common=C.F.transform_points(np.asarray(candidates),self.inverse)
        roots,dist,_=trimesh.proximity.closest_point_naive(support,common)
        k=int(np.argmin(dist));i,f=indices[k];self.end=common[k]
        self.start=roots[k]-(self.end-roots[k])/dist[k]*.002
        if f==0:self.inner=np.roll(xy,-i,axis=0);self.outer=np.roll(outer,-i,axis=0)
        else:
            order=[(i+1+j)%len(xy) for j in range(len(xy))]
            self.inner=np.vstack([xy[i]*(1-f)+xy[(i+1)%len(xy)]*f,xy[order]])
            self.outer=np.vstack([outer[i]*(1-f)+outer[(i+1)%len(xy)]*f,outer[order]])
        center=(self.inner+self.outer)/2
        self.cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.roll(center,-1,axis=0)-center,axis=1))]
        self.prefix=[G.md.Manifold()]
        for i in range(len(self.inner)):
            j=(i+1)%len(self.inner)
            self.prefix.append(self.prefix[-1]+prism(np.array([self.inner[i],self.inner[j],self.outer[j],self.outer[i]])))
        full=self.prefix[-1]
        assert np.isclose(G.volume(full),band.area*HEIGHT,rtol=1e-8)
        assert G.volume(full^prism(xy))<1e-14
        self.solid=G.transform(full,self.inverse);self.mesh=C.posed(G.unpack(full),self.inverse)
        self.link=rod(self.start,self.end)
        assert G.volume(G.solid(self.link)^self.solid)>1e-14

    def mesh_at(self,amount):
        if amount>=1:return self.mesh
        distance=amount*self.cumulative[-1]
        i=min(int(np.searchsorted(self.cumulative,distance,side='right')-1),len(self.inner)-1)
        f=(distance-self.cumulative[i])/(self.cumulative[i+1]-self.cumulative[i]);j=(i+1)%len(self.inner)
        value=self.prefix[i]
        if f>1e-8:
            value=value+prism(np.array([self.inner[i],(1-f)*self.inner[i]+f*self.inner[j],
                                      (1-f)*self.outer[i]+f*self.outer[j],self.outer[i]]))
        return C.posed(G.unpack(value),self.inverse)

def prepare():
    inp=C.Inputs('B',D.BASE/'pose2+3+4+7/data/selection.json');inp.base=D.BASE
    data=inp.load(inp.cases[0]);rows=data['rows'];support=data['fixture']
    assert [r['pose'] for r in rows]==['pose_2','pose_3','pose_4','pose_7']
    assert {p.name for p in DEST.parent.iterdir()}==EXPECTED
    hashes={p.name:C.digest(p) for p in DEST.parent.iterdir()}
    sources=C.freeze(TEMP/'execution');shutil.copy2(__file__,TEMP/'execution/render.py');own_hash=C.digest(__file__)
    old=G.solid(support);rings=[Ring(row,support) for row in rows]
    for ring in rings:assert G.volume(G.solid(ring.link)^old)>1e-14
    final=G.unpack(G.combine([old,*[r.solid for r in rings],*[G.solid(r.link) for r in rings]]))
    assert final.is_watertight and len(final.split(only_watertight=False))==1
    ref=rows[0]['fixture_transform'];hulls=[];clouds=[];dots=[];platforms=[]
    for row in rows:
        inverse=np.linalg.inv(row['fixture_transform'])
        hull=C.F.transform_points(row['hull'],inverse);cloud=C.F.transform_points(row['cloud'],inverse)
        assert len(cloud)==32768 and np.max(np.abs(C.F.transform_points(hull,row['fixture_transform'])-row['hull']))<1e-12
        # Fixed representatives of the existing samples, plus exact hull vertices; no new loads.
        ids=np.linspace(0,len(cloud)-1,256,dtype=int)
        points=np.unique(np.vstack([cloud[ids],hull]),axis=0)
        hulls.append(hull);clouds.append(cloud);dots.append(points)
        mesh=trimesh.Trimesh(hull,np.array([[0,i,i+1] for i in range(1,len(hull)-1)]),process=False)
        platform=C.V.piece(mesh,row['color'],PLATFORM_ALPHA);platform['unlit']=True;platforms.append(platform)
    points=C.F.transform_points(np.vstack([support.vertices,*hulls]),ref);focus=(points.min(0)+points.max(0))/2
    all_points=C.F.transform_points(np.vstack([final.vertices,*hulls]),ref)
    span=2*max(float(np.abs((all_points-focus)@basis(a)[:2].T).max()) for a in np.linspace(0,2*np.pi,361))*1.12
    bodies=[];cameras=[]
    for row in rows:
        body=C.posed(data['body'],row['object_transform']);bodies.append(C.V.CPU.placed(C.V.piece(body,C.V.GRAY,.18)))
        cameras.append(C.V.camera_for(np.vstack([C.F.transform_points(np.vstack([final.vertices,*hulls]),row['fixture_transform']),body.vertices])))
    right_span=max(cam[2] for cam in cameras);cameras=[(f,b,right_span) for f,b,_ in cameras]
    renderer=C.V.Renderer();left_support=C.V.piece(support,C.V.BLUE,.78);right_support=C.V.piece(support,C.V.BLUE,.82)
    final_right=None

    def frame(t):
        nonlocal final_right
        camera=(focus,basis(2*np.pi*min(t/7.75,1)),span)
        link_amount=smooth((t-GROW_START)/1.0);ring_amount=smooth((t-GROW_START-1.0)/4.0)
        material=[]
        for ring in rings:
            if link_amount>1e-5:material.append(C.V.piece(rod(ring.start,ring.start+(ring.end-ring.start)*link_amount),C.V.BLUE,.82))
            if ring_amount>1e-5:material.append(C.V.piece(ring.mesh_at(ring_amount),C.V.BLUE,.82))
        def parts(T,camera,size,support_piece):
            result=[placed(support_piece,T),*[placed(p,T) for p in material]]
            for hull,points,platform,row in zip(hulls,dots,platforms,rows):
                if t<GROW_START:
                    result.append(placed(platform,T))
                    border=C.H.outline_piece(C.F.transform_points(hull,T),row['color'],camera,size,half_width_pixels=.8)
                    border['colors']=np.broadcast_to(C.V.CPU.rgb(row['color']),border['triangles'].shape)
                    border['opacity']=.28;result.append(C.V.CPU.placed(border))
                scatter=C.V.points_piece(C.F.transform_points(points,T),row['color'],camera,size,
                                         half_size_pixels=1.05 if size==MAIN else .65)
                scatter['opacity']=.8;result.append(C.V.CPU.placed(scatter))
            return result
        main=renderer.render(parts(ref,camera,MAIN,left_support),camera,MAIN)
        if t>=5.75 and final_right is not None:panel=final_right
        else:
            panel=Image.new('RGB',(700,900),'white')
            for i,(row,body,cam) in enumerate(zip(rows,bodies,cameras)):
                pieces=parts(row['fixture_transform'],cam,SMALL,right_support);pieces.append(body)
                panel.paste(renderer.render(pieces,cam,SMALL),(7+(i%2)*346,55+(i//2)*450))
            if t>=5.75:final_right=panel
        image=Image.new('RGB',(1600,900),'white');image.paste(main,(0,0));image.paste(panel,(900,0))
        draw=ImageDraw.Draw(image);draw.line((900,0,900,899),fill='#eef0f2',width=1);draw.line((901,450,1599,450),fill='#eef0f2',width=1)
        return image
    def verify():
        inp.verify()
        for path,h in sources.items():assert C.digest(ROOT/path)==C.digest(TEMP/'execution/executed_sources'/path)==h
        assert C.digest(__file__)==C.digest(TEMP/'execution/render.py')==own_hash
        assert {p.name:C.digest(p) for p in DEST.parent.iterdir()}==hashes
        print('Saved input, source and original video hashes match.',flush=True)
    C.save(TEMP/'original_videos.json',hashes)
    C.save(TEMP/'metadata.json',dict(duration_seconds=8,frames=192,platform_opacity=PLATFORM_ALPHA,
        platform_and_border_removed_at_seconds=GROW_START,growth_starts_at_seconds=GROW_START,
        blue_support_opacity_unchanged=True,displayed_original_points_per_pose=[len(p) for p in dots],
        points_preserved_after_platform_removal=True,rings_hollow=True,components=1,text=False,grid=False))
    print(json.dumps(dict(platform_opacity=PLATFORM_ALPHA,platform_removed_at=GROW_START,
                         displayed_points=[len(p) for p in dots],blue_support_unchanged=True)),flush=True)
    return frame,verify

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--encode',action='store_true');args=parser.parse_args()
    frame,verify=prepare()
    if not args.encode:
        for t,name in [(0,'start'),(.75,'platform_removed'),(3,'growing'),(7.75,'finished')]:frame(t).save(TEMP/(name+'.png'))
        verify();return
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1600x900','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18',
        '-profile:v','main','-pix_fmt','yuv420p','-movflags','+faststart',str(TEMP/'video.mp4')],stdin=subprocess.PIPE)
    start=time.monotonic()
    try:
        for i in range(FPS*SECONDS):
            encoder.stdin.write(frame(i/FPS).tobytes())
            if i%24==0:print(json.dumps(dict(frame=i,total=192,elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
        encoder.stdin.close();assert encoder.wait()==0
    except BaseException:encoder.kill();encoder.wait();raise
    verify();print('Encoding complete.',flush=True)

if __name__=='__main__':main()
