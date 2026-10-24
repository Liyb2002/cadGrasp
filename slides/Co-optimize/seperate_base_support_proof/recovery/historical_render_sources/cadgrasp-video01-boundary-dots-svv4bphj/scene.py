"""The exact same grown fixture is the input of all four presentation videos."""
from pathlib import Path
import json,sys
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
from transparency import Renderer
C=D.C
FPS,SECONDS,MAIN,SMALL=24,8,900,340
WIDTH,HEIGHT,RADIUS=.004,.005,.003
GROW_START=.75
NAMES=['01_hulls_and_support.mp4','02_ground_collision.mp4','03_lift_and_layout.mp4','04_work_cones.mp4']

def smooth(x):
    x=float(np.clip(x,0,1));return x*x*(3-2*x)

def basis(angle):
    angle+=np.deg2rad(75.)
    c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
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

def rod(start,end,radius=RADIUS):
    v=end-start;length=float(np.linalg.norm(v))
    T=trimesh.geometry.align_vectors([0,0,1],v/length);T[:3,3]=(start+end)/2
    return trimesh.creation.cylinder(radius=radius,height=length,sections=20,transform=T)

def placed(p,T):return C.V.CPU.placed(p,rotation=T[:3,:3],point=T[:3,3])

def compose(main,panel):
    image=Image.new('RGB',(1600,900),'white');image.paste(main,(0,0));image.paste(panel,(900,0))
    draw=ImageDraw.Draw(image);draw.line((900,0,900,899),fill='#eef0f2',width=1);draw.line((901,450,1599,450),fill='#eef0f2',width=1)
    return image

class Ring:
    def __init__(self,row,support):
        self.inverse=np.linalg.inv(row['fixture_transform']);xy=row['hull'][:,:2]
        edge=np.roll(xy,-1,axis=0)-xy
        out=np.c_[edge[:,1],-edge[:,0]]/np.linalg.norm(edge,axis=1)[:,None]
        outer=np.array([np.linalg.solve(np.array([out[i-1],out[i]]),np.array([out[i-1]@xy[i],out[i]@xy[i]])+WIDTH) for i in range(len(xy))])
        inside,outside=Polygon(xy),Polygon(outer);assert outside.covers(inside)
        band=outside.difference(inside);assert len(band.interiors)==1 and band.intersection(inside).area<1e-16
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
        center=(self.inner+self.outer)/2;self.cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.roll(center,-1,axis=0)-center,axis=1))]
        self.prefix=[G.md.Manifold()]
        for i in range(len(self.inner)):
            j=(i+1)%len(self.inner)
            self.prefix.append(self.prefix[-1]+prism(np.array([self.inner[i],self.inner[j],self.outer[j],self.outer[i]])))
        full=self.prefix[-1];assert np.isclose(G.volume(full),band.area*HEIGHT,rtol=1e-8)
        assert G.volume(full^prism(xy))<1e-14
        self.solid=G.transform(full,self.inverse);self.mesh=C.posed(G.unpack(full),self.inverse);self.link=rod(self.start,self.end)
        assert G.volume(G.solid(self.link)^self.solid)>1e-14

    def mesh_at(self,amount):
        if amount>=1:return self.mesh
        distance=amount*self.cumulative[-1]
        i=min(int(np.searchsorted(self.cumulative,distance,side='right')-1),len(self.inner)-1)
        f=(distance-self.cumulative[i])/(self.cumulative[i+1]-self.cumulative[i]);j=(i+1)%len(self.inner)
        value=self.prefix[i]
        if f>1e-8:value=value+prism(np.array([self.inner[i],(1-f)*self.inner[i]+f*self.inner[j],(1-f)*self.outer[i]+f*self.outer[j],self.outer[i]]))
        return C.posed(G.unpack(value),self.inverse)

class Scene:
    def __init__(self):
        self.inp=C.Inputs('B',D.BASE/'pose2+3+4+7/data/selection.json');self.inp.base=D.BASE
        self.data=self.inp.load(self.inp.cases[0]);self.rows=self.data['rows'];self.old=self.data['fixture']
        assert [r['pose'] for r in self.rows]==['pose_2','pose_3','pose_4','pose_7']
        self.rings=[Ring(row,self.old) for row in self.rows]
        old=G.solid(self.old)
        for ring in self.rings:assert G.volume(G.solid(ring.link)^old)>1e-14
        value=G.combine([old,*[r.solid for r in self.rings],*[G.solid(r.link) for r in self.rings]])
        assert len(value.decompose())==1
        candidate=G.unpack(value);assert candidate.is_watertight and len(candidate.split(only_watertight=False))==1
        saved=TEMP/'support01.npz'
        if saved.exists():
            with np.load(saved,allow_pickle=False) as arrays:
                self.final=trimesh.Trimesh(arrays['vertices'].copy(),arrays['faces'].copy(),process=False)
            assert np.isclose(self.final.volume,candidate.volume,rtol=1e-12)
        else:
            self.final=candidate
            np.savez(saved,vertices=self.final.vertices,faces=self.final.faces)
        self.solid=G.solid(self.final);self.support_hash=C.digest(saved)
        self.ref=self.rows[0]['fixture_transform'];self.hulls=[];self.dots=[];self.boundary=[];self.platforms=[]
        for row in self.rows:
            inverse=np.linalg.inv(row['fixture_transform']);hull=C.F.transform_points(row['hull'],inverse)
            cloud=C.F.transform_points(row['cloud'],inverse);assert len(cloud)==32768
            assert np.max(np.abs(C.F.transform_points(hull,row['fixture_transform'])-row['hull']))<1e-12
            ids=np.linspace(0,len(cloud)-1,256,dtype=int);self.dots.append(np.unique(np.vstack([cloud[ids],hull]),axis=0))
            # Display-only markers on exact straight hull edges, every <= 2 mm.
            edge_markers=[]
            for a,b in zip(hull,np.roll(hull,-1,axis=0)):
                count=max(1,int(np.ceil(np.linalg.norm(b-a)/.002)))
                edge_markers.extend(a+(b-a)*f for f in np.arange(count)/count)
            boundary=np.asarray(edge_markers);assert np.ptp(boundary@row['fixture_transform'][2,:3])<1e-12
            self.boundary.append(boundary);self.hulls.append(hull)
            mesh=trimesh.Trimesh(hull,np.array([[0,i,i+1] for i in range(1,len(hull)-1)]),process=False)
            platform=C.V.piece(mesh,row['color'],.06);platform['unlit']=True;self.platforms.append(platform)
        self.full_pieces=[C.V.piece(self.old,C.V.BLUE,.78)]
        for ring in self.rings:self.full_pieces.extend([C.V.piece(ring.link,C.V.BLUE,.82),C.V.piece(ring.mesh,C.V.BLUE,.82)])
        self.body_world=[C.posed(self.data['body'],row['object_transform']) for row in self.rows]
        points=C.F.transform_points(np.vstack([self.old.vertices,*self.hulls]),self.ref)
        self.focus=(points.min(0)+points.max(0))/2
        all_points=C.F.transform_points(np.vstack([self.final.vertices,*self.hulls]),self.ref)
        self.span=2*max(float(np.abs((all_points-self.focus)@basis(a)[:2].T).max()) for a in np.linspace(0,2*np.pi,361))*1.12
        cameras=[]
        for row,body in zip(self.rows,self.body_world):
            cameras.append(C.V.camera_for(np.vstack([C.F.transform_points(np.vstack([self.final.vertices,*self.hulls]),row['fixture_transform']),body.vertices])))
        right_span=max(cam[2] for cam in cameras);self.cameras=[(f,b,right_span) for f,b,_ in cameras]
        self.renderer=Renderer()

    def fixture_parts(self,T):return [placed(p,T) for p in self.full_pieces]

    def verify(self):
        self.inp.verify();assert C.digest(TEMP/'support01.npz')==self.support_hash

