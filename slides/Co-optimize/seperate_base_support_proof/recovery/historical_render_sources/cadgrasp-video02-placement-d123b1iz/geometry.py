from pathlib import Path
import sys,json
import numpy as np
import trimesh
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
ROOT=Path('/home/yli581/Desktop/cadGrasp')
PROOF=ROOT/'slides/Co-optimize/seperate_base_support_proof'
TEMP=Path(__file__).resolve().parent
sys.path.insert(0,str(PROOF))
from saved_inputs import C,D,G

def smooth(x):
    x=float(np.clip(x,0,1));return x*x*(3-2*x)

def rod(a,b,radius=.003):
    d=b-a;length=np.linalg.norm(d);T=trimesh.geometry.align_vectors([0,0,1],d/length)
    T[:3,3]=(a+b)/2
    return trimesh.creation.cylinder(radius=radius,height=length,sections=20,transform=T)

def offset(xy,width):
    edge=np.roll(xy,-1,axis=0)-xy
    normals=np.c_[edge[:,1],-edge[:,0]]/np.linalg.norm(edge,axis=1)[:,None]
    outer=np.array([np.linalg.solve([normals[i-1],normals[i]],
        [normals[i-1]@xy[i]+width,normals[i]@xy[i]+width]) for i in range(len(xy))])
    return outer,normals

class Model:
    def __init__(self):
        self.inputs=C.Inputs('B',D.BASE/'illegal/pose11+19/data/selection.json');self.inputs.base=D.BASE
        self.data=self.inputs.load(self.inputs.cases[0]);self.rows=self.data['rows'];self.old=self.data['fixture']
        self.start=next(r for r in self.rows if r['pose']=='pose_19');self.end=next(r for r in self.rows if r['pose']=='pose_11')
        self.T0=self.start['fixture_transform'];self.T1=self.end['fixture_transform'];self.inverse=np.linalg.inv(self.T0)
        path=PROOF/'certificates/illegal/pose11+19/report.json'
        self.inputs.protected[C.relative(path)]=C.digest(path)
        certificate=json.loads(path.read_text())['ground_certificates'][0]
        transform=self.T1@self.inverse
        np.testing.assert_allclose(transform[2,:2],certificate['inequality_a_xy'],atol=1e-12)
        assert abs(transform[2,3]-certificate['inequality_offset_m'])<1e-12
        values=self.start['hull'][:,:2]@transform[2,:2]+transform[2,3]
        bound=min(0.,np.min(self.start['object_ground']@transform[2,:2]+transform[2,3]))
        assert np.all(values<bound-1e-9)
        self.xy=self.start['hull'][:,:2]
        lengths=np.linalg.norm(np.roll(self.xy,-1,axis=0)-self.xy,axis=1)
        self.gap_edge=int(np.argmax(lengths));self.gap_length=min(.018,lengths[self.gap_edge]*.6)
        self.full_min=self.frame_solid(.004);base_mesh=G.unpack(self.full_min)
        floor=C.F.transform_points(base_mesh.vertices,self.T0)
        contacts=floor[np.abs(floor[:,2])<1e-9,:2];hull=ConvexHull(contacts)
        assert np.max(self.xy@hull.equations[:,:2].T+hull.equations[:,2])<1e-10
        components=self.old.split(only_watertight=False);links=[]
        accumulated=components[0]
        for component in components[1:]:
            close,distance,_=trimesh.proximity.closest_point_naive(accumulated,component.vertices)
            i=int(np.argmin(distance));a=close[i];b=component.vertices[i];d=(b-a)/np.linalg.norm(b-a)
            links.append(rod(a-d*.002,b+d*.002));accumulated=trimesh.util.concatenate([accumulated,component])
        candidates=base_mesh.triangles_center
        close,distance,_=trimesh.proximity.closest_point_naive(self.old,candidates)
        i=int(np.argmin(distance));a=close[i];b=candidates[i];d=(b-a)/np.linalg.norm(b-a)
        links.append(rod(a-d*.002,b+d*.002))
        self.core=G.combine([G.solid(self.old),*[G.solid(r) for r in links]])
        self.core_mesh=G.unpack(self.core)
        assert len((self.core+self.full_min).decompose())==1
        self.maximum=self.expanded_solid(.029)
        self.maximum_mesh=G.unpack(self.core+self.maximum)
        # Reproduce the previous video's exact construction, without altering it.
        for padding,expected in [(0.,7.010550933361167),(.029,15.902352339994682)]:
            below=G.transform(self.expanded_solid(padding),self.T1).trim_by_plane((0.,0.,-1.),0.)
            assert abs(G.volume(below)*1e6-expected)<1e-9
        self.inputs.verify()

    def frame_solid(self,width,padding=0.):
        inner=self.xy if padding==0 else offset(self.xy,padding)[0]
        outer,normals=offset(self.xy,width+padding)
        band=Polygon(outer).difference(Polygon(inner))
        i=self.gap_edge;j=(i+1)%len(self.xy);middle=(inner[i]+inner[j])/2
        tangent=self.xy[j]-self.xy[i];tangent/=np.linalg.norm(tangent);normal=normals[i];h=self.gap_length/2
        cut=Polygon([middle-tangent*h-normal*.005,middle+tangent*h-normal*.005,
                     middle+tangent*h+normal*(width+.015),middle-tangent*h+normal*(width+.015)])
        band=band.difference(cut)
        assert band.geom_type=='Polygon' and not band.interiors
        value=G.md.CrossSection([np.array(band.exterior.coords)[:-1]*1000],G.md.FillRule.NonZero).extrude(5.)
        return G.transform(value,self.inverse)

    def expanded_solid(self,padding):
        if padding<1e-10:return self.full_min
        outer_ring=self.frame_solid(.004,padding=padding)
        first=(self.xy+offset(self.xy,.004)[0])/2
        last=(offset(self.xy,padding)[0]+offset(self.xy,padding+.004)[0])/2
        links=[]
        for i in [0,len(self.xy)//3,2*len(self.xy)//3]:
            a,b=C.F.transform_points(np.array([[*first[i],.0025],[*last[i],.0025]]),self.inverse);d=(b-a)/np.linalg.norm(b-a)
            link=G.solid(rod(a-d*.001,b+d*.001,radius=.0025))
            assert G.volume(link^self.full_min)>1e-14 and G.volume(link^outer_ring)>1e-14
            links.append(link)
        result=G.combine([self.full_min,outer_ring,*links]);assert len(result.decompose())==1
        return result
