from pathlib import Path
import json,sys
import numpy as np
import trimesh
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

ROOT=Path('/home/yli581/Desktop/cadGrasp')
PROOF=ROOT/'slides/Co-optimize/seperate_base_support_proof'
TEMP=Path(__file__).resolve().parent
sys.path.insert(0,str(PROOF))
import diagnose as D
import base_geometry as G
C=D.C

def smooth(x):
    x=float(np.clip(x,0,1));return x*x*(3-2*x)

def rod(a,b,radius=.003):
    direction=b-a;length=np.linalg.norm(direction)
    T=trimesh.geometry.align_vectors([0,0,1],direction/length);T[:3,3]=(a+b)/2
    return trimesh.creation.cylinder(radius=radius,height=length,sections=20,transform=T)

def offset(xy,width):
    edge=np.roll(xy,-1,axis=0)-xy
    normal=np.c_[edge[:,1],-edge[:,0]]/np.linalg.norm(edge,axis=1)[:,None]
    outer=np.array([np.linalg.solve([normal[i-1],normal[i]],
        [normal[i-1]@xy[i]+width,normal[i]@xy[i]+width]) for i in range(len(xy))])
    return outer,normal

class Model:
    def __init__(self):
        self.inputs=C.Inputs('B',D.BASE/'illegal/pose11+19/data/selection.json');self.inputs.base=D.BASE
        self.data=self.inputs.load(self.inputs.cases[0])
        self.rows=self.data['rows'];self.old=self.data['fixture']
        self.start=next(r for r in self.rows if r['pose']=='pose_19')
        self.end=next(r for r in self.rows if r['pose']=='pose_11')
        self.T0=self.start['fixture_transform'];self.T1=self.end['fixture_transform']
        self.inverse=np.linalg.inv(self.T0)
        certificate_path=PROOF/'certificates/illegal/pose11+19/report.json'
        self.inputs.protected[C.relative(certificate_path)]=C.digest(certificate_path)
        saved=json.loads(certificate_path.read_text())['ground_certificates'][0]
        self.a=np.array(saved['inequality_a_xy']);self.b=saved['inequality_offset_m']
        transform=self.T1@self.inverse
        np.testing.assert_allclose(transform[2,:2],self.a,atol=1e-12)
        assert abs(transform[2,3]-self.b)<1e-12
        bound=min(0.,np.min(self.start['object_ground']@self.a+self.b))
        values=self.start['hull'][:,:2]@self.a+self.b
        assert np.all(values<bound-1e-9),values
        assert abs(bound-saved['attainable_lower_bound_m'])<1e-12
        assert abs(bound-values.min()-saved['separation_gap_m'])<1e-12
        self.xy=self.start['hull'][:,:2]
        _,normals=offset(self.xy,.004)
        # An actual opening through one straight edge; all original corners stay.
        lengths=np.linalg.norm(np.roll(self.xy,-1,axis=0)-self.xy,axis=1)
        self.gap_edge=int(np.argmax(lengths));self.gap_length=min(.018,lengths[self.gap_edge]*.6)
        self.full_min=self.frame_solid(.004)
        self.min_base_mesh=G.unpack(self.full_min)
        world=C.F.transform_points(self.min_base_mesh.vertices,self.T0)
        floor_points=world[np.abs(world[:,2])<1e-9,:2]
        hull=ConvexHull(floor_points)
        assert np.max(self.xy@hull.equations[:,:2].T+hull.equations[:,2])<1e-10
        assert world[:,2].min()>-1e-10
        components=self.old.split(only_watertight=False)
        links=[]
        if len(components)>1:
            # Retain every original part; join them with the same blue short rods.
            accumulated=components[0]
            for component in components[1:]:
                close,distance,_=trimesh.proximity.closest_point_naive(accumulated,component.vertices)
                i=int(np.argmin(distance));a=close[i];b=component.vertices[i]
                direction=(b-a)/np.linalg.norm(b-a)
                links.append(rod(a-direction*.002,b+direction*.002))
                accumulated=trimesh.util.concatenate([accumulated,component])
        # Attach the open frame without closing its opening.
        candidates=self.min_base_mesh.triangles_center
        close,distance,_=trimesh.proximity.closest_point_naive(self.old,candidates)
        i=int(np.argmin(distance));a=close[i];b=candidates[i];direction=(b-a)/np.linalg.norm(b-a)
        links.append(rod(a-direction*.002,b+direction*.002))
        self.core=G.combine([G.solid(self.old),*[G.solid(r) for r in links]])
        self.core_mesh=G.unpack(self.core)
        complete=self.core+self.full_min
        assert len(complete.decompose())==1
        self.complete=G.unpack(complete);assert self.complete.is_watertight
        self.maximum=self.expanded_solid(.029)
        assert G.volume(self.full_min-self.maximum)<1e-14
        for padding in [0.,.008,.019,.029]:
            candidate=self.expanded_solid(padding)
            below=G.transform(candidate,self.T1).trim_by_plane((0.,0.,-1.),0.)
            assert G.volume(below)>1e-9
            print('Open base outward expansion mm',padding*1000,'below ground cm3',G.volume(below)*1e6,flush=True)
        self.record=dict(case='illegal/pose11+19',start_pose='pose_19',end_pose='pose_11',
            opening_width_mm=self.gap_length*1000,rim_width_mm=4,max_outward_expansion_mm=29,
            all_original_material_retained=True,connected_complete_fixture=True,
            demand_hull_vertices=len(self.xy),all_demand_vertices_outside_legal_halfplane=True,
            ground_inequality_a_xy=self.a.tolist(),ground_inequality_offset_m=self.b,
            maximum_legal_system_contact_height_value_m=float(bound),
            worst_demand_gap_m=float(bound-values.min()),
            best_demand_gap_m=float(bound-values.max()),
            proof_independent_of_base_size_topology_and_closure=True,
            object_ground_contacts_included=True,
            scope='unchanged saved relative placements and final installed poses',
            original_step4_force_results_preserved=True)
        C.save(TEMP/'geometry_report.json',self.record)

    def frame_solid(self,width,padding=0.):
        inner=self.xy if padding==0 else offset(self.xy,padding)[0]
        outer,normals=offset(self.xy,width+padding)
        band=Polygon(outer).difference(Polygon(inner))
        i=self.gap_edge;j=(i+1)%len(self.xy);middle=(inner[i]+inner[j])/2
        tangent=self.xy[j]-self.xy[i];tangent/=np.linalg.norm(tangent);normal=normals[i]
        h=self.gap_length/2
        cut=Polygon([middle-tangent*h-normal*.005,middle+tangent*h-normal*.005,
                     middle+tangent*h+normal*(width+.015),middle-tangent*h+normal*(width+.015)])
        band=band.difference(cut)
        assert band.geom_type=='Polygon' and len(band.interiors)==0
        section=G.md.CrossSection([np.array(band.exterior.coords)[:-1]*1000],G.md.FillRule.NonZero)
        value=section.extrude(5.)
        return G.transform(value,self.inverse)

    def expanded_solid(self,padding):
        if padding<1e-10:return self.full_min
        outer_ring=self.frame_solid(.004,padding=padding)
        first=(self.xy+offset(self.xy,.004)[0])/2
        last=(offset(self.xy,padding)[0]+offset(self.xy,padding+.004)[0])/2
        links=[]
        for i in [0,len(self.xy)//3,2*len(self.xy)//3]:
            a,b=C.F.transform_points(np.array([[*first[i],.0025],[*last[i],.0025]]),self.inverse)
            direction=(b-a)/np.linalg.norm(b-a)
            link=G.solid(rod(a-direction*.001,b+direction*.001,radius=.0025))
            assert G.volume(link^self.full_min)>1e-14 and G.volume(link^outer_ring)>1e-14
            links.append(link)
        result=G.combine([self.full_min,outer_ring,*links])
        assert len(result.decompose())==1
        return result

    def verify(self):self.inputs.verify()

if __name__=='__main__':
    m=Model();m.verify();print(json.dumps(m.record,indent=2))
