"""Lateral connection width, independent of the original contact wall thickness.

Feet are checked in their construction plane. Round/box links are checked in
two longitudinal planes. References identify construction features, not stock
that must be retained: deleting an entire feature is allowed. Original contact
material supplies attachment context without an 8 mm wall-thickness gate.
"""
import json
from pathlib import Path
import numpy as np
import manifold3d as md
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely import make_valid
from wrap_dimensions import MINIMUM_WIDTH_M,PREVIOUS_FOOT_WIDTH_M

WIDTH_TOLERANCE_M = 1e-6
AREA_TOLERANCE_M2 = 1e-14


def polygons(value):
    if value.is_empty:return []
    if value.geom_type=='Polygon':return [value] if value.area>AREA_TOLERANCE_M2 else []
    if not hasattr(value,'geoms'):return []
    return [p for child in value.geoms for p in polygons(child)]


def projection(value):
    """Signed-solid orthogonal projection, including holes, in metres."""
    return section_geometry(md.CrossSection(value.project().to_polygons(),md.FillRule.Positive))


def section_geometry(section):
    """Convert a real cross-section, preserving nested solid islands."""
    from deterministic_base import SCALE
    loops=section.to_polygons()
    positive=[];negative=[]
    for contour in loops:
        p=np.asarray(contour)*SCALE
        if len(p)<3:continue
        signed=np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
        (positive if signed>0 else negative).append(make_valid(Polygon(p)))
    # Assign each hole to its immediate outer boundary. Subtracting all holes
    # after union would erase real positive islands inside a projected ring.
    shells=[p for v in positive for p in polygons(v)]
    holes=[[] for _ in shells]
    for v in negative:
        for hole in polygons(v):
            point=hole.representative_point()
            owners=[i for i,p in enumerate(shells) if p.area>=hole.area-AREA_TOLERANCE_M2 and p.covers(point)]
            if owners:holes[min(owners,key=lambda i:shells[i].area)].append(hole)
    return unary_union([p.difference(unary_union(holes[i])) for i,p in enumerate(shells)])


def footprint_solid(footprint, low, high):
    from deterministic_base import SCALE
    contours=[]
    for p in polygons(footprint):
        p=orient(p,sign=1.)
        contours.append(np.asarray(p.exterior.coords)[:-1]/SCALE)
        contours.extend(np.asarray(r.coords)[:-1]/SCALE for r in p.interiors)
    if not contours:return md.Manifold()
    return md.CrossSection(contours,md.FillRule.NonZero).extrude((high-low)/SCALE).translate((0,0,low/SCALE))


def link_frames(a,b,vertices=None):
    a=np.asarray(a);b=np.asarray(b);axis=b-a
    if np.linalg.norm(axis)<1e-10:return []
    axis=axis/np.linalg.norm(axis)
    helper=np.eye(3)[int(np.argmin(abs(axis)))]
    side=np.cross(axis,helper);side/=np.linalg.norm(side)
    if vertices is not None and len(vertices)>2:
        cloud=np.asarray(vertices)-a
        transverse=cloud-np.outer(cloud@axis,axis)
        _,vectors=np.linalg.eigh(transverse.T@transverse)
        side=vectors[:,-1];side-=axis*np.dot(side,axis);side/=np.linalg.norm(side)
    normal=np.cross(axis,side)
    frames=[]
    for y,z in [(side,normal),(normal,-side)]:
        T=np.eye(4);T[:3,:3]=np.array([axis,y,z]);T[:3,3]=-T[:3,:3]@a
        frames.append(T)
    return frames


def foot_tracks(row):
    # Measure each straight member over the original hull edge, avoiding the
    # pointed miter extension at a corner. A full 8 mm strip must pass.
    p=np.asarray(orient(Polygon(row['hull'][:,:2]),sign=1.).exterior.coords);tracks=[]
    for a,b in zip(p[:-1],p[1:]):
        length=float(np.linalg.norm(b-a))
        if length<=WIDTH_TOLERANCE_M:continue
        tangent=(b-a)/length;normal=np.array([-tangent[1],tangent[0]])
        a=a-normal*(PREVIOUS_FOOT_WIDTH_M-MINIMUM_WIDTH_M/2)
        Q=np.eye(4);Q[0,:2]=tangent;Q[1,:2]=normal
        Q[0,3]=-float(a@tangent);Q[1,3]=-float(a@normal)
        tracks.append(dict(frame=Q@row['fixture_transform'],range=(0.,length),
            width_window=(-3*MINIMUM_WIDTH_M,3*MINIMUM_WIDTH_M),depth_window=(-WIDTH_TOLERANCE_M,.005+WIDTH_TOLERANCE_M)))
    return tracks


def feature_for_solid(name,value,rows=()):
    from deterministic_base import moved,unpack
    for row in rows:
        T=row['fixture_transform'];p=unpack(moved(value,T)).vertices
        if p[:,2].min()>=-1e-7 and p[:,2].max()<=.005+1e-7:
            return dict(name=name,kind='ground_patch',solid=value,frames=[T])
    p=unpack(value).vertices;center=p.mean(0)
    _,_,axes=np.linalg.svd(p-center,full_matrices=False)
    if np.linalg.det(axes)<0:axes[-1]*=-1
    span=np.ptp((p-center)@axes.T,axis=0)
    T=np.eye(4);T[:3,:3]=axes;T[:3,3]=-axes@center
    Q=np.eye(4);Q[:3,:3]=np.array([axes[0],axes[2],-axes[1]]);Q[:3,3]=-Q[:3,:3]@center
    frames=[T,Q]
    kind='solid_link' if span[2]>=.8*span[1] else 'flat_stock'
    return dict(name=name,kind=kind,solid=value,frames=frames)


class ConnectionWidthGuard:
    def __init__(self,core,features,minimum=MINIMUM_WIDTH_M):
        from deterministic_base import solid,union
        self.core=solid(core) if hasattr(core,'vertices') else core
        self.features=features;self.minimum=minimum
        self.planar=union([f['solid'] for f in features if f['kind'] in ('foot_band','flat_stock','ground_patch')])
        self.previous=None;self.previous_cache={}

    def assess(self,mesh,previous=None):
        from deterministic_base import solid,volume
        from wrap_width_profiles import feature_profiles
        value=solid(mesh) if hasattr(mesh,'vertices') else mesh
        if previous is not self.previous:
            self.previous=previous;self.previous_cache={}
        checks=[];plane_count=0
        current_core=value^self.core
        for feature in self.features:
            kept=value ^ feature['solid']
            if kept.is_empty() or volume(kept)<=1e-13:continue
            if feature['kind'] in ('ground_patch','flat_stock','solid_link'):
                from wrap_neck_width import assess
                row=assess(value,kept,feature,current_core,self.minimum,previous)
            else:row=feature_profiles(value,kept,feature,self.minimum,previous,self.previous_cache,current_core)
            checks.append(row);plane_count+=row['planes_checked']
        return dict(passed=all(r['passed'] for r in checks),minimum_width_m=self.minimum,
            measurement_revision=4,
            absolute_passed=all(r['absolute_passed'] for r in checks),
            no_new_narrow_connections_checked=previous is not None,
            scope='in_plane_feet_and_connections',wall_thickness_constrained=False,
            actual_saved_solid_measured=True,nominal_diameter_only=False,
            feature_count=len(self.features),planes_checked=plane_count,features=checks)

    def require(self,mesh,previous=None):
        result=self.assess(mesh,previous)
        if not result['passed']:
            bad=[r for r in result['features'] if not r['passed']]
            raise ValueError('Connection width rejected actual solid: '+str([r['feature'] for r in bad]))
        return result

    def save(self,out):
        from feasible_geometry import C
        from deterministic_base import unpack
        from exact_mesh import save_mesh
        rows=[]
        for k,feature in enumerate(self.features):
            path=out/f'width_features/{k:03d}.obj'
            save_mesh(unpack(feature['solid']),path)
            rows.append(dict(name=feature['name'],kind=feature['kind'],
                frames=[T.tolist() for T in feature['frames']],mesh=C.relative(path),sha256=C.digest(path),
                **(dict(tracks=[dict(t,frame=np.asarray(t['frame']).tolist()) for t in feature['tracks']]) if 'tracks' in feature else {})))
        C.save(out/'data/width_features.json',dict(complete=True,minimum_width_m=self.minimum,
            wall_thickness_constrained=False,features=rows))

    @classmethod
    def load(cls,out,core):
        from feasible_geometry import C
        from deterministic_base import solid
        record=json.loads((out/'data/width_features.json').read_text());features=[]
        for row in record['features']:
            path=Path(out)/'width_features'/Path(row['mesh']).name
            if C.digest(path)!=row['sha256']:raise ValueError('Width feature changed')
            value=solid(C.trimesh.load(path,force='mesh',process=False))
            features.append(dict(name=row['name'],kind=row['kind'],
                frames=[np.array(T) for T in row['frames']],solid=value,
                **(dict(tracks=row['tracks']) if 'tracks' in row else {})))
        return cls(core,features,record['minimum_width_m'])
