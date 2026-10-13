"""Conservative self-occlusion clipping of outward work half-rays.

Remove an angular sector only when every source point in a work triangle is
blocked by a convex planar object facet. Three source-corner tests certify
that fact: the facet is convex, and parallel forward-ray intersection
points interpolate. Uncertain sectors remain reserved, so no reachable ray
is discarded. This preprocessing is independent of pose translation/layout.
"""
from functools import lru_cache
import hashlib
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import constrained_delaunay_triangles
from work_access import WorkAccess, tangent_frame, _contains
from co_common import S, G, union


def clip_polygon(points, normal, offset):
    values=points@normal+offset
    if values.min()>=0:return points
    if values.max()<0:return np.empty((0,2))
    result=[]
    for p,q,a,b in zip(points,np.roll(points,-1,axis=0),values,np.roll(values,-1)):
        if a>=0:result.append(p)
        if (a>=0)!=(b>=0):result.append(p+a/(a-b)*(q-p))
    return np.asarray(result).reshape(-1,2)


def convex_occluders(mesh):
    """Merge exactly planar convex facets; retain other faces as triangles."""
    polygons=[];merged=set();extent=max(float(mesh.extents.max()),1e-12)
    for faces in mesh.facets:
        vertices=np.unique(mesh.vertices[mesh.faces[faces].reshape(-1)],axis=0)
        n=mesh.face_normals[faces[0]];frame=tangent_frame(n)
        if np.ptp(vertices@n)>extent*1e-10:continue
        planar=vertices@frame
        hull=ConvexHull(planar)
        if abs(hull.volume-mesh.area_faces[faces].sum())>max(1e-14,hull.volume*1e-10):continue
        polygons.append(vertices[hull.vertices]);merged.update(map(int,faces))
    polygons.extend(tri for k,tri in enumerate(mesh.triangles) if k not in merged)
    groups={}
    for polygon in polygons:groups.setdefault(len(polygon),[]).append(polygon)
    return {k:np.asarray(v) for k,v in groups.items()}


def shadow_angles(occluder_groups, source, normal, frame, ring, ray_offset):
    """Angular regions blocked for ALL source positions, not a ray sample."""
    origins=source+ray_offset*normal
    blocked=[]
    for corners,polygons in occluder_groups.items():
        vectors=polygons[:,None,:,:]-origins[None,:,None,:]
        forward=(vectors@normal).max(axis=2).min(axis=1)>1e-10
        vectors=vectors[forward]
        if not len(vectors):continue
        next_vectors=np.roll(vectors,-1,axis=2)
        normals=np.cross(vectors,next_vectors)
        centers=vectors.mean(axis=2)
        determinant=np.einsum('osvc,osc->osv',normals,centers)
        scale=np.linalg.norm(vectors,axis=3)*np.linalg.norm(next_vectors,axis=3)*np.linalg.norm(centers,axis=2)[:,:,None]
        nonsingular=np.all(np.abs(determinant)>1e-10*scale,axis=(1,2))
        normals=normals[nonsingular]*np.sign(determinant[nonsingular])[:,:,:,None]
        if not len(normals):continue
        normals/=np.linalg.norm(normals,axis=3)[:,:,:,None]
        coefficients=(normals@frame).reshape(-1,3*corners,2)
        # Shrink certified shadow halfspaces slightly rather than claim a
        # borderline numerical direction blocked. d=n+frame*s, with d.n=1.
        offsets=(normals@normal).reshape(-1,3*corners)-1e-10
        values=np.einsum('ocd,pd->ocp',coefficients,ring)+offsets[:,:,None]
        possible=np.all(values.max(axis=2)>=0,axis=1)
        for planes,b in zip(coefficients[possible],offsets[possible]):
            polygon=ring
            for a,c in zip(planes,b):
                polygon=clip_polygon(polygon,a,c)
                if len(polygon)<3:break
            if len(polygon)>=3:
                region=Polygon(polygon)
                if region.area>1e-12:blocked.append(region)
    return unary_union(blocked) if blocked else Polygon()


def convex_cells(region):
    if region.is_empty:return []
    polygons=[region] if region.geom_type=='Polygon' else list(region.geoms)
    cells=[]
    for polygon in polygons:
        if polygon.area<=0:continue
        if not polygon.interiors and abs(polygon.convex_hull.area-polygon.area)<1e-13:
            cells.append(np.asarray(polygon.exterior.coords)[:-1])
        else:
            for triangle in constrained_delaunay_triangles(polygon).geoms:
                if triangle.area>0 and polygon.covers(triangle):
                    cells.append(np.asarray(triangle.exterior.coords)[:-1])
    return cells


class ShadowClippedWorkAccess(WorkAccess):
    def __init__(self, mesh, face_ids, half_angle_deg, ray_offset_m=1e-5):
        super().__init__(mesh,face_ids,half_angle_deg)
        self.ray_offset_m=float(ray_offset_m);self.cells=[];families=[];self.shadow_rows=[];cell_faces=[]
        occluders=convex_occluders(mesh)
        for face,triangle,normal,ring in zip(self.face_ids,self.triangles,self.normals,self.rings):
            frame=tangent_frame(normal);angular_ring=ring@frame;original=Polygon(angular_ring)
            blocked=shadow_angles(occluders,triangle,normal,frame,angular_ring,self.ray_offset_m)
            allowed=original.difference(blocked)
            angular_cells=convex_cells(allowed)
            covered=unary_union([Polygon(c) for c in angular_cells])
            # Geometry splitting must not silently drop a reachable sliver.
            # Any uncertain numerical split falls back to the original cone.
            fallback=not allowed.is_empty and not covered.buffer(1e-12).covers(allowed)
            if fallback:angular_cells=[angular_ring]
            face_cells=[];face_families=[]
            for angular in angular_cells:
                slopes=angular@frame.T
                try:hull=ConvexHull(self.cell_points(triangle,normal,slopes,1.))
                except QhullError:
                    fallback=True;break
                # A clipped off-axis cell may have VALID side planes with
                # a.n>0. Remove only the far cap whose normal is exactly +n.
                planes=hull.equations[hull.equations[:,:3]@normal<1-1e-8]
                _,unique=np.unique(np.round(planes,10),axis=0,return_index=True)
                face_cells.append((triangle,normal,slopes))
                face_families.append(planes[np.sort(unique)])
            if fallback:
                face_cells=[(triangle,normal,ring)]
                hull=ConvexHull(self.cell_points(triangle,normal,ring,1.))
                face_families=[hull.equations[hull.equations[:,:3]@normal<1-1e-8]]
            self.cells.extend(face_cells);families.extend(face_families)
            cell_faces.extend([int(face)]*len(face_cells))
            self.shadow_rows.append(dict(face_id=int(face),numerical_fallback_to_original_cone=fallback,
                retained_convex_angular_cells=len(face_cells),
                removed_angular_area_fraction=0. if fallback else float(max(0.,1-allowed.area/original.area))))
        self.cell_face_ids=np.asarray(cell_faces,int)
        self.counts=np.array([len(p) for p in families],np.int64)
        self.planes=np.zeros((len(families),max(self.counts,default=0),4))
        for k,p in enumerate(families):self.planes[k,:len(p)]=p
        self._solids={}

    def contains_points(self, points):
        return _contains(np.ascontiguousarray(points,dtype=np.float64),self.planes,self.counts)

    def solid(self,length):
        key=round(float(length),9)
        if key not in self._solids:
            self._solids[key]=union([S.solid(G.hull_mesh(self.cell_points(t,n,r,key))) for t,n,r in self.cells])
        return self._solids[key]

    def definition(self):
        result=super().definition()
        result.pop('self_occluded_directions_also_reserved',None)
        result.update(outward_half_rays=True,ray_parameter_min=0.,bidirectional_lines=False,
            certified_self_occluded_sectors_reserved=False,uncertain_occlusion_reserved_conservatively=True,
            self_occlusion_policy='remove only angular sectors certified blocked for the entire source triangle; retain uncertainty conservatively',
            ray_offset_m=self.ray_offset_m,continuous_source_triangle=True,
            shadow_proof='three source vertices hit the same convex planar object facet along parallel positive rays; interpolate over source triangle',
            convex_angular_cell_count=len(self.cells),
            source_faces_with_shadow_clipping=sum(r['removed_angular_area_fraction']>1e-10 for r in self.shadow_rows),
            per_face_shadow_clipping=self.shadow_rows)
        return result


_MESHES={}


@lru_cache(maxsize=128)
def _cached_access(mesh_key,face_ids,half_angle_deg,ray_offset_m):
    return ShadowClippedWorkAccess(_MESHES[mesh_key],face_ids,half_angle_deg,ray_offset_m)


def shadow_clipped_access(mesh,original,ray_offset_m=1e-5):
    digest=hashlib.sha256(np.asarray(mesh.vertices,dtype=np.float64).tobytes()+
        np.asarray(mesh.faces,dtype=np.int64).tobytes()).hexdigest()
    _MESHES.setdefault(digest,mesh)
    return _cached_access(digest,tuple(map(int,original.face_ids)),float(original.half_angle_deg),float(ray_offset_m))
