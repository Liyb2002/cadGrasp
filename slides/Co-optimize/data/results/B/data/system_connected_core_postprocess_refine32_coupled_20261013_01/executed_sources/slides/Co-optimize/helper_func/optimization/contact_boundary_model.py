"""Continuous contact polygons clipped by nominal exit shadows.

Leading triangle prisms supply exact linear shadow boundaries. Only contact
polygons are clipped; no support solid is built. Initial overlapping object
interiors use vertex signed-distance interpolation (guidance only).
"""
import numpy as np
import igl
from rtree import index as rtree_index
from shapely.geometry import Polygon,GeometryCollection
from shapely import make_valid,set_precision
from shapely.ops import unary_union
from co_common import U,transform_points
from physics_guided_cone_iterative import cone_projection


def clean_polygon(shape):
    if not shape.is_valid:shape=make_valid(shape)
    return set_precision(shape,1e-10,mode='valid_output')


def clip_barycentric(polygon,levels):
    if not len(polygon):return polygon
    values=polygon@levels;result=[]
    for j,current in enumerate(polygon):
        previous=polygon[j-1];a=values[j-1];b=values[j]
        if (a<=0)!=(b<=0):result.append(previous+(current-previous)*a/(a-b))
        if b<=0:result.append(current)
    return np.asarray(result).reshape(-1,3)


def available_polygons(triangles,levels,own_valid):
    points=[];indices=[];areas=np.zeros(len(triangles))
    for i in np.flatnonzero(own_valid & ~np.any(np.all(levels>0,axis=2),axis=0)):
        poly=np.eye(3)
        for blocker in range(len(levels)):
            poly=clip_barycentric(poly,levels[blocker,i])
            if len(poly)<3:break
        if len(poly)<3:continue
        vertices=poly@triangles[i]
        area=.5*np.linalg.norm(np.cross(vertices[1:-1]-vertices[0],vertices[2:]-vertices[0]),axis=1).sum()
        if area<1e-14:continue
        points.append(vertices);indices.extend([i]*len(vertices));areas[i]=area
    return np.concatenate(points) if points else np.empty((0,3)),np.asarray(indices,int),areas


def prism_planes(triangle,normal,delta):
    normals=[-normal,normal];bounds=[-normal@triangle[0],normal@(triangle[0]+delta)]
    center=triangle.mean(axis=0)+delta/2
    for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
        n=np.cross(b-a,delta);n/=np.linalg.norm(n)
        if n@(center-a)>0:n=-n
        normals.append(n);bounds.append(n@a)
    return np.asarray(normals),np.asarray(bounds)


class ContactBoundaryModel:
    def __init__(self,search):
        self.search=search;self.shadow_cache={};self.pair_cache={};self.state_cache={}
        tri=search.tri
        self.u=tri[:,1]-tri[:,0];self.u/=np.linalg.norm(self.u,axis=1)[:,None]
        self.v=np.cross(search.mesh.face_normals[search.src],self.u)
        self.xy=np.stack([np.einsum('tvc,tc->tv',tri-tri[:,0,None],self.u),
                          np.einsum('tvc,tc->tv',tri-tri[:,0,None],self.v)],axis=2)
        self.polygons=[clean_polygon(Polygon(xy)) for xy in self.xy]

    def shadows(self,direction):
        key=tuple(direction)
        if key not in self.shadow_cache:
            search=self.search;ids=np.flatnonzero(search.mesh.face_normals@direction>1e-9)
            delta=search.length*direction;planes=[];bounds=[]
            prop=rtree_index.Property();prop.dimension=3
            tree=rtree_index.Index(properties=prop)
            for j,source in enumerate(ids):
                tri=search.mesh.triangles[source];n,b=prism_planes(tri,search.mesh.face_normals[source],delta)
                vertices=np.vstack([tri,tri+delta]);box=np.r_[vertices.min(0)-1e-10,vertices.max(0)+1e-10]
                tree.insert(j,box);planes.append((n,b));bounds.append(box)
            self.shadow_cache[key]=(tree,planes)
            if len(self.shadow_cache)>64:self.shadow_cache.pop(next(iter(self.shadow_cache)))
        return self.shadow_cache[key]

    def blocked(self,relative,direction):
        key=(tuple(relative),tuple(direction))
        if key in self.pair_cache:return self.pair_cache[key]
        search=self.search;tree,planes=self.shadows(direction);blocked=[]
        initial=None
        if np.linalg.norm(relative)>1e-10:
            points=search.tri.reshape(-1,3)+relative
            distances,_,_,_=igl.signed_distance(points,np.asarray(search.mesh.vertices,dtype=np.float64),
                np.asarray(search.mesh.faces,dtype=np.int64),igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
            initial=distances.reshape(-1,3) # negative is inside original object
        for i,triangle in enumerate(search.tri):
            if np.linalg.norm(relative)<1e-10 and search.mesh.face_normals[search.src[i]]@direction>1e-9:
                blocked.append(self.polygons[i]);continue
            pieces=[];query=triangle+relative
            if initial is not None:
                poly=clip_barycentric(np.eye(3),initial[i]+1e-10)
                if len(poly)>=3:
                    shape=clean_polygon(Polygon(poly@self.xy[i]))
                    if shape.area>1e-16:pieces.append(shape)
            box=np.r_[query.min(0)-1e-10,query.max(0)+1e-10]
            fully_blocked=False
            for j in tree.intersection(box):
                n,b=planes[j];levels=query@n.T-b
                # Tangential sweep surfaces are not penetrating regions.
                if np.any(np.min(levels,axis=0)>1e-10):continue
                flat=np.max(np.abs(levels),axis=0)<1e-10
                if np.any(flat & (n@search.mesh.face_normals[search.src[i]]>0)):continue
                poly=np.eye(3)
                for axis in range(len(b)):
                    poly=clip_barycentric(poly,levels[:,axis])
                    if len(poly)<3:break
                if len(poly)<3:continue
                shape=clean_polygon(Polygon(poly@self.xy[i]))
                if shape.area<=1e-16:continue
                if shape.area>=self.polygons[i].area*(1-1e-9):
                    pieces=[self.polygons[i]];fully_blocked=True;break
                pieces.append(shape)
            blocked.append(clean_polygon(unary_union(pieces)) if pieces else GeometryCollection())
        self.pair_cache[key]=blocked
        if len(self.pair_cache)>128:self.pair_cache.pop(next(iter(self.pair_cache)))
        return blocked

    def geometry(self,directions,offsets,owners=None):
        owners=tuple(range(self.search.n)) if owners is None else tuple(sorted(set(owners)))
        key=(directions.tobytes(),offsets.tobytes(),owners)
        if key in self.state_cache:return self.state_cache[key]
        search=self.search;rays=[];areas=[];all_polygons=[]
        for owner,(task,T) in enumerate(search.states):
            if owner not in owners:
                rays.append(None);areas.append(None);all_polygons.append(None);continue
            colocated=np.linalg.norm(offsets-offsets[owner],axis=1)<1e-10
            valid=np.max(search.mesh.face_normals[search.src]@directions[colocated].T,axis=1)<=1e-9
            blocked=[self.blocked(offsets[owner]-offsets[j],directions[j]) for j in range(search.n)]
            points=[];sources=[];area=[];polygons=[]
            for i,triangle in enumerate(search.tri):
                if not valid[i]:
                    polygons.append(GeometryCollection());area.append(0.);continue
                pieces=[row[i] for row in blocked if not row[i].is_empty]
                polygon=self.polygons[i].difference(unary_union(pieces)) if pieces else self.polygons[i]
                polygons.append(polygon);area.append(polygon.area)
                components=[polygon] if polygon.geom_type=='Polygon' else list(polygon.geoms) if hasattr(polygon,'geoms') else []
                for component in components:
                    if component.geom_type!='Polygon' or component.area<1e-14:continue
                    xy=np.asarray(component.exterior.coords)[:-1]
                    # Interior holes do not change a convex wrench hull.
                    points.extend(triangle[0]+xy[:,0,None]*self.u[i]+xy[:,1,None]*self.v[i])
                    sources.extend([search.src[i]]*len(xy))
            points=np.asarray(points).reshape(-1,3);sources=np.asarray(sources,int)
            normals=-task.domain.mesh.face_normals[sources];world=transform_points(points,T)
            reaction=U.heads(np.c_[normals,np.cross(world-task.domain.com,normals)],task.scale)
            rays.append(np.vstack([search.floors[owner],reaction]));areas.append(np.asarray(area));all_polygons.append(polygons)
        result=dict(rays=rays,areas=areas,polygons=all_polygons)
        self.state_cache[key]=result
        if len(self.state_cache)>64:self.state_cache.pop(next(iter(self.state_cache)))
        return result

    def evaluate(self,directions,offsets,targets):
        geometry=self.geometry(directions,offsets,[k for k,i,t in targets]);values=[]
        for k,index,target in targets:
            result=cone_projection(geometry['rays'][k],target)
            if result['kkt_max_violation']>1e-7:raise RuntimeError('boundary cone projection KKT unresolved')
            values.append(result['loss'])
        return dict(losses=values,loss=max(values),sum_loss=float(sum(values)),geometry=geometry)

    def changes(self,before,after):
        released=[];locked=[]
        for old,new in zip(before['polygons'],after['polygons']):
            if old is None or new is None:
                released.append(None);locked.append(None);continue
            released.append(sum(b.difference(a).area for a,b in zip(old,new)))
            locked.append(sum(a.difference(b).area for a,b in zip(old,new)))
        return dict(released_area_m2=released,newly_locked_area_m2=locked)
