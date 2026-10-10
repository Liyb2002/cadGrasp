"""Moving contact boundaries on original faces; no solid Boolean in evaluation.

Signed halfspace unions describe complete original work cones and nominal
exits. Their restriction to a subdivided contact triangle is interpolated and
clipped. The resulting vertices MOVE continuously between topology events.
This is a local boundary approximation, not final mesh contact certification.
"""
from collections import OrderedDict
import hashlib
import numpy as np
from numba import njit
from scipy.spatial import ConvexHull, QhullError
from co_common import transform_points, U


@njit(cache=True)
def union_field(points,planes,counts):
    result=np.full(len(points),np.inf)
    for i in range(len(points)):
        for cell in range(len(counts)):
            distance=-np.inf
            for j in range(counts[cell]):
                a,b,c,d=planes[cell,j]
                value=a*points[i,0]+b*points[i,1]+c*points[i,2]+d
                if value>distance:distance=value
            if distance<result[i]:result[i]=distance
    return result


def exit_planes(mesh,direction,length):
    """Union of outgoing triangle prisms plus body equals the full sweep."""
    rows=[]
    for triangle,normal in zip(mesh.triangles,mesh.face_normals):
        dot=float(normal@direction)
        if dot<=1e-10:continue
        family=[np.r_[-normal,normal@triangle[0]],
                np.r_[normal,-normal@triangle[0]-length*dot]]
        center=triangle.mean(0)+length/2*direction
        for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
            side=np.cross(b-a,direction);norm=np.linalg.norm(side)
            if norm<=1e-15:break
            side/=norm
            if (center-a)@side>0:side=-side
            family.append(np.r_[side,-side@a])
        if len(family)==5:rows.append(family)
    return np.asarray(rows,float).reshape(-1,5,4),np.full(len(rows),5,np.int64)


def barycentric_cells(depth):
    cells=np.eye(3)[None]
    for _ in range(depth):
        children=[]
        for a,b,c in cells:
            ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
            children.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
        cells=np.asarray(children)
    points,inverse=np.unique(cells.reshape(-1,3),axis=0,return_inverse=True)
    return points,inverse.reshape(-1,3)


def clip_field(triangle,values):
    vertices=[]
    for a,b,fa,fb in zip(triangle,np.roll(triangle,-1,axis=0),values,np.roll(values,-1)):
        if fa>=0:vertices.append(a)
        if (fa>=0)!=(fb>=0):vertices.append(a+fa/(fa-fb)*(b-a))
    return np.asarray(vertices,float).reshape(-1,3)


class ContactBoundaries:
    def __init__(self,model,depth=1):
        self.model=model;self.depth=depth;self.bary,self.cells=barycentric_cells(depth)
        self.grid=np.einsum('bv,fvc->fbc',self.bary,model.mesh.triangles)
        self.exit_cache=OrderedDict();self.body_cache=OrderedDict();self.work_cache=OrderedDict()
        self.proximity=model.mesh.nearest
        self.evaluations=0
        from .surface import SurfaceCuts
        self.surface=SurfaceCuts(model)

    def fields(self,layout,owner,points,source_ids):
        model=self.model;result=np.full(len(points),np.inf)
        normals=model.mesh.face_normals[source_ids]
        for blocker in layout.active:
            relative=np.linalg.inv(layout.placements[blocker])@layout.placements[owner]
            local=transform_points(points,relative)
            key=(owner,blocker,relative.tobytes(),points.tobytes())
            if key not in self.work_cache:
                access=model.work_rays[blocker]
                self.work_cache[key]=union_field(local,access.planes,access.counts)
                if len(self.work_cache)>64:self.work_cache.popitem(last=False)
            result=np.minimum(result,self.work_cache[key])
            # Coincident original bodies cannot cut their own exterior wall.
            # Skipping this zero-distance boundary avoids clipping all fields
            # to epsilon and destroying meaningful tangential derivatives.
            if not np.allclose(relative,np.eye(4),atol=1e-12,rtol=0):
                if key not in self.body_cache:
                    origins=local+model.epsilon*(normals@relative[:3,:3].T)
                    self.body_cache[key]=-self.proximity.signed_distance(origins)
                    if len(self.body_cache)>64:self.body_cache.popitem(last=False)
                result=np.minimum(result,self.body_cache[key])
            direction=layout.placements[blocker,:3,:3].T@layout.directions[blocker]
            dkey=direction.tobytes()
            if dkey not in self.exit_cache:
                self.exit_cache[dkey]=exit_planes(model.mesh,direction,model.length)
                if len(self.exit_cache)>32:self.exit_cache.popitem(last=False)
            planes,counts=self.exit_cache[dkey]
            origins=local+model.epsilon*(normals@relative[:3,:3].T)
            result=np.minimum(result,union_field(origins,planes,counts))
        return result

    def supply(self,layout,owner):
        model=self.model;self.evaluations+=1
        direction=layout.placements[owner,:3,:3].T@layout.directions[owner]
        ids=model.allowed[owner][model.mesh.face_normals[model.allowed[owner]]@direction<=1e-9]
        local,normal,area=self.surface.vertices(layout,owner,ids)
        task=model.tasks[owner];native=model.native[owner]
        world=transform_points(local,native);inward=-normal@native[:3,:3].T
        raw=np.c_[inward,np.cross(world-task.domain.com,inward)]
        heads=U.heads(raw,task.scale)
        rays=np.vstack([model.floors[owner],heads])
        return rays,dict(contact_vertices=len(local),contact_area_m2=area,
                         contact_normal_signature=hashlib.sha256(np.unique(normal,axis=0).tobytes()).hexdigest(),
                         boundary_subdivision_depth=self.depth,solid_booleans=0,
                         method='surface_polygons_cut_by_individual_work_and_exit_prisms',
                         contact_geometry_certified=False)
