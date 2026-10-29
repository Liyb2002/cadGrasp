"""Whole-triangle perspective first hit, including all object occluders."""
import numpy as np
import shapely
from shapely import STRtree
from .work_disks import tangent_frame

RECIPROCAL_DEPTH_TOLERANCE=1e-9
PROJECTED_AREA_TOLERANCE=1e-16


def visible_faces(mesh, head, forward, candidate_ids=None):
    head=np.asarray(head);forward=np.asarray(forward);forward=forward/np.linalg.norm(forward)
    basis=np.column_stack([tangent_frame(forward)[:,:2],forward])
    q=(mesh.triangles-head)@basis
    normals=mesh.face_normals@basis
    offset=np.einsum('ij,ij->i',mesh.face_normals,mesh.triangles[:,0]-head)
    front=offset < -1e-12
    if candidate_ids is None:candidate_ids=np.flatnonzero(front & (q[:,:,2].min(1)>1e-8))
    ids=np.asarray(candidate_ids,dtype=int)
    result=np.zeros(len(mesh.faces),bool)
    ids=ids[front[ids] & (q[ids,:,2].min(1)>1e-8)]
    if not len(ids):return result
    # Clip only triangles that cross the camera plane. Whole triangles behind
    # it cannot occlude a forward ray. Do not discard crossing occluders.
    near=1e-8;all_front=q[:,:,2].min(1)>=near
    blocker_ids=list(np.flatnonzero(all_front & (np.abs(offset)>1e-12)))
    vertices=[v for v in q[blocker_ids]]
    crossing=np.flatnonzero((q[:,:,2].min(1)<near)&(q[:,:,2].max(1)>near)&(np.abs(offset)>1e-12))
    for f in crossing:
        polygon=[]
        for a,b in zip(q[f],np.roll(q[f],-1,axis=0)):
            ina,inb=a[2]>=near,b[2]>=near
            if ina:polygon.append(a)
            if ina!=inb:polygon.append(a+(b-a)*((near-a[2])/(b[2]-a[2])))
        if len(polygon)>=3:
            for j in range(1,len(polygon)-1):
                blocker_ids.append(int(f));vertices.append(np.array([polygon[0],polygon[j],polygon[j+1]]))
    blocker_ids=np.asarray(blocker_ids,dtype=int);vertices=np.asarray(vertices)
    projected=vertices[:,:,:2]/vertices[:,:,2,None]
    polygons=shapely.polygons(projected)
    valid=shapely.area(polygons)>PROJECTED_AREA_TOLERANCE
    polygons=polygons[valid];blocker_ids=blocker_ids[valid]
    own_projection=q[ids,:,:2]/q[ids,:,2,None]
    own_polygons=shapely.polygons(own_projection)
    good=shapely.area(own_polygons)>PROJECTED_AREA_TOLERANCE
    ids=ids[good];own_polygons=own_polygons[good];result[ids]=True
    if not len(ids):return result
    own,other=STRtree(polygons).query(own_polygons,predicate='intersects')
    a,b=ids[own],blocker_ids[other];different=a!=b
    a,b,own,other=a[different],b[different],own[different],other[different]
    overlap=shapely.intersection(own_polygons[own],polygons[other])
    good=shapely.area(overlap)>PROJECTED_AREA_TOLERANCE
    a,b,overlap=a[good],b[good],overlap[good]
    uv,index=shapely.get_coordinates(overlap,return_index=True)
    coefficients=normals/np.where(np.abs(offset)>1e-12,offset,1.)[:,None]
    delta=(coefficients[b]-coefficients[a])[index]
    difference=np.einsum('ij,ij->i',uv,delta[:,:2])+delta[:,2]
    result[a[index[difference>RECIPROCAL_DEPTH_TOLERANCE]]]=False
    return result
