"""Bounded smoothing of actual free surfaces, with exact contact/core pins."""
import numpy as np
import trimesh
import manifold3d as md
from scipy.sparse import coo_matrix,diags
from scipy.spatial import cKDTree
from deterministic_base import solid,unpack,SCALE


def nearest_points(mesh,points):
    values=[]
    for start in range(0,len(points),256):
        _,distance,_=trimesh.proximity.closest_point(mesh,points[start:start+256]);values.append(distance)
    return np.concatenate(values)if values else np.empty(0)


def inside_points(mesh,points):
    return np.concatenate([mesh.contains(points[start:start+256])for start in range(0,len(points),256)])if len(points)else np.empty(0,bool)


def flow(mesh,certificate,frames,object_cloud,mandatory=None,final_support=False,displacement=.0009,
         pin_contacts=True,pin_bounds=True,curvature_subdivisions=None):
    base_surface=solid(mesh).simplify(0.)
    if not pin_contacts:
        # Remove redundant nearly planar tessellation before actual curvature
        # construction. The proposed result is still clipped and recertified.
        base_surface=base_surface.simplify(.00004/SCALE)
    dense=base_surface.refine_to_length(.003/SCALE)
    if dense.num_tri()>160000:dense=base_surface.refine_to_length(.005/SCALE)
    surface=unpack(dense);surface.merge_vertices(digits_vertex=12)
    original=surface.vertices.copy();vertices=original.copy();pinned=np.zeros(len(vertices),bool)
    for k in certificate.layout.active if pin_contacts else []:
        body=certificate.mesh.copy();body.apply_transform(certificate.layout.placements[k])
        distance=nearest_points(body,original);pinned|=distance<.0003
    cloud=np.vstack([object_cloud]+[original@T[:3,:3].T+T[:3,3]for T in frames])
    bounds=np.array([cloud.min(0),cloud.max(0)])
    for T in frames:
        world=original@T[:3,:3].T+T[:3,3]
        if pin_bounds:
            for axis in range(3):
                for side in [0,1]:pinned|=abs(world[:,axis]-bounds[side,axis])<1e-9
        if final_support:pinned|=abs(world[:,2])<.0004
    if mandatory is not None and not mandatory.is_empty():
        core=unpack(mandatory);distance=nearest_points(core,original)
        pinned|=distance<.0002;pinned|=inside_points(core,original)
    edges=surface.edges_unique;row=np.r_[edges[:,0],edges[:,1]];col=np.r_[edges[:,1],edges[:,0]]
    neighbors=coo_matrix((np.ones(len(row)),(row,col)),shape=(len(vertices),len(vertices))).tocsr()
    neighbors=diags(1/np.maximum(np.asarray(neighbors.sum(1)).ravel(),1))@neighbors
    for _ in range(8):
        candidate=vertices+.4*(neighbors@vertices-vertices);delta=candidate-original
        lengths=np.linalg.norm(delta,axis=1);delta*=np.minimum(1.,displacement/np.maximum(lengths,1e-12))[:,None]
        vertices=original+delta;vertices[pinned]=original[pinned]
    surface.vertices=vertices
    base=solid(surface)
    smooth=base.smooth_out(180,.7).to_mesh64()
    positions=np.array(smooth.vert_properties,dtype=np.float64)[:,:3]
    faces=np.array(smooth.tri_verts,dtype=np.uint64)
    tangents=np.array(smooth.halfedge_tangent,dtype=np.float64)
    fixed=(cKDTree(vertices[pinned]).query(positions*SCALE)[0]<1e-8)if pinned.any()else np.zeros(len(positions),bool)
    linear=np.zeros_like(tangents);linear[:,:,:3]=positions[np.roll(faces,-1,axis=1)]-positions[faces];linear[:,:,3]=1.
    protected_edges=fixed[faces]&fixed[np.roll(faces,-1,axis=1)]
    protected_edges|=np.all(fixed[faces],axis=1)[:,None]
    protected_edges|=~np.isfinite(tangents).all(axis=2)
    tangents[protected_edges]=linear[protected_edges]
    curved=md.Manifold(md.Mesh64(np.ascontiguousarray(positions),np.ascontiguousarray(faces),
        halfedge_tangent=np.ascontiguousarray(tangents)))
    curved=(curved.refine(curvature_subdivisions)if curvature_subdivisions is not None and len(faces)<100000
            else curved.refine_to_length(.0025/SCALE))
    curved=curved.simplify(.00002/SCALE)
    if mandatory is not None and not mandatory.is_empty():curved=curved+mandatory
    return (curved^solid(mesh)).simplify(0.),dict(actual_surface_flow=True,maximum_vertex_displacement_m=displacement,
        contact_pin_distance_m=.0003,iterations=8,refinement_factor=2,pinned_vertices=int(pinned.sum()),
        input_vertices=len(original),protected_surface_edges=int(protected_edges.sum()),
        pinned_face_tangents_are_linear=True,nonfinite_tangents_replaced=True,
        nearest_query_batch_size=256,geometry_edited=True,display_only=False,
        all_contact_surfaces_pinned=pin_contacts,bbx_extrema_pinned=pin_bounds)
