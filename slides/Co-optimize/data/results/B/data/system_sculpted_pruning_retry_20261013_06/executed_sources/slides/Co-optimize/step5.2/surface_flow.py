"""Bounded smoothing of actual free surfaces, with exact contact/core pins."""
import numpy as np
import trimesh
import manifold3d as md
from scipy.sparse import coo_matrix,diags
from scipy.spatial import cKDTree
from deterministic_base import solid,unpack,SCALE


def flow(mesh,certificate,frames,object_cloud,mandatory=None,final_support=False,displacement=.0009):
    dense=solid(mesh).refine_to_length(.003/SCALE)
    if dense.num_tri()>240000:dense=solid(mesh).refine_to_length(.005/SCALE)
    surface=unpack(dense);surface.merge_vertices(digits_vertex=12)
    original=surface.vertices.copy();vertices=original.copy();pinned=np.zeros(len(vertices),bool)
    for k in certificate.layout.active:
        body=certificate.mesh.copy();body.apply_transform(certificate.layout.placements[k])
        _,distance,_=trimesh.proximity.closest_point(body,original);pinned|=distance<.0003
    cloud=np.vstack([object_cloud]+[original@T[:3,:3].T+T[:3,3]for T in frames])
    bounds=np.array([cloud.min(0),cloud.max(0)])
    for T in frames:
        world=original@T[:3,:3].T+T[:3,3]
        for axis in range(3):
            for side in [0,1]:pinned|=abs(world[:,axis]-bounds[side,axis])<1e-9
        if final_support:pinned|=abs(world[:,2])<.0004
    if mandatory is not None and not mandatory.is_empty():
        core=unpack(mandatory);_,distance,_=trimesh.proximity.closest_point(core,original)
        pinned|=distance<.0002;pinned|=core.contains(original)
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
        halfedge_tangent=np.ascontiguousarray(tangents))).refine(2)
    curved=curved.simplify(.00002/SCALE)
    if mandatory is not None and not mandatory.is_empty():curved=curved+mandatory
    return curved^solid(mesh),dict(actual_surface_flow=True,maximum_vertex_displacement_m=displacement,
        contact_pin_distance_m=.0003,iterations=8,refinement_factor=2,pinned_vertices=int(pinned.sum()),
        input_vertices=len(original),protected_surface_edges=int(protected_edges.sum()),
        pinned_face_tangents_are_linear=True,nonfinite_tangents_replaced=True,
        geometry_edited=True,display_only=False)
