"""Bounded smoothing of actual free surfaces, with exact contact/core pins."""
import numpy as np
import trimesh
from scipy.sparse import coo_matrix,diags
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
    curved=solid(surface).smooth_out(180,.7).refine(2)
    if mandatory is not None and not mandatory.is_empty():curved=curved+mandatory
    return curved^solid(mesh),dict(actual_surface_flow=True,maximum_vertex_displacement_m=displacement,
        contact_pin_distance_m=.0003,iterations=8,refinement_factor=2,pinned_vertices=int(pinned.sum()),
        input_vertices=len(original),geometry_edited=True,display_only=False)
