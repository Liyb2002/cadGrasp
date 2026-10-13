"""Original force laws on contact polygons of the actual constructed solid."""
import numpy as np
from whole_search.common import C,transform_mesh
from whole_search.classify import classify


def certify_material_contacts(model,layout,mesh):
    masks={};supplies={};classifiers={};contacts={}
    for k in layout.active:
        Q=layout.placements[k];direction=Q[:3,:3].T@layout.directions[k]
        allowed=model.allowed[k][model.mesh.face_normals[model.allowed[k]]@direction<=1e-9]
        triangles,sources=C.contact_boundary(transform_mesh(model.mesh,Q),mesh,allowed)
        native_map=model.native[k]@np.linalg.inv(Q)
        points=C.transform_points(triangles.reshape(-1,3),native_map)
        normals=np.repeat(-model.tasks[k].domain.mesh.face_normals[sources],3,axis=0)
        raw=np.c_[normals,np.cross(points-model.tasks[k].domain.com,normals)]
        rays=C.I.merge_columns(model.floor_columns(layout,k),C.U.heads(raw,model.tasks[k].scale))
        mask,info=classify(rays,model.tasks[k].targets)
        masks[k]=mask;supplies[k]=rays;classifiers[k]=info
        contacts[k]=dict(triangle_count=len(triangles),
            area_m2=float(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2))
    return dict(masks=masks,supplies=supplies,classifiers=classifiers,contacts=contacts,
        counts={str(k):int(mask.sum())for k,mask in masks.items()},
        force_contact_representation='actual_constructed_material_contact_polygon_vertices',
        original_force_laws_unchanged=True,original_demand_count_per_pose=32768)
