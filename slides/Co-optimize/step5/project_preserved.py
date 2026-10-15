"""Reinforce a round tree while preserving certified functional surfaces."""
import numpy as np
import manifold3d as md
from deterministic_base import solid,unpack,union,volume,SCALE
from prune_support import material_components
from anchor_network import ball,path_body
from tube_support import capsule
from periphery import PlaneForceFilter
from project_backbone import coarse_ball,coarse_path


def interface_mask(checker,bodies,half_depth=.0001,padding=.00002):
    boxes=[]
    for body in bodies:
        for k,(triangles,sources)in PlaneForceFilter(checker,body).contacts.items():
            for face in np.unique(sources):
                points=triangles[sources==face].reshape(-1,3)
                n=checker.mesh.face_normals[face]@checker.layout.placements[k][:3,:3].T
                n=n/np.linalg.norm(n);axis=np.eye(3)[int(np.argmin(abs(n)))]
                u=np.cross(n,axis);u/=np.linalg.norm(u);v=np.cross(n,u);basis=np.array([u,v,n])
                local=points@basis.T;lo=local.min(0);hi=local.max(0)
                lo[:2]-=padding;hi[:2]+=padding;lo[2]-=half_depth;hi[2]+=half_depth
                vertices=np.array([[x,y,z]for x in [lo[0],hi[0]]for y in [lo[1],hi[1]]for z in [lo[2],hi[2]]])@basis
                boxes.append(md.Manifold.hull_points(np.ascontiguousarray(vertices/SCALE)))
    return union(boxes),len(boxes)


def project(mesh,full,intermediate,nodes,tree_paths,radius,reference,domain,checker,old_full,old_intermediate):
    certified=max(reference+.00005,radius-.00015)
    if certified>=radius:raise ValueError('No positive inner-radius margin')
    working=union([path_body(path,radius+.0002)for a,b,path in tree_paths]+[ball(nodes[0],radius+.0002)])^domain
    mask,boxes=interface_mask(checker,[old_full,old_intermediate])
    outputs=[]
    for body,original in [(full,old_full),(intermediate,old_intermediate)]:
        value=(solid(body)+working).simplify(0.)
        value=((value-mask)+(solid(original)^mask)).simplify(0.)
        parts=material_components(value)
        if len(parts)>1:
            ranked=[(max(volume(ball(c,certified)-part)for c in nodes),-volume(part),part)for part in parts]
            loss,_,value=min(ranked,key=lambda r:r[:2])
            if loss>1e-12:raise ValueError('Preserved product does not contain every inner node')
        result=unpack(value);body.vertices=result.vertices.copy();body.faces=result.faces.copy();outputs.append(value)
    value=coarse_ball(nodes[0],radius);attached={0};pending=list(tree_paths)
    while pending:
        i=next((i for i,(a,b,path)in enumerate(pending)if a in attached or b in attached),None)
        if i is None:raise ValueError('Preserved tree lost a connected node')
        a,b,path=pending.pop(i);value=(value+coarse_path(path,radius)).simplify(0.);attached.update([a,b])
    value=(value^outputs[0]^outputs[1]).simplify(0.);packet=material_components(value)[0]
    primitives=[ball(c,certified)for c in nodes]
    primitives.extend(capsule(a,b,certified)for first,last,path in tree_paths for a,b in zip(path,path[1:]))
    maxima=[]
    for container in [packet]+outputs:
        loss=max(volume(primitive-container)for primitive in primitives)
        if loss>1e-12:raise ValueError('Preserved interface cuts the required inner backbone: '+str(loss))
        maxima.append(loss)
    result=unpack(packet)
    if len(material_components(solid(result)))!=1:raise ValueError('Preserved tree certificate is disconnected')
    mesh.vertices=result.vertices.copy();mesh.faces=result.faces.copy()
    return certified,dict(certified_inner_radius_m=certified,reference_radius_m=reference,
        outer_certificate_radius_m=radius,every_inner_ball_and_capsule_checked=True,
        maximum_inner_primitive_losses_m3=maxima,native_primitive_tolerance_m3=1e-12,
        connected_material_components=1,preserved_interface_boxes=boxes,
        interface_half_depth_m=.0001,interface_tangent_padding_m=.00002,
        original_force_and_geometry_tolerances_unchanged=True)
