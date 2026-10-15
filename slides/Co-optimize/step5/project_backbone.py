"""Construct a coarse round certificate inside both actual saved supports."""
import numpy as np
import manifold3d as md
from deterministic_base import solid,unpack,union,volume,SCALE
from prune_support import material_components
from anchor_network import ball,path_body
from tube_support import capsule


def coarse_ball(point,radius):
    return md.Manifold.sphere(radius/SCALE,24).translate(tuple(np.asarray(point)/SCALE))


def coarse_path(path,radius):
    return union([md.Manifold.batch_hull([coarse_ball(a,radius),coarse_ball(b,radius)])
        for a,b in zip(path,path[1:])])


def project(mesh,full,intermediate,nodes,tree_paths,radius,reference,domain,*,proof_inset_m=.00005):
    if proof_inset_m<.00005:raise ValueError('Inner proof must stay inside the outer certificate')
    certified=radius-proof_inset_m
    if certified<=reference:raise ValueError('Projected backbone falls below reference radius')
    working=union([path_body(path,radius+.0002)for a,b,path in tree_paths]+
        [ball(nodes[0],radius+.0002)])^domain
    outputs=[];removed=[]
    for body in [full,intermediate]:
        value=(solid(body)+working).simplify(0.)
        parts=material_components(value)
        if len(parts)>1:
            ranked=[]
            for part in parts:
                losses=[volume(ball(c,certified)-part)for c in nodes]
                ranked.append((max(losses,default=0.),-volume(part),part))
            loss,_,value=min(ranked,key=lambda r:r[:2])
            if loss>1e-12:raise ValueError('No connected product component contains all interior anchors')
            removed.append([volume(p)for p in parts if p is not value])
        result=unpack(value);body.vertices=result.vertices.copy();body.faces=result.faces.copy();outputs.append(value)
    value=coarse_ball(nodes[0],radius);attached={0};pending=list(tree_paths)
    while pending:
        i=next((i for i,(a,b,path)in enumerate(pending)if a in attached or b in attached),None)
        if i is None:raise ValueError('Certificate traversal lost a node')
        a,b,path=pending.pop(i);value=(value+coarse_path(path,radius)).simplify(0.);attached.update([a,b])
    value=(value^outputs[0]^outputs[1]).simplify(0.)
    pieces=material_components(value)
    if not pieces:raise ValueError('Round certificate has no material inside actual supports')
    packet=pieces[0]
    primitives=[ball(c,certified)for c in nodes]
    primitives.extend(capsule(a,b,certified)for first,last,path in tree_paths for a,b in zip(path,path[1:]))
    maxima=[]
    for container in [packet]+outputs:
        maximum=max((volume(primitive-container)for primitive in primitives),default=0.)
        if maximum>1e-12:raise ValueError('Actual support misses a required full-radius inner capsule: '+str(maximum))
        maxima.append(maximum)
    result=unpack(packet)
    if len(material_components(solid(result)))!=1:raise ValueError('Projected certificate remains disconnected')
    mesh.vertices=result.vertices.copy();mesh.faces=result.faces.copy()
    return certified,dict(certified_inner_radius_m=certified,reference_radius_m=reference,proof_inset_m=proof_inset_m,
        outer_certificate_radius_m=radius,certificate_sphere_segments=24,
        every_inner_ball_and_capsule_checked=True,maximum_inner_primitive_losses_m3=maxima,
        native_primitive_tolerance_m3=1e-12,connected_material_components=1,
        discarded_unattached_added_material_components_m3=removed,
        actual_product_reinforcement_radius_m=radius+.0002)
