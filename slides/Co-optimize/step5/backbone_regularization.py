"""Remove detached Boolean shells only after proving an inner round tree."""
import numpy as np
from deterministic_base import solid,unpack,volume
from prune_support import material_components
from anchor_network import ball
from tube_support import capsule


def regularize(mesh,nodes,tree_paths,radius,reference_radius):
    parts=material_components(solid(mesh))
    if len(parts)==1:return mesh,radius,None
    main=parts[0];others=[volume(p)for p in parts[1:]]
    if sum(others)>1e-9:raise ValueError('Backbone contains a substantial disconnected component')
    certified_radius=radius-.00005
    if certified_radius<=reference_radius:raise ValueError('Inner round tree is below the reference width')
    losses=[]
    for point in nodes:losses.append(volume(ball(point,certified_radius)-main))
    segments=0
    for a,b,path in tree_paths:
        for first,last in zip(path,path[1:]):
            losses.append(volume(capsule(first,last,certified_radius)-main));segments+=1
    maximum=max(losses,default=0.)
    if maximum>1e-12:raise ValueError('Detached component contains part of the required inner round tree')
    result=unpack(main)
    if len(material_components(solid(result)))!=1:raise ValueError('Regularized inner-tree certificate is disconnected')
    return result,certified_radius,dict(discarded_detached_shell_volumes_m3=others,
        outer_design_radius_m=radius,certified_inner_radius_m=certified_radius,
        reference_radius_m=reference_radius,nodes=len(nodes),segments=segments,
        maximum_missing_inner_primitive_m3=maximum,inner_primitive_tolerance_m3=1e-12,
        every_inner_ball_and_capsule_checked=True,connected_material_components=1)
