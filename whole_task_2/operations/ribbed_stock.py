"""Retain the fitted contact skin and replace solid casting fill with posts.

All posts are real material. Working/exit exclusions, all installation floors,
connectivity and the original complete load certificate are checked afterward.
"""
from itertools import product
import numpy as np
from whole_task.step3 import independent_geometry as G
from .geometry import moved_mesh


def contact_skin(model):
    if not hasattr(model,'source'):return model.outer
    if not hasattr(model,'complete_contact_skin'):
        from whole_task.step1.common import ROOT
        import trimesh
        path=ROOT/'whole_task/data'/model.name/'step4_contact_skin_5mm.npz'
        if path.exists():
            with np.load(path)as z:
                np.testing.assert_array_equal(z['source_vertices'],model.source.vertices)
                np.testing.assert_array_equal(z['source_faces'],model.source.faces)
                assert float(z['box_halfwidth_m'])==.005
                skin=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        else:
            value=G.solid(model.source).minkowski_sum(G.box([-.005]*3,[.005]*3))
            skin=G.mesh(value)
            np.savez_compressed(path,vertices=skin.vertices,faces=skin.faces,
                source_vertices=model.source.vertices,source_faces=model.source.faces,box_halfwidth_m=.005)
        model.complete_contact_skin=skin
        if path not in model.inputs:model.inputs.append(path)
    return model.complete_contact_skin


def world_stock(model,data,pose,pitch,width):
    cache=getattr(model,'ribbed_stock_cache',None)
    if cache is None:model.ribbed_stock_cache=cache={}
    key=(pose.tobytes(),float(pitch),float(width))
    if key not in cache:
        dense=data['stock_world']
        lo,hi=dense.bounds
        # The fitted vertex-offset skin can have zero outward thickness on
        # sharp concave faces. A nonconvex Minkowski neighborhood preserves a
        # finite material layer around EVERY original contact face. The new
        # design may regenerate this skin: volume and legality are checked
        # on the complete result, rather than requiring a stock subset.
        skin=G.solid(moved_mesh(contact_skin(model),pose))
        base=G.box([lo[0]-.001,lo[1]-.001,0.],
                   [hi[0]+.001,hi[1]+.001,model.policy['plate_thickness_m']])
        axes=[np.linspace(lo[k]+width/2,hi[k]-width/2,max(2,int(np.ceil((hi[k]-lo[k])/pitch))+1))
              for k in(0,1)]
        posts=[G.box([x-width/2,y-width/2,0.],[x+width/2,y+width/2,hi[2]+.001])
               for x,y in product(*axes)]
        value=G.union([skin,G.solid(dense)^G.union([base,*posts])])
        cache[key]=G.mesh(value)
    return cache[key]
