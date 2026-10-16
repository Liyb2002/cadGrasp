"""Widen initial clipped connections laterally, retaining the original 5 mm wall.

This is part of construction: every action only grows the same solid, remains
inside the original legal space, and is
saved for movie01. No material is added after pruning begins.
"""
import numpy as np
from shapely.ops import unary_union
from wrap_width import (ConnectionWidthGuard,projection,footprint_solid,polygons,
                        WIDTH_TOLERANCE_M,AREA_TOLERANCE_M2)
from wrap_dimensions import MINIMUM_WIDTH_M


def widen_initial(data,mesh,value,proof,checks,validator,out,history):
    from feasible_geometry import C
    from deterministic_base import moved,unpack,volume,box,SCALE,solid
    from exact_mesh import save_mesh
    from prune_support import material_components
    from deterministic_base import union
    from wrap_dimensions import physical_component_count
    baseline=data['fixture'];features=data['width_features']
    bounds=np.array(checks['bounds_world_m'])
    domain=validator.witness-validator.objects
    frames={np.round(r['fixture_transform'],11).tobytes():r['fixture_transform'] for r in data['rows']}
    serial=max(int(p.stem) for p in (out/'construction').glob('*.obj'))+1
    for iteration in range(4):
        guard=ConnectionWidthGuard(baseline,features)
        assessment=guard.assess(mesh)
        if assessment['passed']:break
        names={r['feature'] for r in assessment['features'] if not r['passed']}
        changed=False
        for feature in features:
            if feature['name'] not in names:continue
            candidates=[]
            for T in feature['frames']:
                local=moved(feature['solid'],T);p=projection(local)
                if p.is_empty:continue
                radius=(MINIMUM_WIDTH_M-WIDTH_TOLERANCE_M)/2
                opened=p.buffer(-radius,quad_segs=32).buffer(radius,quad_segs=32)
                loss=p.difference(opened).buffer(-WIDTH_TOLERANCE_M/2).buffer(WIDTH_TOLERANCE_M/2)
                significant=[]
                for q in polygons(loss):
                    span=float(np.linalg.norm(np.subtract(q.bounds[2:],q.bounds[:2])))
                    if q.area>max(AREA_TOLERANCE_M2,span*WIDTH_TOLERANCE_M):significant.append(q)
                repair=unary_union(significant)
                if repair.is_empty:repair=p
                footprint=p.buffer(MINIMUM_WIDTH_M/2,quad_segs=16).intersection(repair.buffer(MINIMUM_WIDTH_M,quad_segs=16))
                low,high=unpack(local).bounds[:,2]
                proposal=moved(footprint_solid(footprint,low,high),np.linalg.inv(T)) ^ domain
                attached=[];grown=value
                for part in material_components(proposal):
                    if volume(part ^ grown)<=1e-13:continue
                    candidate=grown+part
                    if physical_component_count(candidate)==1:
                        attached.append(part);grown=candidate
                proposal=union(attached)
                added=volume(proposal-value)
                if added>1e-13:candidates.append((added,proposal,T))
            if not candidates:continue
            # First try the smaller legal lateral reinforcement. Its depth
            # remains exactly the feature's existing depth in this frame.
            added,proposal,T=min(candidates,key=lambda p:p[0])
            path=out/'candidate.obj'
            try:
                candidate_mesh=save_mesh(unpack((value+proposal).simplify(0.)),path)
                candidate,certificate,accepted=validator.check(candidate_mesh,width=False)
                if volume(value-candidate)>1e-10:raise ValueError('Width growth removed prior material')
            except (ValueError,RuntimeError) as error:
                history.append(dict(operation='lateral_reinforcement_rejected',feature=feature['name'],reason=str(error)))
                continue
            value,mesh,proof,checks=candidate,candidate_mesh,certificate,accepted
            feature['solid']=feature['solid']+proposal
            target=out/f'construction/{serial:03d}.obj';save_mesh(mesh,target);serial+=1
            history.append(dict(operation='lateral_connection_reinforcement',feature=feature['name'],
                iteration=iteration,added_cm3=added*1e6,material_cm3=checks['material_cm3'],
                plane_world_to_local=T.tolist(),wall_thickness_constrained=False,
                original_contact_body_thickened=False,bbx_l=checks['bbx_l'],mesh=C.relative(target)))
            print('WIDEN_CONNECTION',feature['name'],round(added*1e6,4),flush=True)
            changed=True
        if not changed:break
    guard=ConnectionWidthGuard(baseline,features)
    mesh=save_mesh(mesh,out/'initial_support.obj')
    guard.save(out)
    guard=ConnectionWidthGuard.load(out,baseline);assessment=guard.assess(mesh)
    return mesh,value,proof,checks,history,guard,assessment
