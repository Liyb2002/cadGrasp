"""Independent finite-width connectors to immutable Step 4 bearing pads."""
from itertools import combinations,product
import numpy as np
from step1.needs import COORD
from step2_local_support import insertion as D
from step5_connect_support import layout as L, routing as T, solids as S, motion as M

DEPTH_FACTORS=(1., .75, .5, .25, .125)


def pad_parts(foot):
    parts=[]
    for polygon in foot['pads_xy_m']:
        xy=np.asarray(polygon,float);bottom=COORD.lift_floor(xy)
        parts.append(D.engine.hull_mesh(np.vstack([bottom,bottom+[0,0,foot['height_m']]])))
    return parts


def angles(allowed):
    values=L.candidate_angles(allowed)
    values.extend(a for a in (0.,90.,180.,270.) if D.contains(allowed,a))
    return sorted(set(values),key=lambda a:(min(abs((a-b+180)%360-180) for b in (0,90,180,270)),a))


def build_one(mesh,contact,direction,foot,depth,work,edge_budget=800):
    """No sibling geometry or sibling success is required to construct this part."""
    scale=float(mesh.extents.max());cid=contact['candidate_id']
    assert foot['candidate_id']==cid and foot['fixed_for_step5']
    pads=pad_parts(foot)
    report=dict(candidate_id=cid,passed=False,status='not_constructed',footprint_unchanged=True,
                attempts=[],failure_is_global_impossibility_proof=False)
    surface=work.check_surface(contact['triangles_m'],contact['source_faces'])
    report['contact_surface_check']=surface
    if not surface['passed']:
        report['status']='contact_surface_clearance_not_verified';return report,None
    pad_check=work.check_parts(pads)
    report['fixed_pad_work_volume_check']=pad_check
    if not pad_check['passed']:
        report['status']='fixed_pads_clearance_not_verified';return report,None
    possible=[]
    for angle in angles(direction['certified_directions']):
        vector=T.G.frame(angle)[0]
        if M.sweep_check(mesh,pads,vector)['passed']:possible.append(angle)
    report['angles_with_clear_fixed_pad_sweep_deg']=possible
    if not possible:
        report['status']='no_fixed_pad_insertion_in_angle_menu';return report,None
    backings=[]
    for factor in DEPTH_FACTORS:
        heads=D.Analyzer(mesh,depth*factor).heads(contact)
        check=work.check_parts(heads)
        if check['passed']:backings.append((factor,heads))
    if not backings:
        report['status']='no_verified_contact_backing';return report,None
    # All non-pad material must stay above the floor, so the saved Step 4
    # footprint remains the complete actual bearing region.
    backings=[(factor,heads) for factor,heads in backings
              if all(head.vertices[:,2].min()>scale*1e-10 for head in heads)]
    if not backings:
        report['status']='contact_backing_adds_unrecorded_floor_contact';return report,None
    # Try inexpensive shapes across the whole angle menu before spending a
    # roadmap budget on an early difficult angle. Reuse verified edges in both
    # passes; the fixed pads and all final acceptance checks stay identical.
    routers={angle:T.Router(mesh,work,angle,edge_budget=edge_budget) for angle in possible}
    for allow_roadmap,angle in product((False,True),possible):
        router=routers[angle]
        for factor,heads in backings:
            attempt=dict(bearing_deg=angle,backing_depth_factor=factor,connected_pad_count=0,
                         spatial_search_enabled=allow_roadmap)
            report['attempts'].append(attempt)
            routes=[];bars=[];labels=[]
            for index,polygon in enumerate(foot['pads_xy_m']):
                anchor=np.asarray(polygon).mean(axis=0)
                route=router.connect(heads,anchor,float(foot['height_m']),allow_roadmap=allow_roadmap)
                if route is None:
                    attempt['failed_pad_index']=index;break
                routes.append(dict(pad_index=index,**route['record']))
                bars.extend(route['parts']);labels.extend(f'routed_bar_{index:02d}_{k:03d}' for k in range(len(route['parts'])))
                attempt['connected_pad_count']+=1
            print('  ',cid,'angle',angle,'backing',factor,'roadmap',allow_roadmap,
                  'connected pads',len(routes),'/',len(pads),flush=True)
            if len(routes)!=len(pads):continue
            if any(part.vertices[:,2].min()<=scale*1e-10 for part in bars):
                attempt['rejection']='connector_adds_unrecorded_floor_contact';continue
            parts=list(heads)+bars+pads
            labels=[f'contact_head_{k:03d}' for k in range(len(heads))]+labels+[f'ground_pad_{k:03d}' for k in range(len(pads))]
            try:
                joined,solid=S.union_parts(parts,scale)
                if not solid['one_solid']:
                    attempt['rejection']='disconnected_or_nonmanifold_union';continue
                sweep=M.sweep_check(mesh,parts,router.direction)
                if not sweep['passed']:
                    attempt['rejection']='complete_object_sweep';continue
                clearance=work.check_parts(parts,labels)
                if not clearance['passed']:
                    attempt['rejection']='complete_work_volume_clearance';continue
                from step5_connect_support.surface_check import surface_distances
                interface=np.unique(np.vstack([contact['triangles_m'].reshape(-1,3),contact['triangles_m'].mean(axis=1)]),axis=0)
                distance=float(surface_distances(joined,interface).max())
                if distance>scale*1e-9:
                    attempt['rejection']='contact_interface_not_preserved';continue
            except (ValueError,RuntimeError) as error:
                attempt['rejection']=str(error);continue
            corners=COORD.lift_floor(np.concatenate(foot['pads_xy_m']))
            plan=dict(candidate_id=cid,bearing_deg=angle,direction=router.direction,
                backing_depth_factor=factor,routes=routes,ground_height_m=foot['height_m'],
                ground_polygons_xy_m=[np.asarray(p) for p in foot['pads_xy_m']],ground_corners_m=corners,
                ground_area_m2=foot['bearing_area_m2'],anchor_xy_m=np.asarray(foot['center_xy_m']),
                construction='Retained contact backing and routed bars to every fixed Step 4 pad')
            report.update(passed=True,status='individual_connection_and_insertion_verified',
                bearing_deg=angle,backing_depth_factor=factor,solid=solid,
                object_sweep=sweep,work_volume_check=clearance,maximum_contact_distance_m=distance)
            return report,dict(plan=plan,parts=parts,labels=labels,joined=joined,solid=solid)
    report['status']='no_connector_in_fixed_foot_search_menu'
    return report,None


def compatible_subset(assembly,count):
    """Keep independently valid solids even when a pair or assembly order fails.

    Exhaustive subset selection for <=12 parts; deterministic greedy fallback for
    larger sets. This chooses among constructed parts, not all possible routes.
    """
    tolerance=assembly['volume_tolerance_m3']
    eligible=[i for i in range(count)
              if all(not assembly.get(key) or assembly[key][i]['passed']
                     for key in ('object_sweeps','final_work_volume_checks'))]
    conflicts={tuple(sorted((p['first'],p['second']))) for p in assembly['pair_checks'] if p['final_intersection_m3']>tolerance}
    edges=assembly['precedence_edges']
    def check(ids):
        if any(a in ids and b in ids for a,b in conflicts):return None
        mapping={old:new for new,old in enumerate(ids)}
        local=[(mapping[a],mapping[b]) for a,b in edges if a in mapping and b in mapping]
        order=L.installation_order(len(ids),local)
        return None if order is None else [ids[k] for k in order]
    if len(eligible)<=12:
        for size in range(len(eligible),0,-1):
            for ids in combinations(eligible,size):
                order=check(ids)
                if order is not None:return list(ids),order
    else:
        selected=[]
        for i in eligible:
            if check(selected+[i]) is not None:selected.append(i)
        return selected,check(selected)
    return [],[]


def search(mesh,contacts,directions,feet,depth,work,builder=build_one,on_result=None):
    if len(directions)!=len(contacts):
        raise ValueError('Every contact must have an insertion-direction record')
    lookup={f['candidate_id']:f for f in feet};records=[];modules=[]
    for contact,direction in zip(contacts,directions):
        cid=contact['candidate_id']
        if cid not in lookup:
            record,module=dict(candidate_id=cid,passed=False,status='missing_step4_footprint'),None
        else:
            try:
                record,module=builder(mesh,contact,direction,lookup[cid],depth,work)
            except (ValueError,RuntimeError) as error:
                record,module=dict(candidate_id=cid,passed=False,status='individual_geometry_unresolved',reason=str(error)),None
        records.append(record)
        if module is not None:modules.append(module)
        if on_result is not None:on_result(record,module)
        print(cid,record['status'],flush=True)
    return records,modules
