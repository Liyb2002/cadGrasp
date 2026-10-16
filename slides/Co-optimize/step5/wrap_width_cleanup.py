"""Delete optional narrow construction stock from the current actual solid.

These are proposals, never a retained-material mask. Every proposal goes
through the same connectivity, legal-space, floor, force and width guards.
"""
import numpy as np
from wrap_dimensions import MINIMUM_WIDTH_M


def cleanup(deleter, rounds=3):
    from deterministic_base import box,moved,volume,union,unpack
    from wrap_width import projection,footprint_solid,WIDTH_TOLERANCE_M
    from wrap_dimensions import physical_component_count
    guard=deleter.validator.width_guard
    if guard is None:return
    def propose(cut,operation,detail):
        candidate=deleter.value-cut
        if deleter.attempt(candidate,operation,detail):return True
        inherited=deleter.value^guard.core
        protected=candidate+inherited
        if volume(protected-candidate)>1e-13:
            if deleter.attempt(protected,operation+'_preserve_contact_body',detail):return True
            if physical_component_count(protected)>1:
                pieces=[p for p in protected.decompose() if float(p.volume())>0]
                main=max(pieces,key=lambda p:float(p.volume()))
                if deleter.attempt(main,operation+'_preserve_body_and_delete_dependents',
                    dict(detail,proposed_components=len(pieces))):return True
        if candidate.is_empty() or physical_component_count(candidate)==1:return False
        # A second, explicitly larger deletion also removes every dependent
        # branch separated by this cut. Only its one-piece result is tested;
        # no disconnected checkpoint is ever accepted or hidden by a cutoff.
        pieces=[p for p in candidate.decompose() if float(p.volume())>0]
        if not pieces:return False
        main=max(pieces,key=lambda p:float(p.volume()))
        return deleter.attempt(main,operation+'_and_dependent_branches',
            dict(detail,proposed_components=len(pieces),dependent_material_removed_m3=volume(candidate-main)))
    def measured_cuts(assessment,value):
        from wrap_width import section_geometry,polygons
        from deterministic_base import SCALE
        byname={f['name']:f for f in guard.features};cuts=[]
        for result in assessment['features']:
            if result['absolute_passed']:continue
            feature=byname[result['feature']]
            for group_id,group in enumerate(result['groups']):
                if group['absolute_passed']:continue
                if 'tracks' in feature:
                    track=feature['tracks'][group_id];T=np.asarray(track['frame'])
                    for sample in group['alternatives'][0].get('narrow_sections',[]):
                        x=sample['axial_m'];y=sample['width_position_m'];zlo,zhi=track['depth_window']
                        cuts.append(moved(box(np.array([x-.004,y-.004,zlo]),np.array([x+.004,y+.004,zhi])),np.linalg.inv(T)))
                    continue
                for sample in group.get('combined_transverse_width',{}).get('narrow_sections',[]):
                    T=np.asarray(sample.get('cut_frame_world_to_local',feature['frames'][0]));x=sample['axial_m']
                    bounds=sample.get('section_bounds_yz_m')
                    sections=[]
                    if bounds is not None:sections=[bounds]
                    else:
                        Q=np.eye(4);Q[:3,:3]=[[0,1,0],[0,0,1],[1,0,0]]
                        actual=section_geometry(moved(value,Q@T).slice(x/SCALE))
                        for p in polygons(actual):
                            vertices=np.asarray(p.minimum_rotated_rectangle.exterior.coords)
                            if np.linalg.norm(np.diff(vertices,axis=0),axis=1).max()<MINIMUM_WIDTH_M-1e-6:sections.append(p.bounds)
                    for a,b,c,d in sections:
                        cuts.append(moved(box(np.array([x-.004,a-1e-6,b-1e-6]),np.array([x+.004,c+1e-6,d+1e-6])),np.linalg.inv(T)))
        return cuts
    for iteration in range(rounds):
        check=guard.assess(deleter.mesh)
        if check['passed']:return
        # Remove a measured narrow member in one proposal, including any new
        # narrow cut ends. Shaving an already narrow fin one millimetre at a
        # time would violate the same width guard before it disappears.
        mask=union([]);candidate=deleter.value
        for refinement in range(4):
            assessment=guard.assess(candidate)
            cuts=measured_cuts(assessment,candidate)
            if not cuts:break
            mask=mask+union(cuts)
            if propose(mask,'remove_narrow_region_group',dict(iteration=iteration,refinement=refinement,measured_regions=len(cuts))):
                break
            candidate=(deleter.value-mask)+(deleter.value^guard.core)
        check=guard.assess(deleter.mesh)
        if check['passed']:return
        failures={r['feature']:r for r in check['features'] if not r['passed']}
        changed=False
        features=sorted((f for f in guard.features if f['name'] in failures),
                        key=lambda f:volume((f['solid']-guard.core)^deleter.value))
        narrow_cuts=[]
        for feature in features:
            optional=feature['solid']-guard.core
            detail=dict(feature=feature['name'],iteration=iteration)
            result=failures[feature['name']]
            # Target actual narrow sections on a compound connection without
            # removing its entire enclosing construction box or old wall.
            if 'tracks' not in feature:
                local_changed=False
                for group in result['groups']:
                    samples=group.get('combined_transverse_width',{}).get('narrow_sections',[])
                    for sample in sorted(samples,key=lambda s:s['width_m'])[:12]:
                        T=np.asarray(sample.get('cut_frame_world_to_local',feature['frames'][0]));x=sample['axial_m']
                        yz=sample.get('plane_positions_m',[0.,0.])
                        bounds=sample.get('section_bounds_yz_m',[yz[0]-.004,yz[1]-.004,yz[0]+.004,yz[1]+.004])
                        for depth in (.001,.004):
                            region=moved(box(np.array([x-depth,bounds[0]-1e-6,bounds[1]-1e-6]),
                                np.array([x+depth,bounds[2]+1e-6,bounds[3]+1e-6])),np.linalg.inv(T))
                            if propose(region,'remove_narrow_cross_section',dict(detail,axial_m=x,depth_m=depth)):
                                changed=True;local_changed=True;break
                        if local_changed:break
                    if local_changed:break
                if local_changed:continue
            regularized=False
            for plane_id,frame in enumerate(feature['frames']):
                frame=np.asarray(frame);local=moved(feature['solid']^deleter.value,frame)
                if local.is_empty():continue
                p=projection(local);radius=(MINIMUM_WIDTH_M-WIDTH_TOLERANCE_M)/2
                opened=p.buffer(-radius,quad_segs=32).buffer(radius,quad_segs=32)
                narrow=p.difference(opened.buffer(WIDTH_TOLERANCE_M/4))
                if narrow.is_empty:continue
                zlo,zhi=unpack(local).bounds[:,2]
                region=moved(footprint_solid(narrow.buffer(WIDTH_TOLERANCE_M/4),
                    zlo-WIDTH_TOLERANCE_M,zhi+WIDTH_TOLERANCE_M),np.linalg.inv(frame))
                if propose(region,'remove_narrow_planar_branches',dict(detail,plane=plane_id)):
                    changed=True;regularized=True;break
            if regularized:continue
            T=np.asarray(feature['frames'][0]);local=moved(feature['solid'],T)
            zlo,zhi=unpack(local).bounds[:,2]
            footprint=projection(local).buffer(WIDTH_TOLERANCE_M,join_style=2)
            whole=moved(footprint_solid(footprint,zlo-WIDTH_TOLERANCE_M,zhi+WIDTH_TOLERANCE_M),np.linalg.inv(T))
            if propose(whole,'remove_narrow_feature',detail):
                changed=True;continue
            if 'tracks' not in feature:
                narrow_cuts.append(whole);continue
            # Keep the actual wide portions of a wrapped hull whenever a
            # whole-feature deletion would lose required floor coverage.
            for track_id,(track,group) in enumerate(zip(feature['tracks'],result['groups'])):
                if group['absolute_passed']:continue
                T=np.asarray(track['frame']);lo,hi=track['range']
                zlo,zhi=track['depth_window']
                bad=group['alternatives'][0].get('narrow_sections',[]) if group['alternatives'] else []
                spans=[]
                if bad:
                    spans.append((min(v['axial_m'] for v in bad)-MINIMUM_WIDTH_M/2,
                                  max(v['axial_m'] for v in bad)+MINIMUM_WIDTH_M/2))
                spans.append((lo,hi))
                for a,b in spans:
                    for half_width in (MINIMUM_WIDTH_M/2,MINIMUM_WIDTH_M,1.5*MINIMUM_WIDTH_M,3*MINIMUM_WIDTH_M):
                        region=moved(box(np.array([a-1e-6,-half_width,zlo]),
                                         np.array([b+1e-6,half_width,zhi])),np.linalg.inv(T))
                        cut=region
                        if (a,b)==spans[0] and half_width==MINIMUM_WIDTH_M/2:narrow_cuts.append(cut)
                        if propose(cut,'remove_narrow_hull_arc',
                            dict(detail,track=track_id,axial_range_m=[a,b],half_width_m=half_width)):
                            changed=True;break
        if not changed and narrow_cuts:
            changed=propose(union(narrow_cuts),'remove_narrow_arc_group',
                dict(features=[f['name'] for f in features],iteration=iteration))
        if not changed and len(features)>1:
            cut=union([f['solid'] for f in features])
            changed=propose(cut,'remove_narrow_feature_group',
                dict(features=[f['name'] for f in features],iteration=iteration))
        if not changed:return
