"""Direct lateral sections of actual retained connecting members."""
import numpy as np
import shapely
from shapely.geometry import Point
from wrap_dimensions import MINIMUM_WIDTH_M

PITCH_M=.001
TOLERANCE_M=1e-6


def _intervals(geometry):
    if geometry.is_empty:return []
    if geometry.geom_type=='LineString':
        p=np.asarray(geometry.coords)[:,1];low,high=float(p.min()),float(p.max())
        return [(low,high)] if high-low>TOLERANCE_M else []
    if hasattr(geometry,'geoms'):return [v for g in geometry.geoms for v in _intervals(g)]
    return []


def intervals(geometry):
    result=[]
    for low,high in sorted(_intervals(geometry)):
        if result and low<=result[-1][1]+TOLERANCE_M:
            result[-1]=(result[-1][0],max(result[-1][1],high))
        else:result.append((low,high))
    return result


def sections(geometry,xs,y_low,y_high):
    """Exact polygon-edge scan lines, including closed vertical boundaries."""
    from wrap_width import polygons
    edges=[]
    for p in polygons(geometry):
        for ring in [p.exterior,*p.interiors]:
            points=np.asarray(ring.coords);edges.extend(zip(points[:-1],points[1:]))
    if not edges:return [[] for _ in xs]
    edges=np.asarray(edges);a,b=edges[:,0],edges[:,1]
    dx=b[:,0]-a[:,0];vertical=np.abs(dx)<1e-15
    va,vb=a[vertical],b[vertical]
    a,b,dx=a[~vertical],b[~vertical],dx[~vertical]
    low=np.minimum(a[:,0],b[:,0]);high=np.maximum(a[:,0],b[:,0]);slope=(b[:,1]-a[:,1])/dx
    output=[]
    for x in xs:
        crossing=(low<=x)&(x<high)
        ys=np.sort(a[crossing,1]+(x-a[crossing,0])*slope[crossing])
        spans=[(max(y_low,c),min(y_high,d)) for c,d in zip(ys[::2],ys[1::2])]
        at=np.abs(va[:,0]-x)<=1e-13
        spans.extend((max(y_low,min(c,d)),min(y_high,max(c,d)))
            for c,d in zip(va[at,1],vb[at,1]))
        merged=[]
        for c,d in sorted(spans):
            if d-c<=TOLERANCE_M:continue
            if merged and c<=merged[-1][1]+TOLERANCE_M:merged[-1]=(merged[-1][0],max(merged[-1][1],d))
            else:merged.append((float(c),float(d)))
        output.append(merged)
    return output


def profile(footprint,focus,low,high,y_low,y_high,minimum=MINIMUM_WIDTH_M,previous=None,extra_xs=()):
    from wrap_width import polygons
    if focus.is_empty or high-low<=TOLERANCE_M:
        return dict(passed=True,active=False,samples=0)
    events=[]
    for p in polygons(focus):
        for ring in [p.exterior,*p.interiors]:events.extend(np.asarray(ring.coords)[:,0])
    events=np.array(events)
    xs=np.unique(np.r_[np.linspace(low,high,max(2,int(np.ceil((high-low)/PITCH_M))+1)),
                       events-TOLERANCE_M,events+TOLERANCE_M,extra_xs])
    xs=xs[(xs>=low)&(xs<=high)]
    selected=sections(focus,xs,y_low,y_high);actual=sections(footprint,xs,y_low,y_high)
    groups=[];current=[]
    for x,seed,section in zip(xs,selected,actual):
        spans=seed
        if not spans:
            if current:groups.append(current);current=[]
            continue
        a,b=min(spans,key=lambda v:max(v[0],-v[1],0.))
        marker=(a+b)/2
        matching=[(c,d) for c,d in section if d>=a+TOLERANCE_M and c<=b-TOLERANCE_M]
        width=max((d-c for c,d in matching),default=0.)
        current.append((float(x),marker,width))
    if current:groups.append(current)
    checked=[];free_caps=0
    for group in groups:
        start,end=group[0][0],group[-1][0]
        left_free=not footprint.covers(Point(start-2*TOLERANCE_M,group[0][1]))
        right_free=not footprint.covers(Point(end+2*TOLERANCE_M,group[-1][1]))
        # At a junction the member turns into another member/contact plate;
        # that cap is not a shaft in this member's fixed width plane.
        left=minimum/2;right=minimum/2
        free_caps+=int(left_free)+int(right_free)
        shaft=[row for row in group if start+left-TOLERANCE_M<=row[0]<=end-right+TOLERANCE_M]
        if not shaft and not left_free and not right_free:
            # A short bridge joining material at both ends is still a
            # connection. End-cap exclusion must not hide its entire neck.
            shaft=[min(group,key=lambda row:abs(row[0]-(start+end)/2))]
        checked.extend(shaft)
    bad=[row for row in checked if row[2]<minimum-TOLERANCE_M]
    newly_bad=[];before=[]
    if previous is not None:
        old_sections=sections(previous,[r[0] for r in checked],y_low,y_high)
        for (x,y,w),old in zip(checked,old_sections):
            old_width=max((b-a for a,b in old if a-TOLERANCE_M<=y<=b+TOLERANCE_M),default=0.)
            before.append(old_width)
            if w<minimum-TOLERANCE_M and old_width>w+TOLERANCE_M:newly_bad.append((x,y,w))
    return dict(passed=not bad if previous is None else not newly_bad,absolute_passed=not bad,
        transition_passed=not newly_bad,active=bool(groups),samples=len(checked),
        axial_pitch_m=PITCH_M,critical_polygon_vertices_included=True,
        measured_minimum_width_m=min((row[2] for row in checked),default=None),
        previous_minimum_width_m=min(before,default=None),
        minimum_width_m=minimum,wall_thickness_constrained=False,
        free_terminal_caps=free_caps,attachment_cap_length_m=minimum/2,
        narrow_sections=[dict(axial_m=x,width_position_m=y,width_m=w) for x,y,w in bad],
        newly_narrow_sections=[dict(axial_m=x,width_position_m=y,width_m=w) for x,y,w in newly_bad],
        _widths=[(x,y,w,before[k] if previous is not None else None) for k,(x,y,w) in enumerate(checked)])


def feature_profiles(value,kept,feature,minimum=MINIMUM_WIDTH_M,previous=None,previous_cache=None,core=None):
    from deterministic_base import moved,unpack,box
    from wrap_width import projection
    if 'tracks' in feature:
        groups=[[track] for track in feature['tracks']]
    else:groups=[[dict(frame=T) for T in feature['frames']]]
    results=[]
    for group_id,group in enumerate(groups):
        alternatives=[];prepared=[];common_xs=[]
        for plane_id,spec in enumerate(group):
            T=np.asarray(spec['frame']);reference=moved(feature['solid'],T)
            bounds=unpack(reference).bounds
            low,high=spec.get('range',(bounds[0,0],bounds[1,0]))
            y_low,y_high=spec.get('width_window',(bounds[0,1]-minimum,bounds[1,1]+minimum))
            z_low,z_high=spec.get('depth_window',(bounds[0,2]-TOLERANCE_M,bounds[1,2]+TOLERANCE_M))
            region=box(np.array([low-minimum,y_low,z_low]),np.array([high+minimum,y_high,z_high]))
            local_kept=moved(kept,T)^region
            if 'tracks' in feature:
                # Assign this edge only its own 8 mm corridor. A nearby
                # differently oriented edge is measured in its own frame.
                local_kept=local_kept^box(np.array([low,-minimum/2,z_low]),
                                          np.array([high,minimum/2,z_high]))
            if local_kept.is_empty():continue
            focus=projection(local_kept)
            if core is not None and not core.is_empty():
                # Exempt the actual retained contact body in this plane.
                # A 3-D Boolean difference can manufacture nanometre boundary
                # sheets on the old 5 mm walls and change shaft end detection.
                trusted=projection(moved(core,T)^region)
                focus=focus.difference(trusted.buffer(TOLERANCE_M))
            if focus.is_empty:continue
            footprint=projection(moved(value,T)^region)
            before=None
            if previous is not None:
                key=(feature['name'],group_id,plane_id)
                if previous_cache is not None and key in previous_cache:before=previous_cache[key]
                else:
                    before=projection(moved(previous,T)^region)
                    if previous_cache is not None:previous_cache[key]=before
            from wrap_width import polygons
            for p in polygons(focus):
                for ring in [p.exterior,*p.interiors]:
                    xs=np.asarray(ring.coords)[:,0]
                    common_xs.extend(xs-TOLERANCE_M);common_xs.extend(xs+TOLERANCE_M)
            prepared.append((footprint,focus,low,high,y_low,y_high,before))
        for footprint,focus,low,high,y_low,y_high,before in prepared:
            row=profile(footprint,focus,low,high,y_low,y_high,minimum,before,common_xs)
            alternatives.append(row)
        active=[r for r in alternatives if r['active']]
        absolute=not active or any(r['absolute_passed'] for r in active)
        passed=absolute
        combined=None
        if len(active)>1:
            # A curved 5 mm sheet can rotate its width direction along the
            # connection. Select the width at EACH station, rather than
            # requiring one fixed viewing plane to be wide everywhere.
            stations={}
            for plane_id,a in enumerate(active):
                for x,y,w,old in a['_widths']:
                    key=round(x,10);entry=stations.setdefault(key,dict(x=x,width=0.,previous=0.,markers={}))
                    entry['width']=max(entry['width'],w)
                    entry['markers'][plane_id]=y
                    if old is not None:entry['previous']=max(entry['previous'],old)
            # Orthogonal projections can overstate a rotated square's width.
            # Resolve near-threshold stations on their real 2-D cross-section,
            # using its oriented major dimension, not its wall thickness.
            ambiguous=[r for r in stations.values() if r['width']<=minimum*np.sqrt(2)+TOLERANCE_M]
            if ambiguous:
                from wrap_width import section_geometry,polygons
                from deterministic_base import SCALE
                Q=np.eye(4);Q[:3,:3]=[[0,1,0],[0,0,1],[1,0,0]]
                frame=Q@np.asarray(group[0]['frame'])
                local_value=moved(value,frame);local_focus=moved(kept,frame)
                local_core=moved(core,frame) if core is not None else None
                local_previous=moved(previous,frame) if previous is not None else None
                for r in ambiguous:
                    seed=section_geometry(local_focus.slice(r['x']/SCALE))
                    if local_core is not None:
                        seed=seed.difference(section_geometry(local_core.slice(r['x']/SCALE)).buffer(TOLERANCE_M))
                    marker=Point(r['markers'].get(0,0.),r['markers'].get(1,0.))
                    def measured(shape):
                        section=section_geometry(shape.slice(r['x']/SCALE))
                        choices=[p for p in polygons(section) if p.intersection(seed).area>1e-14]
                        if not choices:return 0.
                        p=min(choices,key=lambda p:p.distance(marker))
                        if shape is local_value:r['section_bounds_yz_m']=list(p.bounds)
                        points=np.asarray(p.minimum_rotated_rectangle.exterior.coords)
                        return float(np.linalg.norm(np.diff(points,axis=0),axis=1).max())
                    r['width']=measured(local_value)
                    if local_previous is not None:r['previous']=measured(local_previous)
            bad=[r for r in stations.values() if r['width']<minimum-TOLERANCE_M]
            worse=[r for r in bad if r['previous']>r['width']+TOLERANCE_M]
            absolute=not bad;passed=absolute if previous is None else not worse
            combined=dict(passed=passed,absolute_passed=absolute,samples=len(stations),
                transverse_width_direction_may_rotate=True,wall_thickness_constrained=False,
                oriented_cross_sections_near_threshold=len(ambiguous),
                measured_minimum_width_m=min((r['width'] for r in stations.values()),default=None),
                narrow_sections=[dict(axial_m=r['x'],width_m=r['width'],
                    plane_positions_m=[r['markers'].get(0,0.),r['markers'].get(1,0.)],
                    **(dict(section_bounds_yz_m=r['section_bounds_yz_m']) if 'section_bounds_yz_m' in r else {})) for r in bad])
        elif previous is not None and active and not absolute:
            best=max((r.get('previous_minimum_width_m') or minimum) for r in active)
            passed=any(r['transition_passed'] and ((r.get('previous_minimum_width_m') or minimum)>=min(minimum,best)-TOLERANCE_M) for r in active)
        for row in alternatives:row.pop('_widths',None)
        results.append(dict(passed=passed,absolute_passed=absolute,alternatives=alternatives,
            **(dict(combined_transverse_width=combined) if combined is not None else {})))
    return dict(feature=feature['name'],kind=feature['kind'],passed=all(r['passed'] for r in results),
        absolute_passed=all(r['absolute_passed'] for r in results),
        groups=results,planes_checked=sum(len(g['alternatives']) for g in results))
