"""Measure planar feet and compound stock at connecting necks, not free corners.

Parallel sides and separated wide regions propose local neck directions. The
actual solid is sliced normal to each direction; its oriented major transverse
dimension is width. The other transverse dimension may be the 5 mm wall.
"""
import numpy as np
from shapely.geometry import Point,LineString
from shapely.ops import nearest_points,unary_union
from shapely.geometry.polygon import orient
from wrap_dimensions import MINIMUM_WIDTH_M

TOL=1e-6
PITCH=.001


def candidates(footprint,focus,minimum):
    from wrap_width import polygons
    edges=[]
    for p in polygons(footprint.simplify(TOL/4,preserve_topology=True)):
        p=orient(p,sign=1.)
        for ring in [p.exterior,*p.interiors]:
            points=np.asarray(ring.coords)
            for a,b in zip(points[:-1],points[1:]):
                length=np.linalg.norm(b-a)
                if length>TOL:edges.append((a,b,(b-a)/length,length))
    if not edges:return
    starts=np.array([e[0] for e in edges]);ends=np.array([e[1] for e in edges]);directions=np.array([e[2] for e in edges])
    seen=set();cosine=np.cos(np.deg2rad(2.))
    def emit(point,tangent):
        key=tuple(np.r_[np.round(point/PITCH,3),np.round(tangent,5)])
        if key in seen:return None
        seen.add(key);return point,tangent
    for i,(a,b,t,length) in enumerate(edges):
        n=np.array([-t[1],t[0]])
        ids=np.flatnonzero(directions@t < -cosine)
        for j in ids[ids>i]:
            coordinates=(np.array([starts[j],ends[j]])-a)@t
            low=max(0.,float(coordinates.min()));high=min(length,float(coordinates.max()))
            if high-low<TOL:continue
            at=(low+high)/2;fraction=(at-coordinates[0])/(coordinates[1]-coordinates[0])
            target=starts[j]+fraction*(ends[j]-starts[j]);source=a+at*t
            width=float((target-source)@n)
            if not TOL<width<minimum*np.sqrt(2)+TOL:continue
            middle=(source+target)/2
            if not footprint.covers(LineString([source+n*TOL,target-n*TOL])):continue
            if focus.distance(Point(middle))>minimum/2:continue
            first=middle+(low-at-2*TOL)*t;last=middle+(high-at+2*TOL)*t
            if high-low<minimum and (not footprint.covers(Point(middle-minimum/2*t))
                    or not footprint.covers(Point(middle+minimum/2*t))):
                continue
            left=0. if footprint.covers(Point(first)) else minimum/2
            right=0. if footprint.covers(Point(last)) else minimum/2
            lo,hi=low+left,high-right
            if hi<lo:
                # A terminal cap can taper; a short bridge attached at both
                # ends is still checked at its middle.
                if left or right:continue
                values=[at]
            else:values=np.linspace(lo,hi,max(1,int(np.ceil((hi-lo)/PITCH))+1))
            for x in values:
                fraction=(x-coordinates[0])/(coordinates[1]-coordinates[0])
                q=starts[j]+fraction*(ends[j]-starts[j]);p=a+x*t
                if not footprint.covers(LineString([p+n*TOL,q-n*TOL])):continue
                row=emit((p+q)/2,t)
                if row is not None:yield row
    # Nonparallel hourglass necks also separate the eroded wide regions.
    radius=(minimum-TOL)/2
    for p in polygons(footprint):
        wide=polygons(p.buffer(-radius,quad_segs=32))
        for i,first in enumerate(wide):
            for second in wide[i+1:]:
                a,b=nearest_points(first,second);a=np.array(a.coords[0]);b=np.array(b.coords[0])
                distance=np.linalg.norm(b-a)
                if distance<TOL:continue
                if not p.buffer(TOL/4).covers(LineString([a,b])):
                    continue
                tangent=(b-a)/distance
                for f in np.linspace(0,1,max(2,int(np.ceil(distance/PITCH))+1)):
                    point=a+f*(b-a)
                    if not p.covers(Point(point)) or focus.distance(Point(point))>minimum/2:continue
                    row=emit(point,tangent)
                    if row is not None:yield row


def assess(value,kept,feature,core,minimum=MINIMUM_WIDTH_M,previous=None):
    from deterministic_base import moved,unpack,box,SCALE
    from wrap_width import projection,section_geometry,polygons
    bad=[];worse=[];count=0;minimum_measured=None
    for plane_id,T in enumerate(feature['frames']):
        T=np.asarray(T);reference=moved(feature['solid'],T);bounds=unpack(reference).bounds
        low=bounds[0].copy();high=bounds[1].copy();low[:2]-=minimum;high[:2]+=minimum
        low[2]-=TOL;high[2]+=TOL
        region=box(low,high);focus=projection(moved(kept,T)^region)
        if focus.is_empty:continue
        footprint=projection(moved(value,T)^region)
        trusted=projection(moved(core,T)^region)
        focus=focus.difference(trusted.buffer(TOL))
        if focus.is_empty:continue
        for point,tangent in candidates(footprint,focus,minimum):
            normal=np.array([-tangent[1],tangent[0]])
            Q=np.eye(4);Q[:3,:3]=[[normal[0],normal[1],0],[0,0,1],[tangent[0],tangent[1],0]]
            Q[0,3]=-float(normal@point);Q[2,3]=-float(tangent@point)
            frame=Q@T
            local_value=moved(value,frame)
            actual=section_geometry(local_value.slice(0.))
            inherited=section_geometry(moved(core,frame).slice(0.))
            seed=section_geometry(moved(kept,frame).slice(0.)).difference(inherited.buffer(TOL))
            if seed.is_empty:continue
            line=LineString([(0,low[2]-minimum),(0,high[2]+minimum)])
            for p in polygons(actual):
                contact=p.intersection(seed)
                crossing=p.intersection(line)
                if contact.area<=1e-14 or crossing.is_empty:continue
                markers=[]
                for part in getattr(crossing,'geoms',[crossing]):
                    if part.geom_type=='LineString' and part.length>TOL:
                        coords=np.asarray(part.coords);markers.append(Point(0,float((coords[:,1].min()+coords[:,1].max())/2)))
                if not markers or all(inherited.buffer(TOL).covers(m) for m in markers):continue
                vertices=np.asarray(p.minimum_rotated_rectangle.exterior.coords)
                width=float(np.linalg.norm(np.diff(vertices,axis=0),axis=1).max())
                # Hull bands have a specified lateral construction width;
                # general sheets may turn their width direction in 3-D.
                count+=1;minimum_measured=width if minimum_measured is None else min(minimum_measured,width)
                if width>=minimum-TOL:continue
                # A projected corner can coincide with another member in the
                # view. Check this actual 3-D section at both sides of the
                # 4 mm cap before calling it a connecting neck.
                left=section_geometry(local_value.slice(-minimum/2/SCALE))
                right=section_geometry(local_value.slice(minimum/2/SCALE))
                if left.intersection(p).area<=1e-14 or right.intersection(p).area<=1e-14:
                    continue
                old_width=0.
                if previous is not None:
                    old=section_geometry(moved(previous,frame).slice(0.))
                    choices=[q for q in polygons(old) if q.intersection(contact).area>1e-14]
                    if choices:
                        q=min(choices,key=lambda q:q.distance(markers[0]))
                        pts=np.asarray(q.minimum_rotated_rectangle.exterior.coords)
                        old_width=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).max())
                # Convert the slicing frame to the pruning helper's X-axis
                # convention, preserving this actual measured section.
                K=np.eye(4);K[:3,:3]=[[0,0,1],[1,0,0],[0,1,0]]
                row=dict(axial_m=0.,width_m=width,previous_width_m=old_width,
                    plane_positions_m=[0.,float(markers[0].y)],section_bounds_yz_m=list(p.bounds),
                    cut_frame_world_to_local=(K@frame).tolist(),plane=plane_id)
                bad.append(row)
                if previous is not None and old_width>width+TOL:worse.append(row)
    absolute=not bad;passed=absolute if previous is None else not worse
    summary=dict(passed=passed,absolute_passed=absolute,samples=count,minimum_width_m=minimum,
        measured_minimum_width_m=minimum_measured,wall_thickness_constrained=False,
        free_corners_are_not_connecting_necks=True,narrow_sections=bad,newly_narrow_sections=worse)
    return dict(feature=feature['name'],kind=feature['kind'],passed=passed,absolute_passed=absolute,
        planes_checked=len(feature['frames']),groups=[dict(passed=passed,absolute_passed=absolute,
            alternatives=[summary],combined_transverse_width=summary)])
