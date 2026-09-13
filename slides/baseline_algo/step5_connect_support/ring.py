"""A narrow convex boundary ring, partitioned into separately owned ground arcs.

All lengths are metres in the Step 1 world frame. The ring lies OUTSIDE the
expanded demand polygon. Convex edge strips and convex clipping cells preserve
an exact convex decomposition for continuous solid sweeps.
"""
from itertools import combinations
import numpy as np
from shapely.geometry import Polygon,Point,LineString
from shapely.ops import unary_union


def make(required,expansion,scale,center=None,width_fraction=.025,height_fraction=.018):
    if expansion<1:raise ValueError('A demand boundary may only expand')
    required=np.asarray(required,float)
    center=required.mean(axis=0) if center is None else np.asarray(center,float)
    if not Polygon(required).contains(Point(center)):raise ValueError('Expansion center must be strictly inside the demand hull')
    inner=center+expansion*(required-center)
    edges=np.roll(inner,-1,axis=0)-inner
    normals=np.c_[edges[:,1],-edges[:,0]]/np.linalg.norm(edges,axis=1)[:,None]
    width=width_fraction*scale;height=height_fraction*scale
    offsets=np.sum(normals*(inner-center),axis=1)+width
    outer=np.array([np.linalg.solve(normals[[i-1,i]],offsets[[i-1,i]])+center for i in range(len(inner))])
    strips=[np.array([inner[i],outer[i],outer[(i+1)%len(inner)],inner[(i+1)%len(inner)]]) for i in range(len(inner))]
    return dict(center_xy_m=center,expansion=float(expansion),inner_xy_m=inner,outer_xy_m=outer,
                edge_strips_xy_m=strips,width_m=width,height_m=height,
                area_m2=float(Polygon(outer).area-Polygon(inner).area))


def ray_exit(ring,point,withdrawal):
    """Farthest forward ray intersection, including rays starting outside the hull."""
    point=np.asarray(point,float);direction=np.asarray(withdrawal,float)
    vertices=ring['inner_xy_m'];hits=[]
    for p,q in zip(vertices,np.roll(vertices,-1,axis=0)):
        matrix=np.column_stack([direction,p-q])
        if abs(np.linalg.det(matrix))<1e-14*np.linalg.norm(q-p):continue
        t,u=np.linalg.solve(matrix,p-point)
        if t>=-1e-12 and -1e-10<=u<=1+1e-10:hits.append(max(0.,float(t)))
    if not hits:return None
    distance=max(hits)
    return point+distance*direction,distance


def clip(vertices,normal,offset):
    """Clip a convex polygon to normal.x <= offset without widening the cell."""
    if not len(vertices):return np.empty((0,2))
    result=[]
    for p,q in zip(vertices,np.roll(vertices,-1,axis=0)):
        vp=float(p@normal-offset);vq=float(q@normal-offset)
        if vp<=0:result.append(p)
        if (vp<0 and vq>0) or (vp>0 and vq<0):result.append(p+(q-p)*(vp/(vp-vq)))
    if len(result)<3:return np.empty((0,2))
    result=np.asarray(result)
    keep=np.linalg.norm(result-np.roll(result,1,axis=0),axis=1)>1e-14
    return result[keep] if keep.sum()>=3 else np.empty((0,2))


def partition(ring,anchors,weights=None):
    """Power cells shift the cuts between adjacent anchors; each owns a ground arc.

    The full ring is partitioned, rather than independently placing disconnected
    pads. Requiring each anchor inside its cell preserves the head-to-ring join.
    """
    center=ring['center_xy_m'];seeds=np.asarray(anchors)-center
    weights=np.zeros(len(seeds)) if weights is None else np.asarray(weights,float)
    groups=[]
    for i,seed in enumerate(seeds):
        constraints=[]
        for j,other in enumerate(seeds):
            if i==j:continue
            n=2*(other-seed)
            if np.linalg.norm(n)<1e-12:return None
            b=float(other@other-seed@seed+weights[i]-weights[j])
            if seed@n>b+1e-13:return None
            constraints.append((n,b))
        polygons=[]
        for strip in ring['edge_strips_xy_m']:
            polygon=strip-center
            for n,b in constraints:
                polygon=clip(polygon,n,b)
                if len(polygon)<3:break
            if len(polygon)>=3 and Polygon(polygon).area>1e-18:polygons.append(polygon+center)
        if not polygons:return None
        combined=unary_union([Polygon(p) for p in polygons])
        # Some float64 seams appear as zero-distance components. Solid union and
        # full-ring area checks below decide actual acceptance, not this seed rule.
        if combined.distance(Point(anchors[i]))>1e-9:return None
        connected=combined.buffer(ring['width_m']*1e-8)
        if len(seeds)>1 and (connected.geom_type!='Polygon' or len(connected.interiors)>0):return None
        groups.append(polygons)
    return groups


def polygons_of(plan):
    return [Polygon(p) for p in plan['ground_polygons_xy_m']]


def check(ring,plans,scale):
    expected=Polygon(ring['outer_xy_m']).difference(Polygon(ring['inner_xy_m']))
    polygons=[p for plan in plans for p in polygons_of(plan)]
    supplied=unary_union(polygons)
    missing=float(expected.difference(supplied).area)
    extra=float(supplied.difference(expected).area)
    owners=[unary_union(polygons_of(p)) for p in plans]
    overlaps=[float(a.intersection(b).area) for a,b in combinations(owners,2)]
    tolerance=1e-10*scale**2
    closed=np.vstack([ring['inner_xy_m'],ring['inner_xy_m'][0]])
    uncovered=float(LineString(closed).difference(supplied.buffer(scale*1e-9)).length)
    return dict(passed=bool(len(plans)>0 and missing<=tolerance and extra<=tolerance and
                           max(overlaps,default=0.)<=tolerance and uncovered<=scale*1e-8),
                expected_area_m2=float(expected.area),actual_area_m2=float(supplied.area),
                missing_area_m2=missing,extra_area_m2=extra,maximum_owner_overlap_m2=max(overlaps,default=0.),
                uncovered_boundary_length_m=uncovered,area_tolerance_m2=tolerance,
                endpoint_contact_allowed=True,closed_ring_is_assembled_from_separate_pieces=True)


def record(ring):
    return {key:(value.tolist() if isinstance(value,np.ndarray) else value)
            for key,value in ring.items() if key!='edge_strips_xy_m'}
