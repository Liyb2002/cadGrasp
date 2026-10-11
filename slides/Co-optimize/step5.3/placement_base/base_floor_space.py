"""Finite-window ground screening; it never certifies a connected solid.

All work cones are unbounded. Exit screening uses a conservative convex
sweep. Surviving 2D area is shrunk by a recorded finite foot clearance.
Failure is for this layout and window, not for every possible translation.
"""
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import unary_union

import stage5_common as C
import base_seating as B


def clip(vertices, normal, offset):
    vertices=np.asarray(vertices).reshape(-1,2)
    if not len(vertices):return vertices
    values=vertices@normal+offset
    if values.min()>=-1e-12:return vertices
    if values.max()<-1e-12:return np.empty((0,2))
    output=[]
    for first,last,f,g in zip(vertices,np.roll(vertices,-1,axis=0),values,np.roll(values,-1)):
        if f>=-1e-12:output.append(first)
        if (f>=-1e-12)!=(g>=-1e-12):output.append(first+(last-first)*np.clip(f/(f-g),0.,1.))
    return np.asarray(output).reshape(-1,2)


def screen(access, seating, *, extent=.5, clearance=.0025):
    moved=B.shifted(access.data,seating)
    relative=np.asarray(seating['object_relative_world_translation_m'])
    shifts=np.einsum('nij,nj->ni',access.inverse_rotations,relative)
    reports=[]
    for i,row in enumerate(moved['rows']):
        hull=row['hull'][:,:2];center=hull.mean(0)
        rectangle=np.array([[-extent,-extent],[extent,-extent],[extent,extent],[-extent,extent]])+center
        inverse=np.linalg.inv(row['fixture_transform'])
        region=rectangle.copy()
        for state in moved['states']:
            trans=state@inverse
            region=clip(region,trans[2,:2],trans[2,3])
        if len(region)<3:
            reports.append(dict(pose=row['pose'],passed=False,reason='empty_floor_region',legal_ground_vertices_world_m=[]));continue
        obstacles=[]
        for j,kind,face,equations in access.families:
            normal=equations[:,:3]@inverse[:3,:3]
            offset=equations[:,:3]@inverse[:3,3]+equations[:,3]-equations[:,:3]@shifts[j]
            polygon=region.copy()
            for n,b in zip(normal,offset):
                polygon=clip(polygon,-n[:2],-b)
                if len(polygon)<3:break
            if len(polygon)>=3:
                shape=Polygon(polygon)
                if shape.area>1e-18:obstacles.append(shape)
        legal=Polygon(region).difference(unary_union(obstacles)).buffer(-clearance)
        pieces=list(legal.geoms) if legal.geom_type=='MultiPolygon' else ([legal] if legal.geom_type=='Polygon' else [])
        points=np.vstack([np.asarray(p.exterior.coords)[:-1] for p in pieces]) if pieces else np.empty((0,2))
        if len(row['object_ground']):points=np.vstack([points,row['object_ground']])
        if len(points)>=3 and np.linalg.matrix_rank(points-points.mean(0),tol=1e-10)==2:
            available=ConvexHull(points);points=points[available.vertices]
            gap=max(0.,float(np.max(hull@available.equations[:,:2].T+available.equations[:,2])))
            passed=gap<=1e-9
        else:passed=False;gap=None
        reports.append(dict(pose=row['pose'],passed=passed,maximum_demand_hull_gap_m=gap,
                            legal_ground_area_m2=float(legal.area),
                            legal_ground_vertices_world_m=np.c_[points,np.zeros(len(points))].tolist()))
    return dict(passed=all(p['passed'] for p in reports),per_pose=reports,
                viewport_half_width_m=extent,ground_clearance_m=clearance,
                exact_solid_or_contact_acceptance=False)
