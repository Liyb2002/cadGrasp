"""Local inward force caps and a certified outer polytope for continuous loads."""
from functools import lru_cache
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull


def tangent_frame(normals):
    normals=np.asarray(normals,float)
    ref=np.eye(3)[np.argmin(np.abs(normals),axis=-1)]
    first=np.cross(normals,ref)
    first/=np.linalg.norm(first,axis=-1,keepdims=True)
    return first,np.cross(normals,first)


@lru_cache(maxsize=16)
def outer_cap(half_angle_deg, sides=16, polar_intervals=2):
    """Contain {r*d: 0<=r<=1, angle(d,+Z)<=alpha} without shrinking it.

    Cone tangent planes bound the circular cone from outside. Unit-sphere
    tangent planes bound the unit ball from outside. Their bounded intersection
    therefore contains the full force cap. No random sampling defines this hull.
    """
    if not 0<float(half_angle_deg)<90 or sides<8 or polar_intervals<1:
        raise ValueError('Expected a nondegenerate inward cone and adequate outer cover')
    angle=np.radians(half_angle_deg);phi=np.arange(sides)*2*np.pi/sides
    planes=[np.array([0.,0.,-1.,0.]),np.array([0.,0.,1.,-1.])]
    planes += [np.array([np.cos(p),np.sin(p),-np.tan(angle),0.])for p in phi]
    for theta in np.linspace(0,angle,polar_intervals+1)[1:]:
        planes += [np.array([np.sin(theta)*np.cos(p),np.sin(theta)*np.sin(p),np.cos(theta),-1.])for p in phi]
    planes=np.array(planes)
    vertices=HalfspaceIntersection(planes,np.array([0.,0.,.5])).intersections
    vertices[np.linalg.norm(vertices,axis=1)<1e-12]=0.
    vertices=np.unique(vertices,axis=0)
    if np.max(vertices@planes[:,:3].T+planes[:,3])>1e-11:
        raise ValueError('Unresolved force-cover vertex')
    # A second V->H construction is retained for independent sample containment.
    hull=ConvexHull(vertices)
    return vertices,planes,hull.equations


def certificate_description(domain):
    if domain.direction_model=='fixed_world_needle_axis':
        return dict(kind='exact_fixed_axis_vertex_hull',outer_approximation=False)
    cfg=domain.data['load']['continuous_cover']
    vertices,_,_=outer_cap(np.degrees(domain.half_angle),cfg['azimuth_sides'],cfg['polar_intervals'])
    return dict(kind=('world_force_cap_outer_polytope_x_surface_convex_hull_vertices'
        if domain.direction_model=='world_needle_direction_cone' else 'local_force_cap_outer_polytope_x_triangle_vertices'),outer_approximation=True,
        azimuth_sides=cfg['azimuth_sides'],polar_intervals=cfg['polar_intervals'],
        force_polytope_vertex_count=len(vertices),maximum_force_norm_ratio=float(np.linalg.norm(vertices,axis=1).max()),
        guarantees='passing all outer targets suffices for every surface point, direction and magnitude',
        failure_means='outer-cover failure alone does not prove failure inside the true cone')
