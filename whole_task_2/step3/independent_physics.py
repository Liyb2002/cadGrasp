"""Actual contact polygons and paired-load equilibrium for each cradle."""
import numpy as np
from .legacy import kernel as K
from whole_task_2.common.wrenches import endpoint_demands,sample_domain


def demands(domain,policy):
    endpoints=endpoint_demands(domain)
    samples=sample_domain(domain,policy['sample_count_per_task'],policy['sample_seed'],check_visibility=False)
    return endpoints,samples


LINE=1e-12   # m^2: a contact triangle smaller than this is a line or a point where the support only meets the object
             #   edge-on, not a contact (B+big_01 A1 turned into B1's seat held 1.47% of its samples on such slivers alone)


def area_contacts(triangles,sources):
    """The contact triangles with area: the ones of a line or a point dropped (LINE)."""
    if not len(triangles):return triangles,sources
    keep=__import__('trimesh').triangles.area(triangles)>LINE
    return triangles[keep],sources[keep]


def contacts(body,support,work_ids,com):
    allowed=np.setdiff1d(np.arange(len(body.faces)),work_ids)
    triangles,sources=area_contacts(*K.contact_boundary(body,support,allowed))
    if not len(triangles):raise RuntimeError('No actual contact polygons remain')
    q=triangles.reshape(-1,3);n=np.repeat(-body.face_normals[sources],3,axis=0)
    scale=np.r_[np.ones(3),np.ones(3)/body.extents.max()]
    full=K.I.merge_columns(K.U.floor(np.empty((0,6)),scale),
        K.U.heads(np.c_[n,np.cross(q-com,n)],scale))
    return triangles,sources,full,scale


HELD=.99   # a support holds a task when more than this share of its paired (force, moment) samples hold (user, 2026-10-28)


def outer_checked(domain):
    """Are the continuous outer targets classified?  Not under a local cone (a robot-free plan): its outer polytope
    pairs every face's vertices with the cone (tens of thousands of targets), and only the samples decide."""
    return domain.direction_model not in ('local_inward_normal_cone','local_axis_cone')


def certificate(body,support,domain,endpoints,samples,*,search=False):
    """The paired loads on the actual contacts: passed when more than HELD of the original samples hold.  The
    continuous outer targets are classified and reported (outer_checked), not required.  search is kept for the
    callers: its preflight (reject at the first outer target that fails) assumed every demand had to hold."""
    tri,source,full,scale=contacts(body,support,domain.work_ids,domain.com)
    outer=endpoints['endpoint_need_wrench'] if outer_checked(domain) else np.empty((0,6))
    targets=np.vstack([outer,samples['need_wrench']])*scale
    mask,solver=K.J.classify(full,targets)
    n=len(outer); share=float(mask[n:].mean())
    report=dict(passed=bool(share>HELD),held_sample_share=share,required_sample_share=HELD,contact_polygon_triangle_count=len(tri),
        contact_area_m2=float(__import__('trimesh').triangles.area(tri).sum()),
        original_sample_count=len(samples['need_wrench']),passed_original_sample_count=int(mask[n:].sum()),
        continuous_outer_target_count=n,passed_continuous_outer_target_count=int(mask[:n].sum()),
        continuous_world_direction_cone_verified=bool(mask[:n].all()) if n else None,
        surface_hull_vertex_count=len(endpoints.get('application_hull_vertex_ids',())),
        unreduced_application_vertex_count=int(endpoints.get('unreduced_source_vertex_count',0)),
        application_hull_maximum_residual_m=float(endpoints.get('application_hull_maximum_residual_m',0.)),
        paired_forces_and_moments=True,force_half_angle_deg=float(np.rad2deg(domain.half_angle)),
        force_direction_model=domain.direction_model,force_cone_axis_world=domain.axis.tolist(),solver=solver,
        classification_tolerance=K.J.C.FEASIBILITY_TOL,
        contact_model='frictionless nonnegative unbounded normal reactions',object_floor_reactions_used=False)
    return report,dict(contact_triangles_world_m=tri,contact_source_face_ids=source,
        supply_7d=full,wrench_scale=scale,endpoint_passed=mask[:n],sample_passed=mask[n:])
