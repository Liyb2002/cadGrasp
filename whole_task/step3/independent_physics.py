"""Actual contact polygons and paired-load equilibrium for each cradle."""
import numpy as np
from .legacy import kernel as K
from whole_task.common.wrenches import endpoint_demands,sample_domain


def demands(domain,policy):
    endpoints=endpoint_demands(domain)
    samples=sample_domain(domain,policy['sample_count_per_task'],policy['sample_seed'],check_visibility=False)
    return endpoints,samples


def contacts(body,support,work_ids,com):
    allowed=np.setdiff1d(np.arange(len(body.faces)),work_ids)
    triangles,sources=K.contact_boundary(body,support,allowed)
    if not len(triangles):raise RuntimeError('No actual contact polygons remain')
    q=triangles.reshape(-1,3);n=np.repeat(-body.face_normals[sources],3,axis=0)
    scale=np.r_[np.ones(3),np.ones(3)/body.extents.max()]
    full=K.I.merge_columns(K.U.floor(np.empty((0,6)),scale),
        K.U.heads(np.c_[n,np.cross(q-com,n)],scale))
    return triangles,sources,full,scale


def certificate(body,support,domain,endpoints,samples,*,search=False):
    tri,source,full,scale=contacts(body,support,domain.work_ids,domain.com)
    if search:
        # Candidate rejection needs one failed/unresolved LP, not a complete
        # classification of thousands of failed demands. Acceptance below
        # still checks every original sample and continuous-cover target.
        outer=endpoints['endpoint_need_wrench']*scale
        for index in np.unique(np.linspace(0,len(outer)-1,32).astype(int)):
            if K.J.C.W.solve(full,K.J.C.U.target(outer[index:index+1],full.shape[1])[0]) is None:
                return dict(passed=False,preflight_only=True,unpassed_outer_index=int(index)),{}
    targets=np.vstack([endpoints['endpoint_need_wrench'],samples['need_wrench']])*scale
    mask,solver=K.J.classify(full,targets)
    n=len(endpoints['endpoint_need_wrench'])
    report=dict(passed=bool(mask.all()),contact_polygon_triangle_count=len(tri),
        contact_area_m2=float(__import__('trimesh').triangles.area(tri).sum()),
        original_sample_count=len(samples['need_wrench']),passed_original_sample_count=int(mask[n:].sum()),
        continuous_outer_target_count=n,passed_continuous_outer_target_count=int(mask[:n].sum()),
        continuous_world_direction_cone_verified=bool(mask[:n].all()),
        surface_hull_vertex_count=len(endpoints['application_hull_vertex_ids']),
        unreduced_application_vertex_count=int(endpoints['unreduced_source_vertex_count']),
        application_hull_maximum_residual_m=float(endpoints['application_hull_maximum_residual_m']),
        paired_forces_and_moments=True,force_half_angle_deg=float(np.rad2deg(domain.half_angle)),
        force_cone_axis_world=domain.axis.tolist(),solver=solver,
        classification_tolerance=K.J.C.FEASIBILITY_TOL,
        contact_model='frictionless nonnegative unbounded normal reactions',object_floor_reactions_used=False)
    return report,dict(contact_triangles_world_m=tri,contact_source_face_ids=source,
        supply_7d=full,wrench_scale=scale,endpoint_passed=mask[:n],sample_passed=mask[n:])
