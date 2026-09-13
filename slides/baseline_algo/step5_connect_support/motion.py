"""Fixed-direction contact compatibility and continuous convex-piece sweeps."""
from fractions import Fraction
import numpy as np
import trimesh

from step2_local_support import surface as S
from step5_connect_support import ground as F

# Same normalized local-contact tolerance as the Step 2/3 angle classifier.
# This gate is only preliminary: every accepted assembly still needs its solid sweep.
NORMAL_DOT_TOLERANCE=1e-12


def insertion_vector(direction):
    direction=np.asarray(direction,float)
    if direction.shape!=(3,) or not np.isfinite(direction).all():
        raise ValueError('Expected a finite world-frame insertion vector')
    if abs(np.linalg.norm(direction)-1)>1e-12 or abs(direction[2])>1e-12:
        raise ValueError('Step 5 requires the a horizontal unit direction selected for this independent support')
    return direction


def exact_projection(triangle,direction):
    """Exact sign of the oriented source-face cross product dotted with motion.

    Inputs are the stored binary floats. Retain the exact sign as a diagnostic;
    the preliminary gate separately labels normals within the numerical tangent band.
    """
    v=[[Fraction(float(x)) for x in row] for row in triangle]
    e=[x-y for x,y in zip(v[1],v[0])];f=[x-y for x,y in zip(v[2],v[0])]
    normal=[e[1]*f[2]-e[2]*f[1],e[2]*f[0]-e[0]*f[2],e[0]*f[1]-e[1]*f[0]]
    return sum(n*Fraction(float(x)) for n,x in zip(normal,direction))


def contact_motion(mesh,contacts,direction):
    """Necessary endpoint condition for the WHOLE rigid assembly.

    At a final contact p, withdrawal is p-s*a. For an outward object normal n,
    n.(p-s*a-p)=-s*n.a. A positive n.a enters the object immediately.
    This condition concerns insertion direction, not the vertical hemisphere.
    """
    a=insertion_vector(direction);scale=float(mesh.extents.max())
    records=[]
    for contact in contacts:
        faces=contact['source_faces'];triangles=contact['triangles_m']
        measured=S.areas(triangles)
        np.testing.assert_allclose(measured,contact['triangle_areas_m2'],rtol=1e-9,atol=1e-30)
        projections={int(f):exact_projection(mesh.triangles[f],a) for f in np.unique(faces)}
        dots={int(f):float(mesh.face_normals[f]@a) for f in np.unique(faces)}
        blocked=sorted(f for f,value in projections.items() if value>0 and dots[f]>NORMAL_DOT_TOLERANCE)
        forbidden=np.isin(faces,blocked)
        record=dict(candidate_id=contact['candidate_id'],source_faces=np.unique(faces).tolist(),
            blocked_source_faces=blocked,contact_area_m2=float(measured.sum()),
            blocked_area_m2=float(measured[forbidden].sum()),
            blocked_area_fraction=float(measured[forbidden].sum()/measured.sum()),
            maximum_outward_normal_dot_insertion=float((mesh.face_normals[faces]@a).max()),
            near_tangent_source_faces=sorted(f for f,value in dots.items() if abs(value)<=NORMAL_DOT_TOLERANCE),
            condition_passed=not blocked,witness=None)
        if blocked:
            ids=np.flatnonzero(forbidden)
            # Prefer a well-inside, appreciable patch for the numerical illustration.
            index=int(max(ids,key=lambda i:measured[i]*float(mesh.face_normals[faces[i]]@a)))
            face=int(faces[index]);point=triangles[index].mean(axis=0)
            normal=mesh.face_normals[face]
            # Put the illustrative point on the original source plane.
            point-=((point-mesh.triangles[face,0])@normal)*normal
            witness=dict(source_face=face,patch_triangle_index=index,point_m=point.tolist(),
                outward_normal=normal.tolist(),outward_normal_dot_insertion=float(normal@a),
                oriented_cross_dot_insertion_exact=str(projections[face]),
                necessary_condition_violated=True,numerical_interior_point_found=False)
            distances=scale*.001*2.**(-np.arange(20))
            moved=point-distances[:,None]*a
            inside=mesh.contains(moved)
            candidates=np.flatnonzero(inside)
            if len(candidates):
                _,distance,_=trimesh.proximity.closest_point(mesh,moved[candidates])
                robust=np.flatnonzero(distance>scale*1e-9)
            else:
                robust=[]
            if len(robust):
                k=int(robust[0]);j=int(candidates[k])
                witness.update(numerical_interior_point_found=True,
                    withdrawal_distance_m=float(distances[j]),
                    pre_final_point_m=moved[j].tolist(),
                    distance_to_object_surface_m=float(distance[k]))
            record['witness']=witness
        records.append(record)
    return dict(passed=all(r['condition_passed'] for r in records),
        condition='n_out dot insertion_direction <= 0, with numerical near-tangencies deferred to the complete solid sweep',
        normal_dot_tolerance=NORMAL_DOT_TOLERANCE,
        sign_check='Exact rational cross-product diagnostics; positive normalized dots above the shared tangent tolerance reject immediately',
        not_a_sufficient_sweep_test=True,contacts=records)


def sweep_check(mesh,parts,direction,minimum_length=0.):
    """Exact sweep geometry per convex primitive, evaluated with mesh booleans.

    Do not take the convex hull of the complete nonconvex support: doing so
    would fill its openings. Translation distributes over the union of parts.
    """
    a=insertion_vector(direction);scale=float(mesh.extents.max())
    vertices=np.concatenate([p.vertices for p in parts])
    length=max(float(minimum_length),float((vertices@a).max()-(mesh.vertices@a).min()+.1*scale))
    swept=F.swept_pieces(parts,a,length)
    tolerance=1e-11*scale**3
    end=F.intersection_volumes(mesh,parts,scale)
    volumes=F.intersection_volumes(mesh,swept,scale)
    below=[max(0.,-float(p.vertices[:,2].min())) for p in parts]
    start_gap=float((mesh.vertices@a).min()-((vertices-length*a)@a).max())
    return dict(passed=bool(max(end+volumes,default=0.)<=tolerance and
                           max(below,default=0.)<=scale*1e-10 and start_gap>0),
        direction=a.tolist(),withdrawal_direction=(-a).tolist(),length_m=length,
        start_separation_m=start_gap,per_part_final_intersection_m3=end,
        per_part_sweep_intersection_m3=volumes,per_part_floor_penetration_m=below,
        volume_tolerance_m3=tolerance,geometric_tolerance_m=scale*1e-10,
        method='Union of convex_hull(P, P-L*a) for every convex primitive; float64 manifold intersections',
        continuous_translation_checked=True,trajectory_samples_used_for_acceptance=False,
        boundary_contact_allowed=True,exact_arithmetic_boolean_claimed=False)
