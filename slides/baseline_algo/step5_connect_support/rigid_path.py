"""Finite rigid withdrawal search with translation and rotation.

LPs propose local motions, never certify global impossibility. Sweeps of rotating
convex cells use endpoint hulls enlarged by a circular-arc deviation bound, with
adaptive subdivision. Accepted trajectories are checked continuously at the
reported mesh-boolean and floor tolerances, not just at animation frames.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from step2_local_support import insertion as D
from step5_connect_support import routing as T


def constraints(mesh, contacts, floor_points, origin, scale):
    rows = []
    for c in contacts:
        points = c['triangles_m'].reshape(-1, 3)
        normals = np.repeat(mesh.face_normals[c['source_faces']], 3, axis=0)
        rows.append(np.c_[normals, np.cross((points-origin)/scale, normals)])
    points = np.asarray(floor_points).reshape(-1, 3)
    normals = np.tile([0., 1., 0.], (len(points), 1))
    rows.append(np.c_[normals, np.cross((points-origin)/scale, normals)])
    return np.unique(np.vstack(rows), axis=0)


def local_motions(mesh, contacts, floor_points, origin, scale):
    matrix = constraints(mesh, contacts, floor_points, origin, scale)
    objectives = np.vstack([np.eye(6), -np.eye(6), np.random.default_rng(5107).normal(size=(12, 6))])
    motions = []; diagnostics = []
    for objective in objectives:
        result = linprog(objective, A_ub=-matrix, b_ub=np.zeros(len(matrix)),
            bounds=[(-1., 1.)]*6, method='highs',
            options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
        diagnostics.append(dict(status=int(result.status), objective=result.fun if result.success else None))
        if not result.success or np.max(np.abs(result.x)) < 1e-7: continue
        motion = result.x/np.max(np.abs(result.x))
        if np.min(matrix@motion) < -2e-9: continue
        if all(np.linalg.norm(motion-other) > 1e-5 for other in motions): motions.append(motion)
    # Explicit pure translations are useful when the LP returns a rotational
    # extreme. Floor constraints rule out downward withdrawal from y=0.
    for v in np.vstack([np.eye(3), -np.eye(3)]):
        motion = np.r_[v, np.zeros(3)]
        if np.min(matrix@motion) >= -2e-9 and all(np.linalg.norm(motion-m) > 1e-5 for m in motions):
            motions.insert(0, motion)
    return motions, dict(constraint_count=len(matrix), candidate_count=len(motions), lp_diagnostics=diagnostics,
        includes_rotation=True, includes_floor=True, global_impossibility_claimed=False,
        scope='Finite local first-order LP proposals with numerical tolerances; no global nonexistence certificate')


def transform(points, origin, scale, motion, amount):
    rotation = Rotation.from_rotvec(amount*np.asarray(motion[3:])).as_matrix()
    return (np.asarray(points)-origin)@rotation.T+origin+amount*scale*np.asarray(motion[:3])


def rotation_enclosure(points, origin, motion, scale, first, last):
    """Sagitta padding stays in the rotation plane; exact sinusoid floor minimum."""
    points=np.asarray(points); omega=np.asarray(motion[3:]); speed=float(np.linalg.norm(omega))
    p=transform(points,origin,scale,motion,first);q=transform(points,origin,scale,motion,last)
    if speed < 1e-14:return np.vstack([p,q]),float(min(p[:,1].min(),q[:,1].min()))
    axis=omega/speed;r=points-origin
    parallel=(r@axis)[:,None]*axis; radial=r-parallel; cross=np.cross(axis,radial)
    radius=float(np.linalg.norm(radial,axis=1).max());angle=speed*(last-first)
    padding=radius*(1-np.cos(angle/2))
    helper=np.eye(3)[np.argmin(np.abs(axis))];u=np.cross(axis,helper);u/=np.linalg.norm(u);v=np.cross(axis,u)
    corners=np.array([x*u+y*v for x in (-1.,1.) for y in (-1.,1.)])*padding
    enclosure=(np.vstack([p,q])[:,None,:]+corners).reshape(-1,3)
    # z(s)=C+A*cos(speed*s)+B*sin(speed*s)+D*s. Evaluate endpoints
    # and every derivative zero within this (at most pi/3) interval.
    A=radial[:,1];B=cross[:,1];C=origin[1]+parallel[:,1];D=scale*motion[1]
    low=float(min(p[:,1].min(),q[:,1].min()))
    amp=np.hypot(A,B);ids=np.flatnonzero((amp>1e-15*scale)&(abs(D)<=speed*amp))
    for k in ids:
        phi=np.arctan2(A[k],B[k]);theta=np.arccos(np.clip(-D/(speed*amp[k]),-1,1))
        for sign in (-1,1):
            root=sign*theta-phi
            start=int(np.ceil((speed*first-root)/(2*np.pi)))
            end=int(np.floor((speed*last-root)/(2*np.pi)))
            for winding in range(start,end+1):
                t=(root+2*np.pi*winding)/speed
                low=min(low,float(C[k]+A[k]*np.cos(speed*t)+B[k]*np.sin(speed*t)+D*t))
    return enclosure,low


def segment(scene, parts, origin, motion, first, last, budget, depth=0):
    if budget['remaining'] <= 0: return False
    budget['remaining'] -= 1; budget['checked'] += 1
    angle = float(np.linalg.norm(motion[3:])*(last-first))
    if angle > np.pi/3:
        mid = (first+last)/2
        return (segment(scene, parts, origin, motion, first, mid, budget, depth+1) and
                segment(scene, parts, origin, motion, mid, last, budget, depth+1))
    okay = True
    for part in parts:
        points, low = rotation_enclosure(part.vertices, origin, motion, scene.scale, first, last)
        if low < -scene.scale*1e-10:
            okay = False; break
        swept = D.engine.hull_mesh(points)
        if scene.volume(swept) > 1e-11*scene.scale**3:
            okay = False; break
    if okay: return True
    if depth >= 12: return False
    mid = (first+last)/2
    # An actual intermediate collision is sufficient to reject this proposal.
    for part in parts:
        moved = part.copy(); moved.vertices = transform(part.vertices, origin, scene.scale, motion, mid)
        if not scene.clear(moved, ground=True): return False
    return (segment(scene, parts, origin, motion, first, mid, budget, depth+1) and
            segment(scene, parts, origin, motion, mid, last, budget, depth+1))


def separated(scene, points):
    low, high = points.min(axis=0), points.max(axis=0)
    gaps = np.r_[low-scene.mesh.bounds[1], scene.mesh.bounds[0]-high]
    return bool(np.max(gaps) > .025*scene.scale)


def search(scene, contacts, parts, origin, interval_budget=240):
    vertices = np.vstack([p.vertices for p in parts])
    floor = vertices[np.abs(vertices[:, 1]) <= scene.scale*1e-10]
    motions, local = local_motions(scene.mesh, contacts, floor, origin, scene.scale)
    report = dict(passed=False, status='no_nonzero_first_order_exit_found' if not motions else 'no_trajectory_in_finite_menu',
        local_motion=local, attempts=[], rotation_allowed=True,
        continuous_sweep_verified=False, global_impossibility_claimed=False,
        interval_budget=interval_budget, geometric_tolerance_m=scene.scale*1e-10,
        volume_tolerance_m3=1e-11*scene.scale**3)
    budget = dict(remaining=interval_budget, checked=0)
    for motion in motions:
        first = 0.; amounts = [0.]
        attempt = dict(motion=motion.tolist(), reached_amount=0.); report['attempts'].append(attempt)
        for last in (.02, .08, .2, .5, 1., 2., 4.):
            if not segment(scene, parts, origin, motion, first, last, budget): break
            amounts.append(last); attempt['reached_amount'] = last
            points = transform(vertices, origin, scene.scale, motion, last)
            if separated(scene, points):
                report.update(passed=True, status='rigid_trajectory_verified', continuous_sweep_verified=True,
                    origin_m=np.asarray(origin).tolist(), length_scale_m=scene.scale,
                    motion=motion.tolist(), final_withdrawal_amount=last, segment_amounts=amounts,
                    path_definition='p(s)=R(s*omega)*(p_final-origin)+origin+s*D*v; insertion reverses s',
                    method='endpoint convex hull plus in-plane circular-arc padding; exact sinusoidal floor bound; adaptive continuous mesh sweeps',
                    checked_intervals=budget['checked'])
                return report
            first = last
        if budget['remaining'] <= 0:
            report['status'] = 'trajectory_interval_budget_exhausted'; break
    report['checked_intervals'] = budget['checked']
    return report


def replay(scene, parts, report):
    assert report['passed'] and report['continuous_sweep_verified']
    budget = dict(remaining=max(1000, report['checked_intervals']*2), checked=0)
    origin = np.asarray(report['origin_m']); motion = np.asarray(report['motion'])
    assert abs(report['length_scale_m']-scene.scale) <= scene.scale*1e-12
    for first, last in zip(report['segment_amounts'][:-1], report['segment_amounts'][1:]):
        assert segment(scene, parts, origin, motion, first, last, budget)
    points = transform(np.vstack([p.vertices for p in parts]), origin, scene.scale, motion, report['final_withdrawal_amount'])
    assert separated(scene, points)
    return True
