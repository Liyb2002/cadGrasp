"""Contact-preserving bodies whose inactive branches form the landing feet.

This is a geometric prototype, not a stiffness or stress optimization.  Foot
sizes are fitted against the original load samples; body routes follow a
conservative envelope of both workpieces' withdrawal obstacles.
"""
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial.distance import cdist

from step2_local_support import geometry as G, circles as P, withdrawal as W
from step0_pose_selection import equilibrium as Q
from step0_pose_selection.floor_points import pressure_centers
from step5_base.bearing import bearing_rays, grounded_matrix, seed_probes
from step4_connect_support.baseline_current import solids as SOL


def world(points, basis, offset):
    return (points-offset)@basis.T


def sole(parameters, side, height, radius=.010, thickness=.006):
    """Two rounded ends, a flat real sole, and a six-millimetre body."""
    start, end, y = parameters
    angles = np.linspace(0., 2*np.pi, 12, endpoint=False)
    xy = np.vstack([np.c_[x+radius*np.cos(angles), y+radius*np.sin(angles)]
                    for x in (start, end)])
    levels = (0., thickness) if side == 0 else (height-thickness, height)
    return G.hull_mesh(np.vstack([np.c_[xy, np.full(len(xy), z)] for z in levels]))


def fit_feet(problems, groups, bases, offsets, height):
    """Fit two actual soles per pose, with every accepted reduction re-solved.

    Rectangle fitting is only an initializer. Rounded soles, collision offsets,
    and independently shortened toes/heels are checked again using their own
    floor vertices. Screening uses a subset; acceptance always uses all samples.
    """
    analyzers = [W.Analyzer(p.domain.mesh, P.DEPTH_FRACTION*p.domain.mesh.extents.max(),
                           dict(vectors=[b[:, 0].tolist()])) for p, b in zip(problems, bases)]
    cache = [(*bearing_rays(p.domain, g, p.floor, 64.),
              Q.padded_targets(p.targets/p.scale, p.scale, 12)) for p, g in zip(problems, groups)]
    trials = []

    def bearing(side, local, full=True):
        p = problems[side]
        xy = world(local, bases[side], offsets[side])[:, :2]
        points, normals, owners, targets = cache[side]
        matrix, _ = grounded_matrix(points, normals, owners, p.domain.com, p.scale,
                                    [dict(pads_xy_m=[xy])], 64.)
        solver = Q.BatchSolver(matrix)
        if not solver.solve(targets[seed_probes(targets, 128)])['passed']:
            return False
        return bool(solver.solve(targets)['passed']) if full else True

    def rectangle(side, rect):
        x0, y0, x1, y1 = rect
        return np.array([[x0, y0, side*height], [x1, y0, side*height],
                         [x1, y1, side*height], [x0, y1, side*height]])

    def clear(mesh):
        return all(a.test([SimpleNamespace(vertices=world(mesh.vertices, b, o))], b[:, 0])['clear']
                   for a, b, o in zip(analyzers, bases, offsets))

    def actual(side, parameters):
        return bearing(side, np.concatenate([sole(row, side, height).vertices for row in parameters]))

    result = []
    for k, p in enumerate(problems):
        cloud, _ = pressure_centers(p.targets/p.scale, p.domain.com)
        cloud = np.c_[cloud, np.zeros(len(cloud))]@bases[k]+offsets[k]
        rect = np.r_[cloud[:, :2].min(0)-.012, cloud[:, :2].max(0)+.012]
        for _ in range(12):
            if bearing(k, rectangle(k, rect)):
                break
            rect += [-.010, -.010, .010, .010]
        else:
            raise RuntimeError('No feasible footprint in the finite initialization search')
        for step in (.010, .005):
            for axis in (2, 3, 0, 1):
                for _ in range(20):
                    candidate = rect.copy(); candidate[axis] += step*(1 if axis < 2 else -1)
                    if np.any(candidate[2:]-candidate[:2] < .020) or not bearing(k, rectangle(k, candidate)):
                        break
                    rect = candidate
        # 10 mm round-end radius consumes the initializer's 10 mm cushion.
        x0, y0, x1, y1 = rect
        parameters = np.array([[x0, x1, y0], [x0, x1, y1]])
        for j in range(2):
            for _ in range(30):
                if clear(sole(parameters[j], k, height)):
                    break
                parameters[j, 2] += .002*(1 if j else -1)
            else:
                raise RuntimeError('Could not place a collision-free sole')
        for _ in range(20):
            if actual(k, parameters):
                break
            parameters[:, 0] -= .004; parameters[:, 1] += .004
        else:
            raise RuntimeError('Actual sole footprint failed the sample loads')
        for step in (.015, .0075):
            for j in range(2):
                for end in (0, 1):
                    for _ in range(20):
                        candidate = parameters.copy(); candidate[j, end] += step*(1 if end == 0 else -1)
                        if candidate[j, 1]-candidate[j, 0] < .020 or not actual(k, candidate):
                            break
                        parameters = candidate
        parameters[:, 0] -= .003; parameters[:, 1] += .003
        if not actual(k, parameters) or not all(clear(sole(row, k, height)) for row in parameters):
            raise RuntimeError('Final fitted soles failed validation')
        result.append(parameters)
        trials.append(dict(pose=p.pose, rectangle_initializer_m=rect.tolist(),
                           sole_parameters_m=parameters.tolist(), all_original_samples_passed=True))
        print('Fitted branch soles:', p.pose, parameters*1000, flush=True)
    return np.asarray(result), trials


def build(problems, groups, head_world, bases, offsets, height, foot_parameters=None):
    if foot_parameters is None:
        foot_parameters, fitting = fit_feet(problems, groups, bases, offsets, height)
    else:
        foot_parameters = np.asarray(foot_parameters, float)
        fitting = 'replayed saved fitted dimensions; final whole-body checks rerun'
    triangles = np.concatenate([(p.domain.mesh.vertices@b+o)[p.domain.mesh.faces]
                                for p, b, o in zip(problems, bases, offsets)])
    lower = triangles[:, :, 1:].min(1); upper = triangles[:, :, 1:].max(1)
    rear = triangles[:, :, 0].max(1)
    sphere = np.asarray(trimesh.creation.icosphere(subdivisions=2).vertices)
    parts, labels, roots, nodes = [], [], {}, {}

    def required(a, b, radius):
        low = np.minimum(a[1:], b[1:])-radius-.002
        high = np.maximum(a[1:], b[1:])+radius+.002
        mask = ((upper >= low) & (lower <= high)).all(1)
        return float(rear[mask].max()+radius+.002) if mask.any() else -.15

    def add(mesh, label):
        parts.append(mesh); labels.append(label)

    def route(a, b, label, radius=.007):
        a, b = np.asarray(a), np.asarray(b)
        count = max(2, int(np.linalg.norm(a[1:]-b[1:])/.007)+1)
        path = np.linspace(a, b, count)
        limits = [required(x, y, radius) for x, y in zip(path[:-1], path[1:])]
        for j in range(count):
            path[j, 0] = max(path[j, 0], *limits[max(j-1, 0):min(j+1, count-1)])
        # Outward-only smoothing keeps each segment outside the conservative
        # obstacle bound while avoiding unnecessary steps in the body outline.
        for _ in range(4):
            smoothed = np.convolve(np.pad(path[:, 0], 1, mode='edge'), [.25, .5, .25], mode='valid')
            path[:, 0] = np.maximum(path[:, 0], smoothed)
        for x, y in zip(path[:-1], path[1:]):
            vertices = np.vstack([x+sphere*radius, y+sphere*radius])
            vertices[:, 2] = np.clip(vertices[:, 2], .001, height-.001)
            add(G.hull_mesh(vertices), label)
        nodes.setdefault(label, []).extend(path)
        return path

    shared = next(iter({c['candidate_id'] for c in groups[0]} & {c['candidate_id'] for c in groups[1]}))
    for k, (contacts, cells, basis, offset) in enumerate(zip(groups, head_world, bases, offsets)):
        for contact, source in zip(contacts, cells):
            label = contact['candidate_id']; local = [v@basis+offset for v in source]
            vertices = np.concatenate(local)
            center = contact['center_m']@basis+offset
            root = center.copy()
            root[0] = max(vertices[:, 0].max()+.005, required(center, center, .009))
            for cell in local:
                end = cell.copy(); end[:, 0] = root[0]+.003
                add(G.hull_mesh(np.vstack([cell, end])), label)
            add(G.hull_mesh(root+sphere*.009), label)
            roots[k, label] = root
            nodes.setdefault(label, []).append(root)

    roles = []
    for side in range(2):
        inactive = sorted([c['candidate_id'] for c in groups[1-side] if c['candidate_id'] != shared],
                          key=lambda label: roots[1-side, label][1])
        for label, parameters in zip(inactive, foot_parameters[side]):
            add(sole(parameters, side, height), label)
            _, end, y = parameters
            ankle = np.array([end-.008, y, .010 if side == 0 else height-.010])
            path = route(roots[1-side, label], ankle, label, .007)
            heel = ankle.copy(); heel[2] = .005 if side == 0 else height-.005
            heel_path = route(path[-1], heel, label, .007)
            # The collision envelope can move an ankle behind its force-fitted
            # sole. Keep that sole physically attached: a narrow heel extends
            # only in +X, within the already clear sole's withdrawal extrusion.
            back = heel_path[-1, 0]+.004
            if back > end:
                levels = (0., .006) if side == 0 else (height-.006, height)
                add(G.hull_mesh(np.array([[x, yy, z] for x in (end-.004, back)
                                         for yy in (y-.006, y+.006) for z in levels])), label)
        roles.append(dict(pose=problems[side].pose, active_heads=[c['candidate_id'] for c in groups[side]],
                          landing_branch_ids=inactive, separate_external_base=False))
    route(roots[0, shared], roots[1, shared], shared, .007)
    scale = max(height, np.ptp(np.concatenate([p.vertices for p in parts]), axis=0).max())
    joined, record = SOL.union_parts(parts, scale)
    bridges = []
    if not record['one_solid']:
        for label in dict.fromkeys(c['candidate_id'] for group in groups for c in group if c['candidate_id'] != shared):
            a, b = np.asarray(nodes[label]), np.asarray(nodes[shared])
            i, j = np.unravel_index(np.argmin(cdist(a, b)), (len(a), len(b)))
            route(a[i], b[j], label)
            bridges.append(label)
            joined, record = SOL.union_parts(parts, scale)
            if record['one_solid']:
                break
    if not record['one_solid']:
        raise RuntimeError(f'Branch bodies did not form one solid: {record}')
    visual = []
    for label in dict.fromkeys(labels):
        mesh, branch_record = SOL.union_parts([p for p, owner in zip(parts, labels) if owner == label], scale)
        if not branch_record['one_solid']:
            raise RuntimeError(f'Disconnected branch: {label}')
        visual.append((label, mesh))
    design = dict(kind='inactive_branch_bodies_with_integral_soles', external_frame=False,
                  foot_parameters_m=foot_parameters.tolist(), sole_radius_m=.010, sole_thickness_m=.006,
                  body_radius_m=.007, roles=roles, fitting=fitting, added_bridge_branches=bridges,
                  routing='conservative common withdrawal envelope, checked again on complete convex pieces',
                  optimization_scope='finite coordinate search for sole length; no global or strength optimum')
    return joined, parts, labels, visual, record, design
