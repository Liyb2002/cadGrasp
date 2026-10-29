"""Bidirectional RRT-Connect with checked edges, plus geometry-derived SO101 grasps."""
import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


class PlanningFailure(RuntimeError):
    def __init__(self, message, **detail):
        super().__init__(message)
        self.detail = detail


def edge_samples(a, b, joint_step, reach, linear_step):
    """A conservative rotational lever-arm bound supplements the joint-angle resolution."""
    n = max(1, int(np.ceil(np.abs(b - a).max() / joint_step)),
            int(np.ceil(np.abs(b - a).sum() * reach / linear_step)))
    return [a + (b - a) * (i / n) for i in range(1, n + 1)]


def rrt_connect(a, b, valid, edge, lower, upper, rng, policy):
    """Both endpoints and every extension/shortcut are checked; None means bounded search failed."""
    if not valid(a) or not valid(b):
        return None
    if edge(a, b):
        return [a, b]
    trees = [([a.copy()], [-1]), ([b.copy()], [-1])]
    began = time.monotonic()

    def extend(tree, target):
        nodes, parents = tree
        i = int(np.argmin(np.linalg.norm(np.asarray(nodes) - target, axis=1)))
        d = target - nodes[i]
        length = np.linalg.norm(d)
        new = target.copy() if length <= policy['rrt_step_rad'] else nodes[i] + d * policy['rrt_step_rad'] / length
        if not edge(nodes[i], new):
            return None, False
        nodes.append(new)
        parents.append(i)
        return len(nodes) - 1, length <= policy['rrt_step_rad']

    def trace(tree, i):
        nodes, parents = tree
        out = []
        while i >= 0:
            out.append(nodes[i])
            i = parents[i]
        return out

    for iteration in range(policy['rrt_iterations']):
        if time.monotonic() - began > policy['rrt_seconds']:
            break
        x, y = trees[iteration % 2], trees[1 - iteration % 2]
        target = b if iteration % 10 == 0 and x is trees[0] else rng.uniform(lower, upper)
        i, _ = extend(x, target)
        if i is None:
            continue
        while time.monotonic() - began <= policy['rrt_seconds']:
            j, reached = extend(y, x[0][i])
            if j is None:
                break
            if reached:
                path = trace(x, i)[::-1] + trace(y, j)[1:]
                if x is not trees[0]:
                    path.reverse()
                for _ in range(policy['shortcut_tries']):
                    if len(path) < 3:
                        break
                    u, v = sorted(rng.choice(len(path), 2, replace=False))
                    if v - u > 1 and edge(path[u], path[v]):
                        path = path[:u + 1] + path[v:]
                return path
    return None


def contact_directions(normals, half_angle_deg, count):
    """Inward rays in a normal cone, including its centre; never force the span to equal the normal."""
    inward = -np.asarray(normals, float)
    out = [inward]
    if half_angle_deg == 0:
        return out
    reference = np.eye(3)[np.argmin(np.abs(inward), axis=1)]
    u = np.cross(inward, reference)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(inward, u)
    for i in range(count - 1):
        angle = np.radians(half_angle_deg * (1. if i % 2 == 0 else .5))
        azimuth = i * np.pi * (3 - np.sqrt(5))
        out.append(np.cos(angle) * inward + np.sin(angle) * (np.cos(azimuth) * u + np.sin(azimuth) * v))
    return out


def grasp_candidates(mesh, robot, source, target, rng, policy, statistics=None):
    """Both contact normals must lie in the configured cone around their opposing span directions."""
    statistics = statistics if statistics is not None else {}
    p, faces = trimesh.sample.sample_surface(mesh, policy['grasp_samples'], seed=rng)
    normals = mesh.face_normals[faces]
    directions = np.vstack(contact_directions(normals, policy.get('contact_cone_half_angle_deg', 20.),
                                              policy.get('contact_directions_per_sample', 5)))
    origins = np.tile(p, (len(directions) // len(p), 1)) + directions * 1e-5
    hit, rays, triangles = [], [], []
    for start in range(0, len(directions), 4096):
        h, r, f = mesh.ray.intersects_location(origins[start:start + 4096], directions[start:start + 4096], multiple_hits=False)
        hit.extend(h)
        rays.extend(r + start)
        triangles.extend(f)
    proposals = []
    preferred = source[:3, :3].T @ np.array([0., 0., -1.]) + target[:3, :3].T @ np.array([0., 0., -1.])
    cosine = np.cos(np.radians(policy.get('contact_normal_tolerance_deg', 20.)))
    pairs, seen = [], set()
    for end, ray, face2 in zip(hit, rays, triangles):
        sample = ray % len(p)
        start = p[sample]
        delta = end - start
        width = np.linalg.norm(delta)
        if width < .006 or width > robot.gripper['maximum_grasp_width_m']:
            continue
        c = delta / width
        alignment = [float(normals[sample] @ (-c)), float(mesh.face_normals[face2] @ c)]
        if min(alignment) < cosine - 1e-8:
            continue
        try:
            jaw = robot.jaw_for_width(width)
        except ValueError:
            continue
        key = tuple(np.round(np.r_[start, end] / .001).astype(int))
        if key in seen:
            continue
        seen.add(key)
        height = min((pose[:3, :3] @ ((start + end) / 2) + pose[:3, 3])[2] for pose in (source, target))
        pairs.append((height - .05 * width, start, end, c, width, jaw, int(faces[sample]), int(face2), alignment))
    pairs.sort(key=lambda item: -item[0])
    statistics.update(surface_samples=len(p), rays_cast=len(directions), contact_pairs_before_limit=len(pairs))
    limit = policy.get('contact_pair_limit', 3000)
    if limit:
        pairs = pairs[:limit]
    statistics['contact_pairs'] = len(pairs)
    for _, start, end, c, width, jaw, face1, face2, alignment in pairs:
        native = robot.moving_pad(jaw) - robot.fixed_pad
        native /= np.linalg.norm(native)
        # Complete an orthonormal basis around the contact span.
        z = np.array([0., 0., 1.]) - native * native[2]
        z /= np.linalg.norm(z)
        local = np.column_stack([native, z, np.cross(native, z)])
        approach = preferred - c * (preferred @ c)
        if np.linalg.norm(approach) < .01:
            reference = np.eye(3)[np.argmin(np.abs(c))]
            approach = reference - c * (reference @ c)
        approach /= np.linalg.norm(approach)
        for angle in np.linspace(-np.pi, np.pi, policy['grasp_rolls'], endpoint=False):
            a = Rotation.from_rotvec(c * angle).apply(approach)
            basis = np.column_stack([c, a, np.cross(c, a)])
            rotation = basis @ local.T
            g = np.eye(4)
            g[:3, :3] = rotation
            g[:3, 3] = start - rotation @ robot.fixed_pad
            down = min(-(pose[:3, :3] @ rotation[:, 2])[2] for pose in (source, target))
            height = min((pose[:3, :3] @ ((start + end) / 2) + pose[:3, 3])[2] for pose in (source, target))
            score = height + .12 * down - .05 * width
            proposals.append((score, dict(object_from_gripper=g, jaw_rad=float(jaw), width_m=float(width),
                contacts_object_m=[start, end], contact_faces=[face1, face2],
                contact_normal_angles_deg=np.degrees(np.arccos(np.clip(alignment, -1, 1))).tolist())))
    proposals.sort(key=lambda item: -item[0])
    statistics['grasp_candidates'] = len(proposals)
    return [p for _, p in proposals]
