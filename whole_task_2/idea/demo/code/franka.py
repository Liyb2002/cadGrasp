"""Pick and place with a Franka Panda and its hand, as a real cell plans it: a held body (the bunny, the support, or the
support with the bunny in its seat) from one pose to another among obstacles.  The demo's carrier arm; not a
contribution of the method.

A transfer of a body from pose A to pose B (`Mover.transfer`):
  - it leaves A along a straight line `out` (world: a certified exit, a lift off the table) and reaches B along one
    from B shifted by `into` (an exit, a drop onto the table): Cartesian lines, the IK tracked every `line_step`;
  - its grasp: a pad centre, an approach and a closing direction on the body (antipodal pairs and rim pinches, the
    hand clear of the body and of what surrounds it at A and at B), tried best first; the hand comes in along the
    approach from `pre` behind with the fingers open, they close, and after the place they open and the hand backs out;
  - every free motion (to the grasp, the carry, away to the park pose) by RRT-Connect in joint space (Kuffner and
    LaValle 2000), each edge checked every `resolution`, then shortcut, then timed within the joint speed limits;
  - every configuration checked (World.clear): the Panda's collision hulls (links, hand, fingers at their opening; each
    as its 26-DOP) against dense surface samples of the support and the bunny (`spacing`) and against the table top,
    against the waiting work arm (vertex in hull) and against its own distant links; the held body's surface samples
    `clearance` off the scene and the table, outside the work arm and the Panda's links.
The model is MuJoCo Menagerie's franka_emika_panda (Apache-2.0): its collision hulls, its visual meshes.
"""
import time
import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT
from whole_task_2.step5 import joined_video as J

MODEL = ROOT/'slides/Co-optimize/idea/assets/franka_emika_panda/panda.xml'
# Joint speed limits (rad/s, Franka's datasheet), `speed` of them in free motion, `line_speed` (m/s) along lines; the
# hand's `stroke`, the fingers open `margin` (m) off a grasp each side; `ready` the Panda's usual start pose; IK `seeds` random starts after the given ones; lines tracked every `line_step`
# (m); RRT-Connect: `step` (rad, joint space), `resolution` (rad) of the edge checks, at most `iterations` or
# `seconds`, then `shortcuts` tries; the held body `clearance` (m) off the scene; scenes sampled every `spacing` (m); the
# fingertips (Panda) longer than the stock ones by `fingertip`[0], slim (width, thickness).
PANDA = dict(velocity=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61), speed=.5, line_speed=.12, stroke=.08, margin=.006,
             ready=(0., -.785, 0., -2.356, 0., 1.571, .785), seeds=16, line_step=.005, pre=.06,
             step=.35, resolution=np.radians(1.5), iterations=4000, seconds=20., shortcuts=120, clearance=.005, grip_seconds=.35,
             spacing=.002, fingertip=(.05, .015, .008))


def finger(opening):
    """Each finger joint for a gap `opening` between the pads (the pads' faces .0015 off a finger's axis)."""
    return float(np.clip((opening-.003)/2, 0., .04))


def frame(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p; return T


def shifted(T, v):
    T = np.array(T, float); T[:3, 3] += v; return T


# The 26-DOP's directions: the axes, the face and the corner diagonals of a piece's own frame (exact for boxes in it).
DOP = np.array([d for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)], float)-1.
DOP /= np.linalg.norm(DOP, axis=1)[:, None]


def pieces(points):
    """Convex pieces as step1.arm's Check keeps them (normals, offsets, centre, radius), each the 26-DOP of its points: the
    half-space along each of DOP's directions that holds them (it contains their hull, with few planes)."""
    out = []
    for v in points:
        v = np.asarray(v, float); v = v[ConvexHull(v).vertices]; c = v.mean(0)
        out.append((DOP, -(v@DOP.T).max(0), c, 1.1*float(np.linalg.norm(v-c, axis=1).max())))
    return out


def solids(points):
    """Convex pieces for touch: each one's 26-DOP planes (pieces) and its hull's vertices, in its own frame."""
    return [(np.c_[n, d], np.asarray(v, float)[ConvexHull(v).vertices]) for (n, d, _, _), v in zip(pieces(points), points)]


class Cloud:
    """Dense surface samples of a body (its frame) with a tree; `at` places them."""
    def __init__(self, points):
        self.points = np.asarray(points, float); self.tree = cKDTree(self.points)

    @classmethod
    def of(cls, mesh, spacing=PANDA['spacing']):
        return cls(trimesh.sample.sample_surface_even(mesh, int(mesh.area/spacing**2), radius=spacing*.7, seed=0)[0])

    def at(self, M):
        return Cloud(self.points@M[:3, :3].T+M[:3, 3])


def pierced(pieces, frames, cloud, margin=0.):
    """Does a sample of the cloud lie in a piece (its planes grown by `margin`) at its frame."""
    for (n, d, c, r), M in zip(pieces, frames):
        R, p = M[:3, :3], M[:3, 3]; idx = cloud.tree.query_ball_point(R@c+p, r+margin)
        if idx and (((cloud.points[idx]-p)@R)@n.T+d-margin <= 0).all(1).any(): return True
    return False


def placed_solid(piece, M):
    """A convex piece at frame M: its planes (normals, offsets), vertices, centre and radius in the world."""
    E, v = piece; n = E[:, :3]@M[:3, :3].T; P = v@M[:3, :3].T+M[:3, 3]; c = P.mean(0)
    return n, E[:, 3]-n@M[:3, 3], P, c, float(np.linalg.norm(P-c, axis=1).max())


def overlap(a, b):
    """Do two placed convex pieces overlap: a vertex of one inside the other."""
    na, da, Pa, ca, ra = a; nb, db, Pb, cb, rb = b
    if np.linalg.norm(ca-cb) > ra+rb: return False
    return bool(((Pa@nb.T+db) <= 0).all(1).any() or ((Pb@na.T+da) <= 0).all(1).any())


def touch(A, frames_a, B, frames_b):
    """Do two sets of convex pieces at their frames overlap: a vertex of one inside a hull of the other."""
    A, B = [placed_solid(a, M) for a, M in zip(A, frames_a)], [placed_solid(b, M) for b, M in zip(B, frames_b)]
    return any(overlap(a, b) for a in A for b in B)


def inside(points, solid, M):
    """Is any point inside a convex piece at frame M."""
    E, _ = solid; P = (points-M[:3, 3])@M[:3, :3]
    return bool(((P@E[:, :3].T+E[:, 3]) <= 0).all(1).any())


def hand_hulls(template, opening):
    """The hand's hulls in the TCP frame (Panda.hand_template), the fingers `opening` apart."""
    fixed, left, right = template; s = finger(opening)
    return fixed+[v+[0., s, 0.] for v in left]+[v-[0., s, 0.] for v in right]


def box(lo, hi):
    """A box mesh between corners lo and hi."""
    b = trimesh.creation.box(extents=np.subtract(hi, lo)); b.apply_translation(np.add(lo, hi)/2); return b


class Panda:
    """The Panda with its base at `base` (on the table), turned `yaw` about the vertical, its hand with long slim
    fingertips (`fingertip`: added length, width, thickness; m): forward kinematics of every part, IK of the TCP (between
    the pads; z the approach, y the closing direction), collision hulls, visual meshes."""
    def __init__(self, base, yaw, policy=PANDA):
        m = self.model = mujoco.MjModel.from_xml_path(str(MODEL)); self.data = mujoco.MjData(m); self.policy = policy
        m.body_pos[m.body('link0').id] = base; m.body_quat[m.body('link0').id] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
        self.base, self.yaw = np.asarray(base, float), float(yaw)
        self.lower, self.upper = m.jnt_range[:7, 0].copy(), m.jnt_range[:7, 1].copy()
        self.velocity, self.ready, self.hand = np.asarray(policy['velocity']), np.asarray(policy['ready']), m.body('hand').id
        fingers = {m.body('left_finger').id, m.body('right_finger').id}
        # the fingertips: a bar on each finger from its root, the inner face where the stock pad's is (.0015 off the axis),
        # the stock finger's 54 mm plus `fingertip` long; the pads' centre 9.5 mm from the tip, as the stock pad's
        extra, fw, ft = policy['fingertip']; length = .054+extra; self.tip = box([-fw/2, .0015, 0.], [fw/2, .0015+ft, length])
        self.tcp = float(m.body_pos[m.body('left_finger').id][2])+length-.0095
        # collision parts: (body, frame in the body, hull vertices)
        self.parts = [(int(m.geom_bodyid[g]), self.local(g), (lambda v: v[ConvexHull(v).vertices])(self.vertices(g)))
                      for g in range(m.ngeom) if m.geom_group[g] == 3 and m.geom_bodyid[g] not in fingers]
        self.parts += [(b, np.eye(4), np.asarray(self.tip.vertices)) for b in sorted(fingers)]
        self.hulls = [v for _, _, v in self.parts]; self.pieces, self.solids = pieces(self.hulls), solids(self.hulls)
        names = [m.body(b).name for b, _, _ in self.parts]
        level = {f'link{i}': i for i in range(8)} | dict(hand=8, left_finger=8, right_finger=8)
        self.names, self.level = names, np.array([level[n] for n in names])
        self.fingers = [i for i, n in enumerate(names) if n in ('left_finger', 'right_finger')]
        self.links = [i for i, n in enumerate(names) if self.level[i] <= 6]                   # what a held body must stay out of
        self.visual = [(('geom', g), v, f, c) for g, v, f, c in self.meshes() if m.geom_bodyid[g] not in fingers]
        self.visual += [(('body', b), np.asarray(self.tip.vertices), np.asarray(self.tip.faces), (.22, .23, .25)) for b in sorted(fingers)]
        # own pairs checked: three or more links apart, apart in the ready pose
        F = self.frames(self.ready, .04)
        self.pairs = [(i, j) for i in range(len(names)) for j in range(i+1, len(names)) if abs(self.level[i]-self.level[j]) >= 3
                      and not touch([self.solids[i]], [F[i]], [self.solids[j]], [F[j]])]

    def local(self, g):
        m = self.model; R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g]); return frame(R.reshape(3, 3), m.geom_pos[g])

    def vertices(self, g):
        m = self.model
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            k = m.geom_dataid[g]; a = m.mesh_vertadr[k]; return np.asarray(m.mesh_vert[a:a+m.mesh_vertnum[k]], float)
        return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)*m.geom_size[g]

    def meshes(self):
        """(geom, vertices, faces, colour) of every visual mesh of the model, in its geom's frame."""
        m = self.model; out = []
        for g in range(m.ngeom):
            if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
            k = m.geom_dataid[g]; a, f = m.mesh_vertadr[k], m.mesh_faceadr[k]
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            out.append((g, np.asarray(m.mesh_vert[a:a+m.mesh_vertnum[k]], float), np.asarray(m.mesh_face[f:f+m.mesh_facenum[k]]), tuple(rgba[:3])))
        return out

    def visuals(self):
        """(vertices, faces, colour) of every visual part (the stock meshes, the fingertips), as visual_frames places them."""
        return [(v, f, c) for _, v, f, c in self.visual]

    def visual_frames(self, q, opening):
        self.pose(q, opening); d = self.data; out = []
        for (kind, i), *_ in self.visual:
            out.append(frame(d.geom_xmat[i].reshape(3, 3), d.geom_xpos[i]) if kind == 'geom' else frame(d.xmat[i].reshape(3, 3), d.xpos[i]))
        return out

    def pose(self, q, opening):
        d = self.data; d.qpos[:7] = q; d.qpos[7:9] = finger(opening); mujoco.mj_kinematics(self.model, d)

    def frames(self, q, opening):
        """World frames of the collision parts at joints q, the fingers `opening` apart."""
        self.pose(q, opening); d = self.data; F = np.zeros((len(self.parts), 4, 4))
        for i, (b, L, _) in enumerate(self.parts): F[i] = frame(d.xmat[b].reshape(3, 3), d.xpos[b])@L
        return F

    def tcp_frame(self, q=None):
        """The TCP at joints q (or at the last pose)."""
        if q is not None: self.pose(q, .04)
        R = self.data.xmat[self.hand].reshape(3, 3); return frame(R, self.data.xpos[self.hand]+R[:, 2]*self.tcp)

    def hand_template(self):
        """The hand's, the fingers' and the wrist's (link7) hulls in the TCP frame, the fingers closed: (fixed, left, right)."""
        F = self.frames(self.ready, .003); Ti = np.linalg.inv(self.tcp_frame())
        def at(i): M = Ti@F[i]; return self.hulls[i]@M[:3, :3].T+M[:3, 3]
        return ([at(i) for i, n in enumerate(self.names) if n in ('hand', 'link7')], [at(i) for i, n in enumerate(self.names) if n == 'left_finger'],
                [at(i) for i, n in enumerate(self.names) if n == 'right_finger'])

    def solve(self, T, seeds, rest=None, iterations=150):
        """Joints putting the TCP on T (world): damped least squares from each seed in turn, the null space drawn toward
        `rest` (the seed itself when None), within the joint limits.  None when no seed converges."""
        m, d = self.model, self.data; jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for seed in seeds:
            q = np.clip(np.asarray(seed, float), self.lower+1e-4, self.upper-1e-4); home = q.copy() if rest is None else rest
            for _ in range(iterations):
                d.qpos[:7] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                R = d.xmat[self.hand].reshape(3, 3); p = d.xpos[self.hand]+R[:, 2]*self.tcp
                e = np.r_[T[:3, 3]-p, Rotation.from_matrix(T[:3, :3]@R.T).as_rotvec()]
                if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4: return q
                mujoco.mj_jac(m, d, jp, jr, p, self.hand); A = np.vstack([jp[:, :7], jr[:, :7]])
                dq = A.T@np.linalg.solve(A@A.T+1e-4*np.eye(6), e)+(np.eye(7)-np.linalg.pinv(A)@A)@(.1*(home-q))
                q = np.clip(q+np.clip(dq, -.2, .2), self.lower+1e-4, self.upper-1e-4)
        return None

    def line(self, q, T0, T1):
        """Joints moving the TCP straight from T0 (at joints q) to T1 (the same rotation): tracked every line_step, no
        joint turning more than .15 rad a step.  None when lost."""
        n = max(1, int(np.ceil(np.linalg.norm(T1[:3, 3]-T0[:3, 3])/self.policy['line_step']))); out = []
        for u in np.arange(1, n+1)/n:
            p = self.solve(shifted(T0, u*(T1[:3, 3]-T0[:3, 3])), [q], rest=q)
            if p is None or np.abs(p-q).max() > .15: return None
            out.append(p); q = p
        return out


class World:
    """What the Panda clears in one motion: clouds by name (Cloud, world), the table top at height `table`, the waiting
    work arm (solids at frames)."""
    def __init__(self, panda, clouds, table, work, policy=PANDA):
        self.panda, self.clouds, self.table, self.work, self.policy = panda, clouds, table, work, policy
        self.work_placed = [] if work is None else [placed_solid(s, M) for s, M in zip(*work)]

    def clear(self, q, opening, held=None, skip=(), touching=None):
        """Is the Panda at q (fingers `opening` apart) clear: its moving hulls off every cloud (the fingers not vs
        `touching`) and above the table, off the work arm and its own distant links; the held body (points in the TCP
        frame) `clearance` off every cloud and the table but those in `skip`, outside the work arm and the Panda's links."""
        P = self.panda; F = P.frames(q, opening); moving = [i for i in range(len(F)) if P.level[i] > 0]   # the base stands on the table
        S = [placed_solid(s, M) for s, M in zip(P.solids, F)]
        if min(S[i][2][:, 2].min() for i in moving) < self.table: return False
        for name, cloud in self.clouds.items():
            keep = [i for i in moving if name != touching or i not in P.fingers]
            if pierced([P.pieces[i] for i in keep], F[keep], cloud): return False
        if any(overlap(a, b) for a in S for b in self.work_placed): return False
        if any(overlap(S[i], S[j]) for i, j in P.pairs): return False
        if held is not None:
            T = P.tcp_frame(); W = held@T[:3, :3].T+T[:3, 3]; c = self.policy['clearance']
            if 'table' not in skip and W[:, 2].min() < self.table+c: return False
            for name, cloud in self.clouds.items():
                if name not in skip and np.isfinite(cloud.tree.query(W, distance_upper_bound=c)[0]).any(): return False
            if self.work is not None and any(inside(W, s, M) for s, M in zip(*self.work)): return False
            if any(inside(W, P.solids[i], F[i]) for i in P.links): return False
        return True


def connect(a, b, ok, lower, upper, rng, p=PANDA):
    """RRT-Connect between joints a and b, every edge checked every p['resolution']; then shortcut.  The path, or None."""
    def free(x, y):
        n = int(np.ceil(np.abs(y-x).max()/p['resolution'])); return all(ok(x+(y-x)*t) for t in np.arange(1, n+1)/n)
    if free(a, b): return [a, b]
    trees = [([a], [-1]), ([b], [-1])]; began = time.monotonic()
    def extend(tree, target):
        nodes, parent = tree; i = int(np.argmin(np.linalg.norm(np.asarray(nodes)-target, axis=1))); d = target-nodes[i]; L = np.linalg.norm(d)
        new = target if L <= p['step'] else nodes[i]+d/L*p['step']
        if not free(nodes[i], new): return None, False
        nodes.append(new); parent.append(i); return len(nodes)-1, L <= p['step']
    def trace(tree, i):
        nodes, parent = tree; out = []
        while i >= 0: out.append(nodes[i]); i = parent[i]
        return out                                                                    # from node i back to the root
    for it in range(p['iterations']):
        if time.monotonic()-began > p['seconds']: return None
        A, B = trees[it % 2], trees[1-it % 2]
        i, _ = extend(A, rng.uniform(lower, upper))
        if i is None: continue
        while True:
            j, reached = extend(B, A[0][i])
            if j is None: break
            if reached:
                path = trace(A, i)[::-1]+trace(B, j)[1:]
                return shortcut(path if A is trees[0] else path[::-1], free, rng, p['shortcuts'])
    return None


def shortcut(path, free, rng, tries):
    """Random shortcutting: two points of the path joined straight where that edge is free."""
    path = list(path)
    for _ in range(tries):
        if len(path) < 3: break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j-i > 1 and free(path[i], path[j]): path = path[:i+1]+path[j:]
    return path


def timed(path, panda, fps, p=PANDA):
    """Joints per frame along a joint path: each leg the slowest joint's time at `speed` of its limit, eased over the whole."""
    path = np.asarray(path, float)
    legs = np.r_[0., np.cumsum([(np.abs(b-a)/(panda.velocity*p['speed'])).max() for a, b in zip(path, path[1:])])]
    n = max(2, int(np.ceil(1.3*legs[-1]*fps)))
    if legs[-1] <= 0: return [path[-1]]*n
    t = J.ease(np.linspace(0, 1, n+1)[1:])*legs[-1]
    return [np.array([np.interp(x, legs, path[:, j]) for j in range(7)]) for x in t]


def along(Q, length, fps, p=PANDA):
    """(joints, how far along) per frame along a tracked line `length` (m) long at line_speed, eased; the joints Q
    tracked at even steps along it."""
    Q = np.asarray(Q, float); n = max(2, int(np.ceil(length/p['line_speed']*fps))); s = np.linspace(0, 1, len(Q))
    return [(np.array([np.interp(u, s, Q[:, j]) for j in range(7)]), u) for u in J.ease(np.linspace(0, 1, n+1)[1:])]


class Failed(ValueError):
    pass


class Mover:
    """Pick and place for one Panda: transfers, each from a start configuration to an end one (its park pose)."""
    def __init__(self, panda, fps, seed=0, policy=PANDA):
        self.panda, self.fps, self.policy, self.rng = panda, fps, policy, np.random.default_rng(seed)

    def ik(self, T, world, opening, refs, **check):
        """Joints with the TCP on T, clear in `world`: from the reference joints, then the ready pose, then random seeds."""
        P = self.panda; seeds = [*refs, P.ready, *self.rng.uniform(P.lower, P.upper, (self.policy['seeds'], 7))]
        for s in seeds:
            q = P.solve(T, [s], rest=P.ready)
            if q is not None and world.clear(q, opening, **check): return q
        return None

    def lined(self, q, T0, T1, world, opening, **check):
        Q = self.panda.line(q, T0, T1)
        if Q is None or not all(world.clear(x, opening, **check) for x in Q): return None
        return Q

    def free(self, a, b, world, opening, **check):
        P = self.panda
        return connect(np.asarray(a, float), np.asarray(b, float), lambda q: world.clear(q, opening, **check), P.lower, P.upper, self.rng, self.policy)

    def transfer(self, body, A, B, out, into, grasps, before, after, carry, q0, q1, skip_out=(), skip_in=(), tries=24):
        """The body (points in its frame) from pose A to pose B: out of A along `out`, into B from B shifted by `into`.
        `grasps`: (centre, approach, closing, width) in the body's frame, best first; `before`, `after`, `carry`: the
        Worlds at A (the body among the scenes), at B (likewise) and without the body; q0, q1: the Panda's joints
        before and after (q1 None: it stays where it backed out to).  Returns [(joints, held, opening, body pose)] per frame; the held body's pose follows the TCP."""
        P, p = self.panda, self.policy; failures = {}
        for g, a, c, w in grasps[:tries]:
            for sign in (1., -1.):
                G = np.eye(4); G[:3, 0] = np.cross(sign*c, a); G[:3, 1] = sign*c; G[:3, 2] = a; G[:3, 3] = g
                Gi = np.linalg.inv(G); held = body@Gi[:3, :3].T+Gi[:3, 3]; o = min(p['stroke'], w+2*p['margin'])
                TA, TB = A@G, B@G; aw, bw = TA[:3, 2], TB[:3, 2]
                why = self.attempt(held, Gi, A, B, out, into, TA, TB, aw, bw, o, w, before, after, carry, q0, q1, skip_out, skip_in)
                if isinstance(why, str): failures[why] = failures.get(why, 0)+1; continue
                return why
        raise Failed(f'No transfer: {failures}')

    def attempt(self, held, Gi, A, B, out, into, TA, TB, aw, bw, o, w, before, after, carry, q0, q1, skip_out, skip_in):
        """One grasp: the IK and lines first (cheap), then the free motions.  The frames, or why it failed."""
        P, p, fps = self.panda, self.policy, self.fps
        qA = self.ik(TA, before, o, [q0], touching=None)
        if qA is None: return 'grasp IK'
        pre = self.lined(qA, TA, shifted(TA, -aw*p['pre']), before, o)
        if pre is None: return 'approach line'
        lift = self.lined(qA, TA, shifted(TA, out), carry, w, held=held, skip=skip_out)
        if lift is None: return 'out line'
        qB = self.ik(TB, carry, w, [lift[-1]], held=held, skip=skip_in)
        if qB is None: return 'place IK'
        down = self.lined(qB, TB, shifted(TB, into), carry, w, held=held, skip=skip_in)
        if down is None: return 'in line'
        back = self.lined(qB, TB, shifted(TB, -bw*p['pre']), after, o)
        if back is None: return 'retreat line'
        to = self.free(q0, pre[-1], before, o)
        if to is None: return 'free to the grasp'
        move = self.free(lift[-1], down[-1], carry, w, held=held)
        if move is None: return 'free carry'
        away = [back[-1]] if q1 is None else self.free(back[-1], q1, after, o)
        if away is None: return 'free away'
        # the frames: (joints, held, opening, the body's pose): on the lines the body exactly on its line (the joints
        # tracking it), in the free carry where the hand has it
        grip = max(2, int(round(p['grip_seconds']*fps))); L = lambda v: float(np.linalg.norm(v)); close = J.ease(np.linspace(0, 1, grip+1)[1:])
        F = [(q, False, o, A) for q in timed(to, P, fps)]+[(q, False, o, A) for q, _ in along([pre[-1], *pre[::-1][1:], qA], p['pre'], fps)]
        F += [(qA, True, o+(w-o)*u, A) for u in close]
        F += [(q, True, w, shifted(A, out*u)) for q, u in along([qA, *lift], L(out), fps)]
        F += [(q, True, w, P.tcp_frame(q)@Gi) for q in timed(move, P, fps)]
        F += [(q, True, w, shifted(B, into*(1-u))) for q, u in along([down[-1], *down[::-1][1:], qB], L(into), fps)]
        F += [(qB, True, w+(o-w)*u, B) for u in close]
        F += [(q, False, o, B) for q, _ in along([qB, *back], p['pre'], fps)]+[(q, False, o, B) for q in timed(away, P, fps)]
        return F
