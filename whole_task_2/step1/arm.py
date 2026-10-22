"""The arm behind the head: can the work arm (KUKA or SO101) hold the needle on a face, without its body touching the object?

reach.py certifies the needle and the head (everything within the head radius
behind the needle).  Here the rest of the arm: for a (face, side) pair the
object is placed as that side places it, and at each of the face's test points
(three vertices and the centroid) the work arm must
  - have an inverse-kinematics solution putting the tip there along the pair's
    certified needle axis, inside joint limits and above the floor (Arm.solve);
  - keep every link clear of the object: the convex hull of each link's collision
    mesh (a superset of the link) meets no object triangle.
The triangle-hull test is exact: a vertex inside the hull is a hit, three vertices
beyond one hull plane are clear, anything else is clipped against every plane.
A solution that touches the object is retried from the arm's other seeds before
the pair is refused.  Joint angles are kept so that validation
replays them with forward kinematics alone.
"""
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from .reach import test_points, tip_contact
from .workspace import points_inside_box

OK, NO_IK, ARM_HITS, TABLE_HITS, POINT_OUTSIDE_BOX = 0, 1, 2, 3, 4
POSITION_TOL, AXIS_TOL = 1e-4, 1e-3          # Arm.solve's acceptance


def clipped(polygon, normals, offsets):
    """Does the convex polygon keep any part inside every plane n.x + d <= 0?"""
    for n, d in zip(normals, offsets):
        s = polygon@n+d; inside = s <= 0
        if inside.all(): continue
        if not inside.any(): return False
        out = []
        for i in range(len(polygon)):
            j = (i+1) % len(polygon)
            if inside[i]: out.append(polygon[i])
            if inside[i] != inside[j]: out.append(polygon[i]+(polygon[j]-polygon[i])*(s[i]/(s[i]-s[j])))
        polygon = np.asarray(out)
    return True


class Check:
    def __init__(self, mesh, setup):
        from .reach_video import work_arm
        self.mesh = mesh; self.arm = arm = work_arm(setup); self.floor = float(setup['base_position_world_m'][2])-.001
        self.table_bounds = np.asarray(setup['table_bounds_world_m']) if 'table_bounds_world_m' in setup else None
        self.table = None
        bounds = setup['surface_workspace']['bounds_world_m'] if 'surface_workspace' in setup else None
        self.work_bounds = np.asarray(bounds) if bounds is not None else None
        self.length=setup['needle_length_m'];self.needle_radius=setup['needle_radius_m'];self.head_radius=setup['head_radius_m']
        if self.table_bounds is not None:
            import trimesh
            self.table=trimesh.creation.box(bounds=self.table_bounds).triangles
        self.hulls = []
        for v in arm.collisions:
            h = ConvexHull(v); c = v.mean(0)
            self.hulls.append((h.equations[:, :3], h.equations[:, 3], c, float(np.linalg.norm(v-c, axis=1).max())))
        self.seeds = arm.seeds
        self.placed = {}

    def scene(self, W):
        """The placed object's triangles, a tree on their centroids and the largest centroid-vertex distance."""
        key = W.tobytes()
        if key not in self.placed:
            T = self.mesh.triangles@W[:3, :3].T+W[:3, 3]
            C = T.mean(1)
            self.placed = {key: (T, cKDTree(C), float(np.linalg.norm(T-C[:, None], axis=2).max()))}
        return self.placed[key]

    def hits(self, q, W):
        T, tree, rho = self.scene(W)
        for (n, d, c, r), (R, p) in zip(self.hulls, self.arm.forward(q, geometry=True)[2]):
            center=R@c+p;near = tree.query_ball_point(center, r+rho);candidates=T[near]
            if self.table is not None and np.all(center+r>=self.table_bounds[0]) and np.all(center-r<=self.table_bounds[1]):
                candidates=np.concatenate([candidates,self.table])
            if not len(candidates): continue
            X = (candidates-p)@R; s = X@n.T+d                                  # (k, 3, planes), link frame
            if (s <= 0).all(2).any(): return True
            for k in np.flatnonzero(~(s > 0).all(1).any(1)):
                if clipped(X[k], n, d): return True
        return False

    def tool_above_table(self, face, axis, W):
        """Cylinder cross-sections beyond the tip contact must stay above the tabletop."""
        if self.table_bounds is None:return True
        P=test_points(self.mesh)[face]@W[:3,:3].T+W[:3,3];d=W[:3,:3]@axis
        if d[2]>=0:return False
        radial_z=np.sqrt(max(0.,1-d[2]**2));z=self.table_bounds[1,2]
        start=float(tip_contact(self.needle_radius,-self.mesh.face_normals[face]@axis))
        return bool(np.all(P[:,2]-start*d[2]-self.needle_radius*radial_z>=z-1e-9) and
                    np.all(P[:,2]-self.length*d[2]-self.head_radius*radial_z>=z-1e-9))

    def solve(self, point, axis, W, previous):
        """Joints with the tip on point along axis and the body clear of the object (None: no IK, False: always hits)."""
        bounds=getattr(self,'work_bounds',None)
        if bounds is not None and not points_inside_box(point,bounds):return None
        found = False
        for seed in ([previous] if previous is not None else [])+self.seeds:
            try: q = self.arm.solve(point, -axis, seed)
            except ValueError: continue
            found = True
            if bounds is not None:
                tcp,_,_,_=self.arm.forward(q)
                if not points_inside_box(tcp,bounds):continue
            if not self.hits(q, W): return q
        return False if found else None

    def pair(self, face, axis, W, previous=None):
        """(code, joints (4, dof)) for one (face, side) pair; axis in object coordinates."""
        P = test_points(self.mesh)[face]@W[:3, :3].T+W[:3, 3]; d = W[:3, :3]@axis; Q = np.zeros((4, self.arm.dof))
        if self.work_bounds is not None and not points_inside_box(P,self.work_bounds).all():return POINT_OUTSIDE_BOX,Q
        if not self.tool_above_table(face,axis,W):return TABLE_HITS,Q
        for i, p in enumerate(P):
            q = self.solve(p, d, W, previous)
            if q is None: return NO_IK, Q
            if q is False: return ARM_HITS, Q
            Q[i] = previous = q
        return OK, Q

    def replay(self, face, axis, W, Q):
        """Forward kinematics alone: every test point held along axis, within limits, above the floor, body clear."""
        P = test_points(self.mesh)[face]@W[:3, :3].T+W[:3, 3]; d = W[:3, :3]@axis
        if self.work_bounds is not None and not points_inside_box(P,self.work_bounds).all():return False
        if not self.tool_above_table(face,axis,W):return False
        for p, q in zip(P, Q):
            tcp, z, _, _ = self.arm.forward(q)
            if self.work_bounds is not None and not points_inside_box(tcp,self.work_bounds):return False
            if np.linalg.norm(tcp-p) > POSITION_TOL or np.linalg.norm(z-d) > AXIS_TOL: return False
            if np.any(q < self.arm.lower-1e-9) or np.any(q > self.arm.upper+1e-9): return False
            links = self.arm.forward(q, geometry=True)[2]
            if min((v@R.T+o)[:, 2].min() for v, (R, o) in zip(self.arm.collisions[1:], links[1:])) < self.floor: return False
            if self.hits(q, W): return False
        return True


def order(points, cell=.01):
    """A walk over nearby points (rows of `cell`), so each solve starts from its neighbour's joints."""
    row = np.floor(points[:, 0]/cell).astype(int); y = np.where(row % 2 == 0, points[:, 1], -points[:, 1])
    return np.lexsort((y, row))
