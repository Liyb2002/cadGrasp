"""Small surface-following snap lips for the Step6 concept animation.

These are presentation geometry, not fabricated or mechanically certified parts.
The rigid baseline and its bearing certificates are deliberately not modified.
"""
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


class SnapLips:
    def __init__(self, obj, blue, work_ids, direction, rest, stroke):
        self.obj = obj
        self.direction = np.asarray(direction)
        vertices, normals = obj.vertices, obj.vertex_normals
        surface, distance, _ = blue.nearest.on_surface(vertices)
        work_vertices = np.unique(obj.faces[work_ids])
        work_distance = cKDTree(vertices[work_vertices]).query(vertices)[0]
        floor = trimesh.transform_points(vertices, rest)[:, 2]
        allowed = (work_distance > .008) & (floor > .006) & (vertices[:, 2] > .006)
        roots = np.flatnonzero(allowed & (distance < .0035) & (normals @ self.direction > .05))
        if not len(roots):
            raise ValueError('No exposed contact-module root for a snap-lip illustration')
        edges = obj.edges_unique
        edges = edges[allowed[edges].all(axis=1)]
        lengths = np.linalg.norm(vertices[edges[:, 1]]-vertices[edges[:, 0]], axis=1)
        graph = coo_matrix((np.r_[lengths, lengths],
                           (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                          shape=(len(vertices), len(vertices))).tocsr()
        dist, previous = dijkstra(graph, directed=False, indices=roots, return_predecessors=True)
        root_indices = dist.argmin(axis=0)
        shortest = dist[root_indices, np.arange(len(vertices))]
        targets = np.flatnonzero(allowed & (normals @ self.direction < -.20)
                                 & (shortest > .010) & (shortest < .080))
        targets = targets[np.argsort(shortest[targets])]
        self.lips = []
        for end in targets:
            if any(np.linalg.norm(vertices[end]-lip['surface'][-1]) < .020 for lip in self.lips):
                continue
            row = root_indices[end]
            ids = [int(end)]
            while ids[-1] != roots[row]:
                ids.append(int(previous[row, ids[-1]]))
            ids = np.array(ids[::-1])
            p, n = vertices[ids], normals[ids]
            # The root overlaps the existing blue module. The flexible strip
            # follows the object and terminates beyond a local shoulder.
            p = np.vstack([surface[ids[0]], p + n*.0013])
            n = np.vstack([n[0], n])
            lip=dict(surface=vertices[ids], points=p, normals=n,
                     width_m=.005, thickness_m=.0012)
            # Reject visibly intersecting presentation paths. This sparse
            # centerline check is not a finite-thickness collision certificate.
            for amplitude in (.006, .010, .014):
                if amplitude>.45*np.linalg.norm(np.diff(p,axis=0),axis=1).sum():
                    continue
                lip['opening_m']=amplitude
                samples=[]
                for fraction in np.linspace(0,1,25):
                    line=self.deform(lip,fraction)+self.direction*stroke*(1-fraction)
                    samples.extend(a+np.linspace(0,1,8)[:,None]*(b-a)
                                   for a,b in zip(line[:-1],line[1:]))
                samples=np.vstack(samples)
                if trimesh.transform_points(samples,rest)[:,2].min()<.001:
                    continue
                inside=obj.contains(samples)
                penetration=float(obj.nearest.signed_distance(samples[inside]).max()) if inside.any() else 0.
                if penetration<.00005:
                    lip['sampled_centerline_max_penetration_m']=penetration
                    self.lips.append(lip)
                    break
            if len(self.lips) == 2:
                break
        if not self.lips:
            raise ValueError('No local shoulder available for a snap-lip illustration')
        self.center = np.mean(np.vstack([lip['points'] for lip in self.lips]), axis=0)

    def curves(self, installation_fraction):
        """Prescribed opening and recovery; no elastic force is inferred."""
        return [self.deform(lip,installation_fraction) for lip in self.lips]

    @staticmethod
    def deform(lip, installation_fraction):
        u = float(np.clip(installation_fraction, 0, 1))
        opening = np.sin(np.pi*np.clip((u-.35)/.65, 0, 1))**.7
        p, n = lip['points'], lip['normals']
        arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        fraction = arclength/arclength[-1]
        return p + n*(lip['opening_m']*opening*fraction**1.6)[:, None]

    def add_to_scene(self, mujoco, scene, transform, installation_fraction):
        rotation = transform[:3, :3]
        for lip, curve in zip(self.lips, self.curves(installation_fraction)):
            points = trimesh.transform_points(curve, transform)
            normals = lip['normals'] @ rotation.T
            for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
                along = b-a
                length = np.linalg.norm(along)
                if length < 1e-7:
                    continue
                along /= length
                side = np.cross(along, normals[i]+normals[i+1])
                side /= np.linalg.norm(side)
                normal = np.cross(side, along)
                geom = scene.geoms[scene.ngeom]
                scene.ngeom += 1
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([lip['width_m']/2, (length+.0005)/2, lip['thickness_m']/2]),
                    (a+b)/2, np.column_stack([side, along, normal]).ravel(),
                    np.array([.12, .52, .88, 1.]))

    def metadata(self):
        return dict(mechanism='integral compliant snap lips over existing object shoulders',
                    role='retain blue on the object while the robot grasps the object',
                    geometry_scope='presentation only; excluded from baseline bearing and sweep certificates',
                    deformation='prescribed opening/recovery, not an elastic simulation',
                    retention_force_verified=False, relative_slip_verified=False,
                    clip_collision_verified=False, fabricated=False,
                    lips=[dict(points_m=lip['points'].tolist(), width_m=lip['width_m'],
                               thickness_m=lip['thickness_m'],opening_m=lip['opening_m'],
                               sampled_centerline_max_penetration_m=lip['sampled_centerline_max_penetration_m'])
                          for lip in self.lips])
