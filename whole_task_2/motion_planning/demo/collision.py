"""Triangle-mesh collision checks with FCL; pad contacts are small, explicit exceptions."""
from pathlib import Path

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from robot import points, rigid, transform


def geometry(mesh):
    model = fcl.BVHModel()
    model.beginModel(len(mesh.vertices), len(mesh.faces))
    model.addSubModel(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))
    model.endModel()
    return model


def placed(geom, t):
    return fcl.CollisionObject(geom, fcl.Transform(t[:3, :3].copy(), t[:3, 3].copy()))


def contacts(a, b):
    result = fcl.CollisionResult()
    fcl.collide(a, b, fcl.CollisionRequest(num_max_contacts=2000, enable_contact=True), result)
    return result.contacts


class World:
    def __init__(self, robot, body, request, directory):
        self.robot, self.body, self.request = robot, body, request
        self.table = float(request['table_top_z_m'])
        self.geometries = [geometry(p['mesh']) for p in robot.parts]
        self.body_geometry = geometry(body)
        self.obstacles = []
        self.support_names = set()
        for spec in request.get('obstacles', []):
            if spec.get('supports_bunny', False):
                self.support_names.add(spec['name'])
            t = rigid(spec.get('world_from_mesh', np.eye(4)))
            if 'box_extents_m' in spec:
                mesh = trimesh.creation.box(extents=spec['box_extents_m'])
                geom = fcl.Box(*spec['box_extents_m'])
            else:
                mesh = trimesh.load(Path(directory) / spec['mesh'], force='mesh', process=True)
                unit = spec.get('mesh_units', 'm')
                if unit not in ('m', 'mm'):
                    raise ValueError('Obstacle mesh_units must be m or mm')
                mesh.vertices *= .001 if unit == 'mm' else 1.
                geom = geometry(mesh)
            self.obstacles.append((spec['name'], mesh, t, placed(geom, t)))
        self.calls = 0
        self.last_contact_failure = None

    def pad_contact_failure(self, hits, part, object_pose, grasp, gripper_pose, jaw):
        """One policy shared by screening, jaw motion, candidate replay and independent replay."""
        if grasp is None or part['contact'] is None:
            return dict(reason='arm_bunny', part=part['name'])
        which = part['contact']
        centre = np.asarray(grasp['contacts_object_m'][0 if which == 'fixed' else 1])
        locations = np.array([h.pos for h in hits])
        body_local = points(locations, np.linalg.inv(object_pose))
        radius = float(np.linalg.norm(body_local - centre, axis=1).max())
        depth = float(max(h.penetration_depth for h in hits))
        extent = self.robot.gripper.get('contact_patch_half_extents_m', {}).get(which)
        pad_local = points(locations, np.linalg.inv(self.robot.pad_frame(gripper_pose, jaw, which)))
        outside_box = extent is not None and (np.abs(pad_local) > np.asarray(extent) + 1e-8).any()
        if (len(hits) >= 2000 or radius > self.robot.gripper['contact_patch_radius_m'] or
                depth > self.robot.gripper['contact_penetration_m'] or outside_box):
            return dict(reason='outside_pad_contact', part=part['name'], contact_count=len(hits),
                maximum_radius_m=radius, radius_limit_m=self.robot.gripper['contact_patch_radius_m'],
                maximum_penetration_m=depth, penetration_limit_m=self.robot.gripper['contact_penetration_m'],
                maximum_pad_local_abs_m=np.abs(pad_local).max(0).tolist(), pad_half_extents_m=extent)
        return None

    def has_support(self, object_pose):
        """A placement needs the table or a declared, nearby support. This is not a stability proof."""
        if abs(points(self.body.vertices, object_pose)[:, 2].min() - self.table) <= .001:
            return True
        bunny = placed(self.body_geometry, object_pose)
        for name, _, _, obstacle in self.obstacles:
            if name in self.support_names:
                result = fcl.DistanceResult()
                distance = fcl.distance(bunny, obstacle, fcl.DistanceRequest(), result)
                if distance <= .002:
                    return True
        return False

    def check(self, q, jaw, object_pose, grasp=None, support_touch=()):
        """A reason string on failure, otherwise None. No entire hand or fixture is ignored."""
        self.calls += 1
        self.last_contact_failure = None
        r = self.robot
        if (not np.isfinite(q).all() or not np.isfinite(jaw) or
                np.any(q < r.lower - 1e-8) or np.any(q > r.upper + 1e-8) or
                jaw < r.jaw_limits[0] - 1e-8 or jaw > r.jaw_limits[1] + 1e-8):
            return 'joint_limits'
        gripper_pose, frames = r.frames(q, jaw)
        t = rigid(object_pose)
        bp = points(self.body.vertices, t)
        if bp[:, 2].min() < self.table - 2e-5:
            return 'bunny_table'
        objects = []
        for part, geom in zip(r.parts, self.geometries):
            m = frames[part['link']]
            # Base mounting on the table is an explicit, limited exception.
            if part['level'] > 0 and points(part['table_vertices'], m)[:, 2].min() < self.table - 2e-5:
                return 'arm_table:' + part['name']
            objects.append(placed(geom, m))
        bunny = placed(self.body_geometry, t)
        for i, part in enumerate(r.parts):
            hits = contacts(objects[i], bunny)
            if hits:
                failure = self.pad_contact_failure(hits, part, t, grasp, gripper_pose, jaw)
                if failure:
                    self.last_contact_failure = failure
                    return failure['reason'] + ':' + part['name']
            else:
                # Mesh/mesh FCL checks surfaces; detect a part entirely enclosed by the solid bunny too.
                centre = points(points(part['mesh'].centroid, frames[part['link']]), np.linalg.inv(t))[None]
                if ((centre >= self.body.bounds[0]).all() and (centre <= self.body.bounds[1]).all() and
                        self.body.contains(centre).any()):
                    return 'arm_inside_bunny:' + part['name']
        # Non-neighbour links. Parts of one link and directly connected links are mechanical assemblies.
        for i, a in enumerate(r.parts):
            for j in range(i + 1, len(r.parts)):
                b = r.parts[j]
                adjacent = abs(a['level'] - b['level']) <= 1
                if adjacent:
                    continue
                if contacts(objects[i], objects[j]):
                    return 'self_collision:' + a['link'] + ':' + b['link']
        for name, mesh, m, obstacle in self.obstacles:
            if contacts(bunny, obstacle):
                hits = contacts(bunny, obstacle)
                if name not in support_touch or max(h.penetration_depth for h in hits) > 2e-5:
                    return 'bunny_obstacle:' + name
            for part, obj in zip(r.parts, objects):
                if contacts(obj, obstacle):
                    return 'arm_obstacle:' + name + ':' + part['name']
            # BVH tests detect surface intersections. Also reject complete solid enclosure.
            if mesh.is_watertight:
                local = points(bp[[0]], np.linalg.inv(m))
                if ((local >= mesh.bounds[0]).all() and (local <= mesh.bounds[1]).all() and mesh.contains(local).any()):
                    return 'bunny_inside_obstacle:' + name
                centres = np.array([points(part['mesh'].centroid, frames[part['link']]) for part in r.parts])
                local = points(centres, np.linalg.inv(m))
                candidates = ((local >= mesh.bounds[0]) & (local <= mesh.bounds[1])).all(1)
                if candidates.any() and mesh.contains(local[candidates]).any():
                    return 'arm_inside_obstacle:' + name
        return None

    def hand_clear(self, gripper_pose, jaw, object_pose, grasp):
        """Cheap rejection of impossible grasps before solving the five-axis arm."""
        r = self.robot
        self.last_contact_failure = None
        link = gripper_pose @ np.linalg.inv(r.tcp_local)
        moving = link @ r.jaw_local @ transform(Rotation.from_rotvec(r.jaw_axis * jaw).as_matrix())
        bunny = placed(self.body_geometry, object_pose)
        for part, geom in zip(r.parts, self.geometries):
            if part['level'] < 5:
                continue
            m = link if part['level'] == 5 else moving
            if points(part['table_vertices'], m)[:, 2].min() < self.table - 2e-5:
                return False
            obj = placed(geom, m)
            hits = contacts(obj, bunny)
            if hits:
                failure = self.pad_contact_failure(hits, part, object_pose, grasp, gripper_pose, jaw)
                if failure:
                    self.last_contact_failure = failure
                    return False
            if any(contacts(obj, obstacle) for _, _, _, obstacle in self.obstacles):
                return False
        return True
