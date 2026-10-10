"""One fixed fixture placement, with an additional native object pose.

The initial material is exactly the first frame of the adjacent Direction
example, rigidly placed in pose 1. No new wrap or relocation sweep is used.
This is a geometric operation specimen, not the production optimizer.
"""
from pathlib import Path
import json
import sys

import numpy as np

CO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CO / 'helper_func'))
import co_common as C
from trimesh.ray.ray_pyembree import RayMeshIntersector

OUT = Path(__file__).resolve().parents[1] / 'vis'
POSES = ['pose_1', 'pose_2', 'pose_4', 'pose_6']
TOL = 1e-10


def mesh_in(mesh, transform):
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def box(bounds):
    bounds = np.asarray(bounds)
    return C.F.md.Manifold.cube(((bounds[1] - bounds[0]) / C.S.SCALE).tolist()).translate(
        (bounds[0] / C.S.SCALE).tolist())


def work_exclusion(mesh, faces, thickness=.0005):
    return C.union([C.S.solid(C.G.hull_mesh(np.vstack([
        mesh.triangles[f], mesh.triangles[f] + thickness * mesh.face_normals[f]
    ]))) for f in faces])


def bottom_link(a, b, bottom, width=.012, height=.006):
    """A small beam, below the object occupancies, joining two post feet."""
    a, b = np.asarray(a)[:2], np.asarray(b)[:2]
    length = np.linalg.norm(b - a)
    if length < 1e-8:
        return box([[a[0]-width/2, a[1]-width/2, bottom],
                    [a[0]+width/2, a[1]+width/2, bottom+height]])
    mesh = C.trimesh.creation.box([length + width, width, height])
    angle = np.arctan2((b-a)[1], (b-a)[0])
    transform = np.array([[np.cos(angle), -np.sin(angle), 0, (a[0]+b[0])/2],
                          [np.sin(angle), np.cos(angle), 0, (a[1]+b[1])/2],
                          [0, 0, 1, bottom+height/2], [0, 0, 0, 1]])
    mesh.apply_transform(transform)
    return C.S.solid(mesh)


def area(triangles):
    if not len(triangles):
        return 0.
    return float(np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                        triangles[:, 2]-triangles[:, 0]), axis=1).sum()/2)


class Specimen:
    def __init__(self):
        base = CO / 'output/B/pose1+2+4+6'
        self.inputs = [base/'step3/step3.1/registered_object.obj',
                       base/'step3/step3.2/wrapped_support.obj',
                       CO/'operation_demo/direction/vis/source_animation_record.json']
        obj = C.trimesh.load(self.inputs[0], force='mesh', process=False)
        wrap = C.trimesh.load(self.inputs[1], force='mesh', process=False)
        source = json.loads(self.inputs[2].read_text())
        self.directions = np.asarray(source['initial_directions_fixture'])
        self.states = [C.state('B', p) for p in POSES]
        for task, _, _ in self.states:
            self.inputs += task.inputs
        self.transforms = [T for _, T, _ in self.states]
        self.host_transform = self.transforms[0]
        self.host = mesh_in(obj, self.host_transform)
        self.guest = mesh_in(obj, self.transforms[1])
        self.direction = self.host_transform[:3, :3] @ self.directions[0]
        self.direction /= np.linalg.norm(self.direction)
        self.sweep_length = float(source['full_exit_length_m'])
        local_exits = [C.S.solid(C.S.swept_solid(obj, self.sweep_length*d, fan_in=8))
                       for d in self.directions]
        local_seed = C.S.solid(wrap) - C.union(local_exits)
        self.initial = C.S.solid(mesh_in(C.S.unpack(local_seed), self.host_transform))
        # Poses 1/4/6 keep their original fixture-relative configurations.
        # Pose 2 is replaced by the native pose 2 object in the pose 1 fixture.
        self.exits = [C.S.solid(mesh_in(C.S.unpack(local_exits[i]), self.host_transform))
                      for i in [0, 2, 3]]
        self.guest_exit = C.S.solid(C.S.swept_solid(
            self.guest, self.sweep_length*self.direction, fan_in=8))
        self.exits.append(self.guest_exit)
        self.body_solids = [C.S.solid(self.host), C.S.solid(self.guest)]
        old_work = np.unique(np.concatenate([self.states[i][0].domain.work_ids for i in [0, 2, 3]]))
        guest_work = self.states[1][0].domain.work_ids
        self.work = C.union([work_exclusion(self.host, old_work),
                             work_exclusion(self.guest, guest_work)])
        self.forbidden = C.union(self.exits + self.body_solids + [self.work])
        raw_retained = self.initial - self.forbidden
        # Discard isolated cut-off scraps below 0.5 cm3, rather than show them
        # floating as if they were useful parts of one rigid fixture.
        self.retained = C.union([p for p in raw_retained.decompose()
                                 if C.material_volume(p) >= .5e-6])
        self.removed = self.initial - self.retained
        self.bottom = float(C.S.unpack(self.initial).bounds[0, 2]) - .004
        self.posts, self.post_records, self.links = self.grow_posts(guest_work)
        self.connections, self.connection_records = self.connect_retained()
        self.posts += self.connections
        self.added = (C.union(self.posts + self.links) - self.forbidden) - self.retained
        self.final = self.retained + self.added
        self.verify()

    def grow_posts(self, work_faces):
        centers = self.guest.triangles_center
        normals = self.guest.face_normals
        allowed = np.setdiff1d(np.flatnonzero((normals[:, 2] < -.45) &
                          (normals @ self.direction < -.08)), work_faces)
        # Only vertically exposed underside patches: no columns through a body.
        origins = centers[allowed].copy()
        origins[:, 2] = self.bottom - .001
        rays = np.tile([0., 0., 1.], (len(allowed), 1))
        points, ray_ids, hit_faces = RayMeshIntersector(self.guest).intersects_location(
            origins, rays, multiple_hits=False)
        accessible = {int(allowed[r]) for p, r, f in zip(points, ray_ids, hit_faces)
                      if np.linalg.norm(p-centers[allowed[r]]) < 1e-6}
        candidates = sorted(accessible, key=lambda f: -self.guest.area_faces[f])
        # Diverse contacts, rather than an entire wrap or a swept envelope.
        chosen = []
        for f in candidates:
            if all(np.linalg.norm(centers[f, :2]-centers[g, :2]) >= .018 for g in chosen):
                chosen.append(f)
            if len(chosen) == 28:
                break
        posts, records = [], []
        for face in chosen:
            p = centers[face]
            width = .014
            blank = box([[p[0]-width/2, p[1]-width/2, self.bottom],
                         [p[0]+width/2, p[1]+width/2, p[2]+.012]])
            clipped = blank - self.forbidden
            grounded = []
            for part in clipped.decompose():
                part_mesh = C.S.unpack(part)
                if part_mesh.bounds[0, 2] <= self.bottom+1e-7:
                    grounded.append(part)
            if not grounded:
                continue
            candidate = C.union(grounded)
            tri, src = C.contact_boundary(self.guest, C.S.unpack(candidate), allowed)
            contact_area = area(tri)
            new_tri, _ = C.contact_boundary(self.guest, C.S.unpack(candidate-self.retained), allowed)
            if contact_area < 12e-6 or area(new_tri) < 6e-6:
                continue
            if any(np.linalg.norm(p[:2]-np.asarray(r['center_m'])[:2]) < .024 for r in records):
                continue
            posts.append(candidate)
            records.append(dict(source_face=int(face), center_m=p.tolist(),
                                contact_area_mm2=contact_area*1e6,
                                new_contact_area_mm2=area(new_tri)*1e6,
                                source_faces=src.tolist()))
            print('POST', len(posts), 'height_mm', round((p[2]-self.bottom)*1000, 1),
                  'contact_mm2', round(contact_area*1e6, 1), flush=True)
            if len(posts) == 5:
                break
        if len(posts) < 3:
            raise RuntimeError(f'Only {len(posts)} actual underside contacts found')
        # Connect new feet to each other and to the retained blue material.
        retained_mesh = C.S.unpack(self.retained)
        low = retained_mesh.vertices[retained_mesh.vertices[:, 2] < self.bottom + .012]
        if not len(low):
            low = retained_mesh.vertices[np.argsort(retained_mesh.vertices[:, 2])[:30]]
        foot = np.asarray(records[0]['center_m'])
        anchor = low[np.argmin(np.linalg.norm(low[:, :2]-foot[:2], axis=1))]
        links = [bottom_link(foot, anchor, self.bottom)]
        for first, second in zip(records[:-1], records[1:]):
            links.append(bottom_link(first['center_m'], second['center_m'], self.bottom))
        return posts, records, links

    def connect_retained(self):
        """Connect surviving blue islands with path-safe bottom struts."""
        additions, records = [], []
        foot = np.asarray(self.post_records[0]['center_m'])
        raycaster = RayMeshIntersector(C.S.unpack(self.forbidden))
        for _ in range(5):
            current = self.retained + (C.union(self.posts+self.links+additions)-self.forbidden)
            parts = sorted([p for p in current.decompose() if C.material_volume(p) > 1e-12],
                           key=lambda p: -C.material_volume(p))
            if len(parts) == 1:
                return additions, records
            floating = [p for p in parts if C.S.unpack(p).bounds[0, 2] > self.bottom+1e-6]
            if not floating:
                raise RuntimeError('Separated base components remain')
            part = floating[0]
            mesh = C.S.unpack(part)
            centers = mesh.triangles_center
            candidates = np.argsort(centers[:, 2])
            origins = centers.copy()
            origins[:, 2] = self.bottom-.001
            hits, ray_ids, _ = raycaster.intersects_location(origins,
                               np.tile([0., 0., 1.], (len(origins), 1)), multiple_hits=False)
            first_z = np.full(len(centers), np.inf)
            first_z[ray_ids] = hits[:, 2]
            connected = False
            for face in candidates:
                p = centers[face]
                if first_z[face] < p[2]-1e-6:
                    continue
                blank = box([[p[0]-.006, p[1]-.006, self.bottom],
                             [p[0]+.006, p[1]+.006, p[2]+.008]]) - self.forbidden
                for column in blank.decompose():
                    if C.S.unpack(column).bounds[0, 2] > self.bottom+1e-6:
                        continue
                    if C.material_volume(column ^ part) < 1e-10:
                        continue
                    link = bottom_link(foot, p, self.bottom)
                    joined = current + ((column+link)-self.forbidden)
                    new_count = sum(C.material_volume(x) > 1e-12 for x in joined.decompose())
                    if new_count >= len(parts):
                        continue
                    additions += [column, link]
                    records.append(dict(top_m=p.tolist(), joined_volume_cm3=C.material_volume(part)*1e6,
                                        component_count_before=len(parts), component_count_after=new_count))
                    print('CONNECT', len(parts), '->', new_count, 'height_mm',
                          round((p[2]-self.bottom)*1000, 1), flush=True)
                    connected = True
                    break
                if connected:
                    break
            if not connected:
                raise RuntimeError('No collision-free bottom connection found for retained material')
        raise RuntimeError('Connection budget exceeded')

    def verify(self):
        initial_body = C.material_volume(self.initial ^ self.body_solids[1])
        initial_path = C.material_volume(self.initial ^ self.guest_exit)
        body_overlaps = [C.material_volume(self.final ^ b) for b in self.body_solids]
        exit_overlaps = [C.material_volume(self.final ^ e) for e in self.exits]
        work_overlap = C.material_volume(self.final ^ self.work)
        added_mesh = C.S.unpack(self.added)
        tri, src = C.contact_boundary(self.guest, added_mesh,
                        np.setdiff1d(np.arange(len(self.guest.faces)), self.states[1][0].domain.work_ids))
        work_checks = []
        for index, (task, T, _) in enumerate(self.states):
            material = C.S.unpack(self.final)
            if index != 1:
                material.apply_transform(T @ np.linalg.inv(self.host_transform))
            result = C.WORK.check(material, task)
            work_checks.append(dict(pose=task.pose, passed=result['passed'],
                forbidden_intersections=result['forbidden_intersection_count'],
                contained_work_faces=len(result['contained_work_face_ids'])))
        self.report = dict(complete=True, demo_only=True, optimizer_run=False,
            force_acceptance_run=False, full_fixture_accepted=False,
            host_pose='pose_1', guest_pose='pose_2', object='B', pose_set='pose1+2+4+6',
            source_shape='Exact initial Direction frame; fixed in native pose 1',
            fixture_moves=False, guest_world_transform=self.transforms[1].tolist(),
            fixture_world_transform=self.host_transform.tolist(),
            guest_direction_world=self.direction.tolist(), full_exit_length_m=self.sweep_length,
            original_pose2_configuration_replaced=True, relocation_path_carved=False,
            construction='Selected bottom-up posts and short foot beams; clipped against all current bodies, work bands and four complete exits; no object/sweep wrap',
            original_fixture_volume_cm3=C.material_volume(self.initial)*1e6,
            removed_volume_cm3=C.material_volume(self.removed)*1e6,
            added_volume_cm3=C.material_volume(self.added)*1e6,
            final_volume_cm3=C.material_volume(self.final)*1e6,
            initial_guest_body_overlap_cm3=initial_body*1e6,
            initial_guest_exit_overlap_cm3=initial_path*1e6,
            final_body_overlap_m3=body_overlaps,
            final_exit_overlap_m3=exit_overlaps,
            exit_configuration_order=['pose_1_original', 'pose_4_original', 'pose_6_original', 'pose_2_juxtaposed'],
            final_work_band_overlap_m3=work_overlap, volume_tolerance_m3=TOL,
            actual_work_surface_checks=work_checks,
            added_guest_contact_area_mm2=area(tri)*1e6,
            added_guest_contact_source_faces=src.tolist(), posts=self.post_records,
            retained_material_connections=self.connection_records,
            bottom_plane_world_z_m=self.bottom, installed_floor_acceptance_run=False,
            final_component_volumes_cm3=sorted([C.material_volume(p)*1e6 for p in self.final.decompose()], reverse=True),
            provenance=C.provenance(self.inputs, [Path(__file__), Path(C.S.__file__)]))
        assert initial_body > 1e-6 and initial_path > 1e-6, 'Initial blockage not demonstrated'
        assert C.material_volume(self.removed) > 1e-6 and C.material_volume(self.added) > 1e-6
        assert max(body_overlaps + exit_overlaps + [work_overlap]) <= TOL, 'Final collision unresolved'
        assert area(tri) > 20e-6, 'Added material does not contact the guest'
        assert sum(C.material_volume(p) > 1e-12 for p in self.final.decompose()) == 1, 'Fixture disconnected'
        assert all(row['passed'] for row in work_checks), 'Working face obstructed'

    def save(self):
        data = OUT / 'data'
        data.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data/'geometry.npz', **{
            f'{name}_{kind}': getattr(mesh, kind)
            for name, mesh in [(name, C.S.unpack(getattr(self, name)))
                               for name in ['initial', 'retained', 'removed', 'added', 'final']]
                              + [('host', self.host), ('guest', self.guest)]
            for kind in ['vertices', 'faces']})
        (data/'geometry.json').write_text(json.dumps(self.report, indent=2)+'\n')


if __name__ == '__main__':
    specimen = Specimen()
    specimen.save()
    print(json.dumps({key: specimen.report[key] for key in [
        'removed_volume_cm3', 'added_volume_cm3', 'initial_guest_body_overlap_cm3',
        'final_body_overlap_m3', 'final_exit_overlap_m3', 'added_guest_contact_area_mm2',
        'final_component_volumes_cm3']}, indent=2))
