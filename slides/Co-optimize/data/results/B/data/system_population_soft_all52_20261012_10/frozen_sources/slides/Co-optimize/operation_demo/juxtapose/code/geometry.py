"""Add the same fitted wrapping wherever all current sweeps permit it.

The initial fixture is the adjacent Direction example's first frame in pose1.
The juxtaposed pose2 gets a dense fitted shell using the Step3.2 method.
All bodies, complete exits and work exclusions are subtracted from the union.
No posts, foot beams or hand-picked contact patches are constructed.
"""
from pathlib import Path
import json
import sys

import numpy as np

CO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CO / 'helper_func'))
import co_common as C

OUT = Path(__file__).resolve().parents[1] / 'vis'
POSES = ['pose_1', 'pose_2', 'pose_4', 'pose_6']
TOL = 1e-10
GROWTH_STEPS = 8


def mesh_in(mesh, transform):
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def box(bounds):
    bounds = np.asarray(bounds)
    return C.F.md.Manifold.cube(((bounds[1] - bounds[0]) / C.S.SCALE).tolist()).translate(
        (bounds[0] / C.S.SCALE).tolist())


def work_exclusion(mesh, faces, thickness=.0005):
    # Include a 1um inward band so a Boolean boundary cannot leave vertices
    # touching a work-face interior. This is numerical relief, not an exit
    # clearance or a manufacturing tolerance certificate.
    return C.union([C.S.solid(C.G.hull_mesh(np.vstack([
        mesh.triangles[f] - 1e-6 * mesh.face_normals[f],
        mesh.triangles[f] + thickness * mesh.face_normals[f]
    ]))) for f in faces])


def fitted_wrap(mesh, work_faces, thickness):
    """Step3.2's dense, bounded outer skin on EVERY nonworking source face."""
    offsets = C.wrap_offsets(mesh, thickness)
    cells = [C.S.solid(C.G.hull_mesh(C.G.head_cell(mesh, triangle, i, offsets)))
             for i, triangle in enumerate(mesh.triangles)]
    allowed = np.setdiff1d(np.arange(len(mesh.faces)), work_faces)
    return (C.union([cells[i] for i in allowed]) - C.S.solid(mesh)
            - C.union([cells[i] for i in work_faces]))


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
                       CO/'operation_demo/direction/vis/source_animation_record.json',
                       base/'step3/step3.2/data/report.json']
        obj = C.trimesh.load(self.inputs[0], force='mesh', process=False)
        wrap = C.trimesh.load(self.inputs[1], force='mesh', process=False)
        source = json.loads(self.inputs[2].read_text())
        self.thickness = float(json.loads(self.inputs[3].read_text())['maximum_wrap_vertex_displacement_m'])
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
        # Poses1/4/6 keep their original fixture-relative configurations;
        # pose2 is replaced by native pose2 in the fixed pose1 fixture placement.
        self.exits = [C.S.solid(mesh_in(C.S.unpack(local_exits[i]), self.host_transform))
                      for i in [0, 2, 3]]
        self.guest_exit = C.S.solid(C.S.swept_solid(
            self.guest, self.sweep_length*self.direction, fan_in=8))
        self.exits.append(self.guest_exit)
        self.body_solids = [C.S.solid(self.host), C.S.solid(self.guest)]
        old_work = np.unique(np.concatenate([self.states[i][0].domain.work_ids for i in [0, 2, 3]]))
        self.guest_work = self.states[1][0].domain.work_ids
        self.work = C.union([work_exclusion(self.host, old_work),
                             work_exclusion(self.guest, self.guest_work)])
        self.forbidden = C.union(self.exits + self.body_solids + [self.work])
        self.retained = self.initial - self.forbidden
        self.removed = self.initial - self.retained
        self.guest_wrap = fitted_wrap(self.guest, self.guest_work, self.thickness)
        self.available_wrap = self.guest_wrap - self.forbidden
        self.added = self.available_wrap - self.retained
        self.final = self.retained + self.added
        self.verify()

    def verify(self):
        initial_body = C.material_volume(self.initial ^ self.body_solids[1])
        initial_path = C.material_volume(self.initial ^ self.guest_exit)
        body_overlaps = [C.material_volume(self.final ^ b) for b in self.body_solids]
        exit_overlaps = [C.material_volume(self.final ^ e) for e in self.exits]
        work_overlap = C.material_volume(self.final ^ self.work)
        outside_wrap = C.material_volume(self.added - self.guest_wrap)
        missing_wrap = C.material_volume(self.available_wrap - self.final)
        exact_final = (self.initial + self.guest_wrap) - self.forbidden
        formula_error = C.material_volume(self.final-exact_final)+C.material_volume(exact_final-self.final)
        allowed = np.setdiff1d(np.arange(len(self.guest.faces)), self.guest_work)
        tri, src = C.contact_boundary(self.guest, C.S.unpack(self.added), allowed)
        work_checks = []
        for index, (task, T, _) in enumerate(self.states):
            material = C.S.unpack(self.final)
            if index != 1:
                material.apply_transform(T @ np.linalg.inv(self.host_transform))
            result = C.WORK.check(material, task)
            work_checks.append(dict(pose=task.pose, passed=result['passed'],
                forbidden_intersections=result['forbidden_intersection_count'],
                contained_work_faces=len(result['contained_work_face_ids'])))
        parts = [C.material_volume(p)*1e6 for p in self.final.decompose()]
        self.report = dict(complete=True, demo_only=True, optimizer_run=False,
            force_acceptance_run=False, full_fixture_accepted=False,
            host_pose='pose_1', guest_pose='pose_2', object='B', pose_set='pose1+2+4+6',
            source_shape='Exact initial Direction frame; fixed in native pose1',
            fixture_moves=False, guest_world_transform=self.transforms[1].tolist(),
            fixture_world_transform=self.host_transform.tolist(),
            guest_direction_world=self.direction.tolist(), full_exit_length_m=self.sweep_length,
            original_pose2_configuration_replaced=True, relocation_path_carved=False,
            construction='Dense fitted Step3.2 wrap on every nonworking guest face, minus ALL current bodies, work exclusions and four complete exits; no selected posts or foot beams',
            fitted_wrap_maximum_vertex_displacement_m=self.thickness,
            nonworking_guest_face_count=len(allowed), posts_used=False,
            original_fixture_volume_cm3=C.material_volume(self.initial)*1e6,
            full_guest_wrap_volume_cm3=C.material_volume(self.guest_wrap)*1e6,
            removed_volume_cm3=C.material_volume(self.removed)*1e6,
            added_volume_cm3=C.material_volume(self.added)*1e6,
            final_volume_cm3=C.material_volume(self.final)*1e6,
            initial_guest_body_overlap_cm3=initial_body*1e6,
            initial_guest_exit_overlap_cm3=initial_path*1e6,
            final_body_overlap_m3=body_overlaps, final_exit_overlap_m3=exit_overlaps,
            exit_configuration_order=['pose_1_original', 'pose_4_original', 'pose_6_original', 'pose_2_juxtaposed'],
            final_work_band_overlap_m3=work_overlap, volume_tolerance_m3=TOL,
            work_exclusion_inward_relief_m=1e-6,
            added_material_outside_fitted_wrap_m3=outside_wrap,
            available_fitted_material_missing_m3=missing_wrap,
            final_union_minus_forbidden_error_m3=formula_error,
            actual_work_surface_checks=work_checks,
            added_guest_contact_area_mm2=area(tri)*1e6,
            added_guest_contact_source_faces=src.tolist(),
            installed_floor_acceptance_run=False,
            final_component_volumes_cm3=sorted(parts, reverse=True),
            meaningful_component_count=sum(volume > 1e-6 for volume in parts),
            provenance=C.provenance(self.inputs, [Path(__file__), Path(C.S.__file__), Path(C.G.__file__)]))
        assert initial_body > 1e-6 and initial_path > 1e-6, 'Initial blockage not demonstrated'
        assert C.material_volume(self.removed) > 1e-6 and C.material_volume(self.added) > 1e-6
        assert max(body_overlaps+exit_overlaps+[work_overlap]) <= TOL, 'Final collision unresolved'
        assert max(outside_wrap, missing_wrap, formula_error) <= TOL, 'Fitted-wrap coverage changed'
        assert area(tri) > 20e-6, 'Added wrap does not contact the guest'
        assert all(row['passed'] for row in work_checks), 'Working face obstructed'

    def save(self):
        data = OUT / 'data'
        data.mkdir(parents=True, exist_ok=True)
        meshes = [(name, C.S.unpack(getattr(self, name))) for name in
                  ['initial', 'retained', 'removed', 'added', 'final', 'guest_wrap']]
        meshes += [('host', self.host), ('guest', self.guest)]
        # Grow wall thickness over all available surface patches simultaneously,
        # rather than suggest that this operation selects bottom pillars.
        for index in range(1, GROWTH_STEPS+1):
            fraction = index/GROWTH_STEPS
            if index == GROWTH_STEPS:
                grown = self.added
            else:
                thin_wrap = fitted_wrap(self.guest, self.guest_work, self.thickness*fraction)
                grown = thin_wrap ^ self.added
            meshes.append((f'growth_{index}', C.S.unpack(grown)))
            print('FITTED WRAP', index, '/', GROWTH_STEPS, flush=True)
        np.savez_compressed(data/'geometry.npz', **{
            f'{name}_{kind}': getattr(mesh, kind)
            for name, mesh in meshes for kind in ['vertices', 'faces']})
        (data/'geometry.json').write_text(json.dumps(self.report, indent=2)+'\n')
        (data/'work_surface_checks.json').write_text(json.dumps(self.report['actual_work_surface_checks'], indent=2)+'\n')


if __name__ == '__main__':
    specimen = Specimen()
    specimen.save()
    print(json.dumps({key: specimen.report[key] for key in [
        'removed_volume_cm3', 'added_volume_cm3', 'initial_guest_body_overlap_cm3',
        'final_body_overlap_m3', 'final_exit_overlap_m3', 'added_guest_contact_area_mm2',
        'available_fitted_material_missing_m3', 'added_material_outside_fitted_wrap_m3',
        'final_component_volumes_cm3']}, indent=2))
