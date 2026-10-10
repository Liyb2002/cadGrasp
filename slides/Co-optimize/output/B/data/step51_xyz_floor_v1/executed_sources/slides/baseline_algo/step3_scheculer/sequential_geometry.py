"""Task-local head pools for the fixed 3 + shared-one + 2 experiment.

Only active contact sets are certified here. Inactive material, fixture placement,
feet and the complete five-part solid remain Step5 design variables.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh

from step2_local_support import circles as P, surface as S, geometry as G
from step2_local_support import withdrawal as W
from step3_scheculer.pair_geometry import PairGeometry, transform_contact
from step3_scheculer import contacts as I


def active_entries(entries, task):
    return [e for e in entries if task in e['active_tasks']]


def local_group(geometry, entries):
    if not entries or any(not e['valid'] for e in entries):
        return dict(passed=False, reason='empty_or_invalid_active_head_set')
    directions = set(range(len(geometry.catalogues[0]['vectors'])))
    components = set(geometry.paths.labels.tolist())
    for e in entries:
        directions &= set(e['directions'][0])
        components &= set(e['path_components'])
    return dict(passed=bool(directions and components),
                reason='passed' if directions and components else 'no_common_path_or_direction',
                common_direction_ids=sorted(directions), common_path_components=sorted(components))


class SequentialGeometry:
    def __init__(self, problems, count=200, *, global_head_exclusions=False):
        self.problems, self.count = problems, count
        self.global_head_exclusions = global_head_exclusions
        # PairGeometry's constructor/make support a single task. Its pair-only
        # group_check is deliberately replaced by local_group above.
        self.local = [PairGeometry([p], count=count,
            head_exclusion_problems=problems if global_head_exclusions else None) for p in problems]
        frames = [np.asarray(p.domain.data['frame']['T_world_mesh']) for p in problems]
        self.transforms = [[b@np.linalg.inv(a) for b in frames] for a in frames]
        self.cache = {}
        self.pools = [[self.wrap(e, k, (k,)) for e in geometry.initial]
                      for k, geometry in enumerate(self.local)]

    def wrap(self, native, owner, active):
        contact = dict(native['contact'])
        local_index = contact['candidate_index']
        contact.update(candidate_index=owner*self.count+local_index,
                       candidate_id=f'{self.problems[owner].pose}_C{local_index+1:03d}')
        return dict(native, contact=contact, native=native, owner_task=owner,
                    local_index=local_index, active_tasks=tuple(active))

    def view(self, entry, task):
        if entry['owner_task'] == task:
            return dict(entry['native'], contact=entry['contact'])
        contact = transform_contact(entry['contact'], self.transforms[entry['owner_task']][task])
        key = (contact['candidate_id'], contact['radius_m'], task)
        if key in self.cache:
            return self.cache[key]
        geometry = self.local[task]
        row = dict(contact=contact, area=entry['area'], valid=False, reason='work_surface_overlap')
        self.cache[key] = row
        if np.intersect1d(contact['source_faces'], self.problems[task].domain.work_ids).size:
            return row
        row['reason'] = 'contact_floor_clearance'
        if contact['triangles_m'][:, :, 2].min() < S.FLOOR_CLEARANCE_M-geometry.scale*1e-10:
            return row
        if not entry['native']['valid']:
            row['reason'] = entry['native']['reason']
            return row
        # Transform the SAME owner solid. Re-extruding with the other task's
        # bbox-dependent depth would silently change the shared head's shape.
        owner = self.local[entry['owner_task']]
        original = entry['contact']
        transform = self.transforms[entry['owner_task']][task]
        rotation, translation = transform[:3, :3], transform[:3, 3]
        polygons = {int(f): original['triangles_m'][original['source_faces'] == f, 1]
                    for f in np.unique(original['source_faces'])}
        cells = [G.head_cell(owner.mesh, poly, f, owner.clearance.offsets)@rotation.T+translation
                 for f, poly in polygons.items()]
        row['local_clearance'] = dict(valid=True, source='rigid transform of validated owner solid')
        if np.concatenate(cells)[:, 2].min() < -geometry.scale*1e-10:
            row['reason'] = 'head_hits_floor'
            return row
        face = contact['center_face']
        weights = trimesh.triangles.points_to_barycentric(
            owner.mesh.triangles[face][None], original['center_m'][None])[0]
        root = original['center_m']+.5*weights@owner.clearance.offsets[owner.mesh.faces[face]]
        root = root@rotation.T+translation
        row['path_components'] = geometry.paths.ports(root)
        if not row['path_components']:
            row['reason'] = 'no_roadmap_path_witness'
            return row
        normals = geometry.mesh.face_normals[np.unique(contact['source_faces'])]
        heads = [SimpleNamespace(vertices=cell) for cell in cells]
        clear, checks = [], []
        for index, vector in enumerate(geometry.catalogues[0]['vectors']):
            check = (dict(clear=False, reason='contact_normal_blocks_withdrawal')
                     if np.min(normals@vector) < -W.NORMAL_TOL else geometry.analyzers[0].test(heads, vector))
            checks.append(dict(direction_id=index, **check))
            if check['clear']:
                clear.append(index)
        row['directions'] = [clear]
        row['direction_records'] = [dict(checks=checks, same_owner_solid=True,
                                         normal_depth_m=owner.depth, certified_directions=dict(ids=clear))]
        row['valid'] = bool(row['directions'][0])
        row['reason'] = 'valid' if row['valid'] else 'no_horizontal_head_insertion_witness'
        return row

    def shared(self, entry):
        row = dict(entry, active_tasks=(0, 1))
        check = self.view(row, 1)
        row.update(valid=entry['valid'] and check['valid'],
                   reason='valid' if entry['valid'] and check['valid'] else check['reason'])
        return row

    def expand(self, entry, factor):
        native = self.local[entry['owner_task']].expand(entry['native'], factor)
        trial = self.wrap(native, entry['owner_task'], entry['active_tasks'])
        if native['valid'] and len(entry['active_tasks']) == 2:
            trial = self.shared(trial)
        return trial

    def task_check(self, entries, task):
        return local_group(self.local[task], [self.view(e, task) for e in active_entries(entries, task)])

    def group_check(self, entries):
        checks = [self.task_check(entries, k) for k in range(2)]
        return dict(passed=all(c['passed'] for c in checks),
                    reason='passed' if all(c['passed'] for c in checks) else 'active_set_geometry_failed',
                    per_pose=checks, inactive_material_checked=False,
                    complete_fixture_constructed=False)

    def contacts_by_pose(self, entries):
        return [[self.view(e, k)['contact'] for e in active_entries(entries, k)] for k in range(2)]

    def same_center(self, candidate, selected):
        task = candidate['owner_task']
        center = candidate['contact']['center_m']
        for e in selected:
            contact = transform_contact(e['contact'], self.transforms[e['owner_task']][task])
            if np.linalg.norm(center-contact['center_m']) <= self.local[task].scale*1e-8:
                return True
        return False

    def all_pose_head_check(self, entries):
        """Recheck ALL selected solids in ALL poses, including inactive heads.

        This checks original contact patches and actual owner solids. It does
        not require an inactive head to supply force or have its own insertion
        witness in every task. Foot feasibility and whole-body paths are separate.
        """
        rows = []
        for entry in entries:
            owner = entry['owner_task']
            contact = entry['contact']
            geometry = self.local[owner]
            polygons = {int(f): contact['triangles_m'][contact['source_faces'] == f, 1]
                        for f in np.unique(contact['source_faces'])}
            points = np.concatenate([G.head_cell(geometry.mesh, polygon, f, geometry.clearance.offsets)
                                     for f, polygon in polygons.items()])
            for task, problem in enumerate(self.problems):
                transform = self.transforms[owner][task]
                surface = contact['triangles_m']@transform[:3, :3].T+transform[:3, 3]
                solid = points@transform[:3, :3].T+transform[:3, 3]
                overlap = np.intersect1d(contact['source_faces'], problem.domain.work_ids)
                minimum_surface = float(surface[:, :, 2].min())
                minimum_solid = float(solid[:, 2].min())
                tol = self.local[task].scale*1e-10
                passed = not len(overlap) and minimum_surface >= S.FLOOR_CLEARANCE_M-tol and minimum_solid >= -tol
                rows.append(dict(candidate_id=contact['candidate_id'], pose=problem.pose,
                    active=task in entry['active_tasks'], passed=bool(passed),
                    overlapping_work_faces=overlap.tolist(), minimum_contact_height_m=minimum_surface,
                    minimum_head_height_m=minimum_solid))
        return dict(passed=all(row['passed'] for row in rows), per_head_pose=rows,
            contact_floor_clearance_m=S.FLOOR_CLEARANCE_M, inactive_heads_included=True,
            scope='All selected contact surfaces and owner solids in every supplied pose; feet and connectors absent')

    def save(self, folder, save):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for task, (problem, geometry, pool) in enumerate(zip(self.problems, self.local, self.pools)):
            arrays = folder/f'candidates_{problem.pose}.npz'
            I.save_contacts(arrays, [e['contact'] for e in pool if len(e['contact']['triangles_m'])])
            save(folder/f'candidates_{problem.pose}.json', dict(pose=problem.pose, count=self.count,
                coordinate_frame='this task world frame', area_fraction=P.AREA_FRACTION,
                global_head_exclusions=self.global_head_exclusions,
                excluded_work_and_floor_poses=geometry.head_exclusion_poses,
                area_relative_tolerance=P.AREA_REL_TOL, direction_catalogue=geometry.catalogues[0],
                path_roadmap=geometry.paths.record,
                candidates=[dict(id=e['contact']['candidate_id'], valid=e['valid'], reason=e['reason'],
                                 area_m2=e['area'], radius_m=e['contact']['radius_m']) for e in pool],
                artifacts={arrays.name: I.sha256(arrays)}, provenance=dict(inputs=I.hashes(problem.inputs))))
