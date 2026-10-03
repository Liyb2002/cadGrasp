"""Inherited all-pose withdrawal sets for immutable, append-only physical heads."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from step2_local_support import geometry as G
from step3_scheculer import contacts as I
from step3_scheculer.joint_geometry import JointGeometry
from step3_scheculer.pair_geometry import PairGeometry


class SavedCandidateGeometry(JointGeometry):
    """Read exact Step2 patches; recompute missing local certificates lazily.

    The old Step2 format saved surfaces but not direction/path certificates.
    No centers, radii, surfaces, IDs or candidate files are replaced here.
    """
    def __init__(self, folder, problems, count):
        self.problems, self.count = problems, count
        self.global_head_exclusions = True
        self.cache, self.local, self.pools = {}, [], []
        self.input_paths = []
        frames = [np.asarray(p.domain.data['frame']['T_world_mesh']) for p in problems]
        self.transforms = [[b@np.linalg.inv(a) for b in frames] for a in frames]
        for task, problem in enumerate(problems):
            path = Path(folder)/f'candidates_{problem.pose}.json'
            metadata = json.loads(path.read_text())
            arrays = path.with_suffix('.npz')
            if (metadata['count'] != count or not metadata.get('global_head_exclusions') or
                    metadata['excluded_work_and_floor_poses'] != [p.pose for p in problems]):
                raise ValueError(f'Incompatible saved candidate pool: {path}')
            I.check_hashes(metadata['provenance']['inputs'])
            if I.sha256(arrays) != metadata['artifacts'][arrays.name]:
                raise ValueError(f'Changed candidate geometry: {arrays}')
            contacts = {c['candidate_id']: c for c in I.read_contacts(arrays)}
            geometry = PairGeometry([problem], count=count, initialize_candidates=False,
                                    head_exclusion_problems=problems)
            if geometry.catalogues[0] != metadata['direction_catalogue']:
                raise ValueError('Saved direction catalogue differs; cannot reinterpret IDs')
            self.local.append(geometry)
            pool = []
            for index, record in enumerate(metadata['candidates']):
                contact = contacts.get(record['id'])
                if contact is None:
                    if record['valid']:
                        raise ValueError('Valid saved candidate lacks its contact geometry')
                    contact = dict(candidate_id=record['id'], candidate_index=task*count+index,
                                   radius_m=record['radius_m'], triangles_m=np.empty((0, 3, 3)))
                native = dict(contact=contact, valid=record['valid'], reason=record['reason'],
                              area=record['area_m2'])
                pool.append(dict(native, native=native, owner_task=task,
                                 local_index=index, active_tasks=(task,)))
            self.pools.append(pool)
            self.input_paths.extend([path, arrays])
        self.initial = [e for pool in self.pools for e in pool]

    def view(self, entry, task):
        if task == entry['owner_task']:
            native = entry['native']
            if native['valid'] and 'directions' not in native:
                geometry, contact = self.local[task], entry['contact']
                face = contact['center_face']
                weights = trimesh.triangles.points_to_barycentric(
                    geometry.mesh.triangles[face][None], contact['center_m'][None])[0]
                root = contact['center_m']+.5*weights@geometry.clearance.offsets[geometry.mesh.faces[face]]
                paths = geometry.paths.ports(root)
                record = geometry.analyzers[0].analyze(contact)
                ids = record['certified_directions']['ids']
                native.update(path_components=paths, directions=[ids], direction_records=[record],
                              valid=bool(paths and ids),
                              reason='valid' if paths and ids else 'saved_candidate_has_no_local_path')
        return super().view(entry, task)

    def save(self, folder, save):
        pass  # Exact Step2 artifacts are immutable inputs for this rerun.


class WholeHeadWithdrawal:
    """Per-head full-ray checks factor exactly across a union of rigid cells.

    Caches are independent of the prefix. A trial intersects only inherited
    directions, so rejected trials cannot mutate a selected prefix or restore IDs.
    Final replay sweeps all selected cells together using the Step4 analyzer.
    """
    def __init__(self, geometry):
        self.geometry = geometry
        self.cells_cache, self.check_cache = {}, {}

    def initial(self):
        return tuple(tuple(g.catalogues[0]['global_allowed_directions']['ids'])
                     for g in self.geometry.local)

    @staticmethod
    def restrict(state, task, ids):
        result = list(state)
        result[task] = tuple(sorted(set(state[task]) & set(ids)))
        return tuple(result)

    def cells(self, entry, task):
        key = (entry['contact']['candidate_id'], task)
        if key not in self.cells_cache:
            owner, contact = entry['owner_task'], entry['contact']
            geometry = self.geometry.local[owner]
            transform = self.geometry.transforms[owner][task]
            polygons = {int(f): contact['triangles_m'][contact['source_faces'] == f, 1]
                        for f in np.unique(contact['source_faces'])}
            self.cells_cache[key] = [SimpleNamespace(vertices=
                G.head_cell(geometry.mesh, polygon, face, geometry.clearance.offsets)
                @transform[:3, :3].T+transform[:3, 3]) for face, polygon in polygons.items()]
        return self.cells_cache[key]

    def append(self, state, entry):
        result, checks = [], []
        ident = entry['contact']['candidate_id']
        for task, inherited in enumerate(state):
            geometry = self.geometry.local[task]
            vectors = np.asarray(geometry.catalogues[0]['vectors'])
            normals = geometry.mesh.face_normals[np.unique(entry['contact']['source_faces'])]
            clear = []
            for index in inherited:
                key = (ident, task, index)
                if key not in self.check_cache:
                    self.check_cache[key] = (
                        dict(clear=False, reason='contact_normal_blocks_withdrawal')
                        if np.min(normals@vectors[index]) < -1e-10 else
                        geometry.analyzers[0].test(self.cells(entry, task), vectors[index]))
                if self.check_cache[key]['clear']:
                    clear.append(index)
            result.append(tuple(clear))
            checks.append(dict(pose=self.geometry.problems[task].pose,
                inherited_direction_ids=list(inherited), common_direction_ids=clear))
            if not clear:
                return None, dict(passed=False, reason='no_common_whole_head_withdrawal',
                                  blocked_pose=self.geometry.problems[task].pose, per_pose=checks)
        return tuple(result), dict(passed=True, reason='passed', per_pose=checks)

    def record(self, state):
        return dict(passed=all(state), inactive_heads_included=True,
            direction_sets_only_shrink=True, connectors_checked=False,
            per_pose=[dict(pose=p.pose, common_direction_ids=list(ids))
                      for p, ids in zip(self.geometry.problems, state)])

    def verify(self, entries, state):
        report = self.record(state)
        report.update(physical_head_count=len(entries), full_union_replayed=True)
        for task, ids in enumerate(state):
            geometry = self.geometry.local[task]
            cells = [cell for entry in entries for cell in self.cells(entry, task)]
            vectors = np.asarray(geometry.catalogues[0]['vectors'])
            checks = [dict(direction_id=i, **geometry.analyzers[0].test(cells, vectors[i]))
                      for i in ids] if cells else []
            if any(not c['clear'] for c in checks):
                raise RuntimeError('Inherited direction failed independent whole-head replay')
            report['per_pose'][task]['checks'] = checks
        return report
