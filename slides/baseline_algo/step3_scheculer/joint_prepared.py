"""Replay immutable candidate geometry certified by a complete empty-state round.

The singleton direction/component sets are sufficient for every subsequent
append-only group intersection. Inputs, geometry code and artifacts are hashed.
No fixture geometry or new mechanical pass is inferred from this cache.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from step3_scheculer import contacts as I
from step3_scheculer.joint_geometry import JointGeometry
from step3_scheculer.pair_geometry import transform_contact
from step3_scheculer.pair_tasks import input_hashes
from step3_scheculer.run_pairs import save
from step2_local_support.circles import DEPTH_FRACTION


def geometry_sources():
    here = Path(__file__).parent
    return (list((here.parent/'step2_local_support').glob('*.py'))+
            [here/name for name in ('joint_geometry.py', 'joint_prepared.py',
                                    'sequential_geometry.py', 'pair_geometry.py', 'contacts.py')])


def scoring_sources():
    here = Path(__file__).parent
    return (list((here.parent/'step3.1_score_candidate').glob('*.py'))+
            [here/name for name in ('run_joint.py', 'joint_scoring.py', 'pair_scoring.py',
                                    'pair_tasks.py', 'passive_support.py', 'floor_support.py',
                                    'run_sequential.py', 'strict_lp_retry.py')])


def store_prepared(folder, problems, count, first_scores, origin='completed empty-state candidate evaluation'):
    folder = Path(folder)
    rows = first_scores['rows']
    poses = [p.pose for p in problems]
    assert len(rows) == sum(len(json.loads((folder/f'candidates_{p}.json').read_text())['candidates']) for p in poses)
    for i, row in enumerate(rows):
        assert row['index'] == i
        if row['eligible']:
            assert len(row['per_pose_geometry']) == len(poses)
            assert row['active_tasks'] == [k for k, c in enumerate(row['per_pose_geometry']) if c['passed']]
    path = folder/'prepared_first_round.json'
    save(path, first_scores)
    artifacts = [path]+[folder/f'candidates_{p}.{suffix}' for p in poses for suffix in ('json', 'npz')]
    manifest = dict(schema='joint_prepared_geometry_v1', poses=poses, count=count, origin=origin,
        inputs=input_hashes(problems), geometry_code=I.hashes(geometry_sources()),
        scoring_code=I.hashes(scoring_sources()),
        artifacts={p.name: I.sha256(p) for p in artifacts})
    save(folder/'prepared.json', manifest)


def load_prepared(folder, problems, count):
    folder = Path(folder)
    if not (folder/'prepared.json').exists():
        return None
    manifest = json.loads((folder/'prepared.json').read_text())
    if (manifest['poses'] != [p.pose for p in problems] or manifest['count'] != count or
            manifest['inputs'] != input_hashes(problems) or
            manifest['geometry_code'] != I.hashes(geometry_sources())):
        return None
    for name, digest in manifest['artifacts'].items():
        if not (folder/name).is_file() or I.sha256(folder/name) != digest:
            return None
    return PreparedGeometry(folder, problems, count, manifest)


class PreparedGeometry(JointGeometry):
    def __init__(self, folder, problems, count, manifest):
        self.problems, self.count, self.manifest = problems, count, manifest
        self.first_scores = json.loads((folder/'prepared_first_round.json').read_text())
        self.first_scores_valid = manifest['scoring_code'] == I.hashes(scoring_sources())
        rows = self.first_scores['rows']
        self.checks = {r['id']: r['per_pose_geometry'] for r in rows}
        frames = [np.asarray(p.domain.data['frame']['T_world_mesh']) for p in problems]
        self.transforms = [[b@np.linalg.inv(a) for b in frames] for a in frames]
        self.local, self.pools, self.cache = [], [], {}
        for task, problem in enumerate(problems):
            metadata = json.loads((folder/f'candidates_{problem.pose}.json').read_text())
            contacts = {c['candidate_id']: c for c in I.read_contacts(folder/f'candidates_{problem.pose}.npz')}
            labels = sorted({label for checks in self.checks.values() if checks
                             for label in checks[task].get('common_path_components', [])})
            scale = float(problem.domain.mesh.extents.max())
            self.local.append(SimpleNamespace(mesh=problem.domain.mesh, scale=scale,
                depth=scale*DEPTH_FRACTION, catalogues=[metadata['direction_catalogue']],
                paths=SimpleNamespace(labels=np.asarray(labels), record=metadata['path_roadmap'])))
            pool = []
            for index, record in enumerate(metadata['candidates']):
                cid = record['id']
                contact = contacts.get(cid)
                if contact is None:
                    assert not record['valid']
                    contact = dict(candidate_id=cid, candidate_index=task*count+index,
                                   radius_m=record['radius_m'], triangles_m=np.empty((0, 3, 3)))
                native = dict(contact=contact, valid=record['valid'], reason=record['reason'],
                              area=record['area_m2'])
                if native['valid']:
                    check = self.checks[cid][task]
                    assert check['passed']
                    native.update(directions=[check['common_direction_ids']],
                                  path_components=check['common_path_components'])
                pool.append(dict(native, native=native, owner_task=task,
                                 local_index=index, active_tasks=(task,)))
            self.pools.append(pool)
        self.initial = [e for pool in self.pools for e in pool]
        assert [e['contact']['candidate_id'] for e in self.initial] == [r['id'] for r in rows]

    def view(self, entry, task):
        if entry['owner_task'] == task:
            return dict(entry['native'], contact=entry['contact'])
        key = (entry['contact']['candidate_id'], entry['contact']['radius_m'], task)
        if key not in self.cache:
            check = self.checks[key[0]][task]
            row = dict(contact=transform_contact(entry['contact'], self.transforms[entry['owner_task']][task]),
                       area=entry['area'], valid=check['passed'], reason=check['reason'])
            if check['passed']:
                row.update(directions=[check['common_direction_ids']], path_components=check['common_path_components'])
            self.cache[key] = row
        return self.cache[key]

    def save(self, folder, save):
        # The exact candidate artifacts are already present and hash-checked.
        pass
