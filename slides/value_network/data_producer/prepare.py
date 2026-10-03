"""Reuse immutable current baseline loads/contacts; replay singleton path witnesses."""
from pathlib import Path
import sys
import time
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / 'slides/baseline_algo'
sys.path.insert(0, str(BASELINE))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task, input_hashes
from step3_scheculer.pair_geometry import PairGeometry
from step2_local_support import geometry as G
import trimesh


def prepare(name, poses, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    problems, pools, catalogues, source_files = [], [], [], []
    for pose in poses:
        folder = I.OUTPUTS/name/'independent_poses'/pose
        problem = read_task(name, pose, folder=folder/'step_1_needs')
        problems.append(problem)
        metadata_path = folder/'step2_local_support'/f'candidates_{pose}.json'
        contacts_path = metadata_path.with_suffix('.npz')
        metadata = json.loads(metadata_path.read_text())
        if I.sha256(contacts_path) != metadata['artifacts'][contacts_path.name]:
            raise ValueError(f'Candidate artifact changed: {contacts_path}')
        I.check_hashes(metadata['provenance']['inputs'])
        source_files.extend([metadata_path, contacts_path])
        catalogue = metadata['direction_catalogue']
        catalogues.append(np.asarray(catalogue['vectors'], float))
        source = {c['candidate_id']: c for c in I.read_contacts(contacts_path)}
        cache_path = output/f'paths_{pose}.json'
        signature = {**input_hashes([problem]), **I.hashes([metadata_path, contacts_path]),
                     **I.hashes([Path(__file__), BASELINE/'step3_scheculer/pair_geometry.py',
                                 BASELINE/'step3_scheculer/sequential_geometry.py',
                                 BASELINE/'step2_local_support/withdrawal.py'])}
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
        if cached is None or cached['signature'] != signature:
            start = time.monotonic()
            geometry = PairGeometry([problem], count=metadata['count'], initialize_candidates=False)
            np.testing.assert_allclose(geometry.catalogues[0]['vectors'], catalogue['vectors'], atol=1e-12)
            rows = []
            for index, record in enumerate(metadata['candidates']):
                cid = record['id']
                row = dict(record, directions=[], components=[])
                if record['valid']:
                    contact = source[cid]
                    directions = geometry.analyzers[0].analyze(contact)['certified_directions']['ids']
                    face = contact['center_face']
                    weights = trimesh.triangles.points_to_barycentric(
                        geometry.mesh.triangles[face][None], contact['center_m'][None])[0]
                    root = contact['center_m'] + .5*weights@geometry.clearance.offsets[geometry.mesh.faces[face]]
                    components = geometry.paths.ports(root)
                    row.update(directions=directions, components=components,
                               valid=bool(directions and components),
                               reason='valid' if directions and components else 'no_replayed_path_witness')
                rows.append(row)
                if (index+1) % 20 == 0:
                    print('PATHS', pose, index+1, 'of', len(metadata['candidates']), flush=True)
            cached = dict(signature=signature, candidates=rows, elapsed_s=time.monotonic()-start,
                          singleton_paths_replayed=True, catalogue=catalogue)
            I.save(cache_path, cached)
        pool = []
        for index, row in enumerate(cached['candidates']):
            contact = source.get(row['id'])
            pool.append(dict(row, index=index, contact=contact))
        pools.append(pool)
        print('PREPARED', pose, len(pool), 'candidates;', sum(e['valid'] for e in pool), 'legal', flush=True)
    provenance = {**input_hashes(problems), **I.hashes(source_files)}
    return problems, pools, catalogues, provenance


def intersect(pool, indices, field):
    if not indices:
        return set().union(*(set(e[field]) for e in pool if e['valid']))
    values = set(pool[indices[0]][field])
    for index in indices[1:]:
        values.intersection_update(pool[index][field])
    return values


def compatible(pool, indices):
    return all(pool[i]['valid'] for i in indices) and bool(
        intersect(pool, indices, 'directions') and intersect(pool, indices, 'components'))
