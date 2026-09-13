"""Shared actual contact geometry, round paths and provenance for Step 3 and its substages."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from step1.needs import ROOT, OUTPUTS, sha256
from step1.cases import pose_name


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def hashes(paths):
    return {str(Path(p).resolve().relative_to(ROOT)): sha256(p) for p in paths}


def check_hashes(records):
    for relative, digest in records.items():
        path = ROOT/relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f'Missing or stale dependency: {relative}')


def load_npz(path):
    with np.load(path) as source:
        return {key: source[key].copy() for key in source.files}


def folder(name, step, round_number):
    return OUTPUTS/name/pose_name()/step/f'round_{round_number:03d}'


def check_report(path):
    path = Path(path)
    result = json.loads(path.read_text())
    assert result['complete']
    for group in ('inputs', 'code'):
        check_hashes(result['provenance'][group])
    for filename, digest in result.get('artifacts', {}).items():
        if sha256(path.parent/filename) != digest:
            raise RuntimeError(f'Changed artifact: {path.parent/filename}')
    return result


def save_contacts(path, contacts):
    np.savez_compressed(path,
        triangles_m=np.concatenate([p['triangles_m'] for p in contacts]) if contacts else np.empty((0, 3, 3)),
        source_faces=np.concatenate([p['source_faces'] for p in contacts]) if contacts else np.empty(0, int),
        triangle_areas_m2=np.concatenate([p['triangle_areas_m2'] for p in contacts]) if contacts else np.empty(0),
        offsets=np.cumsum([0]+[len(p['triangles_m']) for p in contacts]),
        centers_m=np.array([p['center_m'] for p in contacts]).reshape(-1, 3),
        center_faces=np.array([p['center_face'] for p in contacts], int),
        radius_m=np.array([p['radius_m'] for p in contacts], float),
        candidate_indices=np.array([p['candidate_index'] for p in contacts], int),
        candidate_ids=np.array([p['candidate_id'] for p in contacts], str))


def read_contacts(path):
    z = load_npz(path)
    offsets = z['offsets']
    count = len(z['candidate_indices'])
    assert offsets.shape == (count+1,) and offsets[0] == 0 and offsets[-1] == len(z['triangles_m'])
    assert np.all(np.diff(offsets) > 0) or count == 0
    assert len(set(z['candidate_indices'].tolist())) == count
    contacts = []
    for i in range(count):
        a, b = offsets[i:i+2]
        contacts.append(dict(triangles_m=z['triangles_m'][a:b], source_faces=z['source_faces'][a:b],
            triangle_areas_m2=z['triangle_areas_m2'][a:b], center_m=z['centers_m'][i],
            center_face=int(z['center_faces'][i]), radius_m=float(z['radius_m'][i]),
            candidate_index=int(z['candidate_indices'][i]), candidate_id=str(z['candidate_ids'][i])))
    return contacts


def as_data(contacts):
    return SimpleNamespace(triangles=np.concatenate([p['triangles_m'] for p in contacts]),
        source_faces=np.concatenate([p['source_faces'] for p in contacts]),
        triangle_areas=np.concatenate([p['triangle_areas_m2'] for p in contacts]),
        offsets=np.cumsum([0]+[len(p['triangles_m']) for p in contacts]),
        centers_m=np.array([p['center_m'] for p in contacts]),
        center_faces=np.array([p['center_face'] for p in contacts]),
        radius_m=np.array([p['radius_m'] for p in contacts]), valid=np.ones(len(contacts), bool))


def area(contacts):
    return float(sum(p['triangle_areas_m2'].sum() for p in contacts))


def merge_columns(*groups):
    full = np.vstack(groups)
    ids = np.unique(np.round(full, 13), axis=0, return_index=True)[1]
    return np.ascontiguousarray(full[np.sort(ids)])
