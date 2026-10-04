"""Read and validate the immutable dataset without repeating pose or load search."""
import json
from pathlib import Path
import random
from codes.precompute_objects.work_regions import ROOT, digest


def read_sets(name):
    folder = ROOT/'objects'/name
    path = folder/'pose_sets.json'
    if not path.exists(): return None
    data = json.loads(path.read_text())
    if not data['passed'] or data['pose_manifest_sha256'] != digest(folder/'poses.json'):
        raise ValueError(f'{name}: stale precomputed set manifest')
    return data


def selected_sets(name, size, seed, count):
    if isinstance(count, bool) or not isinstance(count,int) or count < 1:
        raise ValueError('Positive integer group count required')
    data = read_sets(name)
    if data is None: return None
    groups = [row for row in data['sets'] if row['size'] == size]
    if not groups: raise ValueError(f'{name}: no precomputed sets of size {size}')
    random.Random(seed).shuffle(groups)
    return groups[:count]


def verify_files(name, pose):
    data = read_sets(name)
    if data is None: raise ValueError('Missing precomputed dataset')
    folder=ROOT/'objects'/name
    for relative, expected in data['artifacts'].items():
        if relative.startswith(f'poses/{pose}/') and digest(folder/relative) != expected:
            raise ValueError(f'Stale dataset artifact: {folder/relative}')
    return folder/'poses'/pose
