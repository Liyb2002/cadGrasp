"""Read and validate the immutable dataset without repeating pose or load search."""
import json
from pathlib import Path
import random
from codes.precompute_objects.work_regions import ROOT, digest


def read_sets(name, include_supplemental=False):
    folder = ROOT/'objects'/name
    path = folder/'pose_sets.json'
    if not path.exists(): return None
    data = json.loads(path.read_text())
    if not data['passed'] or data['pose_manifest_sha256'] != digest(folder/'poses.json'):
        raise ValueError(f'{name}: stale precomputed set manifest')
    if data.get('category')=='legal_with_common_direction':
        companion=folder/'no_common_direction_pose_sets.json'
        other=json.loads(companion.read_text())
        if other['pose_manifest_sha256']!=data['pose_manifest_sha256'] or other['category']!='legal_without_common_direction':
            raise ValueError(f'{name}: stale or incorrect categorized set manifest')
        groups=data['sets']+other['sets']
        if not include_supplemental:
            by_id={g['id']:g for g in groups}
            groups=[by_id[i] for i in data['canonical_set_ids']]
        data=dict(data,sets=groups,set_count=len(groups),category='legal',
                  collection_files=['pose_sets.json','no_common_direction_pose_sets.json'])
    return data


def read_pose_groups(name, category='legal', canonical_only=False):
    if category not in ['legal','legal_with_common_direction','legal_without_common_direction','illegal','all']:
        raise ValueError('Unknown pose-set category')
    folder=ROOT/'objects'/name
    data=read_sets(name,include_supplemental=not canonical_only)
    if data is None:return []
    legal=data['sets']
    if category.startswith('legal_'):
        want=category=='legal_with_common_direction'
        legal=[g for g in legal if bool(g.get('common_direction',{}).get('common_direction_exists'))==want]
    if category in ['legal','legal_with_common_direction','legal_without_common_direction']:return legal
    illegal_path=folder/'illegal_pose_sets.json'
    illegal=json.loads(illegal_path.read_text())['sets'] if illegal_path.exists() else []
    illegal=[dict(g,id='illegal/'+g['id']) for g in illegal]
    return illegal if category=='illegal' else legal+illegal


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


def read_selected_pose_groups(name, scope='all'):
    """Read the current 5/20/5 experiment selection, preserving output IDs."""
    if scope not in ('all','legal','illegal'):
        raise ValueError('Unknown selected scope')
    folder=ROOT/'objects'/name
    data=json.loads((folder/'selected_pose_sets.json').read_text())
    if data['immutable_input_sha256']['poses.json'] != digest(folder/'poses.json'):
        raise ValueError(f'{name}: stale selected manifest')
    groups=data['sets']
    categories=('legal_with_common_direction','legal_without_common_direction','illegal')
    if [sum(g['category']==c for g in groups) for c in categories] != [5,20,5]:
        raise ValueError(f'{name}: selected counts must be 5/20/5')
    rows=[dict(g,id=('illegal/'+g['id'] if g['category']=='illegal' and not g['id'].startswith('illegal/') else g['id'])) for g in groups]
    if len({g['id'] for g in rows}) != 30:
        raise ValueError(f'{name}: duplicate selected groups')
    return [g for g in rows if scope=='all' or (g['category']=='illegal')==(scope=='illegal')]
