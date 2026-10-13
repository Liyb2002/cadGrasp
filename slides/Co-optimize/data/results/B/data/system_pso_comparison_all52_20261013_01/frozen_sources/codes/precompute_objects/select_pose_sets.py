"""Select 5 common/legal, 20 noncommon/legal and 5 illegal saved-pose groups.

Only collection JSON files change; saved poses, floor load matrices and loads
remain immutable. Classification uses the full-load native-floor matrix.
"""
import itertools
import hashlib
import json
import time
from collections import Counter
import numpy as np
from codes.precompute_objects.common_directions import ROOT, classify, digest, floor_info, build_counterexamples, write

FILES = {
    'legal_with_common_direction': 'pose_sets.json',
    'legal_without_common_direction': 'no_common_direction_pose_sets.json',
    'illegal': 'illegal_pose_sets.json',
}
TARGETS = dict(zip(FILES, [5, 20, 5]))


def balanced(rows, count, sizes):
    pools = {s: sorted((g for g in rows if len(g['poses']) == s), key=lambda g: tuple(int(p.split('_')[1]) for p in g['poses'])) for s in sizes}
    result = []
    while len(result) < count:
        progressed = False
        for size in sizes:
            if pools[size] and len(result) < count:
                result.append(pools[size].pop(0)); progressed = True
        if not progressed:
            raise RuntimeError(f'Only {len(result)} eligible groups for target {count}')
    return result


def run():
    began = time.perf_counter(); report = []
    for folder in sorted(p.parent for p in (ROOT / 'objects').glob('*/poses.json')):
        if not (folder / 'pose_sets.json').exists():
            continue
        immutable = {str(p.relative_to(folder)): digest(p) for p in [folder / 'poses.json', *sorted((folder / 'poses').rglob('*'))] if p.is_file()}
        collections = {k: json.loads((folder / f).read_text()) for k, f in FILES.items()}
        matrix = np.asarray(collections['legal_with_common_direction']['directed_violating_counts'])
        manifest = json.loads((folder / 'poses.json').read_text())['poses']
        names = [r['pose_id'] for r in manifest]; lookup = {p: i for i, p in enumerate(names)}
        normals = np.array([np.asarray(r['T_world_mesh'])[:3, :3].T @ [0., 0., 1.] for r in manifest])
        used = {tuple(sorted(lookup[p] for p in g['poses'])) for data in collections.values() for g in data['sets']}
        additions = Counter()

        def add(indices):
            indices = tuple(sorted(indices))
            if indices in used:
                return
            info = floor_info(indices, matrix); common = classify(normals[list(indices)])
            category = 'illegal' if not info['floor_demands_compatible'] else ('legal_with_common_direction' if common['common_direction_exists'] else 'legal_without_common_direction')
            row = dict(id='pose' + '+'.join(str(i + 1) for i in indices), size=len(indices), poses=[names[i] for i in indices], passed=info['floor_demands_compatible'], common_direction=common, category=category, canonical_precomputed_set=False, original_collection=FILES[category], selection='deterministic_saved_pose_combination', **info)
            collections[category]['sets'].append(row); used.add(indices); additions[category] += 1

        if len(collections['legal_without_common_direction']['sets']) < 20:
            candidates, _ = build_counterexamples(normals, matrix, per_size=7, excluded=used)
            for row in candidates:
                if row['floor_demands_compatible']:
                    add(row['indices'])
                if len(collections['legal_without_common_direction']['sets']) >= 20:
                    break
        # Construct missing illegal groups with varied sizes using unchanged matrix.
        for size in range(2, 7):
            if len(collections['illegal']['sets']) >= 5:
                break
            for indices in itertools.combinations(range(len(names)), size):
                if indices not in used and np.any(matrix[np.ix_(indices, indices)]):
                    add(indices); break
        if len(collections['legal_with_common_direction']['sets']) < 5:
            for size in range(2, 7):
                for indices in itertools.combinations(range(len(names)), size):
                    if indices not in used and not np.any(matrix[np.ix_(indices, indices)]):
                        if classify(normals[list(indices)])['common_direction_exists']:
                            add(indices)
                    if len(collections['legal_with_common_direction']['sets']) >= 5:
                        break
                if len(collections['legal_with_common_direction']['sets']) >= 5:
                    break
        selected = []
        for category, count in TARGETS.items():
            sizes = [4, 5, 6] if category == 'legal_without_common_direction' else [2, 3, 4, 5, 6]
            for g in balanced(collections[category]['sets'], count, sizes):
                indices = [lookup[p] for p in g['poses']]
                info = floor_info(indices, matrix); direction = classify(normals[indices])
                assert info['floor_demands_compatible'] == (category != 'illegal')
                if category != 'illegal':
                    assert direction['common_direction_exists'] == (category == 'legal_with_common_direction')
                selected.append(dict(g, category=category, source_file=FILES[category], common_direction=direction, **info))
            collections[category]['set_count'] = len(collections[category]['sets'])
        assert len(selected) == len({tuple(sorted(g['poses'])) for g in selected}) == 30
        assert immutable == {str(p.relative_to(folder)): digest(p) for p in [folder / 'poses.json', *sorted((folder / 'poses').rglob('*'))] if p.is_file()}
        legal_total = sum(len(collections[k]['sets']) for k in FILES if k != 'illegal')
        for category, data in collections.items():
            metadata_changed = data.get('total_legal_set_count') != legal_total
            data['total_legal_set_count'] = legal_total
            if additions[category] or metadata_changed:
                write(folder / FILES[category], data)
        stats = dict(object=folder.name, selected_counts=TARGETS, selected_size_counts={k: dict(Counter(str(g['size']) for g in selected if g['category'] == k)) for k in FILES}, available_counts={k: len(v['sets']) for k, v in collections.items()}, added_counts=dict(additions))
        write(folder / 'selected_pose_sets.json', dict(schema='cadgrasp_selected_pose_sets_v1', object=folder.name, set_count=30, sets=selected, statistics=stats, selection='deterministic_size_round_robin_then_numeric_pose_order', category_source_files=FILES, immutable_input_sha256=immutable, directed_violating_counts_sha256=hashlib.sha256(json.dumps(matrix.tolist(), separators=(',', ':')).encode()).hexdigest(), scope='Legal means saved full-load native-floor compatibility; no-common means no nonzero floor-allowed direction, certified by LP. No new poses or loads.'))
        report.append(stats); print(folder.name, stats['available_counts'], 'added', dict(additions), flush=True)
    category_report_path = ROOT / 'objects/pose_set_categories.json'
    category_report = json.loads(category_report_path.read_text())
    for entry in category_report['objects']:
        current = next(row for row in report if row['object'] == entry['object'])
        entry['counts'] = current['available_counts']
    write(category_report_path, category_report)
    write(ROOT / 'objects/selected_pose_sets_report.json', dict(schema='cadgrasp_selected_pose_sets_report_v1', object_count=len(report), selected_set_count=30 * len(report), category_totals={k: TARGETS[k] * len(report) for k in FILES}, objects=report, seconds=time.perf_counter() - began))
    return report


if __name__ == '__main__':
    run()
