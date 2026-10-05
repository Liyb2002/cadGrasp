"""Relocate existing pair experiments without rerunning or changing their data.

Old: object/pose_1/stage/pair_pose_6/...; new: object/pose1+6/stage/...
Shared task inputs are copied byte-for-byte into each pair. Referencing JSON is
rewritten in dependency order, including content hashes. Original generator code
hashes remain explicit historical metadata; migration is not a solver rerun.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT, OUTPUTS
from step3_scheculer import pair_tasks as PT


def digest(value):
    return hashlib.sha256(value).hexdigest()


def plan(root):
    pairs = set()
    for p in root.glob('pose_*/step3_scheculer/pair_pose_*'):
        pairs.add(PT.canonical_pair((p.parents[1].name, p.name.removeprefix('pair_'))))
    if not pairs:
        raise ValueError('No old-layout pairs found; nothing to migrate')
    mapping = defaultdict(list)
    junk = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if path.name == '.DS_Store':
            junk.append(path)
            continue
        parts = path.relative_to(root).parts
        if not re.fullmatch(r'pose_[1-9][0-9]*', parts[0]):
            raise ValueError(f'Unexpected pre-existing root entry: {path}')
        pose, stage, *tail = parts
        if tail and tail[0].startswith('pair_pose_'):
            pair = PT.canonical_pair((pose, tail[0].removeprefix('pair_')))
            mapping[path].append(root/PT.pair_name(pair)/stage/Path(*tail[1:]))
        elif stage in ('step_1_needs', 'step4_floor_contact'):
            for pair in sorted(pairs):
                if pose in pair:
                    mapping[path].append(root/PT.pair_name(pair)/stage/pose/Path(*tail))
            if not mapping[path]:
                raise ValueError(f'Task has no corresponding pair: {path}')
        else:
            raise ValueError(f'Unclassified artifact, refusing to discard it: {path}')
    targets = [p for group in mapping.values() for p in group]
    if len(targets) != len(set(targets)) or any(p.exists() for p in targets):
        raise ValueError('Destination collision; no files moved')
    return mapping, pairs, junk


def migrate(root, dry_run=False):
    root = Path(root).resolve()
    mapping, pairs, junk = plan(root)
    print('Pair folders:', ', '.join(PT.pair_name(p) for p in sorted(pairs)), flush=True)
    print('Source files:', len(mapping), 'destination files:', sum(map(len, mapping.values())),
          '(additional files are shared task inputs)', flush=True)
    if dry_run:
        return
    original = {source: source.read_bytes() for source in mapping}
    source_for = {target: source for source, targets in mapping.items() for target in targets}
    old_digest = {source: digest(value) for source, value in original.items()}
    by_digest = defaultdict(list)
    for source, value in old_digest.items():
        by_digest[value].append(source)
    migration_code = {str(p.relative_to(ROOT)): digest(p.read_bytes())
                      for p in (Path(__file__).resolve(), Path(PT.__file__).resolve())}
    rendered, active = {}, set()

    def pair_of(path):
        return path.relative_to(root).parts[0]

    def destinations(sources, pair):
        choices = [p for source in sources for p in mapping[source]]
        local = [p for p in choices if pair_of(p) == pair]
        return local or choices

    def remap_string(value, pair):
        # Longest source prefixes first: a pair stage must not match a task stage.
        def paired(match):
            name, first, stage, second = match.groups()
            return f'output/{name}/{PT.pair_name((first, second))}/{stage}'
        value = re.sub(r'output/([^/\s]+)/((?:pose_)\d+)/([^/\s]+)/pair_(pose_\d+)', paired, value)
        def single(match):
            pose, stage = match.groups()
            choices = [PT.pair_name(p) for p in sorted(pairs) if pose in p]
            selected = pair if pair in choices else choices[0]
            return f'output/{root.name}/{selected}/{stage}/{pose}'
        return re.sub(rf'output/{re.escape(root.name)}/(pose_\d+)/(step_1_needs|step4_floor_contact)(?=/|$)', single, value)

    def rewrite(value, pair):
        if isinstance(value, dict):
            return {remap_string(key, pair): rewrite(item, pair) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item, pair) for item in value]
        if isinstance(value, str):
            if value in by_digest:
                choices = destinations(by_digest[value], pair)
                values = {digest(render(p)) for p in choices}
                if len(values) != 1:
                    raise ValueError(f'Ambiguous content-hash reference in {pair}: {value}')
                return values.pop()
            return remap_string(value, pair)
        return value

    def render(target):
        if target in rendered:
            return rendered[target]
        if target in active:
            raise ValueError(f'Cyclic artifact hash dependency: {target}')
        active.add(target)
        source = source_for[target]
        raw, pair = original[source], pair_of(target)
        if source.suffix == '.json':
            data = json.loads(raw)
            revised = rewrite(data, pair)
            if isinstance(data, dict) and 'code' in data.get('provenance', {}):
                revised['provenance']['generation_code'] = data['provenance']['code']
                revised['provenance']['code'] = migration_code
                revised['provenance']['path_migration'] = dict(
                    source_sha256=old_digest[source], solver_rerun=False,
                    physical_values_changed=False, binary_artifacts_changed=False,
                    manifest=str((root/pair/'step4_floor_contact/path_migration.json').relative_to(ROOT)))
            result = raw if revised == data else (json.dumps(revised, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        elif source.suffix in ('.log', '.txt', '.md'):
            result = remap_string(raw.decode(), pair).encode()
        else:
            result = raw
        rendered[target] = result
        active.remove(target)
        return result

    # Resolve and validate every rewrite before creating the destination tree.
    for target in source_for:
        render(target)
    for target, raw in rendered.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for target, expected in rendered.items():
        if digest(target.read_bytes()) != digest(expected):
            raise RuntimeError(f'Destination verification failed: {target}')
    for source, raw in original.items():
        if source.read_bytes() != raw:
            raise RuntimeError(f'Source changed during migration: {source}')
    for pair in sorted(pairs):
        label = PT.pair_name(pair)
        files = [dict(source=str(source.relative_to(ROOT)), target=str(target.relative_to(ROOT)),
                      source_sha256=old_digest[source], target_sha256=digest(rendered[target]),
                      bytes_unchanged=original[source] == rendered[target])
                 for target, source in source_for.items() if pair_of(target) == label]
        ledger = dict(schema='pair_output_path_migration_v1', complete=True, object=root.name,
            poses=pair, solver_rerun=False, additional_loads_added=False,
            original_generator_hashes_preserved=True, all_destination_bytes_verified=True,
            files=files, provenance=dict(code=migration_code))
        (root/label/'step4_floor_contact/path_migration.json').write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False)+'\n')
    # Delete only the enumerated, byte-verified originals after every copy passes.
    for source in original:
        source.unlink()
    for path in junk:
        path.unlink()
    for folder in sorted(root.glob('pose_*')):
        for path in sorted((p for p in folder.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            path.rmdir()
        folder.rmdir()
    print('Migration complete. Original loads and binary results retained byte-for-byte.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    migrate(OUTPUTS/args.object, args.dry_run)


if __name__ == '__main__':
    main()
