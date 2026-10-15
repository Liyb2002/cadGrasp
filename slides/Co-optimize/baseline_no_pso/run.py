"""Run an isolated cold Juxtapose baseline on immutable native task inputs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COOPT = HERE.parent
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def freeze(directory):
    """Freeze code, redirect its input root, and keep imports in that snapshot."""
    tree = directory / 'executed_sources'
    folders = ['codes/precompute_objects', 'codes/setup', 'objects',
               'slides/tools', 'slides/obj_supp']
    folders += ['slides/Co-optimize/' + name for name in (
        'helper_func', 'vis_func', 'step3.1', 'step3.2', 'step3.3', 'step4.1',
        'step4.2', 'step5.1', 'step5.2', 'step5.3', 'step5', 'evaluation',
        'baseline_no_pso', 'operation_demo')]
    records = []
    for folder in folders:
        for source in sorted((ROOT / folder).rglob('*')):
            if not source.is_file() or source.suffix not in ('.py', '.so', '.cpp', '.h'):
                continue
            if any(p in ('data', 'output', 'runs', '__pycache__', 'assets')
                   for p in source.relative_to(ROOT / folder).parts):
                continue
            relative = source.relative_to(ROOT)
            target = tree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix != '.py':
                shutil.copyfile(source, target)
                records.append(dict(original=str(relative), original_sha256=digest(source),
                    executed=str(target.relative_to(ROOT)), executed_sha256=digest(target),
                    io_root_adaptation=False))
                continue
            original = source.read_text()
            content = original
            # Repository roots locate physical files; HERE/CO locate code.
            lines = content.splitlines(keepends=True)
            replacements = []
            for node in ast.walk(ast.parse(content)):
                if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == 'ROOT' for t in node.targets):
                    expression = ast.get_source_segment(content, node.value)
                    if 'Path(' in expression or 'HERE.parent' in expression or 'COOPT.parent' in expression:
                        begin = sum(map(len, lines[:node.value.lineno-1])) + node.value.col_offset
                        end = sum(map(len, lines[:node.value.end_lineno-1])) + node.value.end_col_offset
                        physical_root = ROOT / 'objects' if relative.as_posix() == 'objects/r6_demand.py' else ROOT
                        replacements.append((begin, end, 'Path(' + repr(str(physical_root)) + ')'))
            for begin, end, replacement in sorted(replacements, reverse=True):
                content = content[:begin] + replacement + content[end:]
            content = content.replace('sys.path.insert(0,str(ROOT))',
                'sys.path.insert(0,' + repr(str(tree)) + ')')
            content = content.replace("str(ROOT / 'slides/tools')",
                repr(str(tree / 'slides/tools')))
            if relative.as_posix() == 'codes/precompute_objects/registry.py':
                content = content.replace("OBJECT_ROOT = Path(__file__).resolve().parents[2] / 'objects'",
                    'OBJECT_ROOT = Path(' + repr(str(ROOT / 'objects')) + ')')
            if relative.as_posix().endswith('whole_search/common.py'):
                for variable in ('HERE', 'CO'):
                    content = content.replace(variable + " = ROOT/'slides/Co-optimize'",
                        variable + ' = Path(' + repr(str(tree / 'slides/Co-optimize')) + ')')
            target.write_text(content)
            records.append(dict(original=str(relative), original_sha256=digest(source),
                executed=str(target.relative_to(ROOT)), executed_sha256=digest(target),
                io_root_adaptation=content != original))
    save(directory / 'sources.json', records)
    return tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', default='pose1+2+3+4+5+6+7+8+9+10')
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--slots', type=int, default=32)
    parser.add_argument('--mode', choices=['juxtapose_only', 'juxtapose_adjusted', 'pso'],
                        default='juxtapose_only')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    if args.slots < 3 or args.rounds < 1:
        parser.error('At least 3 slots and 1 round are required')
    name = args.name or (args.mode + '_' + time.strftime('%Y%m%d_%H%M%S'))
    if Path(name).name != name or name in ('.', '..'):
        parser.error('--name must be a directory name')
    base = COOPT / 'data/results/B' / args.set
    if not (base / 'step4/step4.1/data/report.json').is_file():
        parser.error('No saved Step4.1 for this set')
    directory = HERE / 'runs' / name
    directory.mkdir(parents=True, exist_ok=False)
    tree = freeze(directory)
    protected = {}
    for folder in [ROOT / 'objects/B', base / 'step3', base / 'step4/step4.1']:
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix.lower() not in ('.png', '.jpg', '.mp4', '.html'):
                protected[str(path.relative_to(ROOT))] = digest(path)
    save(directory / 'protected_inputs.json', protected)
    options = dict(output_name=name, iterations=args.rounds, frontier_width=args.slots,
        search_policy='population_soft', initialization_budget=96, require_connected_core=True,
        oracle_work_mode='normal', allow_oracle_fallback=True, allow_coupled_adjustment=True,
        allow_group_seating=True, allow_joint_reseating=True, allow_placement_guidance=True,
        allow_height_repair=True, stop_when_feasible=True, allow_native_sliding=False,
        allow_geometry_validation=False, population_continuous_policy='pso' if args.mode=='pso' else 'differential',
        pso_refinement_rounds=0)
    payload = dict(id=args.set, mode=args.mode, directory=str(directory),
        frozen=str(tree), options=options)
    save(directory / 'payload.json', payload)
    began = time.monotonic()
    save(directory / 'status.json', dict(complete=False, mode=args.mode, set=args.set,
        slots=args.slots, rounds=args.rounds, seed=42, pid=os.getpid()))
    with (directory / 'execution.log').open('w') as stream:
        code = subprocess.call([sys.executable, str(tree / HERE.relative_to(ROOT) / 'worker.py'),
            str(directory / 'payload.json')], cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
    sources = json.loads((directory / 'sources.json').read_text())
    changed_inputs = [name for name, value in protected.items() if digest(ROOT / name) != value]
    changed_code = [row['executed'] for row in sources if digest(ROOT / row['executed']) != row['executed_sha256']]
    audit = dict(passed=not changed_inputs and not changed_code, changed_inputs=changed_inputs,
        changed_executed_sources=changed_code, protected_files=len(protected), source_files=len(sources))
    save(directory / 'integrity_audit.json', audit)
    result = json.loads((directory / 'result.json').read_text()) if (directory / 'result.json').exists() else {}
    status = dict(complete=code==0 and audit['passed'] and bool(result) and 'error' not in result, worker_exit_code=code,
        mode=args.mode, set=args.set, seconds=time.monotonic()-began, result=result,
        integrity_passed=audit['passed'])
    save(directory / 'status.json', status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not status['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
