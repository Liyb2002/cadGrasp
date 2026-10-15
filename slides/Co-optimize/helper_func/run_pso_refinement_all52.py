"""Run Step4 only, from saved registered inputs, with frozen current sources."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
COOPT = ROOT/'slides/Co-optimize'
BASE = COOPT/'data/results/B'
OLD = BASE/'data/system_pso_comparison_all52_20261013_01'
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', COOPT_REPO_ROOT=str(ROOT))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def prepare(system, labels):
    """Relocate code only; keep physical inputs and artifact paths unchanged."""
    frozen = system/'frozen_sources'
    directories = [COOPT/name for name in (
        'helper_func', 'vis_func', 'step3.1', 'step3.2', 'step3.3',
        'step4.1', 'step4.2', 'step5', 'step5.1', 'step5.2', 'step5.3', 'evaluation')]
    directories += [ROOT/'codes', ROOT/'slides/tools', ROOT/'slides/obj_supp',
                    COOPT/'operation_demo']
    paths = set()
    for directory in directories:
        for path in directory.rglob('*'):
            relative = path.relative_to(directory)
            if (path.is_file() and path.suffix in ('.py', '.so', '.cpp', '.h')
                    and not set(relative.parts).intersection({'data', 'output', '__pycache__', 'tests'})
                    and not path.name.startswith('test_')):
                paths.add(path)
    manifest = []
    for source in sorted(paths):
        relative = source.relative_to(ROOT)
        target = frozen/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        adaptations = []
        if source.suffix == '.py':
            code = target.read_text()
            # ROOT means the input/provenance repository in these adapters.
            if (str(relative).startswith('codes/precompute_objects/')
                    or relative.name in ('co_common.py', 'artifact_store.py', 'stage5_shared.py')
                    or str(relative).endswith('legacy_kernel/step1/needs.py')
                    or str(relative).endswith('whole_search/common.py')):
                code, count = re.subn(r'^ROOT\s*=.*$', 'ROOT = Path('+repr(str(ROOT))+')', code, flags=re.M)
                if count:
                    adaptations.append('original physical-input/provenance root')
                code, count = re.subn(r'^OBJECT_ROOT\s*=.*$',
                    'OBJECT_ROOT = Path('+repr(str(ROOT/'objects'))+')', code, flags=re.M)
                if count:
                    adaptations.append('original object root')
            if relative.name == 'co_common.py':
                code = code.replace('HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent.parent',
                    'HERE=Path(__file__).resolve().parents[1];ROOT=Path('+repr(str(ROOT))+')')
                adaptations.append('original physical-input/provenance root')
            if relative.name == 'artifact_store.py':
                for key, value in [('RESULTS', BASE.parent), ('VISUALS', COOPT/'output'),
                                   ('INITIALIZATION_SOURCES', COOPT/'data/initialization_sources')]:
                    code = re.sub(r'^'+key+r'\s*=.*$', key+' = Path('+repr(str(value))+')', code, flags=re.M)
                adaptations.append('original artifact destinations')
            if str(relative).endswith('whole_search/common.py'):
                for key in ('HERE', 'CO'):
                    code = re.sub(r'^'+key+r'\s*=.*$', key+' = Path(__file__).resolve().parents[2]', code, flags=re.M)
                adaptations.append('frozen numerical import paths')
            code = code.replace("ROOT / 'slides/tools'", "Path("+repr(str(frozen/'slides/tools'))+")")
            if relative == Path('slides/Co-optimize/helper_func/_bootstrap.py'):
                code += '\n# Resolve the input readers from this immutable source tree.\nsys.path.insert(0, str(COOPT_ROOT.parents[1]))\nimport codes.precompute_objects\n'
                adaptations.append('frozen input-reader import paths')
            target.write_text(code)
        target.chmod(0o444)
        manifest.append(dict(original=str(relative), original_sha256=sha(source),
            frozen=str(target.relative_to(ROOT)), frozen_sha256=sha(target), io_adaptations=adaptations))
    save(system/'frozen_sources.json', dict(complete=True, files=manifest,
        source_count=len(manifest), source_baseline='current source including actual complete-solid certification',
        numerical_algorithms_modified_by_freezing=False))
    protected = read(OLD/'protected_originals.json')
    assert all((ROOT/p).is_file() and sha(ROOT/p)==h for p,h in protected.items())
    expected = {
        'objects/r6_demand.py':read(ROOT/'objects/r6_domain_index.json')['generator']['sha256'],
        'objects/r6_quadratic.py':read(ROOT/'objects/r6_quadratic_index.json')['generator_sha256'],
    }
    external = []
    for relative, digest in expected.items():
        assert sha(ROOT/relative)==digest
        target=system/'external_execution_sources'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
        target.chmod(0o444)
        external.append(dict(original=relative,archive=str(target.relative_to(ROOT)),sha256=digest,
            execution_location='original immutable objects module; formula-manifest and per-run source hashes'))
    protected.update(expected)
    save(system/'external_numerical_sources.json',dict(files=external,numerical_modules_changed=False))
    save(system/'protected_originals.json', protected)
    save(system/'requested_sets.json', dict(sets=labels, group_count=len(labels)))
    scripts = system/'execution_scripts'
    scripts.mkdir()
    worker = scripts/'step4_worker.py'
    shutil.copyfile(Path(__file__).with_name('step4_worker_runtime.py'),worker)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='pso_refine_all52_step4_20261014_01')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--runtime-cache', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gpu-replay',choices=['off','auto','required'],default='auto')
    parser.add_argument('--scheduling',choices=['original','small-first'],default='small-first')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    system = BASE/'data'/('system_'+args.name)
    directory = BASE/'data'/args.name
    if not args.resume:
        source = read(BASE/'data/pso_all52_20261013_01/batch.json')
        labels = source['sets']
        assert len(labels)==52 and len(set(labels))==52
        system.mkdir(exist_ok=False)
        frozen = prepare(system, labels)
        options = dict(source['options'], output_name=args.name,
            pso_refinement_rounds=args.rounds, pso_refinement_policy='pso', pso_refinement_seed=42,
            runtime_cache=args.runtime_cache, runtime_cache_memory_mb=32)
        options['gpu_replay']=args.gpu_replay
        batch = dict(complete=False, sets=labels, group_count=52, options=options,
            results=[], passed=0, stage='Step4 only', memory_limit_gib=None,
            warm_start=False, extra_refinement_rounds_explicit=True,
            source_snapshot=str(frozen.relative_to(ROOT)), execution_jobs=args.jobs)
        directory.mkdir(exist_ok=False)
        save(directory/'batch.json', batch)
        save(system/'state.json', dict(complete=False, phase='PREPARED', experiment=args.name, jobs=args.jobs))
    else:
        batch = read(directory/'batch.json')
        assert batch['options']['pso_refinement_rounds']==args.rounds
        frozen = system/'frozen_sources'
    manifest = read(system/'frozen_sources.json')['files']
    assert all(sha(ROOT/r['frozen'])==r['frozen_sha256'] for r in manifest)
    if args.prepare_only:
        print('PREPARED', args.name, len(manifest), 'sources', flush=True)
        return
    completed = {r['id'] for r in batch['results']}
    pending = [label for label in batch['sets'] if label not in completed]
    if args.scheduling=='small-first':pending.reverse()
    began = time.monotonic()
    save(system/'state.json', dict(complete=False, phase='STEP4', experiment=args.name,
        jobs=args.jobs, completed=len(completed), orchestrator_pid=os.getpid()))

    def one(label):
        safe = label.replace('/', '__')
        result = directory/'workers'/(safe+'.json')
        payload = directory/'payloads'/(safe+'.json')
        log = directory/'logs'/(safe+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        save(payload, dict(id=label, root=str(ROOT), frozen=str(frozen), options=batch['options'],
            result=str(result), private_output=str(system/'private_output')))
        start = time.monotonic()
        with log.open('w') as output:
            code = subprocess.call([str(ROOT/'.venv/bin/python'),
                str(system/'execution_scripts/step4_worker.py'), str(payload)],
                cwd=ROOT, env=ENV, stdout=output, stderr=subprocess.STDOUT)
        row = read(result) if result.exists() else dict(id=label, passed=False,
            status='runtime_failure', error='Worker exited '+str(code))
        if code:
            row.update(passed=False, status='runtime_failure', error='Worker exited '+str(code))
        row.update(worker_exit_code=code, process_wall_seconds=time.monotonic()-start,
                   execution_log=str(log.relative_to(ROOT)), memory_limit_gib=None)
        save(result, row)
        return row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, label) for label in pending]
        for future in as_completed(futures):
            row = future.result()
            batch['results'].append(row)
            batch.update(passed=sum(r['passed'] for r in batch['results']),
                         seconds=time.monotonic()-began)
            save(directory/'batch.json', batch)
            save(system/'state.json', dict(complete=False, phase='STEP4', experiment=args.name,
                jobs=args.jobs, completed=len(batch['results']), passed=batch['passed'], orchestrator_pid=os.getpid()))
            print('STEP4',len(batch['results']),'/52 PASS',batch['passed'],row['id'],
                  row['passed'],round(row['process_wall_seconds'],1),row.get('error',''),flush=True)
    protected = read(system/'protected_originals.json')
    changes = [p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    frozen_changes = [r['frozen'] for r in manifest if sha(ROOT/r['frozen'])!=r['frozen_sha256']]
    batch.update(complete=len(batch['results'])==52, seconds=time.monotonic()-began,
        protected_unchanged=not changes, frozen_sources_unchanged=not frozen_changes)
    save(directory/'batch.json', batch)
    save(system/'execution_audit.json', dict(passed=not changes and not frozen_changes,
        protected_files=len(protected), frozen_sources=len(manifest),
        changed_originals=changes, changed_frozen_sources=frozen_changes))
    save(system/'state.json', dict(complete=batch['complete'], phase='STEP4_FINISHED',
        experiment=args.name, completed=len(batch['results']), passed=batch['passed']))
    print('ALL52 STEP4 FINISHED',batch['passed'],'/52',flush=True)


if __name__ == '__main__':
    main()
