"""Finish saved-certificate audits and publish the actual PSO stage images."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_pso_campaign import ROOT, BASE, POLICIES, ENV, read, save, sha, verify


def publish_evaluations(system):
    import numpy as np
    import trimesh
    frozen = system / 'frozen_sources/slides/Co-optimize'
    sys.path[:0] = [str(frozen / name) for name in ['evaluation', 'helper_func', 'vis_func', 'step5.1']]
    import _bootstrap
    import step51_render as V
    from metrics import static_bounds
    spec = importlib.util.spec_from_file_location('final_pso_evaluation_renderer', frozen / 'evaluation/run.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    body = trimesh.load(ROOT / 'objects/B/mesh.stl', force='mesh', process=False)
    renderer = V.Renderer()
    experiment = POLICIES['pso']
    post = experiment + '_sculpted'
    cases = read(system / 'pso_evaluation.json')['cases']
    pictures = []
    for case in cases:
        source = BASE / case['id'] / 'step4/step4.2' / experiment
        if case['step4_passed']:
            with np.load(source / 'layout.npz') as a:
                layout = {k: a[k].copy() for k in a.files}
            record = read(source / 'data/report.json')
            nominal = trimesh.load(source / 'support.obj', force='mesh', process=False)
            for stage in ['step5.2', 'step5.3']:
                data = BASE / case['id'] / 'step5' / stage / post
                report_path = data / 'data/report.json'
                if not report_path.exists():
                    continue
                report = read(report_path)
                if not report['passed']:
                    continue
                fixture = trimesh.load(data / report['support'], force='mesh', process=False)
                metric = static_bounds(body.vertices, layout, nominal.vertices, fixture.vertices, record['poses'])
                output = ROOT / 'slides/Co-optimize/output/B' / case['id'] / stage
                evaluation.render(renderer, body, nominal, fixture, layout, metric, output)
                shutil.copy2(output / 'static_occupancy.png', output / 'evaluation.png')
        for stage in ['step4.2', 'step5.1', 'step5.2', 'step5.3']:
            output = ROOT / 'slides/Co-optimize/output/B' / case['id'] / stage
            for picture in output.glob('*.png'):
                pictures.append(dict(id=case['id'], stage=stage,
                    path=str(picture.relative_to(ROOT)), sha256=sha(picture)))
        print('ACTUAL PSO STAGE IMAGES', case['id'], flush=True)
    save(system / 'current_pso_images.json', dict(complete=True, files=pictures,
        actual_saved_geometry=True, force_replayed=False, boolean_rebuilt=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', default='pso_comparison_all52_20261013_01')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    system = BASE / 'data' / ('system_' + args.system)
    with (system / 'finalizer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(system / 'finalizer_state.json', dict(complete=False, active_pid=os.getpid(), status='WAITING'))
        ready = lambda: all((system / name).exists() for name in [
            'pso_evaluation.json', 'differential_evaluation.json', 'comparison_summary.json'])
        while not ready():
            if not args.wait:
                raise RuntimeError('Numerical campaign and Evaluation must finish first')
            state = read(system / 'work_state.json')
            try:
                os.kill(state['active_pid'], 0)
            except ProcessLookupError:
                raise RuntimeError('Numerical campaign exited before final Evaluation')
            time.sleep(5)
        save(system / 'finalizer_state.json', dict(complete=False, active_pid=os.getpid(), status='AUDITING'))
        scripts = Path(__file__).resolve().parent
        for filename in ['audit_pso_campaign.py']:
            with (system / (filename + '.log')).open('w') as stream:
                subprocess.run([str(ROOT / '.venv/bin/python'), str(scripts / filename)],
                               cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)
        protected = read(system / 'protected_originals.json')
        for case in read(system / 'pso_evaluation.json')['cases']:
            output = ROOT / 'slides/Co-optimize/output/B' / case['id'] / 'step4.2'
            for picture in output.glob('*.png'):
                assert str(picture.relative_to(ROOT)) not in protected
                picture.unlink()
            for stage in ['step5.1', 'step5.2', 'step5.3']:
                output = ROOT / 'slides/Co-optimize/output/B' / case['id'] / stage
                report_path = BASE / case['id'] / 'step5' / stage / (POLICIES['pso'] + '_sculpted') / 'data/report.json'
                report = read(report_path) if report_path.exists() else {}
                for picture in output.glob('*.png'):
                    obsolete = picture.name in ['comparison.png', 'detail.png', 'evaluation.png', 'static_occupancy.png']
                    invalid_stage = not case['passed'] and not (
                        stage == 'step5.1' and case.get('step51_source') or
                        report.get('passed') and report.get('selected_step4_experiment') == POLICIES['pso'] and
                        report.get('published_images', {}).get(picture.name) == sha(picture))
                    if obsolete or invalid_stage:
                        assert str(picture.relative_to(ROOT)) not in protected
                        picture.unlink()
        with (system / 'step4_publication.log').open('w') as stream:
            subprocess.run([str(ROOT / '.venv/bin/python'), str(scripts / 'report_population32.py'),
                POLICIES['pso'], '--source-system', args.system], cwd=ROOT, env=ENV,
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        with (system / 'surface_comparison_publication.log').open('w') as stream:
            subprocess.run([str(ROOT / '.venv/bin/python'), str(scripts / 'publish_pso_comparisons.py'),
                '--system', args.system], cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)
        publish_evaluations(system)
        with (system / 'final_report.log').open('w') as stream:
            subprocess.run([str(ROOT / '.venv/bin/python'), str(scripts / 'finalize_pso_results.py')],
                           cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)
        verify(system)
        summary = read(system / 'comparison_summary.json')
        state = read(system / 'work_state.json')
        state.update(complete=True, final_saved_certificate_audit_passed=True,
                     actual_psostage_images_published=True, remaining=[], remaining_extra_audits=[])
        save(system / 'work_state.json', state)
        save(system / 'finalizer_state.json', dict(complete=True, active_pid=os.getpid(), status='COMPLETE', summary=summary))
        print('ALL52 PSO DELIVERY COMPLETE', summary, flush=True)


if __name__ == '__main__':
    main()
