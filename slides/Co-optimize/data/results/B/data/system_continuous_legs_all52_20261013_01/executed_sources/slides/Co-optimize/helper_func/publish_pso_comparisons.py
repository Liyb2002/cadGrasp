"""Reuse the existing renderer for this PSO layout's own saved surface stages."""
import argparse
import subprocess
from run_pso_campaign import ROOT, BASE, POLICIES, ENV, read, save, sha


def publish(system_name):
    system = BASE / 'data' / ('system_' + system_name)
    experiment = POLICIES['pso']
    visual_name = experiment + '_surface_visuals'
    visual = BASE / 'data' / ('system_' + visual_name)
    evaluation = read(system / 'pso_evaluation.json')
    cases = [dict(r, source_experiment=experiment) for r in evaluation['cases'] if r['passed']]
    selection = read(BASE / 'data' / (experiment + '_sculpted') / 'selection.json')
    save(visual / 'selection.json', selection)
    save(visual / 'previous_evaluation.json', dict(cases=[dict(id=r['id'],
        step52_source=str((BASE / r['id'] / 'step5/step5.2' / (experiment + '_rounded')).relative_to(ROOT)),
        step53_source=str((BASE / r['id'] / 'step5/step5.3' / (experiment + '_rounded')).relative_to(ROOT)))
        for r in cases]))
    save(visual / 'evaluation_results.json', dict(evaluation, cases=cases, pictures=[]))
    renderer = ROOT / 'slides/Co-optimize/helper_func/render_sculpted_comparison.py'
    frozen = system / 'frozen_sources' / renderer.relative_to(ROOT)
    assert sha(renderer) == sha(frozen), 'Use the original comparison renderer unchanged'
    if cases:
        subprocess.run([str(ROOT / '.venv/bin/python'), str(renderer), '--experiment', visual_name],
            cwd=ROOT, env=ENV, check=True)
        comparisons = read(visual / 'visual_comparisons.json')
    else:
        comparisons = dict(complete=True, pictures=[])
    comparisons.update(groups=len(cases), requested_groups=52,
        before_source='Own PSO broad rounding and fixed-leg reconstruction',
        after_source='Own PSO final surface rounding',
        saved_geometry_only=True)
    save(visual / 'visual_comparisons.json', comparisons)
    save(system / 'pso_visual_comparisons.json', comparisons)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', default='pso_comparison_all52_20261013_01')
    publish(parser.parse_args().system)


if __name__ == '__main__':
    main()
