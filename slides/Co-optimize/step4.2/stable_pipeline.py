"""Unified XYZ search: original force/torque demands define success."""
import argparse
import json
import sys
from pathlib import Path
import run_stable_gradient
import run_fast_state_seat_gradient
from co_common import HERE, I, save
from whole_search.checked_solution import read_incumbents,select_no_worse


def own_guidance_counts(result, out):
    if result['passed']:
        return None
    report = out / 'search_report.json'
    if not report.exists() or not (out / 'sampled_layout.npz').exists():
        return None
    data = json.loads(report.read_text())
    return data['counts'] or None


def invoke(main, arguments):
    original = sys.argv
    sys.argv = [original[0], *arguments]
    try:
        main()
    finally:
        sys.argv = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets', nargs='+', required=True)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--worker-cpus', type=int, default=6)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--repair-iterations', type=int, default=8)
    parser.add_argument('--demand-nodes', type=int, default=32)
    parser.add_argument('--output-name', default='stable_gradient_xyz_force_v3')
    parser.add_argument('--incumbent-summary',nargs='*',default=[
        'data/stable_gradient_results.json','data/stable_gradient_xyz_v1/pipeline.json',
        'data/stable_gradient_xyz_v2/pipeline.json'],
        help='Saved feasible results retained unless a smaller exported material result is found')
    args = parser.parse_args()
    base = HERE / 'output/B'
    incumbents,incumbent_records=read_incumbents(base,args.sets,args.incumbent_summary)
    common = ['--sets', *args.sets, '--jobs', str(args.jobs),
              '--iterations', str(args.iterations), '--demand-nodes', str(args.demand_nodes)]
    invoke(run_stable_gradient.main, [*common, '--worker-cpus', str(args.worker_cpus),
                                     '--output-name', args.output_name])
    directory = base / 'data' / args.output_name
    first = json.loads((directory / 'batch.json').read_text())
    selected = {r['id']: dict(r, experiment=args.output_name) for r in first['results']}
    repair_guests = [r['id'] for r in first['results']
        if (counts := own_guidance_counts(r, base / r['id'] / 'step4/step4.2' / args.output_name))
        and any(n != 32768 for n in counts.values())]
    repaired = None
    if repair_guests:
        repair_name = args.output_name + '_repair'
        invoke(run_fast_state_seat_gradient.main, ['--sets', *repair_guests, '--jobs', str(args.jobs),
            '--worker-cpus', str(args.worker_cpus), '--iterations', str(args.repair_iterations),
            '--demand-nodes', str(args.demand_nodes), '--resume-experiment', args.output_name,
            '--output-name', repair_name])
        repaired = json.loads((base / 'data' / repair_name / 'batch.json').read_text())
        for r in repaired['results']:
            selected[r['id']] = dict(r, experiment=repair_name, additional_gradient_budget=True)
    attempted={label:dict(r) for label,r in selected.items()}
    for label,row in attempted.items():
        incumbent=incumbents.get(label)
        if incumbent is not None:
            selected[label]=select_no_worse(row,incumbent)
            assert selected[label]['passed'] and selected[label]['volume_cm3']<=incumbent['volume_cm3']
        else:selected[label]=dict(row,retained_incumbent=False)
        save(base/label/'step4/step4.2'/args.output_name/'material_selection.json',selected[label])
    save(directory / 'incumbents.json',dict(checked=incumbent_records,selected=incumbents,
        validated_before_search=True,old_optimized_answers_used_as_start=False))
    save(directory / 'pipeline.json', dict(complete=True, policy='uniform_xyz_force_pass_then_material_selection',
        groups=args.sets, base_budget_passes=sum(r['passed'] for r in first['results']),
        gradient_repair_passes=sum(r['passed'] for r in repaired['results']) if repaired else 0,
        conditional_passes=0,
        passed=sum(r['passed'] for r in selected.values()), selected=selected,
        attempted=attempted,new_search_passed=sum(r['passed'] for r in attempted.values()),
        retained_incumbents=sum(r.get('retained_incumbent',False) for r in selected.values()),
        improved_incumbents=sum(label in incumbents and not r.get('retained_incumbent',False)
                                for label,r in selected.items()),
        material_non_regression_checked=len(incumbents),incumbent_records=incumbent_records,
        gradient_repair_guests=repair_guests,conditioning_guests=[],
        maximum_additional_gradient_rounds=args.repair_iterations,old_optimized_answers_used_as_start=False,
        acceptance='all_original_force_torque_demands_on_search_contacts',
        translation_axis_priority='equal',geometry_recovery_run=False,final_full_demand_recheck_run=False,
        original_demand_checks_retained=True, workpiece_floor_availability_from_actual_height=True, code=I.hashes([Path(__file__),
            Path(run_stable_gradient.__file__), Path(run_fast_state_seat_gradient.__file__),
            Path(__file__).parents[1]/'helper_func/whole_search/checked_solution.py'])))
    print('PIPELINE', sum(r['passed'] for r in selected.values()), '/', len(args.sets), flush=True)


if __name__ == '__main__':
    main()
