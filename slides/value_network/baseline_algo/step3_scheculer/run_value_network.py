"""Step3: fixed-task neural values, hard path masks, final mechanics acceptance."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.value_runtime import RuntimeContext, TRAIN, I
import rollout_fixed
from train import save, digest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=I.OUTPUTS/'B/value_network_step3')
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = TRAIN/'current/best.pt'
    rollout_fixed.Context = RuntimeContext
    # Inference never collects labels or consults a completion bank.
    args.collect = None
    args.checks = args.output/'checks'
    report = rollout_fixed.run(args)
    report['provenance'] = dict(inputs=I.hashes([
        I.OUTPUTS/'B/independent_poses'/p/stage/file
        for p in sorted({p for r in report['results'] for p in r['poses']})
        for stage,files in [('step_1_needs',['needs.json','samples.json']),
                            ('step2_local_support',[f'candidates_{p}.json',f'candidates_{p}.npz'])]
        for file in files]), code=I.hashes([Path(__file__),Path(__file__).with_name('value_runtime.py'),TRAIN/'fixed_model.py',TRAIN/'rollout_fixed.py']))
    for result in report['results']:
        folder = args.output/result['group']
        folder.mkdir(parents=True, exist_ok=True)
        # Contacts are fixed Step2 candidates from the copied baseline.
        for p, ids in zip(result['poses'], result['selected_ids']):
            source = I.OUTPUTS/'B/independent_poses'/p/'step2_local_support'/f'candidates_{p}.npz'
            catalogue = {c['candidate_id']:c for c in I.read_contacts(source)}
            I.save_contacts(folder/f'final_contacts_{p}.npz', [catalogue[i] for i in ids])
        save(folder/'schedule.json', dict(schema='fixed_task_value_network_step3_v1', complete=True,
            object='B', poses=result['poses'], independent=True, shared_heads=False,
            selection='neural_Q_remaining_heads_plus_exit_dispersion', result=result,
            fixture_placement_solved=False, complete_fixture_verified=False,
            checkpoint_sha256=report['checkpoint_sha256'], provenance=report['provenance'],
            artifacts={p.name:digest(p) for p in folder.glob('final_contacts_*.npz')}))
    report['baseline_copy'] = str(I.OUTPUTS.parent)
    report['mechanics_module'] = str(sys.modules['step3_scheculer.pair_scoring'].__file__)
    report['step4_constructed'] = False
    save(args.output/'report.json', report)
    if report['passed_groups'] != report['groups']: raise SystemExit(2)
    return report

if __name__ == '__main__': main()
