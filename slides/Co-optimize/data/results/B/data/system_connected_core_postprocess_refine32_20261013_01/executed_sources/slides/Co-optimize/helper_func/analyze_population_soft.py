"""Summarize saved search events; never replay physics."""
from pathlib import Path
from collections import Counter
import argparse,json

BASE=Path(__file__).resolve().parents[1]/'data/results/B'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment');args=parser.parse_args()
    batch=json.loads((BASE/'data'/args.experiment/'batch.json').read_text())
    cases=[];moves=Counter();weights=Counter();accepted=Counter();events=0;zero_force_owners=0;errors=[]
    for result in batch['results']:
        label=result['id'];exp=result.get('experiment',args.experiment)
        path=BASE/label/'step4/step4.2'/exp/'population32_trace.json'
        if not path.exists():continue
        rows=json.loads(path.read_text())
        if rows:first=rows[0]['events']
        else:
            initial=json.loads((path.parent/'initial_population_summary.json').read_text())
            first=[dict(before=item)for item in initial]
        held=0;accepted_count=0
        for row in rows:
            for event in row['events']:
                events+=1;move=event.get('move',{});moves[move.get('action',event['branch'])]+=1
                weights[event['weight_source']]+=1;held+=event['branch']=='hold';accepted_count+=event['accepted']
                if event['accepted']:accepted[event['reason']]+=1
                owner=event.get('selected_pose')
                if owner is not None and event['before']['per_pose_failed_demands'][str(owner)]==0:zero_force_owners+=1
                if 'error'in event:errors.append(dict(id=label,slot=event['slot'],error=event['error']))
        cases.append(dict(id=label,passed=result['passed'],initial_all_force_passed_slots=sum(not e['before']['failed_demands']for e in first),
            initial_completed_slots=sum(e['before']['completed']for e in first),holds=held,accepted=accepted_count,
            final_juxtaposed=result.get('juxtaposed_pose_count'),final_oracle_status=result.get('oracle_status'),
            first_completed_round=next((r['iteration']for r in rows if r['incumbent']),None),
            fully_held_and_failed=held==len(rows)*32 and not result['passed']))
    summary=dict(experiment=args.experiment,complete=batch['complete'],groups=len(cases),events=events,
        moves=dict(moves),weight_sources=dict(weights),accepted_reasons=dict(accepted),zero_force_failure_owners=zero_force_owners,
        fully_held_failed_groups=[c['id']for c in cases if c['fully_held_and_failed']],errors=errors,cases=cases,
        force_solver_run=False,boolean_run=False)
    out=BASE/'data'/('system_'+args.experiment)/'policy_analysis.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items()if k not in ['cases','errors']},ensure_ascii=False))


if __name__=='__main__':main()
