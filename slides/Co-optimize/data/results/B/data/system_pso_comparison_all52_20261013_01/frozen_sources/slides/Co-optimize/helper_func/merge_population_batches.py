"""Combine disjoint saved executions of one immutable controller."""
from pathlib import Path
import argparse,json

BASE=Path(__file__).resolve().parents[1]/'data/results/B'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment')
    parser.add_argument('--sources',nargs='+',required=True);parser.add_argument('--source-system',required=True)
    args=parser.parse_args();request=json.loads((BASE/'data'/('system_'+args.source_system)/'requested_sets.json').read_text())
    batches=[json.loads((BASE/'data'/e/'batch.json').read_text())for e in args.sources]
    options=dict(batches[0]['options']);options.pop('output_name')
    rows=[]
    for name,batch in zip(args.sources,batches):
        other=dict(batch['options']);other.pop('output_name');assert other==options
        rows.extend(dict(row,experiment=row.get('experiment',name))for row in batch['results'])
    ids=[row['id']for row in rows];assert len(ids)==len(set(ids)) and set(ids)<=set(request['sets'])
    combined=dict(complete=all(b['complete']for b in batches)and set(ids)==set(request['sets']),
        sets=request['sets'],group_count=len(request['sets']),results=rows,passed=sum(r['passed']for r in rows),
        population_size=32,rounds=options['iterations'],warm_start=False,options=options,sources=args.sources,
        source_system=args.source_system,seconds=sum(b.get('seconds',0)for b in batches),
        protected_unchanged=all(b.get('protected_unchanged',False)for b in batches),
        frozen_sources_unchanged=all(b.get('frozen_sources_unchanged',False)for b in batches))
    target=BASE/'data'/args.experiment;target.mkdir(exist_ok=True)
    temporary=target/'batch.writing.json';temporary.write_text(json.dumps(combined,ensure_ascii=False,indent=2)+'\n');temporary.replace(target/'batch.json')
    print('COMBINED',len(rows),'/',len(request['sets']),'PASS',combined['passed'],'complete',combined['complete'])


if __name__=='__main__':main()
