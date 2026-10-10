"""Evaluate coordinated proposals before integrating them into either search."""
import argparse
import json
import time
from common import *
from model import Model
from cohort import candidates
from search import failed_count,rank


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--poses',default='1,2,3,4,5,6,7,8,9,10')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--finalists',type=int,default=4)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    model=Model([f'pose_{i}' for i in args.poses.split(',')])
    layout,_=model.initial()
    targets=[]
    for k in layout.active:
        for axis in range(6):
            values=model.tasks[k].targets[:,axis]
            targets.extend([(k,int(np.argmax(values))),(k,int(np.argmin(values)))])
    targets=list(dict.fromkeys(targets))
    scored=[]
    for index,(kind,trial,detail) in enumerate(candidates(model,layout)):
        value=model.proxy(trial,targets)
        scored.append((value['loss'],value['sum_loss'],value['span_m'],trial,detail,value))
        if index%12==0:
            print('COHORT PROXY',index,'loss',value['loss'],detail,flush=True)
    scored.sort(key=lambda r:r[:3])
    records=[];best=None;seen=set()
    for row in scored:
        signature=(row[4]['radius_fraction'],row[4]['host'])
        if signature in seen:
            continue
        seen.add(signature)
        try:
            result=model.exact(row[3])
            records.append(dict(detail=row[4],proxy=row[5],counts=result['counts'],failed=failed_count(result)))
            if best is None or rank(result)<rank(best):
                best=result
                model.save(best,args.out/'best',dict(strategy='cohort-probe',proposal=row[4],force_guided_radius_prescribed=True))
            if failed_count(result)==0:
                break
        except (RuntimeError,ValueError,AssertionError) as error:
            records.append(dict(detail=row[4],proxy=row[5],error=str(error)))
        C.save(args.out/'probe.json',dict(complete=False,candidates=records))
        if len(records)>=args.finalists:
            break
    C.save(args.out/'probe.json',dict(complete=True,candidates=records,success=best is not None and failed_count(best)==0))


if __name__=='__main__':
    main()
