"""Run new >=7-pose cases without writing any Co-optimize or object inputs."""
import argparse
import hashlib
import json
import shutil
import time
import faulthandler

from common import *
from model import Model
from search import Search

CASES = {
    'seven_chain': list(range(1,8)),
    'seven_hard': [1,2,4,5,6,7,11],
    'seven_spread': [1,4,7,12,21,23,27],
    'eight_chain': list(range(1,9)),
    'pose1-10': list(range(1,11)),
}


def protected_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in CO.rglob('*') if p.is_file() and p.suffix in ['.py','.md']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['all']+list(CASES),default='all')
    parser.add_argument('--mode',choices=['both','joint','incremental'],default='both')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=6)
    parser.add_argument('--finalists',type=int,default=3)
    parser.add_argument('--branch-rounds',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    stack_file=(args.out/'performance_stacks.log').open('w')
    faulthandler.dump_traceback_later(90,repeat=True,file=stack_file)
    before=protected_hashes()
    runtime=args.out/'runtime_sources';runtime.mkdir()
    sources=list((HERE/'code').glob('*.py'))+C.code_sources()+[Path(C.J.__file__),Path(C.U.__file__),Path(C.WORK.__file__),Path(ConservativeExitClearance.sweep.__code__.co_filename)]
    source_hashes={}
    for path in sorted(set(sources)):
        relative=path.resolve().relative_to(ROOT)
        destination=runtime/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        source_hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    C.save(runtime/'manifest.json',dict(startup_sources=source_hashes,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},cases=CASES))
    C.save(args.out/'protected_coopt_sources.json',before)
    cases=CASES if args.case=='all' else {args.case:CASES[args.case]}
    modes=['joint','incremental'] if args.mode=='both' else [args.mode]
    reports=[]
    for case,numbers in cases.items():
        poses=[f'pose_{i}' for i in numbers]
        model=Model(poses)
        for mode in modes:
            out=args.out/case/mode;out.mkdir(parents=True)
            began=time.monotonic()
            try:
                search=Search(model,out,args.iterations,args.finalists,args.branch_rounds,args.seed)
                report=getattr(search,mode)()
                row=dict(case=case,strategy=mode,pose_count=len(poses),passed=report['force_exit_work_passed'],
                         counts=report['counts'],volume_cm3=report['volume_cm3'],footprint_m2=report['maximum_projected_footprint_m2'],
                         hosts=report['hosts'],seconds=time.monotonic()-began,report=str(out/'report.json'))
            except Exception as error:
                row=dict(case=case,strategy=mode,pose_count=len(poses),passed=False,error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-began)
                C.save(out/'error.json',row)
                import traceback
                traceback.print_exc()
            reports.append(row)
            C.save(args.out/'batch.json',dict(complete=False,results=reports))
    after=protected_hashes()
    assert before==after,'Co-optimize source files changed during the experiment'
    C.save(args.out/'batch.json',dict(complete=True,results=reports,coopt_sources_unchanged=True,
                                    passed=sum(r['passed'] for r in reports),case_count=len(reports)))
    print('BATCH COMPLETE',sum(r['passed'] for r in reports),'/',len(reports),flush=True)


if __name__=='__main__':
    main()
