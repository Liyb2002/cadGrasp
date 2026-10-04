"""All saved sets: common-cavity proposals, actual Step4/Step5, publication/audit."""
from pathlib import Path
import argparse,concurrent.futures,subprocess,sys,os
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--jobs',type=int,default=3);args=p.parse_args()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    groups=sorted((I.OUTPUTS/'B').glob('pose*+*'))
    def run(group):
        out=group/'step3_scheculer/dsl_cavity'
        if args.resume:
            try:
                if I.check_report(out/'report.json')['passed']:return
            except (RuntimeError,FileNotFoundError,KeyError):pass
        out.mkdir(parents=True,exist_ok=True)
        with (out/'batch.log').open('w') as log:subprocess.run([sys.executable,str(Path(__file__).with_name('cavity_dsl.py')),group.name],env=env,cwd=I.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('CAVITY BATCH',group.name,'PASS',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,groups))
    from step3_scheculer.cavity_metadata import correct
    from step3_scheculer.cavity_presentation import publish
    correct(groups);publish(groups)
    subprocess.run([sys.executable,str(Path(__file__).with_name('review_cavity_dsl.py'))],env=env,cwd=I.ROOT,check=True)
if __name__=='__main__':main()
