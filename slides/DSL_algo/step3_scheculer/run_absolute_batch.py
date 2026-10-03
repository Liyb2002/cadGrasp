"""Reproducible two-round all-set Step3/Step4/Step5 experiment."""
from pathlib import Path
import argparse,concurrent.futures,subprocess,sys,os,json
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    groups=sorted((I.OUTPUTS/'B').glob('pose*+*'))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    def run(group):
        for stage,script in (('dsl_absolute','run_absolute_dsl.py'),('dsl_absolute_refined','run_absolute_refinement.py')):
            if args.resume:
                try:
                    r=I.check_report(group/'step3_scheculer'/stage/'report.json')
                    if r['passed']:continue
                except (RuntimeError,FileNotFoundError,KeyError):pass
            out=group/'step3_scheculer'/stage;out.mkdir(parents=True,exist_ok=True)
            with (out/'batch.log').open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name(script)),group.name],env=env,cwd=I.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        failures=[]
        for path in (group/'step3_scheculer/dsl_absolute/trials').glob('*common_up*/report.json'):
            report=json.loads(path.read_text())
            if not report.get('passed') and any(c.get('min_fixture_z_m',0)<-1e-9 for c in report.get('checks',[]) or []):failures.append(path)
        if failures:
            stage=group/'step3_scheculer/dsl_absolute_floor';ready=False
            if args.resume:
                try:ready=I.check_report(stage/'report.json')['passed']
                except (RuntimeError,FileNotFoundError,KeyError):pass
            if not ready:
                stage.mkdir(parents=True,exist_ok=True)
                with (stage/'batch.log').open('w') as log:
                    subprocess.run([sys.executable,str(Path(__file__).with_name('absolute_floor_recovery.py')),group.name],env=env,cwd=I.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('ABSOLUTE BOTH ROUNDS',group.name,'PASS',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,groups))
    from step3_scheculer.correct_absolute_metadata import correct
    correct(groups)
    from step3_scheculer.absolute_presentation import publish
    publish(groups)
    subprocess.run([sys.executable,str(Path(__file__).with_name('review_absolute_dsl.py'))],env=env,cwd=I.ROOT,check=True)
if __name__=='__main__':main()
