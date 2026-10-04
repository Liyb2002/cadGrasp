"""Compare stable-contact construction against an accepted saved incumbent."""
import argparse
import os
import shutil
import subprocess
import sys
import experiment as E

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);args=p.parse_args()
    saved=['step3','step4/data/growing_support','step4/data/source_inputs','step5_evaluate','comparison.json']
    for name in args.groups:
        group=E.OUT/name;comparison=group/'comparison.json';previous=E.json.loads(comparison.read_text());assert previous['passed']
        E.I.check_report(group/'step4/data/growing_support/report.json')
        history=group/'history/before_stable_improvement'
        if history.exists():shutil.rmtree(history)
        for rel in saved:
            src=group/rel;dst=history/rel
            if src.is_dir():shutil.copytree(src,dst)
            elif src.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        subprocess.run([sys.executable,str(E.HERE/'stable_contact_recovery.py'),'--groups',name],check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4'))
        current=E.json.loads(comparison.read_text())
        before=previous['step5']['object_and_support_poses']['box_volume_cm3'];after=current['step5']['object_and_support_poses']['box_volume_cm3'] if current['passed'] else None
        kept_new=after is not None and after<before
        if not kept_new:
            for rel in saved:
                src=history/rel;dst=group/rel
                if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
                elif src.exists():shutil.copy2(src,dst)
        E.save(group/'improvement_review.json',dict(previous_volume_cm3=before,candidate_volume_cm3=after,kept_new=kept_new,comparison_uses_actual_step5=True,head_count_penalty=0))
        print('BEST CONSTRUCTED VOLUME',name,min(before,after) if after is not None else before,flush=True)
