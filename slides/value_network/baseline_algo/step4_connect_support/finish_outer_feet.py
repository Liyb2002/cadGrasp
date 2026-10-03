"""Install one reviewed-design candidate as the sole current Step4 geometry."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import hollow_feet as H


def finish(group_name,index,allow_failed=False):
    out=I.OUTPUTS/'B'/group_name/'step4';data=out/'data'
    source=data/'outer_design'/f'candidate_{index:02d}'
    report=I.check_report(source/'outer_report.json');construction=I.check_report(source/'report.json')
    if not report.get('process_access',{}).get('passed'):
        raise ValueError('Candidate has no passed whole-fixture working-surface check')
    if not report['passed'] and not allow_failed:raise ValueError('Candidate did not pass all original checks')
    design=json.loads((source/'design.json').read_text())
    count=sum(len(H.pockets(np.asarray(xy))) for floor in design['feet_xy_m'] for xy in floor)
    design['optional_final_hollowing']=dict(considered=True,admissible_pocket_count=count,applied=False,
        rim_m=H.WALL,rib_m=H.RIB,
        reason='The small soles cannot fit pockets while retaining the border and central rib' if count==0
            else 'Solid reference-style feet retained; no subtraction accepted')
    I.save(source/'design.json',design)
    construction['body_design']=design;construction['artifacts']['design.json']=I.sha256(source/'design.json')
    construction['provenance']['code'].update(I.hashes([Path(__file__),Path(H.__file__)]))
    I.save(source/'report.json',construction)
    # Keep only this generation of material, not an archive of earlier frames.
    target=data/'current_build'
    if target.exists():shutil.rmtree(target)
    source.rename(target)
    for folder in ('outer_design','co_design_body','viewer','search'):
        if (data/folder).exists():shutil.rmtree(data/folder)
    if (target/'preview').exists():shutil.rmtree(target/'preview')
    (target/'outer_report.json').unlink(missing_ok=True)
    for stale in ('independent_review.json','visualization.json','review.log','publish.log','run.log'):
        (data/stale).unlink(missing_ok=True)
    cache=data/'construction_cache'
    for path in cache.iterdir():
        if path.is_file() and path.name not in ('forbidden.npz','sweep0.npz','sweep1.npz','sweep_cache.json'):
            path.unlink()
    shutil.copyfile(target/'fixture.obj',out/'shape.obj')
    report.update(construction=construction,body_directory='data/current_build',
        artifacts={'../shape.obj':I.sha256(out/'shape.obj')})
    report['provenance']['code'].update(I.hashes([Path(__file__),Path(H.__file__)]))
    I.save(data/'report.json',report)
    print('CURRENT OUTER FEET',group_name,'accepted',report['passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group');p.add_argument('--candidate',type=int,default=0)
    p.add_argument('--allow-failed',action='store_true');args=p.parse_args()
    finish(args.group,args.candidate,args.allow_failed)
