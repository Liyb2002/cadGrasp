"""Read-only sampled diagnostic of initial versus mutated solo exits."""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'slides/Co-optimize/helper_func'))
import _bootstrap
from co_common import HERE,I,save
from whole_pipeline import load_layout
from continuous_support.expanded_gradient import ExpandedGradientModel
from whole_search.classify import classify

def main():
    label='pose1+3+5+7+9+12+16+21+23+27'
    base=HERE/'output/B'/label;out=base/'step4/step4.2/loss_first_gradient_v14'
    process=json.loads((out/'process.json').read_text())
    current=load_layout(out/process[-1]['layout'])
    initial=load_layout(base/'step4/step4.1/layout.npz')
    poses=I.check_report(base/'step4/step4.1/data/report.json')['poses']
    m=ExpandedGradientModel(poses,'B',initialization_report=base/'step3/step3.1/data/report.json')
    rows=[];began=time.monotonic()
    for k in current.active:
        row=dict(pose=poses[k],directions={})
        for name,d in [('mutated',current.directions[k]),
                       ('initial_relative',current.placements[k,:3,:3]@initial.directions[k])]:
            q=current.copy();q.active=(k,);q.directions[k]=d
            flags=m.contact_delta.state(q).available[k];rays,columns=m.supply_at_points(k,flags)
            mask,_=classify(rays,m.tasks[k].targets,basis_cache=[],column_ids=columns)
            row['directions'][name]=dict(passed_original_samples=int(mask.sum()),contact_points=int(flags.sum()))
        rows.append(row);print(row,flush=True)
    save(HERE/'output/B/data/solo_direction_probe/result.json',dict(case=label,rows=rows,
         seconds=time.monotonic()-began,only_sampled_point_geometry=True,not_full_case_acceptance=True))

if __name__=='__main__':main()
