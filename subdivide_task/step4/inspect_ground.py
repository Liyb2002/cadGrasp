"""Sensitivity of object balance to floor friction on the saved full solid.

This is a diagnostic, not a substitute for the complete fixture certificate.
Only the four object-floor rays are replaced; real fixture-contact rays,
no-uplift slack, samples and continuous endpoint loads remain unchanged.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .mobile import MobileTaskModel
from subdivide_task.step1.common import DEFAULT_OUTPUT,write_json,sha256
from subdivide_task.step3.legacy import kernel as K
from subdivide_task.common.wrenches import endpoint_demands
from whole_search.model import Layout
from whole_search.classify import classify
from material_force import certify_material_contacts
from system_floor import system_floor_demands
from scipy.spatial import ConvexHull


def inspect(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    model=MobileTaskModel(report['poses'],report['object'],report['output_root'])
    with np.load(folder/'step4.2/layout.npz')as data:layout=Layout.from_arrays(data)
    mesh=K.trimesh.load(folder/'step4.2/complete_support.obj',force='mesh',process=False)
    proof=certify_material_contacts(model,layout,mesh)
    tests=[]
    for mu in [0.,.3,.5,1.]:
        rows=[]
        for k in layout.active:
            task=model.tasks[k];full=proof['supplies'][k]
            original=model.floor_columns(layout,k)
            np.testing.assert_allclose(full[:len(original)],original,atol=1e-13,rtol=0)
            if model.floor_signature(layout,k):
                f=np.array([[mu,0,1],[-mu,0,1],[0,mu,1],[0,-mu,1]])
                floor=K.U.floor(np.c_[f,np.cross(task.floor-task.domain.com,f)],task.scale)
                full=K.I.merge_columns(floor,full[len(original):])
            mask,_=classify(full,task.targets)
            endpoints=endpoint_demands(task.domain)['endpoint_need_wrench']*task.scale
            ends,_=classify(full,endpoints)
            rows.append(dict(pose=model.poses[k],samples_passed=int(mask.sum()),samples_total=len(mask),
                endpoints_passed=int(ends.sum()),endpoints_total=len(ends)))
        tests.append(dict(object_floor_friction_coefficient=mu,poses=rows,
            passed=all(r['samples_passed']==r['samples_total']and r['endpoints_passed']==r['endpoints_total']for r in rows)))
    # Benchmark precisely the projection + hull operation replaced in the
    # oracle, using identical translated COMs and no cache. No whole-run claim.
    timings={}
    for mode in ['original_samples','continuous_endpoints']:
        started=time.perf_counter();queries=0
        for height in np.linspace(0.,.15,20):
            for k,task in enumerate(model.tasks):
                needs=task.targets/task.scale if mode=='original_samples' else model.base_oracle.loads[k]
                xy=system_floor_demands(needs,task.domain.com+[0,0,height])['floor_demands_xy_m']
                ConvexHull(xy);queries+=1
        timings[mode]=dict(seconds=time.perf_counter()-started,queries=queries)
    result=dict(schema='mobile_station_ground_diagnostics_v1',tests=tests,
        scope='object balance only; fixed actual support geometry and all original targets',
        complete_support_sha256=sha256(folder/'step4.2/complete_support.obj'),
        floor_hull_benchmark=timings,
        floor_projection_input_reduction_ratio=sum(len(t.targets)for t in model.tasks)/sum(len(x)for x in model.base_oracle.loads))
    write_json(folder/'ground_diagnostics.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',nargs='?',type=Path,default=DEFAULT_OUTPUT/'B/step4/movable_station')
    print(json.dumps(inspect(p.parse_args().folder),indent=2))
