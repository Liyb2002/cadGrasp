"""Independent frozen pure-assembly-Z construction trial for the third set."""
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'step5.2'));sys.path.insert(0,str(HERE))
import stage5_common as C
import base_seating as B
import base_geometry as G
from base_reactions import load_tasks,ground_coverage,validate_contacts,validate_ground


def main():
    inputs=C.Inputs('B');case=inputs.cases[2];data=inputs.load(case);B.load_demands(data,inputs)
    tasks,body=load_tasks(data,inputs)
    directory=inputs.base/'data/step53_pure_z_probe_v1';directory.mkdir(exist_ok=False)
    code=C.begin(directory,'step5.3/pure_z_trials',inputs,[case]);records=[];started=time.perf_counter()
    for extra in [0.,.1,.3]:
        seating=B.Seating(data).solve('z',extra_lift=extra)
        record=dict(extra_minimum_lift_m=extra,seating=seating);records.append(record)
        print('PURE Z',extra,seating['passed'],seating.get('maximum_displacement_coordinate_mm'),flush=True)
        if seating['passed']:
            moved=B.shifted(data,seating);builder=G.Construction(moved,tasks,body,padding=.14)
            mesh=G.unpack(builder.free);coverage,_=ground_coverage(mesh,moved);record['complete_free_space_ground_coverage']=coverage
            record['components']=[]
            print('FULL GROUND',sum(q['passed'] for q in coverage['per_pose']),len(tasks),flush=True)
            for index,component in enumerate(builder.components()):
                mesh=G.unpack(component);ground,points=ground_coverage(mesh,moved)
                result=dict(component=index,ground_coverage=ground,material_cm3=G.volume(component)*1e6)
                record['components'].append(result)
                if not ground['passed']:continue
                geometry=builder.checks(component);result['geometry']=geometry
                if not geometry['passed']:continue
                contacts,_,_=validate_contacts(mesh,builder,tasks,moved,all_poses=False);result['object_contacts']=contacts
                if not contacts['passed']:continue
                whole,_,_=validate_ground(points,tasks,moved);result['system_ground']=whole
                if not whole['passed']:continue
                result['passed']=True;mesh.export(directory/f'accepted_base_extra_{int(extra*1000)}mm.obj')
        C.save(directory/'trials.json',records)
    force_ran=any('object_contacts' in c for r in records for c in r.get('components',[]))
    execution=C.json.loads((directory/'execution.json').read_text());execution['force_solver_run']=force_ran
    execution['boolean_base_construction_run']=True;C.save(directory/'execution.json',execution)
    C.finish(directory,inputs,code,dict(group=case['id'],trials=records,measured_seconds=time.perf_counter()-started,
        accepted_count=sum(c.get('passed',False) for r in records for c in r.get('components',[]))))


if __name__=='__main__':main()
