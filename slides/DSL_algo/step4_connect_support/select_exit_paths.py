"""Select complete exit paths by actual excluded workspace in fixture coordinates.

Exit-space union is primary; direction closeness is a soft secondary preference.
Beam search ranks exact padded sweep unions inside a finite navigation workspace;
the final constructor ranks accepted fixtures by actual workstation XY footprint.
"""
from pathlib import Path
import numpy as np
from step3_scheculer import contacts as I, exit_options as EXIT
from step4_connect_support import boxed_support as F, build_coupled_saddle as S


def select(tasks,reports,roots,bases,offsets,window,beam_width=3):
    workspace=F.bounded_space(window,bases,offsets)
    mandatory=F.union(roots)
    cube=F.md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
    tolerance=1e-11*S.SCALE**3
    choices=[];diagnostics=[]
    for task,report,b,o in zip(tasks,reports,bases,offsets):
        accepted=[];rows=[]
        for record in report['exit_options']:
            plan=record['plan']
            try:
                raw=EXIT.sweep_mesh(task.domain.mesh,plan)
                installed=F.transform(S.solid(raw),b,o)
                overlap=abs(float((mandatory^installed).volume()))*S.SCALE**3
                padded=installed.minkowski_sum(cube)
                cap=padded^workspace
                row=dict(option_id=plan['id'],kind=plan['kind'],
                    mandatory_root_overlap_m3=overlap,eligible=overlap<=tolerance,
                    excluded_workspace_cm3=float(cap.volume())*S.SCALE**3*1e6)
                if row['eligible']:
                    accepted.append(dict(plan=plan,raw=raw,installed=installed,padded=padded,cap=cap,metrics=row))
            except (RuntimeError,ValueError) as error:
                row=dict(option_id=plan['id'],eligible=False,numerically_unresolved=True,error=str(error))
            rows.append(row)
        choices.append(accepted);diagnostics.append(dict(pose=task.pose,options=rows))
    detail=dict(object_motion='fixed support; piecewise-linear object translation',
        selection_metric='exact padded object sweep union volume inside common fixture-frame navigation workspace',
        cross_pose_angle_penalty=True,angle_is_soft_and_never_a_feasibility_gate=True,finite_navigation_window=window,
        mandatory_roots_checked=True,root_volume_tolerance_m3=tolerance,diagnostics=diagnostics,
        beam_width=beam_width,global_optimality_claim=False)
    if any(not row for row in choices):
        detail.update(passed=False,error='No retained exit option clears all installed real roots for '+', '.join(
            task.pose for task,row in zip(tasks,choices) if not row))
        return [],detail
    states=[dict(selected=[],cap=F.md.Manifold(),cost=0.,angle_loss=0.)]
    for candidates in choices:
        expanded=[]
        for state in states:
            for option in candidates:
                cap=state['cap']+option['cap']
                selected=state['selected']+[option]
                vectors=[]
                for p,b in zip(selected,bases):
                    plan=p['plan']
                    v=np.asarray(plan.get('initial_object_exit_world',plan['object_translation_waypoints_world_m'][1]),float)
                    vectors.append(v/max(np.linalg.norm(v),1e-15)@b)
                angles=[1-np.clip(a@b,-1,1) for i,a in enumerate(vectors) for b in vectors[i+1:]]
                expanded.append(dict(selected=selected,cap=cap,angle_loss=float(np.mean(angles)) if angles else 0.,
                    cost=float(cap.volume())*S.SCALE**3*1e6))
        expanded.sort(key=lambda state:(state['cost']+.01*state['angle_loss'],tuple(p['plan']['id'] for p in state['selected'])))
        states=expanded[:beam_width]
    detail.update(passed=True,candidates=[dict(option_ids=[p['plan']['id'] for p in state['selected']],
        excluded_workspace_cm3=state['cost'],soft_direction_closeness=state['angle_loss'],paths=[p['plan'] for p in state['selected']]) for state in states])
    return states,detail


def sources():return [Path(__file__)]+EXIT.sources()
