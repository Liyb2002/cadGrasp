"""Reuse checked prerequisites; evaluate native carving once, inside Step4.2."""
from prepare_five_pose_group import *
from exit_clearance import ExitClearance


def prepare_group_fast(name,group):
    base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']
    for stage in ['step3.1','step3.2','step3.3']:
        report=base/'step3'/stage/'data/report.json'
        if report.exists():I.check_report(report)
    if not (base/'step3/step3.1/data/report.json').exists():
        states,mesh,_=step31.run(name,group,base/'step3/step3.1')
    else:
        states={p:state(name,p) for p in group['poses']};mesh=states[group['poses'][0]][2]
    if not (base/'step3/step3.2/data/report.json').exists():
        step32.run(name,group,states,mesh,base/'step3/step3.2')
    gate=json.loads((base/'step3/step3.2/data/report.json').read_text())
    if gate['status']=='fail':
        raise CommonSurfaceInfeasible('Full common non-work-surface force upper bound fails; see Step3.2 failure_proof')
    if not (base/'step3/step3.3/data/report.json').exists():build_rings(name,group)
    initial=base/'step4/step4.1/data/report.json';out=initial.parent;out.mkdir(parents=True,exist_ok=True)
    directions_path=out/'native_start_directions.npz'
    if initial.exists():
        I.check_report(initial);report=json.loads(initial.read_text())
        directions=np.array([r['direction_fixture'] for r in report['state_results']])
        if directions_path.exists():
            np.testing.assert_allclose(np.load(directions_path)['directions'],directions,atol=1e-12,rtol=0)
        else:np.savez_compressed(directions_path,directions=directions)
        return directions_path
    seed_path=base/'step3/step3.3/support_with_rings.obj'
    support=trimesh.load(seed_path,force='mesh',process=False)
    combined=np.vstack([support.vertices,mesh.vertices])
    length=max(.5,float(np.linalg.norm(np.ptp(combined,axis=0)))+.02)
    paths={};rows=[];inputs=[seed_path,base/'step3/step3.3/data/report.json',
                            base/'step3/step3.2/data/contacts.npz',ROOT/'objects'/name/'pose_sets.json']
    for pose,(task,T,m) in states.items():
        d,native=step41.initialize_direction(T)
        paths[pose]=dict(direction_fixture=d.tolist(),direction_world=native.tolist())
        rows.append(dict(pose=pose,direction_fixture=d.tolist(),direction_world=native.tolist(),
                         force_covered=None,force_passed=None,load_count=len(task.targets)))
        inputs+=task.inputs
    np.savez_compressed(directions_path,directions=np.array([paths[p]['direction_fixture'] for p in group['poses']]))
    initialization=dict(complete=True,initialization_kind='Each pose withdraws along its own native world +z',
        paths=paths,full_length_m=length,display_length_m=.10,initialization_is_arbitrary=True,
        paths_independent_variables=True,initial_canonical_sweeps_identical=False,
        length_policy='Combined seed/object AABB diagonal plus 20 mm; at least 500 mm')
    report=dict(complete=True,stage='step4.1_native_direction_definition',object=name,pose_set=group['id'],
        initialization=initialization,state_results=rows,initial_support_constructed=False,
        geometry_partition_resolved=None,force_and_path_initialization_gate=None,
        status='native_directions_defined_construction_deferred_to_step4.2',optimized=False,
        full_fixture_accepted=False,exit_clearance=ExitClearance(mesh).metadata,
        policy='Identical native-up directions; Step4.2 first exact call constructs native support and classifies every load once',
        provenance=provenance(inputs,[Path(__file__),HERE/'helper_func/prepare_five_pose_group.py',
            HERE/'step4.1/step41.py',HERE/'helper_func/exit_clearance.py']),
        artifacts={'native_start_directions.npz':I.sha256(directions_path)})
    save(initial,report);I.check_report(initial)
    return directions_path
