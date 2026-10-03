"""Audit feasible commits and actual sharing; no exported-fixture geometry replay."""
from pathlib import Path
import sys,json,itertools
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from step3_scheculer import feasible_dsl as F,contacts as I
from step3_scheculer.run_dsl import saved_task
from step3_scheculer.pair_scoring import J


def load(folder,tasks):
    row=json.loads((folder/'state.json').read_text())
    state=F.State(tuple(tuple(I.read_contacts(folder/f'contacts_{t.pose}.npz')) for t in tasks),
        np.asarray(row['placement']['bases']),np.asarray(row['placement']['offsets']),tuple(row['exit_paths']))
    assert row['physical_head_count']==state.physical_count
    assert row['shared_head_count']==state.shared_count
    for p,d in row['artifacts'].items():assert I.sha256(folder/p)==d
    return state,row


def angle_degrees(tasks,state):
    u=[np.asarray(p['initial_object_exit_world'])@np.asarray(t.domain.data['frame']['T_world_mesh'])[:3,:3] for t,p in zip(tasks,state.paths)]
    return float(np.mean([np.degrees(np.arccos(np.clip(a@b,-1,1))) for a,b in itertools.combinations(u,2)])) if len(u)>1 else 0.


def render_assignment(group,tasks,state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids=list(dict.fromkeys(c['candidate_id'] for row in state.groups for c in row))
    matrix=np.array([[any(c['candidate_id']==ident for c in row) for ident in ids] for row in state.groups],int)
    fig,ax=plt.subplots(figsize=(max(7,len(ids)*.65),3))
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(len(ids)),ids,rotation=60,ha='right',fontsize=7)
    ax.set_yticks(range(len(tasks)),[t.pose for t in tasks]);ax.set_title(f'{group.name}: {state.physical_count} physical heads, {state.shared_count} shared')
    ax.set_xlabel('One column = one actual installed physical head; blue = active contact')
    fig.tight_layout();fig.savefig(group/'step3_scheculer'/F.STAGE/'head_assignments.png',dpi=160);plt.close(fig)



def render_geometry(group,tasks,state,witness):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh
    from step4_connect_support import build_coupled_saddle as S,boxed_support as B
    mesh=trimesh.load(witness.parent/'shape.obj',force='mesh',process=False)
    _,owners,_=F.roots(tasks,state)
    objects=[t.domain.mesh.vertices@b+o for t,b,o in zip(tasks,state.bases,state.offsets)]
    clouds=np.vstack([mesh.vertices]+objects);center=clouds.mean(axis=0);radius=np.ptp(clouds,axis=0).max()*.55
    fig=plt.figure(figsize=(16,4));titles=['Actual physical heads','Independent exit paths','Connected fixture','Fixture with one object pose']
    for panel,title in enumerate(titles):
        ax=fig.add_subplot(1,4,panel+1,projection='3d');ax.set_title(title,fontsize=10)
        if panel in (0,1):
            for index,(owner,cells,contact) in enumerate(owners.values()):
                ax.add_collection3d(Poly3DCollection(contact,facecolors=plt.get_cmap('tab20')(index%20),edgecolors='none'))
        if panel in (2,3):ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolors='#b4b9bd',edgecolors='none',alpha=1.))
        if panel==1:
            for k,(t,vertices,path,b) in enumerate(zip(tasks,objects,state.paths,state.bases)):
                ax.add_collection3d(Poly3DCollection(vertices[t.domain.mesh.faces],facecolors=plt.get_cmap('tab10')(k),edgecolors='none',alpha=.13))
                start=vertices.mean(axis=0);direction=np.asarray(path['initial_object_exit_world'])@b
                ax.quiver(*start,*direction,length=radius*.45,color=plt.get_cmap('tab10')(k),arrow_length_ratio=.2)
                ax.text(*start,t.pose,fontsize=7)
        if panel==3:
            ax.add_collection3d(Poly3DCollection(objects[0][tasks[0].domain.mesh.faces],facecolors='#77acd0',edgecolors='none',alpha=.32))
        if panel in (2,3):
            from shapely.geometry import MultiPoint
            from step0_pose_selection.floor_points import pressure_centers
            for t,b,o in zip(tasks,state.bases,state.offsets):
                points=pressure_centers(t.targets/t.scale,t.domain.com)[0]
                hull=MultiPoint(points).convex_hull
                if hull.geom_type=='Polygon':
                    xy=np.asarray(hull.exterior.coords);v=np.c_[xy,np.zeros(len(xy))]@b+o
                    ax.plot(*v.T,color='#b67f3d',linewidth=1.)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(elev=25,azim=-60)
    fig.suptitle(f'{group.name}: accepted feasible geometry ({state.physical_count} heads, {state.shared_count} shared)',fontsize=12)
    fig.tight_layout();fig.savefig(group/'step4/data'/F.BODY_STAGE/'construction.png',dpi=150);plt.close(fig)


def review(group):
    out=group/'step3_scheculer'/F.STAGE;report=I.check_report(out/'report.json')
    assert report['passed'] and report['constructed']
    poses=['pose_'+v for v in group.name.removeprefix('pose').removesuffix('copied').split('+')]
    tasks=[saved_task(group,p) for p in poses];checker=F.Checker(tasks)
    initial,_=load(out/'initial',tasks);final,_=load(out/'final',tasks)
    assert initial.shared_count==0, 'Independent initialization unexpectedly forced sharing'
    initialization=next(e for e in report['events'] if e['operation']=='initialize_feasible')
    initial_witness=I.check_report(I.ROOT/initialization['witness']);assert initial_witness['passed']
    for state in (initial,final):
        check=checker.check(state);assert check['passed'];assert all(r['covered']==32768 for r in check['per_pose'])
        F.roots(tasks,state)
    before=F.objective(tasks,initial);accepted=0
    dependencies=[out/'report.json',out/'final/report.json',out/'initial/state.json',I.ROOT/initialization['witness']]
    for event in report['events']:
        if event['operation']!='accept_feasible_improvement':continue
        assert np.allclose(event['before'],before)
        assert F.improves(tuple(event['before']),tuple(event['after']))
        witness=I.ROOT/event['witness'];body=I.check_report(witness);assert body['passed']
        state,_=load(witness.parent,tasks);check=checker.check(state);assert check['passed']
        assert np.allclose(F.objective(tasks,state),event['after'])
        assert event['checks']['passed'];assert all(r['covered']==32768 for r in event['checks']['per_pose'])
        F.roots(tasks,state);before=tuple(event['after']);accepted+=1;dependencies.append(witness)
    assert np.allclose(before,F.objective(tasks,final))
    witness=I.ROOT/report['witness'];body=I.check_report(witness);assert body['passed']
    step4=group/'step4/data'/F.BODY_STAGE;published=I.check_report(step4/'report.json')
    assert published['passed'] and published['published_model_identical_to_step3_witness']
    digests=[I.sha256(p) for p in (witness.parent/'shape.obj',out/'final/shape.obj',step4/'shape.obj')]
    assert len(set(digests))==1
    render_assignment(group,tasks,final)
    render_geometry(group,tasks,final,witness)
    row=dict(group=group.name,passed=True,audit_passed=True,initial_heads=initial.physical_count,
        physical_heads=final.physical_count,shared_heads=final.shared_count,covered_counts=report['covered_counts'],
        initial_mean_exit_angle_deg=angle_degrees(tasks,initial),final_mean_exit_angle_deg=angle_degrees(tasks,final),
        initial_angle_loss=F.angle_loss(tasks,initial),final_angle_loss=F.angle_loss(tasks,final),
        accepted_feasible_updates=accepted,original_seed_coverage=report['original_seed_coverage'],
        volume_cm3=report['volume_cm3'],step3_implies_step4=True)
    I.save(out/'audit.json',dict(complete=True,**row,provenance=dict(inputs=I.hashes(dependencies+[witness,step4/'report.json',out/'head_assignments.png',step4/'construction.png',step4/'overview.png']),code=I.hashes([Path(__file__)]))))
    (group/'step3_scheculer/README.md').write_text(f"# Active feasible-incumbent DSL (V5)\n\n{row['initial_heads']} initialized heads → {row['physical_heads']} final physical heads; {row['shared_heads']} shared. All {len(tasks)} poses pass all 32768 original loads, independent exits and full fixture acceptance.\n\n[Final state](dsl_feasible/final/state.json) · [Accepted updates](dsl_feasible/report.json) · [Physical head assignments](dsl_feasible/head_assignments.png) · [Audit](dsl_feasible/audit.json)\n")
    readme=group/'step4/README.md';old=readme.read_text() if readme.exists() else ''
    marker='\n## Historical copied baseline description\n\n'
    if old.startswith('# Active feasible-incumbent DSL'):old=old.split(marker,1)[-1] if marker in old else ''
    prefix=f"# Active feasible-incumbent DSL (V5)\n\nThe accepted fixture has {row['physical_heads']} physical heads, including {row['shared_heads']} shared heads. Every pose passes all 32768 original loads and its own continuous exit. Step4 publishes the identical Step3 construction witness.\n\n[Model](data/dsl_feasible_support/shape.obj) · [Overview](data/dsl_feasible_support/overview.png) · [Geometry illustration](data/dsl_feasible_support/construction.png) · [Report](data/dsl_feasible_support/report.json)\n"
    readme.write_text(prefix+(marker+old if old else ''))
    print('FEASIBLE AUDIT',group.name,row['physical_heads'],row['shared_heads'],accepted,flush=True)
    return row



def render_summary(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in rows if r['passed']];y=np.arange(len(rows))
    fig,ax=plt.subplots(figsize=(10,6))
    initial=np.array([r['initial_heads'] for r in rows]);final=np.array([r['physical_heads'] for r in rows]);shared=np.array([r['shared_heads'] for r in rows])
    ax.barh(y-.18,initial,height=.32,color='#b4b9bd',label='Feasible initialized heads')
    ax.barh(y+.18,final-shared,height=.32,color='#5298bd',label='Final independent heads')
    ax.barh(y+.18,shared,left=final-shared,height=.32,color='#80a886',label='Final shared physical heads')
    for i,(a,b) in enumerate(zip(initial,final)):
        ax.text(a+.15,i-.18,str(a),va='center',fontsize=8);ax.text(b+.15,i+.18,str(b),va='center',fontsize=8)
    ax.set_yticks(y,[r['group'] for r in rows]);ax.invert_yaxis();ax.set_xlabel('Distinct physical heads')
    ax.set_title('B: feasible initialization and feasible optimization results')
    ax.set_xlim(0,max(initial)+4);ax.legend(loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(out/'all_cases.png',dpi=160);plt.close(fig)


def main():
    root=I.OUTPUTS/'B';groups=sorted(root.glob('pose*+*'));rows=[]
    for group in groups:
        try:rows.append(review(group))
        except Exception as error:
            import traceback
            rows.append(dict(group=group.name,passed=False,audit_passed=False,error=str(error),traceback=traceback.format_exc()))
            print('AUDIT FAILED',group.name,str(error),flush=True)
    out=root/'pose2+9+13+15+17/step3_scheculer'/F.STAGE
    summary=dict(complete=True,passed=all(r['passed'] for r in rows),passed_count=sum(r['passed'] for r in rows),results=rows,
        validation_policy='fresh force/contact identity audit; full fixture uses its single construction acceptance; no geometry replay',
        provenance=dict(inputs=I.hashes([g/'step3_scheculer'/F.STAGE/'audit.json' for g,r in zip(groups,rows) if r['passed']]),code=I.hashes([Path(__file__)])))
    render_summary(rows,out)
    summary['initialized_head_total']=sum(r['initial_heads'] for r in rows if r['passed'])
    summary['final_physical_head_total']=sum(r['physical_heads'] for r in rows if r['passed'])
    summary['accepted_feasible_update_total']=sum(r['accepted_feasible_updates'] for r in rows if r['passed'])
    I.save(out/'experiment_summary.json',summary)
    lines=['# Feasible-incumbent DSL: B combinations','',f"Complete feasible fixtures: {summary['passed_count']}/{len(rows)}.",'',
        'Each accepted update preserves full force coverage and its complete construction witness. Different exits and independent head subsets remain allowed. Shared heads count actual coincident geometry, not duplicated labels.','',
        '| Group | Initialized heads | Final physical heads | Shared heads | Exit angle (initial → final) | Feasible updates |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['passed']:lines.append(f"| {r['group']} | {r['initial_heads']} | {r['physical_heads']} | {r['shared_heads']} | {r['initial_mean_exit_angle_deg']:.1f}° → {r['final_mean_exit_angle_deg']:.1f}° | {r['accepted_feasible_updates']} |")
        else:lines.append(f"| {r['group']} | failed: {r['error']} | | | | |")
    lines+=['','Original pose2 inputs were incomplete; initialization repairs keep the 32768/32768 acceptance threshold. A finite search does not establish minimum head count; an unchanged feasible incumbent is a valid result. Direction similarity is a proxy and does not certify reduced final footprint.']
    (out/'experiment_summary.md').write_text('\n'.join(lines)+'\n')
    if not summary['passed']:raise SystemExit('Feasible-incumbent audit incomplete')

if __name__=='__main__':main()
