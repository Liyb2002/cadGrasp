"""Compare nominal pressure using mg over a common 1%-surface reference area.

Baseline contacts are read-only. Solve the original finite loads, minimizing
the largest head's sum of compressive normal forces. These are required force
capacities under this allocation, not unique elastic reactions or local peaks.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
import trimesh
import highspy
from scipy.sparse import csc_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GROUP=ROOT/'slides/baseline_algo/output/B/pose1+3'
OUT=HERE/'pressure_data'


def baseline(pose):
    inputs=GROUP/'step4/data/source_inputs'/pose
    domain_path=inputs/'needs.json';samples_path=inputs/'samples.json'
    contacts_path=GROUP/'step3_scheculer/co_design_reference'/f'contacts_{pose}.npz'
    setup_path=ROOT/'objects/B/history/before_compatible_poses_860a4233e74b/tasks'/pose/'setup.npz'
    d=json.loads(domain_path.read_text());s=json.loads(samples_path.read_text())
    mesh=trimesh.Trimesh(d['geometry']['vertices_m'],d['geometry']['faces'],process=False)
    com=np.array(d['frame']['moment_origin_m']);z=np.load(contacts_path)
    floor=np.load(setup_path)['floor_contact_m']
    rays=np.array([[64,0,1],[-64,0,1],[0,64,1],[0,-64,1.]])
    groups=[]
    for a,b in zip(z['offsets'][:-1],z['offsets'][1:]):
        points=z['triangles_m'][a:b].reshape(-1,3)
        normal=np.repeat(-mesh.face_normals[z['source_faces'][a:b]],3,axis=0)
        w=np.c_[normal,np.cross(points-com,normal)]
        ids=np.unique(np.round(w,12),axis=0,return_index=True)[1]
        groups.append(w[np.sort(ids)])
    head=np.vstack(groups);floor_w=np.c_[rays,np.cross(floor-com,rays)]
    cols=np.vstack([head,floor_w]);n=len(cols);nh=len(head)
    scale=np.r_[np.ones(3),np.ones(3)/mesh.extents.max()]
    eq=np.c_[cols.T*scale[:,None],np.zeros(6)]
    # Original shared no-uplift condition: head Fz >= 0.
    uplift=np.r_[-head[:,2],np.zeros(4),0.]
    capacities=np.zeros((len(groups),n+1));start=0
    for k,g in enumerate(groups):
        capacities[k,start:start+len(g)]=1;capacities[k,-1]=-1;start+=len(g)
    A=csc_matrix(np.vstack([eq,uplift,capacities]))
    lp=highspy.HighsLp();lp.num_col_=n+1;lp.num_row_=len(A.toarray())
    lp.col_cost_=np.r_[np.zeros(n),1.];lp.col_lower_=np.zeros(n+1);lp.col_upper_=np.full(n+1,np.inf)
    lp.row_lower_=np.r_[np.zeros(6),np.full(1+len(groups),-np.inf)]
    lp.row_upper_=np.zeros(7+len(groups))
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_=A.indptr;lp.a_matrix_.index_=A.indices;lp.a_matrix_.value_=A.data
    solver=highspy.Highs();solver.setOptionValue('output_flag',False);solver.setOptionValue('threads',1)
    solver.setOptionValue('primal_feasibility_tolerance',1e-8)
    solver.passModel(lp)
    targets=np.asarray(s['need_wrench']);assert len(targets)==32768
    # Include exact 0.5mg endpoints of each original sampled direction, as
    # requested. The optimum is convex along each magnitude segment: gravity
    # and this endpoint suffice to bound the original intermediate samples.
    push=np.array(s['force_push_mg']);length=np.linalg.norm(push,axis=1)
    endpoint=.5*push/length[:,None];points=np.array(s['pt_m'])
    ends=np.c_[-(endpoint+[0,0,-1]),-np.cross(points-com,endpoint)]
    targets=np.vstack([targets,ends,[0,0,1,0,0,0]])
    max_each=np.zeros(len(groups));worst_ids=np.zeros(len(groups),int);max_cap=0.;worst=None
    max_net=np.zeros(len(groups));net_witnesses=[None]*len(groups)
    offsets=np.cumsum([0]+[len(g) for g in groups]);failed=[];t0=time.monotonic()
    for i,target in enumerate(targets):
        rhs=target*scale
        solver.changeRowsBounds(6,np.arange(6,dtype=np.int32),rhs,rhs);solver.run()
        if solver.getModelStatus()!=highspy.HighsModelStatus.kOptimal:
            failed.append(i);continue
        x=np.asarray(solver.getSolution().col_value);forces=np.array([sum(x[a:b]) for a,b in zip(offsets[:-1],offsets[1:])])
        net=np.array([groups[k][:,:3].T@x[a:b] for k,(a,b) in enumerate(zip(offsets[:-1],offsets[1:]))])
        for k,value in enumerate(np.linalg.norm(net,axis=1)):
            if value>max_net[k]:
                max_net[k]=value
                net_witnesses[k]=dict(index=i,force_on_object_mg=net[k].tolist(),force_on_head_mg=(-net[k]).tolist())
        assert np.max(abs(cols.T@x[:n]-target))<2e-6
        assert np.dot(head[:,2],x[:nh]) >= -2e-6
        assert max(forces) <= x[-1]+2e-6
        update=forces>max_each;max_each[update]=forces[update];worst_ids[update]=i
        if x[-1]>max_cap:max_cap=float(x[-1]);worst=dict(index=i,target=target.tolist(),forces_mg=forces.tolist(),ground_force_mg=(rays.T@x[nh:n]).tolist())
        if i%8192==0:print(pose,i,'/',len(targets),'cap',max_cap,flush=True)
    sources=[domain_path,samples_path,contacts_path,setup_path,Path(__file__)]
    row=dict(pose=pose,head_ids=z['candidate_ids'].tolist(),object_area_m2=mesh.area,
             assumed_each_head_area_m2=.01*mesh.area,
             original_samples=32768,endpoint_samples=32768,gravity_samples=1,
             infeasible_original_count=sum(i<32768 for i in failed),
             infeasible_endpoint_count=sum(32768<=i<65536 for i in failed),
             head_max_compressive_force_mg_in_minimax_solutions=max_each.tolist(),
             head_max_resultant_force_mg_in_minimax_solutions=max_net.tolist(),
             head_resultant_witnesses=net_witnesses,
             head_worst_indices=worst_ids.tolist(),common_head_capacity_mg=max_cap,
             worst=worst,seconds=time.monotonic()-t0,
             source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    print(pose,'RESULT',max_each,'infeasible',len(failed),flush=True)
    return row


def main():
    rows=[baseline(p) for p in ('pose_1','pose_3')]
    belt=json.loads((OUT/'report.json').read_text())
    z=np.load(HERE/'original_geometry.npz');mesh=trimesh.Trimesh(z['object_vertices'],z['object_faces'],process=False)
    ref_area=.01*mesh.area;dock=[]
    for r in belt['records']:
        # p/(mg/Aref): same reference area for meaningful pressure comparison.
        equivalent=r['locked_sampled_max_minimum_tile_pressure_kpa']*1000*ref_area/belt['weight_N']
        face=[];tile=np.array(r['worst']['tile_pressure_kpa']);a=np.array(belt['contact_model']['tile_area_m2']);labels=np.array(belt['contact_model']['labels'])
        for name in dict.fromkeys(labels):face.append(dict(face=str(name),force_mg=float(sum(tile[labels==name]*a[labels==name]*1000)/belt['weight_N'])))
        dock.append(dict(pose=r['pose'],net_force_mg_at_pressure_selected_load=float(np.linalg.norm(r['worst']['interface_reaction_local_N_Nm'][:3])/belt['weight_N']),
                         pressure_equivalent_mg_on_1pct_area=equivalent,faces_at_selected_load=face))
    report=dict(reference_pressure='object weight / (1% of object total surface area)',
        explanation='A value X means the same nominal pressure as X object weights pressing on one 1%-surface patch. It is not X weights of resultant dock force.',
        baseline_policy='For each load minimize max_j sum(normal contact forces on head j); report per-head maxima in chosen optimal allocations. Normal-force sum / assumed area is head-mean pressure, not a local peak; physical force allocations are not unique.',
        caveat='Historical reference has six physical patches/five IDs. Original Step3 floor friction mu=64 and shared no-uplift retained. Dock has no object-ground load sharing and hypothetical ideal retainer; this is not a matched physical-system comparison.',
        baseline=rows,dock=dock)
    (OUT/'comparison_mg.json').write_text(json.dumps(report,indent=2)+'\n')
    from draw_pose3_load import draw
    draw(report)


if __name__=='__main__':main()
