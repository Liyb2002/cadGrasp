"""Matched nominal pressure-capacity comparison, not a material failure claim.

Every surface tile has constant unilateral frictionless pressure. Minimize a
common tile-pressure cap for each identical load, with object-ground reactions.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,hstack,vstack,eye
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from interface_pressure import contact_model

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GROUP=ROOT/'slides/baseline_algo/output/B/pose1+3'


def model(points,normals,areas,labels,com,floor,extent,reference_area):
    # q = pressure/(mg/Aref); force in mg = q * area/Aref * normal.
    forces=normals*(areas/reference_area)[:,None]
    head=np.c_[forces,np.cross(points-com,forces)]
    rays=np.array([[64,0,1],[-64,0,1],[0,64,1],[0,-64,1.]])
    ground=np.c_[rays,np.cross(floor-com,rays)]
    cols=np.vstack([head,ground]);n=len(cols);nh=len(head)
    scale=np.r_[np.ones(3),np.ones(3)/extent]
    eq=csr_matrix(np.c_[cols.T*scale[:,None],np.zeros(6)])
    pressure_cap=hstack([eye(nh,format='csr'),csr_matrix((nh,4)),csr_matrix(-np.ones((nh,1)))])
    no_uplift=csr_matrix(np.r_[-head[:,2],np.zeros(5)][None,:])
    ub=vstack([pressure_cap,no_uplift],format='csr')
    def solve(point,push):
        need=np.r_[-(push+[0,0,-1]),-np.cross(point-com,push)]
        fit=linprog(np.r_[np.zeros(n),1],A_eq=eq,b_eq=need*scale,A_ub=ub,b_ub=np.zeros(nh+1),bounds=(0,None),method='highs')
        if not fit.success:
            if fit.status!=2:raise RuntimeError(fit.message)
            return dict(feasible=False,message=fit.message)
        cap=float(fit.x[-1])
        # Select a low-total-compression allocation among equal-cap solutions.
        objective=np.r_[areas/reference_area,np.zeros(5)]
        allocation=linprog(objective,A_eq=eq,b_eq=need*scale,
            A_ub=no_uplift,b_ub=[0.],bounds=[(0,cap+1e-7)]*nh+[(0,None)]*4+[(cap,cap)],method='highs')
        if not allocation.success:raise RuntimeError(allocation.message)
        x=allocation.x;residual=float(np.max(abs(cols.T@x[:n]-need)))
        assert residual<1e-7
        compression=x[:nh]*areas/reference_area
        group_stats=[]
        for name in dict.fromkeys(labels):
            selected=labels==name
            group_stats.append(dict(name=str(name),peak_tile_pressure_ratio=float(max(x[:nh][selected])),
                average_face_pressure_ratio=float(np.sum(x[:nh][selected]*areas[selected])/sum(areas[selected])),
                normal_force_sum_mg=float(sum(compression[selected])),
                net_force_on_object_mg=(forces[selected].T@x[:nh][selected]).tolist()))
        return dict(feasible=True,minimum_pressure_cap_ratio=cap,groups=group_stats,
                    total_normal_force_sum_mg=float(sum(compression)),
                    overall_net_force_on_object_mg=(forces.T@x[:nh]).tolist(),
                    ground_force_on_object_mg=(rays.T@x[nh:n]).tolist(),wrench_residual=residual,
                    tile_pressure_ratios=x[:nh].tolist())
    return solve


def run(solve,loads):
    records=[]
    for load in loads:
        solution=solve(np.array(load['tool_point_world_m']),np.array(load['tool_force_mg']))
        records.append(dict(sample_id=load['sample_id'],solution=solution))
    good=[r for r in records if r['solution']['feasible']]
    peak=np.array([r['solution']['minimum_pressure_cap_ratio'] for r in good])
    i=int(np.argmax(peak))
    return dict(count=len(loads),feasible_count=len(good),average_minimum_peak_ratio=float(peak.mean()),
                worst_minimum_peak_ratio=float(peak[i]),worst_sample_id=good[i]['sample_id'],records=records)


def calculate():
    stat_path=HERE/'pressure_data/statistics_100.json';stats=json.loads(stat_path.read_text())
    domain_path=GROUP/'step4/data/source_inputs/pose_3/needs.json';d=json.loads(domain_path.read_text())
    mesh=trimesh.Trimesh(d['geometry']['vertices_m'],d['geometry']['faces'],process=False)
    com=np.array(d['frame']['moment_origin_m']);ref=.01*mesh.area
    cp=GROUP/'step3_scheculer/co_design_reference/contacts_pose_3.npz';z=np.load(cp)
    sp=ROOT/'objects/B/history/before_compatible_poses_860a4233e74b/tasks/pose_3/setup.npz';floor=np.load(sp)['floor_contact_m']
    bp=z['triangles_m'].mean(1);bn=-mesh.face_normals[z['source_faces']];ba=z['triangle_areas_m2']
    labels=np.concatenate([np.full(b-a,str(name)) for name,a,b in zip(z['candidate_ids'],z['offsets'][:-1],z['offsets'][1:])])
    base_coarse=run(model(bp,bn,ba,labels,com,floor,mesh.extents.max(),ref),stats['baseline']['records'])
    tri=z['triangles_m'];a,b,c=tri[:,0],tri[:,1],tri[:,2]
    ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
    sub=np.stack([np.stack([a,ab,ca],axis=1),np.stack([ab,b,bc],axis=1),
                  np.stack([ca,bc,c],axis=1),np.stack([ab,bc,ca],axis=1)],axis=1).reshape(-1,3,3)
    base=run(model(sub.mean(1),np.repeat(bn,4,axis=0),np.repeat(ba/4,4),np.repeat(labels,4),com,floor,mesh.extents.max(),ref),stats['baseline']['records'])
    pinned_path=HERE/'original_geometry.npz';pinned=np.load(pinned_path)
    align=np.array(stats['matched_object_registration']['transform'])
    port=align[:3,:3]@pinned['port']+align[:3,3];basis=align[:3,:3]@pinned['basis']
    dock_results=[]
    for resolution in [(8,4),(16,8)]:
        _,tiles=contact_model(*resolution,locked=True)
        dp=np.array(tiles['points_m'])@basis.T+port;dn=np.array(tiles['normals'])@basis.T
        da=np.array(tiles['tile_area_m2']);dl=np.array(tiles['labels'])
        result=run(model(dp,dn,da,dl,com,floor,mesh.extents.max(),ref),stats['baseline']['records'])
        result['grid']=list(resolution);dock_results.append(result)
    coarse,fine=dock_results
    b=np.array([r['solution']['minimum_pressure_cap_ratio'] for r in base['records']]);a=np.array([r['solution']['minimum_pressure_cap_ratio'] for r in fine['records']])
    ratio=a/b
    sources=[stat_path,domain_path,cp,sp,pinned_path,Path(__file__),HERE/'interface_pressure.py']
    report=dict(count=100,quantity='Minimum possible maximum nominal surface-tile pressure for each load, not resultant force and not a predicted elastic peak.',
        pressure_reference='Object weight spread over one contact patch equal to 1% of total object surface area. All plotted values are dimensionless multiples, not mg forces.',
        reference_area_m2=ref,baseline_head_areas_m2=[float(sum(ba[a:b])) for a,b in zip(z['offsets'][:-1],z['offsets'][1:])],
        model='Uniform frictionless pressure on each actual head triangle or ideal snug dock-face tile, mu=64 object-ground point, shared no-uplift, massless rigid support/belt. Minimize common pressure cap; secondary minimize total compression. No force limit added to baseline pipeline.',
        dock='Hypothetical zero-clearance locked rectangular interface, side faces plus full-area retaining face; no geometric placement or base equilibrium certification.',
        limitations='Nominal discretized rigid traction-capacity model, not material or elastic-contact FEA; mean/worst over the same 100 downward-component loads; no global maximum or guaranteed material failure.',
        baseline=base,baseline_coarse=base_coarse,baseline_refinement='Every original head triangle subdivided into four uniform-pressure triangles.',dock_coarse=coarse,dock_refined=fine,
        comparison=dict(ratio_of_average_peak_demands=fine['average_minimum_peak_ratio']/base['average_minimum_peak_ratio'],
            ratio_of_worst_peak_demands=fine['worst_minimum_peak_ratio']/base['worst_minimum_peak_ratio'],
            per_sample_ratio_mean=float(ratio.mean()),per_sample_ratio_min=float(ratio.min()),per_sample_ratio_max=float(ratio.max()),
            dock_higher_sample_count=int(np.sum(a>b)),baseline_higher_sample_count=int(np.sum(b>a))),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    out=HERE/'pressure_data/pressure_comparison_100.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print('BASELINE',base['average_minimum_peak_ratio'],base['worst_minimum_peak_ratio'],flush=True)
    print('DOCK COARSE',coarse['average_minimum_peak_ratio'],coarse['worst_minimum_peak_ratio'],flush=True)
    print('DOCK REFINED',fine['average_minimum_peak_ratio'],fine['worst_minimum_peak_ratio'],flush=True)
    print('COMPARISON',report['comparison'],flush=True)
    return report


def render(report):
    fig,(chart,scatter)=plt.subplots(1,2,figsize=(13,6),gridspec_kw=dict(width_ratios=[1,1.15]))
    b=report['baseline'];d=report['dock_refined'];x=np.arange(2)
    chart.bar(x-.16,[b['average_minimum_peak_ratio'],d['average_minimum_peak_ratio']],width=.3,color='#78a9c0',label='Average of 100')
    chart.bar(x+.16,[b['worst_minimum_peak_ratio'],d['worst_minimum_peak_ratio']],width=.3,color='#d99a45',label='Worst of 100')
    chart.set_xticks(x,['Three-head support','Small locked dock']);chart.set_ylabel('Nominal pressure / reference pressure');chart.legend(frameon=False)
    chart.spines[['top','right']].set_visible(False)
    for k,result in enumerate([b,d]):
        for offset,key in [(-.16,'average_minimum_peak_ratio'),(.16,'worst_minimum_peak_ratio')]:
            v=result[key];chart.text(k+offset,v,f'{v:.2f} x',ha='center',va='bottom',fontsize=12)
    bv=np.array([r['solution']['minimum_pressure_cap_ratio'] for r in b['records']]);dv=np.array([r['solution']['minimum_pressure_cap_ratio'] for r in d['records']])
    limit=max(bv.max(),dv.max())*1.07
    scatter.scatter(bv,dv,color='#528eb1',alpha=.65,s=22)
    scatter.plot([0,limit],[0,limit],'--',color='#777777');scatter.set(xlim=(0,limit),ylim=(0,limit),xlabel='Support nominal pressure / reference',ylabel='Dock nominal pressure / reference')
    scatter.set_aspect('equal',adjustable='box');scatter.spines[['top','right']].set_visible(False)
    scatter.text(.04,.92,'Above line: dock higher\nBelow line: support higher',transform=scatter.transAxes,fontsize=10)
    fig.suptitle('Same 100 loads: compare contact pressure, not net force',fontsize=19,y=.97)
    fig.text(.05,.09,'Reference pressure = object weight distributed over 1% of the object surface. Both models include object-ground support.',fontsize=10)
    fig.text(.05,.047,'Optimistic nominal pressure-capacity model; ideal dock lock and zero clearance. This is not a material failure certificate.',fontsize=10,color='#666666')
    fig.tight_layout(rect=[0,.13,1,.92]);path=HERE.parent/'pressure_comparison_100.png';fig.savefig(path,dpi=170);plt.close(fig);print(path,flush=True)


if __name__=='__main__':render(calculate())
