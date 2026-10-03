"""Conditional pressure lower bounds for a clearance sleeve's edge contact.

Effective edge-band width b is a parameter, NOT measured or inferred. Contact
locations are relaxed to all cavity-face corners, making the bound optimistic.
No material failure conclusion or actual elastic contact width is claimed.
"""
from pathlib import Path
import json,hashlib
from itertools import product
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GROUP=ROOT/'slides/baseline_algo/output/B/pose1+3'


def calculate():
    sp=HERE/'pressure_data/statistics_100.json';stats=json.loads(sp.read_text())
    pp=HERE/'pressure_data/pressure_comparison_100.json';pressure=json.loads(pp.read_text())
    zp=HERE/'original_geometry.npz';z=np.load(zp)
    T=np.array(stats['matched_object_registration']['transform']);B=T[:3,:3]@z['basis'];port=T[:3,:3]@z['port']+T[:3,3]
    dp=GROUP/'step4/data/source_inputs/pose_3/needs.json';domain=json.loads(dp.read_text())
    com=np.array(domain['frame']['moment_origin_m']);ref=.01*domain['geometry']['total_area_m2']
    fp=ROOT/'objects/B/history/before_compatible_poses_860a4233e74b/tasks/pose_3/setup.npz';floor=np.load(fp)['floor_contact_m']
    rays=np.array([[64,0,1],[-64,0,1],[0,64,1],[0,-64,1.]])
    # Use the larger cavity (19x9 mm), not peg (18x8 mm), for a relaxed location
    # set and maximum edge lengths. Entire end-stop opening is also available.
    px=[];ns=[];labels=[]
    for axis in (0,1):
        for sign in (-1,1):
            for cross,zlocal in product([-.0045,.0045] if axis==0 else [-.0095,.0095],[-.0262,-.005]):
                point=[0.,0.,zlocal];point[axis]=sign*(.0095 if axis==0 else .0045);point[1-axis]=cross
                normal=[0.,0.,0.];normal[axis]=-sign
                px.append(point);ns.append(normal);labels.append(2*axis+(sign==1))
    for x,y in product([-.0095,.0095],[-.0045,.0045]):px.append([x,y,-.0262]);ns.append([0,0,1]);labels.append(4)
    points=np.array(px)@B.T+port;normal=np.array(ns)@B.T;labels=np.array(labels)
    head=np.c_[normal,np.cross(points-com,normal)]
    cols=np.vstack([head,np.c_[rays,np.cross(floor-com,rays)]])
    n=len(cols);nh=len(head);no_uplift=np.r_[-normal[:,2],np.zeros(5)]
    loads=stats['baseline']['records'];targets=[]
    for r in loads:
        f=np.array(r['tool_force_mg']);q=np.array(r['tool_point_world_m']);targets.append(np.r_[-(f+[0,0,-1]),-np.cross(q-com,f)])
    # If none of four one-sided-X/one-sided-Y combinations is feasible, some
    # opposite-wall forces are necessary in this relaxed fixed-pose model.
    must_opposed=[]
    for load,need in zip(loads,targets):
        possible=False
        for ix,iy in product((0,1),(2,3)):
            keep=np.r_[np.flatnonzero((labels==ix)|(labels==iy)|(labels==4)),np.arange(nh,n)]
            fit=linprog(np.zeros(len(keep)),A_eq=cols[keep].T,b_eq=need,A_ub=no_uplift[:-1][keep][None,:],b_ub=[0.],bounds=(0,None),method='highs')
            if fit.success:
                assert max(abs(cols[keep].T@fit.x-need))<1e-7
                possible=True;break
            if fit.status!=2:raise RuntimeError(fit.message)
        if not possible:must_opposed.append(load['sample_id'])
    bands=[]
    for width in (.1,.25,.5,1.):
        b=width/1000
        # Each side's total actual contact area is assumed <= edge length * b.
        # End stop gets its entire larger opening, a deliberately optimistic cap.
        areas=np.array([.009*b,.009*b,.019*b,.019*b,.019*.009])
        ub=np.zeros((6,n+1));ub[0]=no_uplift
        for k in range(5):ub[k+1,:nh]=(labels==k);ub[k+1,-1]=-areas[k]/ref
        results=[]
        for load,need in zip(loads,targets):
            fit=linprog(np.r_[np.zeros(n),1],A_eq=np.c_[cols.T,np.zeros(6)],b_eq=need,A_ub=ub,b_ub=np.zeros(6),bounds=(0,None),method='highs')
            if not fit.success:raise RuntimeError(fit.message)
            assert max(abs(cols.T@fit.x[:n]-need))<1e-7
            face_forces=np.array([sum(fit.x[:nh][labels==k]) for k in range(5)])
            assert np.max(face_forces/(areas/ref))<=fit.x[-1]+1e-6
            results.append(dict(sample_id=load['sample_id'],minimum_peak_pressure_lower_bound_ratio=float(fit.x[-1]),face_normal_forces_mg=face_forces.tolist()))
        values=np.array([r['minimum_peak_pressure_lower_bound_ratio'] for r in results]);avg=float(values.mean());worst=float(values.max())
        bands.append(dict(assumed_effective_edge_band_width_mm=width,side_area_upper_bounds_mm2=(areas[:4]*1e6).tolist(),average_pressure_lower_bound_ratio=avg,worst_pressure_lower_bound_ratio=worst,
            average_relative_to_baseline_nominal_peak=avg/pressure['baseline']['average_minimum_peak_ratio'],worst_relative_to_baseline_nominal_peak=worst/pressure['baseline']['worst_minimum_peak_ratio'],records=results))
    # 2D strip model for free angular play: w sec(theta)+L tan(theta)=W.
    angles=[]
    for w,W in ((.018,.019),(.008,.009)):
        theta=np.arccos(w/np.hypot(W,.021))-np.arctan2(.021,W)
        angles.append(dict(peg_width_mm=w*1000,opening_width_mm=W*1000,free_tilt_deg=float(np.degrees(theta)),model='2D tilted strip through 21 mm sleeve; end-stop/torsion coupling not included'))
    out=dict(count=100,claim='Conditional contact-pressure concentration mechanism; no assertion that actual edge-band width equals an example value.',
        reference_pressure='mg / (1% object surface area); normalized values are dimensionless pressure ratios.',
        geometry=dict(peg_mm=[18,8],opening_mm=[19,9],engagement_mm=21,clearance_per_side_mm=.5),
        proof='Peak pressure >= face normal-force sum / actual face contact area. If area <= edge_length*b, pressure >= force/(edge_length*b). LP minimizes this necessary bound over a relaxed contact-location set, so the physical solution cannot have lower peak pressure under the assumed area bounds and fixed model.',
        relaxed_model='Frictionless original unlocked sleeve (no retainer), contact resultants anywhere on full cavity faces via corner generators; full end-stop opening allowed; same object/floor/load points, mu=64 ground and no-uplift. Contact-location compatibility and elasticity not modeled.',
        opposite_contact=dict(fixed_pose_cases_requiring_opposite_wall_forces=len(must_opposed),sample_ids=must_opposed,scope='Infeasibility of all four one-sided-axis combinations in the relaxed fixed-pose force model; not a full finite-displacement contact certificate.'),
        free_play_strip_angles=angles,band_scenarios=bands,
        baseline_nominal_peak=dict(average_ratio=pressure['baseline']['average_minimum_peak_ratio'],worst_ratio=pressure['baseline']['worst_minimum_peak_ratio']),
        limitations='Width b must come from a specified compliance/corner-radius/preload model or measurement; examples do not establish actual pressure. Sharp perfectly rigid edge/line contact has zero area and singular pressure, not a prediction of material failure. Fixed-pose relaxation does not certify actual tilted assembly geometry.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [sp,pp,zp,dp,fp,Path(__file__)]})
    (HERE/'pressure_data/clearance_pressure_bound.json').write_text(json.dumps(out,indent=2)+'\n')
    print('opposite-wall necessity cases',len(must_opposed),flush=True)
    for row in bands:print(row['assumed_effective_edge_band_width_mm'],'mm',row['average_relative_to_baseline_nominal_peak'],row['worst_relative_to_baseline_nominal_peak'],flush=True)
    return out


def render(report):
    fig,(diagram,chart)=plt.subplots(1,2,figsize=(13,6),gridspec_kw=dict(width_ratios=[1,1.25]))
    diagram.add_patch(Rectangle((-10.5,-12),1,24,fc='#d99a45'));diagram.add_patch(Rectangle((9.5,-12),1,24,fc='#d99a45'))
    angle=np.deg2rad(2.7);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    corners=np.array([[-9,-10.5],[9,-10.5],[9,10.5],[-9,10.5]])@R.T
    diagram.add_patch(Polygon(corners,fc='#78a9c0',alpha=.85));diagram.scatter([corners[1,0],corners[3,0]],[corners[1,1],corners[3,1]],s=85,c='#bc3745',zorder=10)
    diagram.annotate('Edge contact',xy=corners[1],xytext=(13,-5),arrowprops=dict(arrowstyle='->',color='#bc3745'),color='#bc3745',fontsize=12)
    diagram.annotate('Edge contact',xy=corners[3],xytext=(-20,6),arrowprops=dict(arrowstyle='->',color='#bc3745'),color='#bc3745',fontsize=12)
    diagram.set(xlim=(-24,25),ylim=(-15,16));diagram.set_aspect('equal');diagram.set_axis_off();diagram.set_title('Clearance + opposing forces\ncan localize load at edges',fontsize=16)
    diagram.text(0,-14,'Schematic, not a solved contact configuration',ha='center',fontsize=9,color='#777777')
    rows=report['band_scenarios'];x=np.arange(len(rows));av=[r['average_relative_to_baseline_nominal_peak'] for r in rows];wo=[r['worst_relative_to_baseline_nominal_peak'] for r in rows]
    chart.bar(x-.16,av,width=.3,color='#78a9c0',label='Average over 100');chart.bar(x+.16,wo,width=.3,color='#d99a45',label='Worst of 100')
    chart.axhline(1,color='#777777',ls='--');chart.set_xticks(x,[f"{r['assumed_effective_edge_band_width_mm']:g} mm" for r in rows]);chart.set_xlabel('ASSUMED effective edge-band width');chart.set_ylabel('Dock pressure lower bound / support nominal peak');chart.legend(frameon=False,fontsize=10)
    chart.spines[['top','right']].set_visible(False)
    for i,(a,w) in enumerate(zip(av,wo)):
        chart.text(i-.16,a,f'{a:.1f} x',ha='center',va='bottom',fontsize=11);chart.text(i+.16,w,f'{w:.1f} x',ha='center',va='bottom',fontsize=11)
    chart.set_ylim(0,max(av+wo)*1.25);chart.set_title('Pressure concentration: conditional bounds',fontsize=16)
    fig.suptitle('Existing 0.5 mm-clearance sleeve: contact width matters',fontsize=19,y=.97)
    fig.text(.05,.085,'Same 100 loads and object-ground model. Contact-area assumptions are parameters; no arbitrary measured-width claim.',fontsize=10)
    fig.text(.05,.043,'An actual contact/compliance model is needed to identify the real width. These conditional bounds do not prove material failure.',fontsize=10,color='#666666')
    fig.tight_layout(rect=[0,.14,1,.93]);path=HERE.parent/'clearance_pressure_bound.png';fig.savefig(path,dpi=170);plt.close(fig);print(path,flush=True)


if __name__=='__main__':render(calculate())
