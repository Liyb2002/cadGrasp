"""Reproduce A1-f's right-edge load and pure-gravity counterexamples.

This diagnostic reads the current two-contact output without changing the design.
"""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linprog

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step3_scheculer.stage_imports import load_stage
from step3_scheculer import contacts as I
from step3_scheculer import verification as V
from step5_connect_support import connect as C, solids as S, visual_details as L
from step4_floor_contact import draw as FD
from step2_local_support import render as D

P=load_stage('score','contribution')
SAMPLE_INDEX=13350

from step1.cases import pose_name


def joint_check(problem,supply,floor,edge_need,out):
    """Give all floor points unlimited horizontal friction; share pad forces.

    This is a relaxation of finite Coulomb friction. All four outer U corners
    are actual floor material. Affine normal-force moments need no interior
    ground generators. The ordinary floor can only push up.
    """
    contact=supply.full[supply.owners>=0].T
    rays=np.array([[0,1,0],[1,0,0],[-1,0,0],[0,0,1],[0,0,-1.]])
    original=np.c_[rays,np.cross(problem.floor-problem.domain.com,rays)]*problem.scale
    base=np.concatenate([np.c_[rays,np.cross(pt-problem.domain.com,rays)]*problem.scale
                         for pt in floor['floor_contact_corners_m']])
    matrix=np.block([[contact,original.T,np.zeros((6,len(base)))],
                     [-contact,np.zeros((6,len(original))),base.T]])
    n=matrix.shape[1];pull=-matrix[:,np.arange(n-20,n,5)]
    extended=np.c_[matrix,pull];objective=np.r_[np.zeros(n),np.ones(4)]
    cases={};arrays=dict(equilibrium_matrix=matrix,pull_columns=pull)
    for label,need in [('gravity',np.r_[-problem.domain.gravity,[0.,0.,0.]]),('edge',edge_need)]:
        target=np.r_[need*problem.scale,np.zeros(6)]
        object_matrix=matrix[:6,:contact.shape[1]+len(original)]
        object_only=linprog(np.ones(object_matrix.shape[1]),A_eq=object_matrix,b_eq=target[:6],
                            bounds=(0,None),method='highs')
        assert object_only.success
        object_residual=float(np.max(np.abs(object_matrix@object_only.x-target[:6])))
        assert object_residual<2e-8
        cop,_,_=FD.F.floor_demands(need[None],problem.domain.com)
        ground_hull=np.vstack([COORD.floor(floor['floor_contact_corners_m']),COORD.floor(problem.floor)])
        inside,_=FD.F.hull_coverage(cop,ground_hull,1e-9*problem.domain.mesh.extents.max())
        assert bool(inside[0])
        statuses={}
        for method in ['highs-ds','highs-ipm']:
            result=linprog(np.zeros(n),A_eq=matrix,b_eq=target,bounds=(0,None),method=method,
                           options=dict(presolve=False))
            statuses[method]=int(result.status)
            assert result.status==2
        # Counterfactual control: only base floor normals may now pull downward.
        result=linprog(objective,A_eq=extended,b_eq=target,bounds=(0,None),method='highs')
        assert result.success and result.fun>1e-6 and result.x.min()>=0
        residual=float(np.max(np.abs(extended@result.x-target)))
        assert residual<2e-8
        cases[label]=dict(joint_equilibrium_feasible=False,numerical_solver_status=statuses,
            object_only_with_floor_friction_feasible=True,object_only_residual_conditioned=object_residual,
            whole_assembly_cop_inside_floor_hull=True,whole_assembly_cop_m=cop[0].tolist(),
            minimum_total_downward_base_floor_force_mg_numeric=float(result.fun),
            counterfactual_equilibrium_residual_conditioned=residual,
            scope='Numerical joint infeasibility and minimum-pull control; not an exact rational optimum certificate.')
        arrays[label+'_target']=target;arrays[label+'_counterfactual_coefficients']=result.x
        arrays[label+'_object_only_coefficients']=object_only.x
    np.savez_compressed(out/'joint_balance_check.npz',**arrays)
    return dict(assumptions='Same pad forces act oppositely on workpiece and massless support; original floor point and all U corners allow unrestricted horizontal friction but nonnegative normal force.',
        equations=['G_contact*lambda + G_original_floor*r0 = need',
                   '-G_contact*lambda + G_base_floor*rb = 0'],
        contact_ray_count=contact.shape[1],cases=cases,
        conclusion='The current base also fails simultaneous workpiece/support equilibrium under a more permissive floor-friction model.')


def plot(problem,contacts,parts,report,out):
    domain=problem.domain;a=np.asarray(report['insertion_direction'])
    basis=D.axes(.8*a+.75*np.cross([0,1,0],a)+[0,.7,0])
    pt=np.asarray(report['edge_load']['pt_m']);force=np.asarray(report['edge_load']['force_push_mg'])
    tip=pt+force/np.linalg.norm(force)*.24*domain.mesh.extents.max()
    ground=D.floor_triangles(domain)
    patches=np.concatenate([p['triangles_m'] for p in contacts])
    triangles=np.concatenate([ground,domain.mesh.triangles,*[p.triangles for p in parts],patches])
    colors=np.tile(D.GREY,(len(domain.mesh.faces),1));colors[domain.work_ids]=D.GREEN
    colors=np.concatenate([np.tile(D.FLOOR,(len(ground),1)),colors,
        *[np.tile([47,139,143] if i<3 else [64,119,165],(len(p.faces),1)) for i,p in enumerate(parts)],
        np.tile(D.ORANGE,(len(patches),1))])
    focus,width=FD.fit(np.vstack([triangles.reshape(-1,3),tip]),basis,1.13)
    picture,_=D.raster(triangles,colors,focus,basis,width,1050,
        overlay=np.arange(len(triangles)-len(patches),len(triangles)),unlit=[0,1])
    ink=ImageDraw.Draw(picture)
    def xy(point):return D.project(point,focus,basis,width,1050)[:2]
    p,q=xy(pt),xy(tip)
    L.arrow(ink,p,q,color='#c83f34',width=7)
    ink.ellipse((*tuple(p-8),*tuple(p+8)),fill='#c83f34',outline='white',width=2)
    ink.text(tuple(p+[-100,-40]),'pt / sample 13350',font=D.font(24),fill='#a52c26')
    ink.text(tuple(q+[10,-4]),'F_push',font=D.font(25),fill='#a52c26')
    for point,label,color in [(problem.floor,'original floor point','#26302e'),(domain.com,'COM','#75589e')]:
        p=xy(point);ink.ellipse((*tuple(p-6),*tuple(p+6)),fill=color,outline='white',width=2)
        ink.text(tuple(p+[12,-10]),label,font=D.font(21),fill=color)
    page=Image.new('RGB',(2200,1350),D.PAPER);ink=ImageDraw.Draw(page)
    ink.text((45,25),'A1-f / Right-edge load / Static equilibrium FAILS',font=D.font(43),fill='#a52c26')
    ink.text((48,88),'Current C005 + C081 connection: insertable geometry, incomplete force coverage.',font=D.font(27),fill=D.INK)
    page.paste(picture,(15,185))
    x=1135
    lines=[(175,'An allowed load at the marked point',31,D.INK),
        (230,'|F_push| = 0.489885 mg',29,D.INK),
        (273,'Angle from inward normal = 25.898° < 30°',26,D.INK),
        (316,'Tool ray reaches the work face.',26,D.INK),
        (395,'Force balance along insertion direction a',30,D.INK),
        (454,'External push:     F_push · a = +0.083745 mg',26,'#a52c26'),
        (500,'Required reaction:       R · a = -0.083745 mg',26,'#a52c26'),
        (546,'Available reactions:    R · a >= 0',29,'#a52c26'),
        (605,'No nonnegative contact-force solution exists.',27,'#a52c26'),
        (710,'Pure gravity also has no equilibrium',31,D.INK),
        (765,'Both pads push strictly toward +a.',26,D.INK),
        (809,'Zero horizontal load forces both pad reactions to 0.',24,D.INK),
        (853,'The original floor point cannot balance gravity torque.',24,D.INK),
        (934,'Step3: 18.91% sampled coverage; stopped at 2 rounds.',25,'#a52c26'),
        (981,'This connection is not a stable support solution.',28,'#a52c26'),
        (1070,'Joint balance also fails with unrestricted floor friction.',24,D.INK),
        (1115,'Edge load would need the floor to pull the base DOWN',24,'#a52c26'),
        (1158,f"by {report['joint_floor_comparison']['cases']['edge']['minimum_total_downward_base_floor_force_mg_numeric']:.4f} mg. The floor cannot pull.",25,'#a52c26')]
    for y,content,size,color in lines:ink.text((x,y),content,font=D.font(size),fill=color)
    ink.text((48,1275),'Current model: frictionless object–support contacts; original object–floor reaction is vertical only. mg = object weight.',font=D.font(24),fill='#65706c')
    page.save(out/'load_check.png')


def run():
    problem=P.Problem('A1-f');root=P.OUTPUTS/'A1-f'/pose_name();out=root/C.STAGE
    domain,contacts,schedule,floor,_,_,inputs=C.read_inputs('A1-f')
    connection=I.check_report(out/'connection.json')
    assert schedule['selected_ids']==['C005','C081']
    assert connection['complete_support_insertion_verified']
    sample=SAMPLE_INDEX;face=int(problem.samples['work_face_index'][sample])
    u,v,theta,phi,magnitude=problem.samples['parameters'][sample]
    case=domain.evaluate(face,u,v,theta,phi,magnitude_mg=magnitude)
    assert bool(case['reachable']) and theta<domain.half_angle and magnitude<domain.k
    np.testing.assert_allclose(case['force_push_mg'],problem.samples['force_push_mg'][sample],atol=1e-14)
    a=floor['insertion_direction'];a_exact=[Fraction(float(x)) for x in a]
    projections=[]
    for contact in contacts:
        normals=-domain.mesh.face_normals[np.unique(contact['source_faces'])]
        exact=[sum(Fraction(float(n))*v for n,v in zip(normal,a_exact)) for normal in normals]
        assert min(exact)>0
        projections.append(dict(candidate_id=contact['candidate_id'],minimum=float(min(exact)),
            maximum=float(max(exact)),minimum_exact=str(min(exact)),all_source_faces_checked=True))
    force_projection=sum(Fraction(float(f))*v for f,v in zip(case['force_push_mg'],a_exact))
    assert force_projection>0
    supply=V.Supply(problem,contacts)
    gravity=np.r_[-domain.gravity,[0.,0.,0.]]
    assert P.W.solve(supply.full,case['need_wrench']*problem.scale) is None
    assert P.W.solve(supply.full,gravity*problem.scale) is None
    edge_proof=V.strict_separator(supply,case['need_wrench'])
    gravity_proof=V.strict_separator(supply,gravity)
    assert edge_proof is not None and gravity_proof is not None
    parts=S.unpack_parts(I.load_npz(out/'connection_geometry.npz'))
    report=dict(object='A1-f',complete=True,status='edge_load_and_pure_gravity_infeasible',
        selected_ids=schedule['selected_ids'],insertion_direction=a.tolist(),
        scope='Static equilibrium under the current frictionless contact model; no claim about the eventual dynamic fall trajectory.',
        assumptions=dict(object_support_reaction='nonnegative inward normals, unlimited magnitude',
            original_object_floor_reaction='vertical normal only',all_supports_connected=True),
        edge_load=dict(sample_index=sample,work_face_index=face,mesh_face_id=int(domain.work_ids[face]),
            parameters=[u,v,theta,phi,magnitude],pt_m=case['q_m'].tolist(),
            force_push_mg=case['force_push_mg'].tolist(),magnitude_mg=magnitude,
            theta_deg=float(np.rad2deg(theta)),reachable=True,need_wrench=case['need_wrench'].tolist(),
            force_dot_insertion_mg=float(force_projection),force_dot_insertion_exact=str(force_projection),
            necessary_reaction_dot_insertion_mg=-float(force_projection),lp_feasible=False,exact_separator=edge_proof),
        contact_inward_normal_dot_insertion=projections,
        pure_gravity=dict(lp_feasible=False,exact_separator=gravity_proof,
            gravity_torque_about_original_floor_point_mgm=np.cross(domain.com-problem.floor,domain.gravity).tolist(),
            explanation='Strictly positive pad projections require zero pad forces under pure gravity; the remaining floor reaction cannot balance the nonzero gravity moment about that point.'),
        scheduler=dict(status=schedule['status'],covered_percent=schedule['covered_percent'],
            geometry_connection_passed=True,static_stability_certified=False),
        provenance=dict(inputs=I.hashes(inputs+[out/'connection.json',out/'connection_geometry.npz']),
            code={**P.code_hashes(),**I.hashes([Path(__file__),Path(V.__file__),Path(D.__file__),Path(FD.__file__),Path(L.__file__)])}))
    report['joint_floor_comparison']=joint_check(problem,supply,floor,case['need_wrench'],out)
    plot(problem,contacts,parts,report,out)
    report['artifacts']={filename:P.sha256(out/filename) for filename in ['load_check.png','joint_balance_check.npz']}
    I.save(out/'load_check.json',report)
    print('A1-f: allowed right-edge sample 13350 and pure gravity both fail; exact separating certificates saved.',flush=True)


if __name__=='__main__':run()
