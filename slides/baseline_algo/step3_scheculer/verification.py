"""Continuous load checks and independent-support counterexamples.

The supply is reconstructed from actual points and normals. Rational solves
check selected reaction bases in the original force and moment units.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step3_scheculer.stage_imports import load_stage
C = load_stage('score', 'contribution')
from step3_scheculer import contacts as I
from step3_scheculer import floor_support as FLOOR

from step1.cases import pose_name


def rational_solve(matrix, target):
    n=len(target)
    rows=[list(row)+[value] for row,value in zip(matrix,target)]
    for col in range(n):
        pivot=max(range(col,n),key=lambda row:abs(rows[row][col]))
        rows[col],rows[pivot]=rows[pivot],rows[col]
        if not rows[col][col]:
            raise ValueError('Singular rational basis')
        divisor=rows[col][col]
        rows[col]=[value/divisor for value in rows[col]]
        for row in range(n):
            if row!=col:
                factor=rows[row][col]
                rows[row]=[a-factor*b for a,b in zip(rows[row],rows[col])]
    return [rows[row][-1] for row in range(n)]


def exact_column(point, normal, com):
    n=[F(float(x)) for x in normal]
    r=[F(float(x))-F(float(c)) for x,c in zip(point,com)]
    return n+[r[1]*n[2]-r[2]*n[1],r[2]*n[0]-r[0]*n[2],r[0]*n[1]-r[1]*n[0]]


class Supply:
    def __init__(self, problem, contacts):
        self.problem=problem
        normals=list(FLOOR.rays())
        points=[problem.floor]*len(normals)
        owners=[-1]*len(normals)
        for j,contact in enumerate(contacts):
            for triangle,face in zip(contact['triangles_m'],contact['source_faces']):
                for point in triangle:
                    points.append(point)
                    normals.append(-problem.domain.mesh.face_normals[face])
                    owners.append(j)
        points=np.array(points);normals=np.array(normals)
        r=points-problem.domain.com
        # Component expansion is independent of contribution.py's np.cross.
        moments=np.c_[r[:,1]*normals[:,2]-r[:,2]*normals[:,1],
                      r[:,2]*normals[:,0]-r[:,0]*normals[:,2],
                      r[:,0]*normals[:,1]-r[:,1]*normals[:,0]]
        # Seventh row counts only head forces; the original object-floor
        # reaction is excluded. The last ray represents N_floor >= 0.
        raw=np.c_[normals,moments,np.where(np.array(owners)>=0,normals[:,1],0.)]
        slack_index=len(FLOOR.rays())
        raw=np.insert(raw,slack_index,np.r_[np.zeros(6),-1.],axis=0)
        points=np.insert(points,slack_index,problem.floor,axis=0)
        normals=np.insert(normals,slack_index,np.zeros(3),axis=0)
        owners.insert(slack_index,-2)
        scale=np.r_[problem.scale,1.]
        ids=np.sort(np.unique(np.round(raw*scale,13),axis=0,return_index=True)[1])
        self.points,self.normals,self.owners=points[ids],normals[ids],np.array(owners)[ids]
        self.raw=raw[ids]
        self.full=self.raw*scale
        np.testing.assert_array_equal(self.full,problem.supply(contacts))
        self.exact_cache={}

    def exact(self,index):
        index=int(index)
        if index not in self.exact_cache:
            owner=self.owners[index]
            self.exact_cache[index]=(exact_column(self.points[index],self.normals[index],self.problem.domain.com)
                +[F(-1) if owner==-2 else F(float(self.normals[index,1])) if owner>=0 else F(0)])
        return self.exact_cache[index]

    def witness(self,target):
        target=C.U.target(target)
        witness=C.W.solve(self.full,target*np.r_[self.problem.scale,1.])
        if witness is None:
            return None
        ids=witness['indices']
        # A boundary load (notably N=0) may use fewer than seven rays.
        # Complete the independent basis with zero-weight rays where possible.
        ids=list(ids)
        if np.linalg.matrix_rank(self.full[ids].T)!=len(ids):
            return None
        for j in qr(self.full.T,mode='economic',pivoting=True)[2]:
            if len(ids)==7: break
            if j not in ids and np.linalg.matrix_rank(self.full[ids+[int(j)]].T)>len(ids):
                ids.append(int(j))
        if len(ids)!=7: return None
        matrix=[[self.exact(j)[i] for j in ids] for i in range(7)]
        exact_target=[F(float(x)) for x in target]
        coefficients=rational_solve(matrix,exact_target)
        if min(coefficients)<0:
            return None
        assert all(sum(matrix[i][j]*coefficients[j] for j in range(7))==exact_target[i] for i in range(7))
        return dict(indices=ids,coefficients_exact=[str(v) for v in coefficients],
                    coefficients_mg=[float(v) for v in coefficients],target=np.asarray(target).tolist(),
                    exact_equilibrium=True,nonnegative=True,passive_support_no_uplift=True)


def domain_extrema(problem, physical_duals):
    """Exact optimization formula: closed triangles, full spherical cap, 0 <= t <= K.

    h.need = h.b0 + t * (-(h_F + h_tau cross (pt-c))).d.
    For each h the cap support is convex in pt, so its maximum on a triangle
    occurs at a vertex. The cap support includes its interior optimum.
    """
    domain=problem.domain
    points=domain.mesh.triangles[domain.work_ids].reshape(-1,3)
    normals=np.repeat(domain.normals,3,axis=0)
    normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    c,s=np.cos(domain.half_angle),np.sin(domain.half_angle)
    records=[]
    for h in physical_duals:
        a=-(h[:3]+np.cross(h[3:6],points-domain.com))
        dot=np.sum(a*normals,axis=1)
        norm=np.linalg.norm(a,axis=1)
        perpendicular=a-dot[:,None]*normals
        cap=np.where(dot>=c*norm,c*0+norm,c*dot+s*np.linalg.norm(perpendicular,axis=1))
        gain=domain.k*np.maximum(0,cap)
        index=int(gain.argmax())
        records.append(dict(maximum=float(h[:3]@(-domain.gravity)+gain[index]),point_index=index,
                            magnitude_mg=domain.k if cap[index]>0 else 0.))
    return records


def outer_box(problem):
    extrema=domain_extrema(problem,np.r_[np.eye(6),-np.eye(6)])
    # An outward numerical margin, in the same conditioned units as scoring.
    pad=1e-9/problem.scale
    upper=np.array([r['maximum'] for r in extrema[:6]])+pad
    lower=-np.array([r['maximum'] for r in extrema[6:]])-pad
    vertices=np.array(list(product(*zip(lower,upper))))
    return lower,upper,vertices


def strict_separator(supply, target):
    """Find a strict separator and check its signs with exact rational geometry."""
    target=C.U.target(target)
    scale=np.r_[supply.problem.scale,1.]
    matrix=np.vstack([np.c_[supply.full,np.ones(len(supply.full))],
                       np.r_[-target*scale,1.][None]])
    result=linprog(np.r_[np.zeros(7),-1.],A_ub=matrix,b_ub=np.zeros(len(matrix)),
                   bounds=[(-1,1)]*7+[(0,None)],method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success or result.x[-1]<=1e-10:
        return None
    h=result.x[:7]*scale
    hf=[F(float(v)) for v in h]
    supply_scores=[sum(a*b for a,b in zip(supply.exact(i),hf)) for i in range(len(supply.full))]
    target_score=sum(F(float(a))*b for a,b in zip(target,hf))
    if max(supply_scores)>0 or target_score<=0:
        return None
    return dict(physical_dual=h.tolist(),max_supply_dot=float(max(supply_scores)),
                target_dot=float(target_score),strict_margin=result.x[-1],
                every_original_generator_checked_with_exact_rational_arithmetic=True)


def reachable_counterexample(problem,supply,H,extrema):
    domain=problem.domain
    order=np.argsort([-r['maximum'] for r in extrema])
    for facet in order[:128]:
        record=extrema[facet]
        if record['maximum']<=1e-7:
            break
        face,vertex=divmod(record['point_index'],3)
        point=domain.mesh.triangles[domain.work_ids[face],vertex]
        h=H[facet,:6]*problem.scale
        n=domain.normals[face]
        a=-(h[:3]+np.cross(h[3:],point-domain.com))
        dot=a@n
        if np.linalg.norm(a)<1e-15:
            direction=n
        elif dot>=np.cos(domain.half_angle)*np.linalg.norm(a):
            direction=a/np.linalg.norm(a)
        else:
            tangent=a-dot*n
            direction=np.cos(domain.half_angle)*n+np.sin(domain.half_angle)*tangent/np.linalg.norm(tangent)
        theta=min(domain.half_angle,float(np.arccos(np.clip(direction@n,-1,1))))
        phi=float(np.arctan2(direction@domain.e2[face],direction@domain.e1[face]))
        corner=np.array([[0.,0.],[1.,0.],[0.,1.]][vertex])
        for epsilon,angle_inset,magnitude_inset in [(.03,np.deg2rad(.1),.001),(.003,np.deg2rad(.01),.0001),(0.,0.,0.)]:
            uv=(1-epsilon)*corner+epsilon/3
            angle=max(0.,theta-angle_inset)
            magnitude=max(0.,record['magnitude_mg']-magnitude_inset)
            case=domain.evaluate(face,*uv,angle,phi,magnitude_mg=magnitude)
            if not bool(case['reachable']) or float(case['need_wrench']@h)<=1e-7:
                continue
            if C.W.solve(supply.full,case['need_wrench']*problem.scale) is not None:
                continue
            separator=strict_separator(supply,case['need_wrench'])
            return dict(work_face_index=face,mesh_face_id=int(domain.work_ids[face]),
                u=float(uv[0]),v=float(uv[1]),theta_rad=angle,theta_deg=float(np.rad2deg(angle)),
                phi_rad=phi,magnitude_mg=magnitude,pt_m=case['q_m'].tolist(),
                force_push_mg=case['force_push_mg'].tolist(),need_wrench=case['need_wrench'].tolist(),
                reachable=True,lp_feasible=False,facet_violation=float(case['need_wrench']@h),
                physical_dual=h.tolist(),strict_exact_separator=separator)
    return None


def continuous_check(problem,contacts,certificate_path=None):
    supply=Supply(problem,contacts)
    if not contacts:
        return dict(status='unresolved',reason='no_selected_heads',
                    rest_equilibrium=C.gravity_check(supply.full,problem.domain,problem.scale),
                    scope='No contact design has been selected; no continuous coverage claim.')
    axis=[supply.witness(target/problem.scale) for target in np.r_[np.eye(6),-np.eye(6)]]
    if all(w is not None for w in axis):
        return dict(status='verified',method='exact_positive_span_of_all_six_wrench_coordinates',
                    scope='All wrenches, hence every work face, every 30-degree direction and every magnitude in [0,0.5mg].',
                    witnesses=axis,ray_count=len(supply.full))
    lower,upper,vertices=outer_box(problem)
    witnesses=[supply.witness(target) for target in vertices]
    if all(w is not None for w in witnesses):
        return dict(status='verified',method='continuous_domain_outer_box_with_exact_nonnegative_reactions',
                    scope='All work-face points, all local cap directions including occluded ones, and the entire closed force-magnitude interval.',
                    lower_wrench=lower.tolist(),upper_wrench=upper.tolist(),
                    numerical_padding_conditioned=1e-9,witnesses=witnesses,ray_count=len(supply.full))
    try:
        H=C.W.cone(supply.full)
        extrema=domain_extrema(problem,H*np.r_[problem.scale,1.])
        case=reachable_counterexample(problem,supply,H,extrema)
        if case is not None:
            return dict(status='counterexample',method='analytic_cap_and_triangle_extrema_then_admissible_interior_load',
                        counterexample=case,facets_checked=len(H),ray_count=len(supply.full))
        if max([r['maximum'] for r in extrema],default=0.) < -1e-7 and certificate_path is not None:
            from step3_scheculer import enclosure as E
            enclosure_attempts=[]
            for sides,bands in [(8,1),(16,2),(32,4)]:
                proof,arrays=E.contain(problem,supply.full,sides,bands)
                enclosure_attempts.append(dict(proof))
                if arrays is not None:
                    certificate_path=Path(certificate_path)
                    certificate_path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(certificate_path,**arrays)
                    proof.update(certificate_file=str(certificate_path.relative_to(C.ROOT)),
                                 certificate_sha256=C.sha256(certificate_path),
                                 enclosure_code_sha256=C.sha256(E.__file__),
                                 enclosure_attempts=enclosure_attempts)
                    return proof
        return dict(status='unresolved',method='outer_box_failed_and_no_verified_reachable_counterexample',
                    largest_analytic_facet_violation=max([r['maximum'] for r in extrema],default=None),
                    enclosure_attempts=locals().get('enclosure_attempts',[]),
                    facet_test_alone_is_not_an_exact_primal_certificate=True)
    except (C.W.QhullError,AssertionError,RuntimeError) as error:
        return dict(status='unresolved',reason=type(error).__name__)


def horizontal_obstruction(problem,contact):
    normals=problem.domain.mesh.face_normals[np.unique(contact['source_faces'])]
    xy=COORD.floor(normals)
    keep=np.linalg.norm(xy,axis=1)>1e-12
    xy=xy[keep]
    # Positive coefficients on ALL projected normals plus rank two prove
    # that n_xy.v >= 0 for all contact normals implies v=0.
    matrix=np.vstack([np.c_[xy.T,xy.sum(axis=0)],np.r_[np.ones(len(xy)),len(xy)]])
    result=linprog(np.r_[np.zeros(len(xy)),-1.],A_eq=matrix,b_eq=[0,0,1],bounds=(0,None),method='highs')
    if not result.success or result.x[-1]<=1e-9 or np.linalg.matrix_rank(xy)<2:
        return dict(horizontal_translation_obstructed=False,scope='Necessary local test only; this does not prove insertability.')
    coefficients=[F(float(v)) for v in result.x[:-1]+result.x[-1]]
    A=np.vstack([xy.T,np.ones(len(xy))])
    basis=qr(A,mode='economic',pivoting=True)[2][:3].tolist()
    exact=[[F(float(v)) for v in row] for row in A]
    rhs=[F(v)-sum(exact[i][j]*coefficients[j] for j in range(len(xy)) if j not in basis)
         for i,v in enumerate([0,0,1])]
    solution=rational_solve([[exact[i][j] for j in basis] for i in range(3)],rhs)
    for j,value in zip(basis,solution):
        coefficients[j]=value
    assert min(coefficients)>0
    assert [sum(a*b for a,b in zip(row,coefficients)) for row in exact]==[0,0,1]
    angles=np.sort(np.arctan2(xy[:,1],xy[:,0]))
    maxgap=float(np.diff(np.r_[angles,angles[0]+2*np.pi]).max())
    return dict(horizontal_translation_obstructed=True,projected_normals=xy.tolist(),
                positive_weights_exact=[str(v) for v in coefficients],rank=2,
                largest_projected_normal_gap_degrees=float(np.rad2deg(maxgap)),
                proof='Positive spanning projected normals force every nonpenetrating horizontal translation to be zero.',
                scope='Rules out a horizontal straight insertion/extraction at fixed orientation; arbitrary paths not tested.')


def independent_support_check(problem,contacts):
    rows=[]
    remaining=[]
    for contact in contacts:
        inward=-problem.domain.mesh.face_normals[np.unique(contact['source_faces'])]
        minimum,maximum=float(inward[:,1].min()),float(inward[:,1].max())
        unusable=maximum < -1e-12
        if not unusable:
            remaining.append(contact)
        rows.append(dict(id=contact['candidate_id'],inward_y_min=minimum,inward_y_max=maximum,
                         all_reactions_must_be_zero_for_an_independent_massless_unanchored_support=unusable,
                         horizontal_insertion=horizontal_obstruction(problem,contact)))
    relaxed=Supply(problem,remaining)
    case=None
    for sample in range(min(64,len(problem.targets))):
        target=problem.targets[sample]/problem.scale
        separator=strict_separator(relaxed,target)
        if separator is not None:
            case=dict(sample_index=sample,pt_m=problem.samples['pt_m'][sample],
                      force_push_mg=problem.samples['force_push_mg'][sample],
                      need_wrench=target.tolist(),separator=separator)
            break
    return dict(status='counterexample' if case else 'unresolved',contacts=rows,counterexample=case,
        assumptions='Each patch is a separate massless support touching only the workpiece and a unilateral floor, without anchors or connections to other supports.',
        necessary_vertical_equation='R_j = sum_i(lambda_ji * inward_normal_ji,z) >= 0',
        consequence='If every inward normal has negative y, every nonnegative reaction coefficient on that support must be zero.',
        current_pipeline_did_not_impose_this_constraint=True)


def run(name):
    problem=C.Problem(name)
    root=C.OUTPUTS/name/pose_name()/'step3_scheculer'
    folder=root/'verification'
    folder.mkdir(parents=True,exist_ok=True)
    results={}
    for variant,path in [('previous_two_contacts',folder/'previous_two_contacts.npz'),
                         ('sampling_only_contacts',folder/'sampling_only_contacts.npz'),
                         ('scheduled_contacts',root/'final_contacts.npz')]:
        if not path.exists():
            continue
        contacts=I.read_contacts(path)
        result=dict(object=name,variant=variant,geometry_sha256=C.sha256(path),
            selected_ids=[c['candidate_id'] for c in contacts],
            continuous_contact_model=continuous_check(problem,contacts,folder/f'{variant}_primal_certificate.npz'),
            independent_floor_supports=independent_support_check(problem,contacts),
            areas_m2=[float(c['triangle_areas_m2'].sum()) for c in contacts],
            radii_m=[c['radius_m'] for c in contacts],
            provenance=dict(inputs=I.hashes(problem.inputs+[path]),code=I.hashes([Path(__file__)])))
        I.save(folder/f'{variant}_verification.json',result)
        results[variant]=dict(continuous_contact_model=result['continuous_contact_model']['status'],
                             independent_floor_supports=result['independent_floor_supports']['status'])
        print(name,variant,results[variant],flush=True)
    I.save(folder/'summary.json',dict(object=name,results=results))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or C.OBJECTS:
        run(name)
