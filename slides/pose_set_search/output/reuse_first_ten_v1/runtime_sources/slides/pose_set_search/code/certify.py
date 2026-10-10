"""Save and replay a constructive nonnegative witness for EVERY saved load.

The original independent LP proposes small bases. The verifier substitutes
extended-precision nonnegative pressures into the seven original equations.
No force capacities, sampled success rate, or changed floor model.
"""
import argparse
import json
import hashlib
from common import *
from classify import replay_coefficients


def certificate(full,targets):
    targets=C.U.target(targets,full.shape[1])
    assignments=np.full(len(targets),-1,np.int32)
    weights=np.zeros((len(targets),7),np.longdouble)
    bases=[];lps=0
    while np.any(assignments<0):
        rows=np.flatnonzero(assignments<0);index=int(rows[0])
        witness=C.C.W.solve(full,targets[index]);lps+=1
        if witness is None:
            raise RuntimeError(f'Original independent LP did not certify demand {index}')
        indices=np.asarray(witness['indices'],int)
        if len(indices)>7:
            raise RuntimeError('LP witness was not a seven-column basic solution')
        basis=full[indices]
        coeff=replay_coefficients(basis,targets[rows])
        residual=np.max(np.abs(coeff @ np.asarray(basis,np.longdouble)-targets[rows]),axis=1)
        rounding=32*np.finfo(np.longdouble).eps*np.max(coeff @ np.abs(np.asarray(basis,np.longdouble)),axis=1)
        passed=residual+rounding<=2e-9
        if not passed.any():
            raise RuntimeError('No extended-precision basic pressure witness')
        padded=np.full(7,-1,int);padded[:len(indices)]=indices
        bases.append(padded)
        accepted=rows[passed]
        assignments[accepted]=len(bases)-1
        weights[accepted,:len(indices)]=coeff[passed]
    return np.asarray(bases),assignments,weights,lps


def replay(full,targets,bases,assignments,weights):
    targets=C.U.target(targets,full.shape[1])
    if np.any(assignments<0) or np.any(weights<0) or not np.isfinite(weights).all():
        raise RuntimeError('Invalid pressure coefficients or uncovered loads')
    maximum=0.
    for i,columns in enumerate(bases):
        rows=np.flatnonzero(assignments==i);indices=columns[columns>=0]
        coeff=weights[rows,:len(indices)]
        basis=np.asarray(full[indices],np.longdouble)
        error=np.max(np.abs(coeff @ basis-np.asarray(targets[rows],np.longdouble)),axis=1)
        rounding=32*np.finfo(np.longdouble).eps*np.max(coeff @ np.abs(basis),axis=1)
        maximum=max(maximum,float(np.max(error+rounding)))
    if maximum>2e-9:
        raise RuntimeError(f'Pressure replay exceeds original tolerance: {maximum}')
    return dict(passed=True,load_count=len(targets),minimum_coefficient=float(weights.min()),
                maximum_coefficient=float(weights.max()),maximum_residual_including_roundoff=maximum,
                lift_constraint_checked_as_seventh_original_equation=True,basis_count=len(bases))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.result/'report.json').read_text())
    layout=np.load(args.result/'layout.npz')
    actual=C.trimesh.load(args.result/'support.obj',process=False)
    states=[C.state(report['object'],p) for p in report['poses']]
    checks=[]
    for k,(pose,state) in enumerate(zip(report['poses'],states)):
        task,native,mesh=state
        saved=np.load(args.result/f'{pose}_force.npz')
        q=layout['placements'][k];direction=q[:3,:3].T @ layout['directions'][k]
        allowed=np.setdiff1d(np.arange(len(mesh.faces)),task.domain.work_ids)
        allowed=allowed[mesh.face_normals[allowed] @ direction<=1e-9]
        tri,src=C.contact_boundary(transform_mesh(mesh,q),actual,allowed)
        regenerated=C.supply(task,native @ np.linalg.inv(q),tri,src)
        # Contact tessellations can change at OBJ roundtrip; certify the actual
        # exported solid's regenerated cone, rather than demand row identity.
        bases,assignments,weights,lps=certificate(regenerated,task.targets)
        path=args.result/f'{pose}_pressure_certificate.npz'
        np.savez_compressed(path,supply_7d=regenerated,bases=bases,assignments=assignments,weights=weights)
        with np.load(path) as data:
            check=replay(data['supply_7d'],task.targets,data['bases'],data['assignments'],data['weights'])
        check.update(pose=pose,original_independent_lps=lps,
                     certificate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     task_inputs_sha256=C.I.hashes(task.inputs))
        checks.append(check)
        C.save(args.result/'pressure_audit.json',dict(complete=False,checks=checks))
        print('CERTIFIED',pose,'loads',check['load_count'],'max residual',check['maximum_residual_including_roundoff'],flush=True)
    C.save(args.result/'pressure_audit.json',dict(complete=True,passed=all(c['passed'] for c in checks),
        checks=checks,actual_exported_mesh_contacts_regenerated=True,
        support_sha256=hashlib.sha256((args.result/'support.obj').read_bytes()).hexdigest(),
        verifier_sources_sha256=C.I.hashes([Path(__file__),Path(__file__).with_name('classify.py')]),
        original_force_model=C.U.description(),original_floor_model=C.FLOOR.description()))


if __name__=='__main__':
    main()
