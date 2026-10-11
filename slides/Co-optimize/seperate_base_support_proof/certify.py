"""Independent conservative geometric certificates, without layout changes.

The work proof uses INSCRIBED 32-sided cones, with the exact original whole
triangles and zero backward clearance. An inward 0.1 micrometre plane margin
makes these a strict subset of the original circular work cones. Their free
space is therefore a generous upper bound on physically usable contacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PROOF = Path(__file__).resolve().parent
sys.path.insert(0, str(PROOF))
import diagnose as D
np = D.np
C = D.C
from scipy.spatial import ConvexHull
from work_access import tangent_frame


def inner_access(body, task):
    angle = float(task.domain.data['load']['cone_half_deg'])
    ids = np.asarray(task.domain.work_ids, int)
    phases = (np.arange(32)+.5)*2*np.pi/32
    families = []
    for face in ids:
        triangle, normal = body.triangles[face], body.face_normals[face]
        tangent = tangent_frame(normal)
        ring = np.tan(np.radians(angle)) * (
            np.cos(phases)[:,None]*tangent[:,0]+np.sin(phases)[:,None]*tangent[:,1])
        top = (triangle[:,None,:]+(normal+ring)[None,:,:]).reshape(-1,3)
        hull = ConvexHull(np.vstack([triangle,top]))
        planes = hull.equations[hull.equations[:,:3]@normal<1e-7].copy()
        _, unique = np.unique(np.round(planes,10),axis=0,return_index=True)
        planes = planes[np.sort(unique)]
        planes[:,3] += 1e-7
        families.append(planes)
    counts = np.array([len(p) for p in families],int)
    planes = np.zeros((len(families),max(counts),4))
    for i,p in enumerate(families):
        planes[i,:len(p)] = p
    return SimpleNamespace(face_ids=ids,counts=counts,planes=planes)


def certify(case):
    inputs = C.Inputs('B', D.BASE/case['id']/'data/selection.json')
    inputs.base = D.BASE
    data = inputs.load(case)
    tasks, body = D.load_tasks(data,inputs)
    accesses = [inner_access(body,task) for task in tasks]
    out = PROOF/'certificates'/case['id']
    out.mkdir(parents=True,exist_ok=False)
    ground = D.growth.floor_certificates(data['rows'])
    work = []
    for index,row in enumerate(data['rows']):
        planes = D.planes_in_floor(data,accesses,index)
        answer = dict(pose=row['pose'], unbounded_proof_found=False)
        for radius in [.25,1.,10.]:
            certificate = D.radial_certificate(planes,radius)
            if not certificate['passed']:
                continue
            bounds = np.array([[-radius]*3,[radius]*3])
            free, forbidden = D.free_contacts(planes,bounds)
            contacts = np.vstack([D.geometry_vertices(free),row['object_ground']])
            cover = D.containment(row['hull'],contacts)
            if not cover['passed']:
                hull = ConvexHull(contacts)
                errors = row['hull'][:,:2]@hull.equations[:,:2].T+hull.equations[:,2]
                k,facet = np.unravel_index(np.argmax(errors),errors.shape)
                equation = hull.equations[facet]
                cover['separator'] = dict(normal_xy=equation[:2].tolist(),
                    offset_m=float(equation[2]), demand_hull_vertex_index=int(k),
                    demand_xy_m=row['hull'][k,:2].tolist(),
                    separation_gap_m=float(errors[k,facet]),
                    maximum_legal_contact_value_m=float(np.max(contacts@equation[:2]+equation[2])))
            answer.update(unbounded_proof_found=True, **cover, radial_certificate=certificate,
                bounding_box_independent=True, own_object_ground_included=True,
                other_floors_bodies_exits_ignored=True,
                connected_material_not_required=True, material_rim_not_required=True,
                cone_approximation='strict inscribed subset of original circular cones',
                positive_inward_plane_margin_m=1e-7)
            # Numerically verify the analytic certificate on widely separated
            # radii. These checks supplement the angular subcover proof.
            angles = np.linspace(0,2*np.pi,2049)[:-1]
            directions = np.c_[np.cos(angles),np.sin(angles)]
            for distance in [radius,radius*100,radius*1e6]:
                points = directions*distance
                inside = np.zeros(len(points),bool)
                for cone in planes:
                    inside |= np.all(points@cone['A'].T+cone['b']<=1e-10,axis=1)
                if not inside.all():
                    raise RuntimeError('Radial angular certificate failed direct halfspace checks')
            answer['independent_radial_point_checks'] = 3*len(directions)
            (out/f'{row["pose"]}_free.wkt').write_text(free.wkt)
            (out/f'{row["pose"]}_work.wkt').write_text(forbidden.wkt)
            np.savez_compressed(out/f'{row["pose"]}_cones.npz',
                **{f'A_{k}':cone['A'] for k,cone in enumerate(planes)},
                **{f'b_{k}':cone['b'] for k,cone in enumerate(planes)})
            break
        work.append(answer)
    inputs.verify()
    record = dict(id=case['id'],pose_count=case['pose_count'],fresh_step4_passed=True,
        poses=data['record']['poses'],step4_experiment=case['experiment'],
        ground_certificates=ground,ground_rejected=bool(ground),
        work_per_pose=work,
        unbounded_work_rejected=any(r['unbounded_proof_found'] and not r['passed'] for r in work),
        layouts_and_native_loads_unchanged=True,
        scope='fixed saved relative layout and installed fixture states; arbitrary base size and topology',
        step4_force_acceptance_not_revoked=True)
    C.save(out/'report.json',record)
    print('CERT',case['id'],'ground',record['ground_rejected'],
          'unbounded_work',record['unbounded_work_rejected'],flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets',nargs='*')
    args = parser.parse_args()
    batch = PROOF/'certificate_runs'/str(time.time_ns())
    batch.mkdir(parents=True,exist_ok=False)
    code = C.freeze(batch)
    progress = json.loads((PROOF/'progress.json').read_text())
    cases = [r['case'] for r in progress['step5_results'] if 4<=r['case']['pose_count']<=6]
    if args.sets:
        cases = [c for c in cases if c['id'] in args.sets]
    results = []
    for case in cases:
        if (PROOF/'certificates'/case['id']/'report.json').exists():
            continue
        results.append(certify(case))
    for p,h in code.items():
        if C.digest(C.ROOT/p)!=h or C.digest(batch/'executed_sources'/p)!=h:
            raise RuntimeError(f'Executed source changed: {p}')
    C.save(batch/'execution.json',dict(complete=True,cases=[r['id'] for r in results],
                                      code=code,source_snapshots_match=True))


if __name__=='__main__':
    main()
