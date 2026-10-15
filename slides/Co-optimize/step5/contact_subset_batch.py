"""Certified contact-face elimination in exterior-first batches, with cache."""
from pathlib import Path
import hashlib,json
import numpy as np
from feasible_geometry import C as S
from periphery import PlaneForceFilter,C as W,classify


def minimal_face_patches(certificate,mesh,cache=None):
    signature=dict(mesh=hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()).hexdigest(),
        placements=hashlib.sha256(certificate.layout.placements.tobytes()).hexdigest(),
        directions=hashlib.sha256(certificate.layout.directions.tobytes()).hexdigest(),
        targets=[hashlib.sha256(t.targets.tobytes()).hexdigest()for t in certificate.tasks])
    if cache is not None:
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
        report=cache/'report.json'
        if report.exists():
            old=json.loads(report.read_text())
            if old['signature']==signature and old['complete']:
                with np.load(cache/'guide.npz')as a:points=a['points'].copy()
                return points,old['records']
    contacts=PlaneForceFilter(certificate,mesh).contacts
    retained=[];records=[]
    for k in certificate.layout.active:
        triangles,sources=contacts[k];Q=certificate.layout.placements[k]
        points=W.transform_points(triangles.reshape(-1,3),certificate.native[k]@np.linalg.inv(Q))
        normals=np.repeat(-certificate.tasks[k].domain.mesh.face_normals[sources],3,axis=0)
        raw=np.c_[normals,np.cross(points-certificate.tasks[k].domain.com,normals)]
        floor=certificate.floors[k]if certificate.grounded[k]else certificate.floors[k][-1:]
        active=np.ones(len(sources),bool);basis=[];attempts=[]
        faces=np.unique(sources)
        order=sorted(faces,key=lambda f:-np.linalg.norm(triangles[sources==f].mean(axis=(0,1))-mesh.center_mass))
        def eliminate(group):
            nonlocal active
            if not len(group):return
            proposed=active&~np.isin(sources,group)
            full=W.I.merge_columns(floor,W.U.heads(raw[np.repeat(proposed,3)],certificate.tasks[k].scale))
            columns=[int.from_bytes(hashlib.blake2b(row.tobytes(),digest_size=8).digest(),'little')for row in full]
            unresolved=None
            try:
                mask,_=classify(full,certificate.tasks[k].targets,basis_cache=basis,column_ids=columns,
                    stop_on_counterexample=True)
                passed=bool(mask.all())
            except RuntimeError as error:
                if 'Original equilibrium unresolved'not in str(error):raise
                # An unresolved proposed deletion is rejected. No mask or
                # tolerance is changed; smaller groups may still be tested.
                passed=False;unresolved=str(error);basis.clear()
            attempts.append(dict(faces=[int(f)for f in group],passed=passed,
                rejected_numerical_uncertainty=unresolved))
            if passed:active=proposed
            elif len(group)>1:
                half=len(group)//2;eliminate(group[:half]);eliminate(group[half:])
        eliminate(order)
        selected=triangles[active];retained.append(selected)
        patches=[]
        for face in np.unique(sources[active]):
            vertices=triangles[active&(sources==face)].reshape(-1,3)
            center=(vertices.min(0)+vertices.max(0))/2
            patches.append(dict(face=int(face),center_m=center.tolist(),
                extent_m=float(np.linalg.norm(vertices-center,axis=1).max())))
        records.append(dict(pose_index=int(k),input_faces=len(faces),
            retained_faces=np.unique(sources[active]).tolist(),
            removed_faces=np.setdiff1d(faces,sources[active]).tolist(),patches=patches,
            attempts=attempts,all_original_demands_checked_on_accepted_removals=True,
            original_demand_count=len(certificate.tasks[k].targets),actual_geometry_changed=False))
        print('CONTACT_SUBSET_BATCH',k,len(faces),'->',len(patches),'queries',len(attempts),flush=True)
        if cache is not None:
            S.save(cache/'partial.json',dict(complete=False,signature=signature,records=records))
    points=np.concatenate(retained).reshape(-1,3)
    if cache is not None:
        np.savez_compressed(cache/'guide.npz',points=points)
        S.save(cache/'report.json',dict(complete=True,signature=signature,records=records,
            force_demand_resampling=False,proposal_only=True))
    return points,records


if __name__=='__main__':
    import argparse
    from feasible_geometry import Layout,ForceCertificate
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('sets',nargs='+')
    args=parser.parse_args()
    inputs=S.Inputs('B',S.COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json',
        'pso_all52_20261013_01_floor')
    for name in args.sets:
        case=next(c for c in inputs.cases if c['id']==name);data=inputs.load(case)
        with np.load(data['selected']/'layout.npz')as a:layout=Layout.from_arrays(a)
        checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],'B')
        source=S.COOPT/'data/results/B'/name/'step5/step5.3/step5_show_all52_20261014_02/support_with_base.obj'
        mesh=S.trimesh.load(source,force='mesh',process=False)
        minimal_face_patches(checker,mesh,S.COOPT/'data/results/B/data/contact_subset_cache_20261015'/name)
