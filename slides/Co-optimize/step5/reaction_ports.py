"""Retain finite local pads around only necessary reaction locations."""
from pathlib import Path
import json,hashlib
import numpy as np
from feasible_geometry import C as S
from periphery import PlaneForceFilter,C as W,classify


def select_ports(checker,stock,face_records,cache=None):
    signature=dict(mesh=hashlib.sha256(stock.vertices.tobytes()+stock.faces.tobytes()).hexdigest(),
        faces=[r['retained_faces']for r in face_records],
        targets=[hashlib.sha256(t.targets.tobytes()).hexdigest()for t in checker.tasks],
        placements=hashlib.sha256(checker.layout.placements.tobytes()).hexdigest(),
        grounded=checker.grounded)
    if cache is not None:
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
        if(cache/'report.json').exists():
            saved=json.loads((cache/'report.json').read_text())
            if saved['signature']==signature and saved['complete']:return saved['ports'],saved['records']
    contacts=PlaneForceFilter(checker,stock).contacts
    ports=[];records=[]
    for k in checker.layout.active:
        triangles,sources=contacts[k]
        faces=next(r['retained_faces']for r in face_records if r['pose_index']==int(k))
        keep=np.isin(sources,faces);triangles=triangles[keep];sources=sources[keep]
        material=triangles.reshape(-1,3);Q=checker.layout.placements[k]
        points=W.transform_points(material,checker.native[k]@np.linalg.inv(Q))
        native_normals=np.repeat(-checker.tasks[k].domain.mesh.face_normals[sources],3,axis=0)
        raw=np.c_[native_normals,np.cross(points-checker.tasks[k].domain.com,native_normals)]
        _,groups=np.unique(np.round(material,12),axis=0,return_inverse=True)
        active=np.ones(len(material),bool);basis=[];attempts=[]
        floor=checker.floors[k]if checker.grounded[k]else checker.floors[k][-1:]
        identifiers=np.unique(groups)
        order=sorted(identifiers,key=lambda i:-np.linalg.norm(material[groups==i].mean(0)-stock.center_mass))
        def eliminate(group):
            nonlocal active
            proposed=active&~np.isin(groups,group)
            full=W.I.merge_columns(floor,W.U.heads(raw[proposed],checker.tasks[k].scale))
            columns=[int.from_bytes(hashlib.blake2b(r.tobytes(),digest_size=8).digest(),'little')for r in full]
            error=None
            try:
                mask,_=classify(full,checker.tasks[k].targets,basis_cache=basis,column_ids=columns,stop_on_counterexample=True)
                passed=bool(mask.all())
            except RuntimeError as failure:
                if 'Original equilibrium unresolved'not in str(failure):raise
                passed=False;error=str(failure);basis.clear()
            attempts.append(dict(locations=[int(i)for i in group],passed=passed,rejected_uncertainty=error))
            if passed:active=proposed
            elif len(group)>1:
                mid=len(group)//2;eliminate(group[:mid]);eliminate(group[mid:])
        eliminate(order)
        for group in np.unique(groups[active]):
            selected=np.flatnonzero(active&(groups==group));index=int(selected[0])
            face_ids=np.unique(np.repeat(sources,3)[selected])
            normal=np.sum(checker.mesh.face_normals[face_ids]@Q[:3,:3].T,axis=0)
            if np.linalg.norm(normal)>1e-12:normal/=np.linalg.norm(normal)
            ports.append(dict(center_m=material[index].tolist(),extent_m=0.,normal_m=normal.tolist(),
                pose_index=int(k),source_faces=face_ids.tolist()))
        record=dict(pose_index=int(k),input_locations=len(identifiers),
            retained_locations=len(np.unique(groups[active])),attempts=attempts,
            all_original_demands_checked_on_accepted_removals=True)
        records.append(record)
        print('REACTION_PORTS',k,len(identifiers),'->',record['retained_locations'],flush=True)
    if cache is not None:S.save(cache/'report.json',dict(complete=True,signature=signature,ports=ports,records=records,
        finite_pads_must_be_constructed_and_physically_certified=True))
    return ports,records
