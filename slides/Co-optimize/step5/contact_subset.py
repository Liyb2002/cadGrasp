"""Select removable contact patches using every original wrench demand.

This is a retention guide, not a physical result: an actual reconstructed or
pruned solid must still pass the complete geometry and demand certificate.
"""
import hashlib
import numpy as np
from feasible_geometry import C as S
from periphery import PlaneForceFilter,C as W,classify


def minimal_face_patches(certificate,mesh):
    contact=PlaneForceFilter(certificate,mesh).contacts
    retained=[];records=[]
    for k in certificate.layout.active:
        triangles,sources=contact[k]
        Q=certificate.layout.placements[k]
        points=W.transform_points(triangles.reshape(-1,3),certificate.native[k]@np.linalg.inv(Q))
        normals=np.repeat(-certificate.tasks[k].domain.mesh.face_normals[sources],3,axis=0)
        raw=np.c_[normals,np.cross(points-certificate.tasks[k].domain.com,normals)]
        floor=certificate.floors[k]if certificate.grounded[k]else certificate.floors[k][-1:]
        ids=np.repeat(sources,3)
        active=np.ones(len(sources),bool);basis=[]
        faces=np.unique(sources)
        center=mesh.center_mass
        order=sorted(faces,key=lambda face:-np.linalg.norm(triangles[sources==face].mean(axis=(0,1))-center))
        attempts=[]
        for face in order:
            proposed=active&(sources!=face)
            selected=np.repeat(proposed,3)
            full=W.I.merge_columns(floor,W.U.heads(raw[selected],certificate.tasks[k].scale))
            columns=[int.from_bytes(hashlib.blake2b(row.tobytes(),digest_size=8).digest(),'little')for row in full]
            mask,_=classify(full,certificate.tasks[k].targets,basis_cache=basis,column_ids=columns,
                            stop_on_counterexample=True)
            passed=bool(mask.all())
            if passed:active=proposed
            attempts.append(dict(face=int(face),removable_in_retention_guide=passed))
        retained.append(triangles[active])
        records.append(dict(pose_index=int(k),input_faces=len(faces),
            retained_faces=np.unique(sources[active]).tolist(),
            removed_faces=np.setdiff1d(faces,sources[active]).tolist(),
            attempts=attempts,original_demand_count=len(certificate.tasks[k].targets),
            all_original_demands_checked_on_accepted_removals=True,
            actual_geometry_changed=False))
        print('CONTACT_SUBSET',k,len(faces),'->',len(np.unique(sources[active])),flush=True)
    points=np.concatenate(retained).reshape(-1,3)
    return points,records


if __name__=='__main__':
    from pathlib import Path
    import argparse,json,sys
    from feasible_geometry import C,Layout,ForceCertificate
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('sets',nargs='+')
    args=parser.parse_args()
    selection=C.COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json'
    inputs=C.Inputs('B',selection,'pso_all52_20261013_01_floor')
    for name in args.sets:
        case=next(c for c in inputs.cases if c['id']==name);data=inputs.load(case)
        with np.load(data['selected']/'layout.npz')as a:layout=Layout.from_arrays(a)
        checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],'B')
        source=C.COOPT/'data/results/B'/name/'step5/step5.3/step5_show_all52_20261014_02/support_with_base.obj'
        mesh=C.trimesh.load(source,force='mesh',process=False)
        points,records=minimal_face_patches(checker,mesh)
        target=C.COOPT/'data/results/B/data/contact_subset_probe_20261015'/name;target.mkdir(parents=True,exist_ok=True)
        C.save(target/'report.json',dict(complete=True,actual_geometry_changed=False,records=records,
            source=C.relative(source),source_sha256=C.digest(source)))
        np.savez_compressed(target/'guide.npz',points=points)
