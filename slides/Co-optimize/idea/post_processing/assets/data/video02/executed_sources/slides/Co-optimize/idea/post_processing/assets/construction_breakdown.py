"""Replay saved contact decisions and expose the recorded 32-link construction.

Contact elimination changes reaction candidates, not the 155.46cm3 stock.
Construction is visualized from saved anchors, paths and the first certified
constructed OBJ. No load solve or numerical algorithm is rerun.
"""
from pathlib import Path
import hashlib,json
import render01 as R
import prepare02 as P
import numpy as np
import trimesh
from scipy.spatial import cKDTree

DATA=P.DATA/'breakdown'
CACHE=R.COOPT/'data/results/B/data/contact_subset_cache_20261015'/R.CASE
OUT=P.RUN/'outside_in'


def prepare():
    import co_common as C
    DATA.mkdir(parents=True,exist_ok=True)
    paths=[CACHE/'report.json',CACHE/'reaction_ports/report.json',OUT/'contact_subset.json',
        OUT/'connection_proposals.json',OUT/'report.json',R.ROOT/'objects/B/mesh.stl',
        P.RUN/'baseline/support.obj',OUT/'checkpoints/0001/support.obj',R.STEP4/'layout.npz']
    protected={R.relative(p):R.digest(p)for p in paths}
    faces=json.loads(paths[0].read_text());locations=json.loads(paths[1].read_text())
    subset=json.loads(paths[2].read_text());network=json.loads(paths[3].read_text())
    result=json.loads(paths[4].read_text())
    construction=next(h['pruning']for h in result['history']if h['operation']=='outside_in_contact_pruning')
    body=trimesh.load(paths[5],force='mesh',process=False)
    stock=trimesh.load(paths[6],force='mesh',process=False)
    if hashlib.sha256(stock.vertices.tobytes()+stock.faces.tobytes()).hexdigest()!=faces['signature']['mesh']:
        raise ValueError('Saved contact trace belongs to another baseline')
    with np.load(paths[8])as saved:
        if not np.array_equal(saved['placements'],np.tile(np.eye(4),(5,1,1))):
            raise ValueError('This geometry replay expects the actual registered five-pose layout')
    face_ids=np.unique(np.concatenate([np.r_[r['retained_faces'],r['removed_faces']]for r in faces['records']]))
    triangles,sources=C.contact_boundary(body,stock,face_ids)
    arrays=dict(contact_triangles_m=triangles,contact_sources=sources)
    records=[]
    for face,location in zip(faces['records'],locations['records']):
        k=face['pose_index'];ids=np.union1d(face['retained_faces'],face['removed_faces'])
        active=np.ones(len(ids),bool);masks=[active.copy()]
        for attempt in face['attempts']:
            if attempt['passed']:active=active&~np.isin(ids,attempt['faces'])
            masks.append(active.copy())
        if not np.array_equal(ids[active],face['retained_faces']):raise ValueError('Face trace mismatch')
        material=triangles[np.isin(sources,face['retained_faces'])].reshape(-1,3)
        _,groups=np.unique(np.round(material,12),axis=0,return_inverse=True)
        unique,first=np.unique(groups,return_index=True);points=material[first]
        if len(points)!=location['input_locations']:raise ValueError('Point groups differ from original trace')
        point_active=np.ones(len(points),bool);point_masks=[point_active.copy()]
        for attempt in location['attempts']:
            if attempt['passed']:point_active=point_active&~np.isin(unique,attempt['locations'])
            point_masks.append(point_active.copy())
        actual=np.array([p['center_m']for p in subset['ports']if p['pose_index']==k])
        if len(actual)!=point_active.sum()or np.max(cKDTree(points[point_active]).query(actual)[0])>1e-11:
            raise ValueError('Selected reaction locations do not match saved original ports')
        arrays.update({f'face_ids_{k}':ids,f'face_masks_{k}':np.array(masks),
            f'points_{k}':points,f'point_masks_{k}':np.array(point_masks)})
        records.append(dict(pose_index=k,input_faces=len(ids),retained_faces=len(face['retained_faces']),
            face_attempts=face['attempts'],input_locations=len(points),retained_locations=int(point_active.sum()),
            point_attempts=location['attempts']))
    port_data=construction['port_data'];anchors=[]
    for port in port_data:
        point=np.asarray(port['anchor_m'])
        if not any(np.linalg.norm(point-a)<1e-12 for a in anchors):anchors.append(point)
    anchors=np.asarray(anchors)
    paths_m=[np.asarray(p)for p in network['paths']]
    if len(paths_m)!=32:raise ValueError('Wrong saved connected construction')
    arrays['anchors_m']=anchors
    arrays['object_ports_m']=np.array([p['center_m']for p in port_data if p['kind']=='object'])
    arrays['ground_ports_m']=np.array([p['center_m']for p in port_data if p['kind']=='ground'])
    seeds=[];stamps=[]
    for i,path in enumerate(paths_m):
        arrays[f'path_{i}']=path
        lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);total=lengths.sum();covered=0.
        for a,b,length in zip(path,path[1:],lengths):
            ts=np.linspace(0.,1.,max(2,int(np.ceil(length/.001))))
            seeds.extend(a+(b-a)*ts[:,None]);stamps.extend(i+(covered+length*ts)/total)
            covered+=length
    # End caps are assigned to the first real link reaching their actual
    # interior anchor. Only existing CP1 triangles are revealed on screen.
    for port in port_data:
        anchor=np.asarray(port['anchor_m'])
        incident=[i for i,p in enumerate(paths_m)if min(np.linalg.norm(p[0]-anchor),np.linalg.norm(p[-1]-anchor))<1e-9]
        # Some original caps already connect through their finite overlap and
        # therefore have no separate routed link in this historical proposal.
        # Assign only their display reveal time to the nearest real path.
        stamp=min(incident)+.5 if incident else float(stamps[cKDTree(np.asarray(seeds)).query(anchor)[1]])
        seeds.append(np.asarray(port['center_m']));stamps.append(stamp)
    mesh=trimesh.load(paths[7],force='mesh',process=False)
    _,nearest=cKDTree(np.asarray(seeds)).query(mesh.triangles_center)
    arrays['construction_face_progress']=np.asarray(stamps)[nearest]/len(paths_m)
    np.savez_compressed(DATA/'arrays.npz',**arrays)
    record=dict(complete=True,records=records,protected_inputs=protected,
        geometry_contact_extraction_only=True,force_solver_rerun=False,
        face_selection_changes_solid=False,point_selection_changes_solid=False,
        contact_stage_material_cm3=abs(float(stock.volume))*1e6,
        selected_object_ports=len(arrays['object_ports_m']),selected_ground_ports=len(arrays['ground_ports_m']),
        unique_internal_anchors=len(anchors),recorded_connections=len(paths_m),
        construction_preview='progressive display of actual CP1 triangles along the saved 32-link tree',
        partial_construction_not_a_certified_complete_support=True,
        all_contact_decisions_replayed_from_original_saved_trials=True,
        actual_cp1_source=R.relative(paths[7]),actual_cp1_sha256=R.digest(paths[7]),
        reconstructed_geometry_from_original_proposals=True)
    R.save(DATA/'preparation.json',record)
    print(json.dumps(dict(passed=True,contact_triangles=len(triangles),poses=records.__len__(),
        faces=[r['retained_faces']for r in records],points=[r['retained_locations']for r in records],
        internal_nodes=len(anchors),connections=len(paths_m))),flush=True)


class Data:
    def __init__(self):
        self.record=json.loads((DATA/'preparation.json').read_text())
        with np.load(DATA/'arrays.npz')as saved:self.arrays={k:saved[k].copy()for k in saved.files}
        self.protected=dict(self.record['protected_inputs'])
        for path in [DATA/'arrays.npz',DATA/'preparation.json']:self.protected[R.relative(path)]=R.digest(path)


if __name__=='__main__':prepare()
