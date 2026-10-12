"""Reconstruct the saved two-head illustration; never read live search outputs."""
from pathlib import Path
import hashlib
import numpy as np
import trimesh
import fixture_geometry as F
from geometry_utils import tube
from source_geometry import oriented_box

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DIRECTION=np.array([.8137976813493735,-.46984631039295466,.3420201433256687])
TERMINALS=np.array([[.12057598926791789,-.04672218596357173,.05568160645908548],
                    [.07709707564498997,-.050241156727484096,.15430054549627092]])


def reconstruct_geometry():
    from step3_scheculer.contacts import read_contacts
    from step2_local_support.insertion import Analyzer,engine
    from step5_connect_support.routing import CORNERS,beam
    object_path=ROOT/'codes/simulation/shape/B/pose_2/object_geometry.npz'
    contact_path=HERE/'original_contacts.npz'
    with np.load(object_path) as z:
        obj=trimesh.Trimesh(z['vertices_m'],z['faces'],process=False)
    contacts=read_contacts(contact_path)
    d=DIRECTION;scale=float(obj.extents.max());radius=.025*scale
    analyzer=Analyzer(obj,.01*scale)
    parts=[];cell_count=0;terminal_errors=[]
    for contact,terminal in zip(contacts,TERMINALS):
        heads=analyzer.heads(contact);parts+=heads;cell_count+=len(heads)
        roots=[]
        for head in heads:
            start=head.vertices.mean(0)
            root=start+.85*(head.vertices-start)
            shifted=root+d*max(.007629700825297719,.13912093820285604-start@d)
            parts.append(engine.hull_mesh(np.vstack([root,shifted])))
            roots.append(shifted)
        shifted=np.vstack(roots)
        terminal_errors.append(float(np.linalg.norm(shifted.mean(0)-terminal)))
        parts.append(engine.hull_mesh(np.vstack([shifted,terminal+radius*CORNERS])))
    parts.append(beam(*TERMINALS,radius,radius))
    x=np.cross([0.,0.,1.],d);x/=np.linalg.norm(x);y=np.cross(d,x)
    basis=np.column_stack([x,y,d]);grip=TERMINALS.mean(0)
    q=grip+x*.035+d*.015
    parts+=tube(np.array([grip,q+d*.012,q]),.004)
    peg=oriented_box(q-d*.013,[.018,.008,.026],basis)
    blue=F.joined(parts+[peg])
    channel=[oriented_box(q-d*.0275,[.025,.015,.003],basis)]
    for sign in (-1,1):
        channel += [oriented_box(q+x*sign*.011-d*.0156,[.003,.015,.0212],basis),
                    oriented_box(q+y*sign*.006-d*.0156,[.025,.003,.0212],basis)]
    socket=F.joined(channel)
    paths=[object_path,contact_path,Path(__file__),HERE/'geometry_report.json']
    meta=dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        source_kind='saved_original_two_head_frame_and_closed_ground_ring',
        selected_contact_ids=[c['candidate_id'] for c in contacts],head_cell_count=cell_count,
        recovered_terminal_error_m=terminal_errors,
        blue_grip_local=grip.tolist(),blue_grip_rod_axis=(TERMINALS[1]-TERMINALS[0]).tolist(),
        multi_pose_bearing_verified=False,connector_and_base_are_illustrative=True)
    return obj,blue,socket,peg,q,basis,np.array([c['center_m'] for c in contacts]),np.unique(np.concatenate([c['source_faces'] for c in contacts])),meta


def geometry():
    """Use the pinned illustration assets even when the baseline is rerun."""
    import json
    path=HERE/'original_geometry.npz'
    with np.load(path) as z:
        meshes=[trimesh.Trimesh(z[name+'_vertices'],z[name+'_faces'],process=False)
                for name in ('object','blue','socket','peg')]
        meta=json.loads(str(z['metadata']))
        meta['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (path,Path(__file__),HERE/'geometry_report.json')}
        return (*meshes,z['port'].copy(),z['basis'].copy(),z['centers'].copy(),z['contact_faces'].copy(),meta)
