"""Read the saved B/pose2 module and add the shared rectangular interface."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'slides/baseline_algo'))
from step5_connect_support import solids as S, belt_geometry as BG, ground as G
from geometry_utils import tube
SOURCE=ROOT/'slides/baseline_algo/output/B/pose_2'
def oriented_box(center,size,basis):
    m=trimesh.creation.box(extents=size)
    m.vertices=m.vertices@basis.T+center
    return m


def modular_geometry(domain,obj):
    """Read the new two-module output without adding a second peg/link."""
    from step5_base import rectangular_dock as J
    source=SOURCE/'step5_base'
    report=json.loads((source/'base.json').read_text())
    if not report['geometry_constructed']:
        raise RuntimeError('B/pose_2 has no current modular geometry to illustrate')
    with np.load(source/'blue.npz') as a:
        contact=trimesh.Trimesh(a['union_vertices_m'],a['union_faces'],process=False)
    q=np.asarray(report['interface']['port_m']);peg,channel,_,_=J.joint(q)
    socket=BG.union_parts(channel,float(obj.extents.max()))[0]
    interface=dict(origin_m=(q-J.UP*(J.LENGTH-J.ENGAGEMENT/2)).tolist(),
        withdrawal_direction=J.UP.tolist(),type=report['interface']['type'],
        slider_length_mm=1000*J.LENGTH,engaged_length_mm=1000*J.ENGAGEMENT,
        peg_cross_section_mm=[1000*J.WIDTH,1000*J.HEIGHT],
        socket_opening_mm=[1000*J.WIDTH,1000*J.HEIGHT],
        sleeve_outer_cross_section_mm=[1000*(J.WIDTH+2*J.WALL),1000*(J.HEIGHT+2*J.WALL)],
        wall_thickness_mm=1000*J.WALL,nominal_clearance_mm=0.,
        docking_stroke_mm=1000*J.STROKE,anti_withdrawal_lock=False,
        limitation='Ideal snug interface from the baseline. Repositioned multi-pose figures do not inherit its single-pose bearing certificate.')
    files=[SOURCE/'step_1_needs/needs.json',source/'base.json',source/'blue.npz',Path(J.__file__)]
    return dict(object=obj,contact=contact),dict(interface=interface,channel=socket,peg=peg,port=q,
        basis=np.eye(3),domain=domain,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def geometry():
    domain=json.loads((SOURCE/'step_1_needs/needs.json').read_text())
    obj=trimesh.Trimesh(domain['geometry']['vertices_m'],domain['geometry']['faces'],process=False)
    report=json.loads((SOURCE/'step6_connect_support/connection.json').read_text())
    if report.get('schema')=='staged_contact_module_stationary_dock_v1':
        return modular_geometry(domain,obj)
    trajectory=json.loads((SOURCE/'step6_connect_support/trajectory.json').read_text())
    d=np.array(trajectory['motion'][:3]);d/=np.linalg.norm(d)
    x=np.cross([0,0,1],d);x/=np.linalg.norm(x)
    y=np.cross(d,x);basis=np.stack([x,y,d],axis=1)
    with np.load(SOURCE/'step6_connect_support/geometry.npz') as a:
        parts=S.unpack_parts(a);labels=a['part_labels'].tolist()
    head_parts=[p for p,l in zip(parts,labels) if l.startswith(('contact_head_','neck_','frame_'))]
    terminals=np.array(report['geometry']['belt']['terminals_m'])
    scale=float(obj.extents.max())
    # A plain rectangular peg enters a rectangular sleeve along -d.
    # No T-profile, lips, latch or active clamp. A fixed end wall stops insertion;
    # translation along +d remains free (load retention is NOT established).
    midpoint=terminals.mean(0)
    q=midpoint+x*.035+d*.015
    head_parts+=tube(np.array([midpoint,q+d*.012,q]),.004)
    peg=oriented_box(q-d*.013,[.018,.008,.026],basis)
    head_parts.append(peg)
    channel=[oriented_box(q-d*.0275,[.025,.015,.003],basis)]
    for sign in (-1,1):
        # Small overlaps between construction boxes make the sleeve a single
        # solid; the actual opening and 21 mm engagement are unchanged.
        channel += [oriented_box(q+x*sign*.011-d*.0156,[.003,.015,.0212],basis),
                    oriented_box(q+y*sign*.006-d*.0156,[.025,.003,.0212],basis)]
    socket=BG.union_parts(channel,scale)[0]
    interface=dict(origin_m=(q-d*.0155).tolist(),withdrawal_direction=d.tolist(),
                     type='rectangular_peg_and_sleeve_with_fixed_end_stop',
                     slider_length_mm=26,engaged_length_mm=21,
                     peg_cross_section_mm=[18,8],socket_opening_mm=[19,9],
                     sleeve_outer_cross_section_mm=[25,15],wall_thickness_mm=3,
                     nominal_clearance_mm=.5,
                     docking_stroke_mm=30,anti_withdrawal_lock=False,
                     limitation='No retention along withdrawal axis; clearance permits play. Load capacity and precision are unverified.')
    contact,crecord=BG.union_parts(head_parts,scale)
    if not crecord['one_solid']:
        raise RuntimeError('The shared contact module must form one connected solid')
    meshes=dict(object=obj,contact=contact)
    files=[SOURCE/'step_1_needs/needs.json',SOURCE/'step6_connect_support/geometry.npz',
           SOURCE/'step6_connect_support/connection.json',SOURCE/'step6_connect_support/trajectory.json',
           SOURCE/'step3_scheculer/final_contacts.npz']
    metadata=dict(interface=interface,channel=socket,peg=peg,port=q,basis=basis,domain=domain,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return meshes,metadata
