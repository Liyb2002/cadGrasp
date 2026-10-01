"""Install blue at rest, grasp object+blue, dock into a stationary orange base.

Geometry uses certified module/docking sweeps and a high-clearance transport.
The video reuses the simulation KUKA model with sampled IK and saved load arrows.
It is not a robot collision, grasp or dynamics certificate.
"""
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation,Slerp
from PIL import ImageDraw
from step1.needs import OUTPUTS,sha256
from step1.cases import pose_name
from step3_scheculer import contacts as I
from step5_connect_support import whole_assembly as A,solids as S,visual_details as V,video
from step5_base import modular as BASE,rectangular_dock as J

SCHEMA='staged_contact_module_stationary_dock_v1'
STAGE='step6_connect_support'


def code_hashes():
    return {**BASE.code_hashes(),**I.hashes([Path(__file__),Path(V.__file__),Path(video.__file__),Path(__file__).with_name('robot_video.py'),Path(__file__).with_name('snap_retention.py')])}


def unpack(data):
    return trimesh.Trimesh(data['union_vertices_m'],data['union_faces'],process=False)


def transport(domain,blue,base,report):
    rest=np.asarray(report['installation']['T_initial_from_task']).copy()
    points=np.vstack([domain.mesh.vertices,blue.vertices]);center=points.mean(axis=0)
    radius=float(np.linalg.norm(points-center,axis=1).max())
    d0=rest[:3,:3]@np.asarray(report['d0_withdrawal_direction'])
    away=np.r_[d0[:2],0.];length=np.linalg.norm(away)
    away=away/length if length>1e-8 else np.array([-1.,0.,0.])
    initial=trimesh.transform_points(points,rest)
    shift=max(0.,np.max(base.vertices@away)+.08-np.min(initial@away))
    rest[:3,3]+=away*shift
    initial=trimesh.transform_points(points,rest)
    gap=float(np.min(initial@away)-np.max(base.vertices@away))
    height=float(base.bounds[1,2]+radius+.08)
    return dict(T_initial_from_task=rest.tolist(),horizontal_placement_offset_m=(away*shift).tolist(),
        center_task_m=center.tolist(),body_radius_m=radius,flight_center_height_m=height,
        initial_base_separating_axis=away.tolist(),initial_base_separation_m=gap,
        initial_installation_length_m=max(.04,float(report['initial_module_sweep']['length_m'])),
        initial_floor_clear=bool(initial[:,2].min()>=-1e-9),
        transport_geometry_verified=bool(gap>0 and initial[:,2].min()>=-1e-9),
        proof='Initial placement and lift separated horizontally from base; high transfer/rotation bounded by a sphere above the entire base; final descent is the certified vertical assembly withdrawal reversed.',
        robot_grasp_assumed=True,robot_kinematics_verified=False,gripper_collisions_verified=False)


def build(name,edge_budget=None):
    domain,contacts,schedule,_,_,_,_=A.read_inputs(name)
    root=OUTPUTS/name/pose_name();source=root/BASE.STAGE;out=root/STAGE;out.mkdir(parents=True,exist_ok=True)
    base=I.check_report(source/'base.json');assert I.check_report(source/'audit.json')['passed']
    # Retain diagnostic history, but remove superseded accepted geometry/video names.
    for file in ('geometry.npz','trajectory.json','insertion.mp4','insertion.gif','video_metadata.json','robot_motion.npz','support.stl','support_mm.stl'):
        (out/file).unlink(missing_ok=True)
    paths=[source/'base.json',source/'audit.json',root/'step3_scheculer/schedule.json']
    ready=base['geometry_constructed'];path=dict(passed=False,status=base['status'],base_stationary=True)
    if ready:
        blue_data=I.load_npz(source/'blue.npz');base_data=I.load_npz(source/'geometry.npz')
        blue,fixed=unpack(blue_data),unpack(base_data)
        moving=S.unpack_parts(blue_data);stationary=S.unpack_parts(base_data)
        plan=transport(domain,blue,fixed,base)
        path=dict(schema=SCHEMA,passed=bool(plan['transport_geometry_verified'] and base['docking']['passed']),
            base_stationary=True,initial_installation=base['initial_module_sweep'],d0_withdrawal_direction=base['d0_withdrawal_direction'],
            docking=base['docking'],transport=plan,continuous_module_and_dock_sweeps_verified=True,
            robot_kinematics_verified=False,grasp_feasibility_assumed=True,
            stages=['install_blue_at_rest','grasp_and_lift_object_plus_blue','dock_object_plus_blue_into_fixed_base'])
        packed=S.pack_parts(moving+stationary,[f'blue_{i}' for i in range(len(moving))]+[f'orange_{i}' for i in range(len(stationary))],trimesh.util.concatenate([blue,fixed]))
        np.savez_compressed(out/'geometry.npz',**packed)
        paths += [source/'blue.npz',source/'geometry.npz']
    I.save(out/'trajectory.json',path)
    passed=bool(ready and base['passed'] and schedule['continuous_coverage_proved'] and path['passed'])
    report=dict(schema=SCHEMA,object=name,pose=pose_name(),stage=STAGE,complete=True,passed=passed,
        status='modular_fixture_verified' if passed else base['status'],selected_ids=schedule['selected_ids'],
        geometry_constructed=ready,trajectory_verified=bool(path['passed']),base_stationary=True,
        separate_printed_modules=True,interface=base.get('interface'),bearing=base.get('bearing',{'continuous_passed':False}),
        geometry=dict(direction_search=base['direction_search'],attempts=[],belt={'passed':ready},trajectory=path),
        robot_kinematics_verified=False,grasp_feasibility_assumed=True,strength_verified=False,
        retention=dict(concept='integral snap lips; robot grasps object after module installation',
                       scope='video presentation geometry only; not included in geometry.npz or bearing/sweep certificates',
                       retention_verified=False),
        provenance=dict(inputs=I.hashes(paths),code=code_hashes()),
        artifacts={f:sha256(out/f) for f in ('geometry.npz','trajectory.json') if (out/f).exists()})
    I.save(out/'connection.json',report);I.save(out/'status.json',dict(complete=True,status=report['status'],connection_sha256=sha256(out/'connection.json')))
    print(name,pose_name(),'modular Step6:',report['status'],flush=True);return report


def audit(name):
    BASE.audit(name)
    domain,_,schedule,_,_,_,_=A.read_inputs(name);root=OUTPUTS/name/pose_name();out=root/STAGE
    report=I.check_report(out/'connection.json');base=I.check_report(root/BASE.STAGE/'base.json')
    if report['geometry_constructed']:
        blue=unpack(I.load_npz(root/BASE.STAGE/'blue.npz'));fixed=unpack(I.load_npz(root/BASE.STAGE/'geometry.npz'))
        actual=transport(domain,blue,fixed,base);assert actual==report['geometry']['trajectory']['transport']
        assert actual['transport_geometry_verified']
    assert report['passed']==bool(base['passed'] and schedule['continuous_coverage_proved'] and report['trajectory_verified'])
    result=dict(complete=True,passed=True,design_passed=report['passed'],base_stationary=True,
        initial_and_docking_sweeps_replayed=report['geometry_constructed'],three_body_bearing_replayed=base['passed'],
        transport_scope='assembly geometry only; connected gripper illustration is not collision/IK/grasp validation',
        provenance=dict(inputs=I.hashes([out/'connection.json',root/BASE.STAGE/'audit.json']),code=code_hashes()))
    I.save(out/'audit.json',result);return result


def smooth(x):
    x=float(np.clip(x,0,1));return x*x*(3-2*x)


def pose_at(t,plan):
    rest=np.asarray(plan['T_initial_from_task']);center=np.asarray(plan['center_task_m'])
    initial=rest[:3,:3]@center+rest[:3,3];height=plan['flight_center_height_m']
    high0=initial.copy();high0[2]=height;high1=center.copy();high1[2]=height
    pre=center+J.UP*J.STROKE
    if t<=5:return rest.copy()
    if t<7:r=rest[:3,:3];c=initial+(high0-initial)*smooth((t-5)/2)
    elif t<10:
        f=smooth((t-7)/3);r=Slerp([0,1],Rotation.from_matrix([rest[:3,:3],np.eye(3)]))([f]).as_matrix()[0];c=high0+(high1-high0)*f
    elif t<12:r=np.eye(3);c=high1+(pre-high1)*smooth((t-10)/2)
    else:r=np.eye(3);c=pre+(center-pre)*smooth((t-12)/2)
    T=np.eye(4);T[:3,:3]=r;T[:3,3]=c-r@center;return T


def scene_arrays(domain,blue,fixed,objT,blueT):
    triangles=[];colors=[]
    def add(mesh,color,T=None):
        p=mesh.triangles if T is None else trimesh.transform_points(mesh.vertices,T)[mesh.faces]
        triangles.append(p);colors.append(np.tile(color,(len(p),1)) if np.asarray(color).ndim==1 else color)
    palette=np.tile(V.R.GREY,(len(domain.mesh.faces),1));palette[domain.work_ids]=V.R.GREEN
    add(domain.mesh,palette,objT)
    if blue is not None:add(blue,V.BLUE,blueT)
    if fixed is not None:add(fixed,[240.,133.,38.])
    points=np.concatenate(triangles).reshape(-1,3);ground=V.floor_triangles(points,.04)
    triangles.append(ground);colors.append(np.tile(V.R.FLOOR,(len(ground),1)))
    return np.concatenate(triangles),np.concatenate(colors),[],[]


def preview(name):
    domain,_,_,_,_,_,_=A.read_inputs(name);root=OUTPUTS/name/pose_name();source=root/BASE.STAGE
    report=I.check_report(source/'base.json');blue=fixed=None
    if report['geometry_constructed']:
        blue=unpack(I.load_npz(source/'blue.npz'));fixed=unpack(I.load_npz(source/'geometry.npz'))
    data=scene_arrays(domain,blue,fixed,np.eye(4),np.eye(4));points=data[0].reshape(-1,3)
    view=V.camera(domain,[],points=points);picture,_=V.render(data,view,900)
    ink=ImageDraw.Draw(picture);ink.text((15,15),'Stationary base / removable contact module',font=V.R.font(23),fill=V.R.INK)
    picture.save(source/'base.png');return picture


def draw(name,static_only=False):
    domain,_,_,_,_,_,_=A.read_inputs(name);root=OUTPUTS/name/pose_name();out=root/STAGE;source=root/BASE.STAGE
    report=I.check_report(out/'connection.json');base=I.check_report(source/'base.json')
    preview(name).save(out/'connection.png')
    artifacts=['connection.png']
    render_code=code_hashes();inputs=[out/'connection.json',source/'base.json']
    if report['trajectory_verified'] and not static_only:
        blue=unpack(I.load_npz(source/'blue.npz'));fixed=unpack(I.load_npz(source/'geometry.npz'))
        from step6_connect_support import robot_video
        plan=report['geometry']['trajectory']['transport']
        robot_video.render(domain,blue,fixed,base,plan,out,report['passed'])
        artifacts += ['insertion.mp4','video_metadata.json','robot_motion.npz']
        render_code.update(I.hashes(robot_video.source_files()))
        inputs.append(root/'step_1_needs/samples.json')
    I.save(out/'views.json',dict(complete=True,base_stationary=True,trajectory_success=report['trajectory_verified'],
        provenance=dict(inputs=I.hashes(inputs),code=render_code),
        artifacts={f:sha256(out/f) for f in artifacts}))
