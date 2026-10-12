"""Separate loading with the original closed ground ring and saved dock poses."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.optimize import linprog
import trimesh

import fixture_geometry as F
from geometry_utils import tube
from source_geometry import oriented_box

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT/'codes/simulation/shape/B/pose_2'
POSE_RECORD = Path(__file__).with_name('geometry_report.json')
HEAD_IDS = ('C140', 'C093')


def ground_balance(mesh, transform, tolerance=1e-7):
    points = trimesh.transform_points(mesh.vertices, transform)
    com = trimesh.transform_points([mesh.center_mass], transform)[0]
    height = points[:, 2].min()
    contacts = points[points[:, 2] <= height+tolerance]
    fit = linprog(np.zeros(len(contacts)),
        A_eq=np.vstack([contacts[:, :2].T, np.ones(len(contacts))]),
        b_eq=np.r_[com[:2], 1.], bounds=(0, None), method='highs')
    return dict(ground_only_equilibrium=bool(fit.success),
        floor_contact_count=len(contacts), floor_tolerance_m=tolerance,
        minimum_z_m=float(height), center_of_mass_m=com.tolist(),
        scope='Rigid mesh, uniform-density center of mass, gravity only; no compliance or disturbance margin.')


def supported_gravity_balance(mesh, transform, face_ids, friction=.3):
    """Object-only equilibrium with fixed blue, using an inscribed friction cone."""
    rotation = transform[:3, :3]
    vertices = trimesh.transform_points(mesh.vertices, transform)
    com = trimesh.transform_points([mesh.center_mass], transform)[0]
    ground = vertices[vertices[:, 2] <= 1e-7]
    points = np.vstack([trimesh.transform_points(mesh.triangles[face_ids].reshape(-1, 3), transform), ground])
    normals = np.vstack([np.repeat(-mesh.face_normals[face_ids]@rotation.T, 3, axis=0),
                         np.tile([0., 0., 1.], (len(ground), 1))])
    columns = []
    for point, normal in zip(points, normals):
        tangent = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
        tangent /= np.linalg.norm(tangent)
        second = np.cross(normal, tangent)
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            force = normal+friction*(np.cos(angle)*tangent+np.sin(angle)*second)
            columns.append(np.r_[force, np.cross(point-com, force)])
    wrench = np.array(columns).T
    need = np.array([0., 0., 1., 0., 0., 0.])
    scaling = np.array([1., 1., 1., 10., 10., 10.])
    fit = linprog(np.ones(wrench.shape[1]), A_eq=wrench*scaling[:, None],
                  b_eq=need*scaling, bounds=(0, None), method='highs')
    residual = float(np.max(abs(wrench@fit.x-need))) if fit.success else None
    return dict(feasible=bool(fit.success and residual < 1e-8),
        solver_status=int(fit.status), friction_coefficient_assumed=friction,
        maximum_equilibrium_residual=residual,
        scope='Object only, fixed blue, rigid unilateral contacts, uniform density, unit weight, '
              '8-ray inscribed friction cone; no force cap, dock/base equilibrium, or task load.')


def build():
    from original_geometry import geometry
    obj, blue, socket, peg, port, basis, centers, source_faces, meta = geometry()
    records = json.loads(POSE_RECORD.read_text())['records']
    d = basis[:, 2]
    cases = []
    for record in records:
        T = np.asarray(record['transform'])
        # Find an initial separating twist at the fixed heads and floor.
        # The finite path is checked against every dock; shape stays fixed.
        normals = np.repeat(obj.face_normals[source_faces]@T[:3, :3].T,3,axis=0)
        points = trimesh.transform_points(obj.triangles[source_faces].reshape(-1,3),T)
        vertices = trimesh.transform_points(obj.vertices,T)
        floor = vertices[vertices[:,2]<1e-7]
        com = trimesh.transform_points([obj.center_mass],T)[0]
        normals = np.vstack([normals,np.tile([0,0,-1],(len(floor),1))])
        points = np.vstack([points,floor])
        matrix = np.c_[normals,np.cross(points-com,normals)/.2]
        fit = linprog([0,0,0,0,0,0,-1],A_ub=np.c_[matrix,np.ones(len(matrix))],
            b_ub=np.zeros(len(matrix)),bounds=[(-1,1)]*6+[(0,None)],method='highs')
        if not fit.success or fit.x[-1]<1e-7:
            raise RuntimeError('No separating object motion for '+record['pose'])
        from scipy.spatial.transform import Rotation
        travel = {'pose_2':.04,'pose_2_tilt_25deg':.04,'pose_4':.04}[record['pose']]
        rotation = Rotation.from_rotvec(fit.x[3:6]*travel/.2).as_matrix()
        withdrawal = np.eye(4);withdrawal[:3,:3]=rotation
        withdrawal[:3,3]=com+fit.x[:3]*travel-rotation@com
        object_pre = withdrawal@T
        object_exit = fit.x[:3]
        work_ids = np.asarray(record['working_area']['face_ids'], int)
        cases.append(dict(pose=record['pose'], pose_kind=record['pose_kind'],
            transform=T, object=obj.copy().apply_transform(T),
            contact=blue.copy().apply_transform(T), channel=socket.copy().apply_transform(T),
            slider_detail=peg.copy().apply_transform(T), basis=T[:3, :3]@basis,
            direction=T[:3, :3]@d, object_exit=object_exit, object_pre=object_pre,
            blue_departure=np.asarray(record['post_release_departure']['direction']),
            port=trimesh.transform_points([port], T)[0],
            work_ids=work_ids, work_area=record['working_area'],
            points=trimesh.transform_points(centers, T), label=record['pose'],
            ground_balance=ground_balance(obj,T),
            supported_gravity_balance=supported_gravity_balance(obj,T,source_faces)))
    from shapely.geometry import MultiPoint, Point
    from geometry_utils import resample
    cloud = np.vstack([c['object'].vertices for c in cases]+[c['port'][None,:] for c in cases])
    footprint = MultiPoint(cloud[:,:2]).convex_hull.buffer(.035)
    xy = resample(footprint,96)
    parts = tube(np.c_[xy,np.full(len(xy),.005)],.005,True)
    for c in cases:
        x,y,d = c['basis'].T
        mount = c['port']-d*.0315-y*.004
        back = mount-d*.020+x*.025
        local = MultiPoint(np.vstack([c['object'].vertices[:,:2],c['port'][None,:2]])).convex_hull.buffer(.023)
        edge = local.exterior.interpolate(local.exterior.project(Point(back[:2])))
        foot = np.array([edge.x,edge.y,.005])
        elbow = np.array([edge.x,edge.y,max(.018,back[2]-.010)])
        outer = footprint.exterior.interpolate(footprint.exterior.project(edge))
        anchor = np.array([outer.x,outer.y,.005])
        parts += [c['channel']]+tube(np.array([anchor,foot,elbow,back,mount]),.0045)
    base = F.joined(parts)
    for c in cases:
        c['base']=base
        c['base_layout']='shared_ground_frame_three_sockets'
    meta['base_mesh_sha256']=hashlib.sha256(base.vertices.tobytes()+base.faces.tobytes()).hexdigest()
    return cases,base,obj,blue,meta
