"""Exact polyhedral collision geometry for the installed, free-body assembly."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import tetgen
import trimesh
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
SHAPE = HERE / 'shape/B/pose_2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(x):
    return ' '.join(format(float(v), '.17g') for v in np.asarray(x).ravel())


def verify_inputs(shape=SHAPE):
    manifest = read(shape / 'manifest.json')
    for name, entry in manifest['files'].items():
        if sha(shape / name) != entry['sha256']:
            raise ValueError(f'Input changed: {name}; prepare a new snapshot first')
    return manifest


def tetrahedra(shape):
    """pY preserves the supplied boundary; no approximate convex decomposition."""
    raw = trimesh.load(shape / 'object_source.stl', force='mesh')
    with np.load(shape / 'object_geometry.npz') as g:
        T = g['T_world_mesh']
        v, f = trimesh.remesh.subdivide(raw.vertices, raw.faces)
        np.testing.assert_allclose(trimesh.transform_points(v, T), g['vertices_m'], atol=1e-13)
        np.testing.assert_array_equal(f, g['faces'])
    world = trimesh.Trimesh(trimesh.transform_points(raw.vertices, T), raw.faces, process=False)
    nodes, cells, _, _ = tetgen.TetGen(world.vertices, world.faces).tetrahedralize(switches='pYQ')
    tris = cells[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    unique, count = np.unique(np.sort(tris, axis=1), axis=0, return_counts=True)
    boundary = unique[count == 1]
    # TetGen retains input vertex numbering under these switches.
    assert len(nodes) == len(world.vertices)
    np.testing.assert_allclose(nodes, world.vertices, atol=1e-14)
    assert set(map(tuple, boundary)) == set(map(tuple, np.sort(world.faces, axis=1)))
    tv = nodes[cells]
    volumes = np.abs(np.linalg.det(tv[:, 1:] - tv[:, :1])) / 6
    np.testing.assert_allclose(volumes.sum(), world.volume, rtol=1e-10)
    assert volumes.min() > 0
    return nodes, cells, dict(method='surface_preserving_TetGen_pYQ', count=len(cells),
        boundary_matches_source_triangles=True, boundary_triangle_count=len(boundary),
        volume_m3=float(volumes.sum()), volume_relative_error=float(abs(volumes.sum()/world.volume-1)),
        minimum_tetrahedron_volume_m3=float(volumes.min()))


def build(shape=SHAPE, timestep=.001, density=1000., timeconst=.01, robots=False):
    manifest = verify_inputs(shape)
    nodes, cells, tet_report = tetrahedra(shape)
    obj = np.load(shape / 'object_geometry.npz')
    support = np.load(shape / 'support_geometry.npz')
    # process=True would weld distinct, extremely close seam vertices: do not use it.
    solid = trimesh.Trimesh(support['union_vertices_m'], support['union_faces'], process=False)
    assert solid.is_watertight and solid.is_winding_consistent and solid.volume > 0
    solid.density = density
    root = ET.Element('mujoco', model='B_pose_2_installed')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', inertiafromgeom='false')
    opt = ET.SubElement(root, 'option', timestep=str(timestep), gravity='0 0 -9.81',
        integrator='implicitfast', solver='Newton', cone='elliptic', iterations='100',
        tolerance='1e-10', noslip_iterations='20', ccd_tolerance='1e-10')
    ET.SubElement(opt, 'flag', multiccd='enable')
    ET.SubElement(root, 'size', memory='128M')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='800')
    ET.SubElement(visual, 'quality', shadowsize='2048')
    ET.SubElement(visual, 'headlight', ambient='.45 .45 .45', diffuse='.65 .65 .65')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', solref=f'{timeconst} 1', solimp='.99 .999 .0001',
        margin='0', gap='0', condim='1', friction='0 0 0', density='0')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', type='skybox', builtin='gradient',
        rgb1='.86 .89 .92', rgb2='.97 .98 .99', width='512', height='512')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -.3 .7', dir='0 .3 -.7', directional='true')
    # Floor coefficients are a fixed backend realization of the no-slip assumption.
    # Priority makes ONLY floor contacts tangential. Nothing is welded to the floor.
    ET.SubElement(world, 'geom', name='floor', type='plane', size='1 1 .01',
        rgba='.92 .93 .95 1', contype='4', conaffinity='3', priority='1',
        condim='3', friction='64 0 0')

    def mesh(name, vertices, faces=None):
        kwargs = dict(name=name, vertex=numbers(vertices))
        if faces is not None:
            kwargs['face'] = ' '.join(map(str, np.asarray(faces).ravel()))
        ET.SubElement(asset, 'mesh', **kwargs)
        return name

    def body(name, mass, com, inertia):
        b = ET.SubElement(world, 'body', name=name)
        ET.SubElement(b, 'freejoint', name=name + '_free')
        I = np.asarray(inertia)
        ET.SubElement(b, 'inertial', pos=numbers(com), mass=str(mass),
            fullinertia=numbers([I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]]))
        return b

    b = body('object', manifest['mass_kg'], manifest['com_world_m'], manifest['inertia_com_world_kg_m2'])
    for k, cell in enumerate(cells):
        ET.SubElement(b, 'geom', name=f'object_cell_{k}', type='mesh', mesh=mesh(f'ot{k}', nodes[cell]),
            contype='1', conaffinity='6', group='3', rgba='.8 .7 .5 0')
    ET.SubElement(b, 'geom', name='object_visual', type='mesh',
        mesh=mesh('object_surface', obj['vertices_m'], obj['faces']),
        contype='0', conaffinity='0', group='2', rgba='.77 .73 .64 1')
    w = trimesh.Trimesh(obj['vertices_m'], obj['faces'][obj['work_face_ids']], process=False)
    w.remove_unreferenced_vertices()
    # Visual overlay only, tiny normal offset to avoid z-fighting.
    ET.SubElement(b, 'geom', name='working_area_visual', type='mesh',
        mesh=mesh('working_surface', w.vertices + 2e-5*w.vertex_normals, w.faces),
        contype='0', conaffinity='0', group='2', rgba='.18 .7 .35 1')
    b = body('support', solid.mass, solid.center_mass, solid.moment_inertia)
    convex_errors = []
    for k, label in enumerate(support['part_labels']):
        a, z = support['vertex_offsets'][k:k+2]
        c, e = support['face_offsets'][k:k+2]
        vertices, faces = support['part_vertices_m'][a:z], support['part_faces'][c:e]
        part = trimesh.Trimesh(vertices, faces, process=False)
        convex_errors.append(abs(ConvexHull(vertices).volume - abs(part.volume)))
        ET.SubElement(b, 'geom', name=str(label), type='mesh', mesh=mesh(f's{k}', vertices, faces),
            contype='2', conaffinity='5', group='3', rgba='.1 .5 .7 0')
    assert max(convex_errors) < 1e-13
    ET.SubElement(b, 'geom', name='support_visual', type='mesh',
        mesh=mesh('support_surface', support['union_vertices_m'], support['union_faces']),
        contype='0', conaffinity='0', group='2', rgba='.15 .46 .69 1')
    if robots:
        from kuka import add_robots
        add_robots(root)
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    report = dict(mujoco_version=mujoco.__version__, object_tetrahedra=tet_report,
        support_convex_parts=len(convex_errors), maximum_convex_volume_error_m3=max(convex_errors),
        support_mass_kg=float(solid.mass), support_density_kg_m3=density,
        support_com_world_m=solid.center_mass.tolist(), support_mesh_watertight=True,
        body_dofs=int(model.nv), equality_constraints=int(model.neq), timestep_s=timestep,
        contact_time_constant_s=timeconst, insertion_path_simulated=False,
        object_support_contact='unilateral_normal_only', floor_contact='unilateral_no_slip_assumption',
        shape_manifest_sha256=sha(shape/'manifest.json'), scene_sha256=hashlib.sha256(xml.encode()).hexdigest())
    return model, report
