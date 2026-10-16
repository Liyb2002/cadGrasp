"""Nested, exterior-first explanatory deletions from the exact end of movie 01.

This builds presentation geometry only. The historical force certificates and
physical solver are never changed or rerun. Each shown change is an actual
Boolean deletion, with a freshly measured world-axis-aligned support box.
"""
from pathlib import Path
import json
import os
import sys

import render01 as R
sys.path.insert(0,str(R.COOPT/'step5.2'))
import numpy as np
import trimesh
import manifold3d as md
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from deterministic_base import solid, unpack, union, moved, volume, SCALE
from continuous_legs import raster

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data/video02'


def load_rows():
    record = json.loads((HERE / 'data/manifest.json').read_text())
    with np.load(HERE / 'data/geometry.npz') as saved:
        rows = [dict(pose=p['pose'], color=p['color'],
                     T=saved[p['pose']+'_T_world_fixture'].copy(),
                     U=saved[p['pose']+'_T_world_object'].copy(),
                     hull_world=saved[p['pose']+'_hull_world_m'].copy(),
                     hull=saved[p['pose']+'_hull_fixture_m'].copy(),
                     dots=saved[p['pose']+'_cloud_fixture_m'][saved[p['pose']+'_display_sample_indices']].copy())
                for p in record['poses']]
    return record, rows


def ball(point, radius):
    return md.Manifold.sphere(radius/SCALE, 16).translate(tuple(np.asarray(point)/SCALE))


def capsule(a, b, radius):
    return md.Manifold.batch_hull([ball(a, radius), ball(b, radius)])


def simplify_path(points, tolerance=.00065):
    if len(points) <= 2:
        return points
    a, b = points[0], points[-1]
    direction = b-a
    t = np.clip((points-a)@direction/max(float(direction@direction), 1e-30), 0., 1.)
    distance = np.linalg.norm(points-a-t[:, None]*direction, axis=1)
    i = int(distance.argmax())
    if distance[i] <= tolerance:
        return np.array([a, b])
    return np.vstack([simplify_path(points[:i+1], tolerance)[:-1], simplify_path(points[i:], tolerance)])


def native_components(value):
    # Units here are cubic metres; ignore only native numerical shell noise.
    return sorted([p for p in value.decompose() if volume(p) > 1e-12], key=volume, reverse=True)


def core_guide(core, rings, ports):
    """A shared interior tree keeps the explanatory object connected as it peels."""
    source = solid(core)
    pitch = .001
    origin = np.floor(core.bounds[0]/pitch)*pitch-pitch
    shape = tuple(np.ceil((core.bounds[1]-origin)/pitch).astype(int)+2)
    inside = raster(source, origin, shape, pitch)
    clearance = ndimage.distance_transform_edt(inside, sampling=pitch).astype(np.float32)
    labels, _ = ndimage.label(inside, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    valid = labels == int(sizes.argmax())
    cells = np.argwhere(valid & (clearance >= .0008))
    points = origin+cells*pitch
    tree = cKDTree(points)
    requests = np.vstack([[r.start for r in rings], [p['center_m'] for p in ports]])
    _, ids = tree.query(requests)
    anchors = points[ids]
    roots = [tuple(cells[i]) for i in ids]
    cost = np.where(valid, 1.+(.003/np.maximum(clearance, .0004))**2, np.inf)
    solver = MCP_Geometric(cost, fully_connected=True)
    cumulative, _ = solver.find_costs([roots[0]], ends=list(set(roots[1:])), find_all_ends=True)
    segments = {}
    paths = []
    for request, anchor, end in zip(requests, anchors, roots):
        if not np.isfinite(cumulative[end]):
            raise ValueError('Presentation tree has an unreachable contact')
        cells_path = np.asarray(solver.traceback(end))
        path = simplify_path(origin+cells_path*pitch)
        if len(path) > 1:
            paths.append(path)
        paths.append(np.array([anchor, request]))
    for path in paths:
        for a, b in zip(path, path[1:]):
            if np.linalg.norm(a-b) < 1e-8:
                continue
            key = tuple(sorted([tuple(np.round(a, 10)), tuple(np.round(b, 10))]))
            segments[key] = (a, b)
    print('PRESENTATION_TREE', dict(grid_shape=shape, requests=len(requests), segments=len(segments)), flush=True)
    fallback = None
    for radius in [.0035, .0045, .006, .009]:
        guide = union([capsule(a, b, radius) for a, b in segments.values()] +
                      [ball(p, radius+.0005) for p in requests])
        trimmed = source ^ guide
        parts = native_components(trimmed)
        print('GUIDE_COMPONENTS', radius, len(parts), [round(volume(p)*1e6,6) for p in parts[:8]], flush=True)
        if fallback is None:
            fallback = (trimmed, radius, len(parts))
        if len(parts) == 1:
            mesh = unpack(trimmed)
            print('CONNECTED_CORE_GUIDE', radius, volume(trimmed)*1e6, 'triangles', len(mesh.faces), flush=True)
            return trimmed, dict(grid_pitch_m=pitch, guide_radius_m=radius,
                contact_count=len(ports), branch_roots=len(rings), segment_count=len(segments),
                source_core_volume_cm3=volume(source)*1e6, retained_core_volume_cm3=volume(trimmed)*1e6,
                presentation_only=True)
    # A coarse presentation field can leave separate protected contact caps.
    # Keep them all. Every subsequent *complete* model is checked for actual
    # connectivity; a proposed cut that severs one of these caps is rejected.
    trimmed, radius, part_count = fallback
    return trimmed, dict(grid_pitch_m=pitch, guide_radius_m=radius,
        contact_count=len(ports), branch_roots=len(rings), segment_count=len(segments),
        source_core_volume_cm3=volume(source)*1e6, retained_core_volume_cm3=volume(trimmed)*1e6,
        protected_guide_components=part_count, complete_models_require_one_component=True,
        presentation_only=True)


def thin_ring(ring, fraction):
    outer = ring.inner+(ring.outer-ring.inner)*fraction
    cells = []
    for i in range(len(ring.inner)):
        j = (i+1) % len(ring.inner)
        cells.append(R.prism(np.array([ring.inner[i], ring.inner[j], outer[j], outer[i]])))
    return moved(union(cells), ring.inverse)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest, rows = load_rows()
    ref = rows[0]['T']
    paths = [HERE/'data/illustrative_support.obj', HERE/'data/geometry.npz', HERE/'data/manifest.json',
             R.MOVIE, R.STEP4/'support.obj', R.ROOT/'objects/B/mesh.stl',
             R.COOPT/'data/results/B/data/contact_subset_cache_20261015'/R.CASE/'reaction_ports/report.json']
    protected = {R.relative(p):R.digest(p) for p in paths}
    original_mesh = trimesh.load(paths[0], force='mesh', process=False)
    original = solid(original_mesh)
    core_mesh = trimesh.load(R.STEP4/'support.obj', force='mesh', process=False)
    core = solid(core_mesh)
    ports = json.loads(paths[-1].read_text())['ports']
    rings = [R.Ring(row, core_mesh) for row in rows]
    rods = union([solid(r.link) for r in rings])
    target_core, tree_record = core_guide(core_mesh, rings, ports)
    current = original
    stages = []
    def commit(candidate, operation, detail, direction=None):
        nonlocal current
        candidate = candidate.simplify(0.)
        loss = volume(current)-volume(candidate)
        if loss < 1e-9:
            return False
        if volume(candidate-current) > 1e-12:
            raise ValueError('A presentation deletion added material')
        parts = native_components(candidate)
        if len(parts) != 1:
            print('REJECT_DISCONNECTED_CUT', operation, detail, len(parts), flush=True)
            return False
        mesh = unpack(candidate)
        if not mesh.is_watertight:
            raise ValueError('Presentation checkpoint is not closed')
        if len(mesh.faces)>250000:
            raise ValueError('Presentation Boolean generated redundant subdivisions')
        removed = (current-candidate).simplify(0.)
        bounds = np.array([R.transform(mesh.vertices, ref).min(0), R.transform(mesh.vertices, ref).max(0)])
        index = len(stages)
        folder = DATA/'stages'/f'{index:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        mesh.export(folder/'support.obj', digits=17)
        if index:
            unpack(removed).export(folder/'removed.obj', digits=17)
        bbox = float(np.prod(bounds[1]-bounds[0])*1000)
        previous = stages[-1]['bbx_l'] if stages else bbox
        stages.append(dict(index=index, operation=operation, detail=detail,
            support=R.relative(folder/'support.obj'), support_sha256=R.digest(folder/'support.obj'),
            removed=None if not index else R.relative(folder/'removed.obj'),
            material_cm3=volume(candidate)*1e6, removed_cm3=loss*1e6 if index else 0.,
            bounds_world_m=bounds.tolist(), bbx_l=bbox, bbx_change_l=bbox-previous,
            bbx_changed=abs(bbox-previous)>1e-8, direction_world=None if direction is None else list(direction),
            subset_verified=True, connected_components=1, watertight=True, force_solver_run=False))
        current = candidate
        print('DELETE_STAGE', index, operation, round(volume(candidate)*1e6, 3),
              'BBX', round(bbox, 6), 'delta', round(bbox-previous, 8), flush=True)
        return True
    # Preserve the byte-identical first movie endpoint for the first frame.
    folder = DATA/'stages/000';folder.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(paths[0], folder/'support.obj')
    bounds = np.array([R.transform(original_mesh.vertices, ref).min(0),R.transform(original_mesh.vertices, ref).max(0)])
    stages.append(dict(index=0, operation='video01_exact_endpoint', detail={},
        support=R.relative(folder/'support.obj'),support_sha256=R.digest(folder/'support.obj'),removed=None,
        material_cm3=volume(original)*1e6, removed_cm3=0., bounds_world_m=bounds.tolist(),
        bbx_l=float(np.prod(bounds[1]-bounds[0])*1000),bbx_change_l=0.,bbx_changed=False,
        direction_world=None,subset_verified=True,connected_components=1,watertight=True,force_solver_run=False))
    ring_values = [r.solid for r in rings]
    # Outer rim material goes first. The exact original hull centerlines and
    # all five feet remain, while true world BBX extrema can move inward.
    for fraction in [.75, .5]:
        order = sorted(range(len(rings)), key=lambda i:-np.max(np.linalg.norm(
            R.transform(rings[i].mesh.vertices, ref)-bounds.mean(0),axis=1)))
        for i in order:
            proposed = list(ring_values);proposed[i] = thin_ring(rings[i], fraction)
            candidate = union([core, rods, *proposed])
            if commit(candidate, 'outer_rim_peel', dict(pose=rows[i]['pose'],remaining_width_m=R.WIDTH*fraction)):
                ring_values = proposed
    mandatory = union([target_core, rods, *ring_values])
    remaining_shell = core
    shown_core = R.transform(core_mesh.vertices, ref)
    center = (shown_core.min(0)+shown_core.max(0))/2
    # Successive exterior halfspaces shave the exposed core. The retained
    # connected round tree protects finite contact/foot connections.
    normals = [np.eye(3)[axis]*sign for axis,sign in [(0,1),(2,1),(1,-1),(0,-1),(2,-1),(1,1)]]
    for fraction in [.70, .38, .10]:
        for normal in normals:
            reach = float(np.max((shown_core-center)@normal))
            offset = float(center@normal+fraction*reach)
            local_normal = ref[:3,:3].T@normal
            local_offset = offset-ref[:3,3]@normal
            inside = remaining_shell.trim_by_plane(tuple(-local_normal),-local_offset/SCALE)
            candidate = inside+mandatory
            if commit(candidate,'outer_core_peel',dict(remaining_directional_fraction=fraction,
                plane_world=[*normal.tolist(),-offset]),normal):
                remaining_shell = inside
    candidate = mandatory
    commit(candidate,'rounded_exterior_finish',dict(guide_radius_m=tree_record['guide_radius_m']))
    if len(stages) < 10 or stages[-1]['material_cm3'] >= stages[0]['material_cm3']*.9:
        raise ValueError('Insufficient visible presentation pruning')
    if any(s['bbx_l']>p['bbx_l']+1e-8 for p,s in zip(stages,stages[1:])):
        raise ValueError('Nested deletion increased the actual displayed BBX')
    record=dict(complete=True,pose_set=R.CASE,stages=stages,tree=tree_record,
        source_video=R.relative(R.MOVIE),initial_geometry_exactly_matches_video01=True,
        reference_fixture_transform=ref.tolist(),bbx_definition='world XYZ AABB of currently retained blue support at the fixed absolute installation',
        bbx_contains_graphical_hulls=False,bbx_forced_to_shrink=False,
        material_only_deletion_steps=sum(not s['bbx_changed'] for s in stages[1:]),
        bbx_reduction_steps=sum(s['bbx_changed'] for s in stages[1:]),
        every_step_is_actual_nested_boolean_subset=True,explanatory_geometry=True,
        force_solver_run=False,numerical_search_run=False,protected_inputs=protected)
    for path,expected in protected.items():
        if R.digest(R.ROOT/path)!=expected:
            raise ValueError('Saved input changed during presentation preparation: '+path)
    R.save(DATA/'stages.json',record)
    print('PREPARATION_COMPLETE',len(stages),record['bbx_reduction_steps'],record['material_only_deletion_steps'],flush=True)
    return record


if __name__=='__main__':
    prepare()
