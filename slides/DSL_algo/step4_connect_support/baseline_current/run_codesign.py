"""Use the copied co-design body algorithm with independent heads on one object."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current.run_independent import read_case as read_independent
from step4_connect_support.baseline_current.run_sequential_k import foot_menu, plain
from step4_connect_support.baseline_current import head_registration as H, build_coupled_saddle as S
from step4_connect_support.baseline_current.codesign_port import build_local_bodies as L
from step4_connect_support.baseline_current.codesign_port import visual_details as V
from step4_connect_support.baseline_current.codesign_port.fixture_view import export_viewer, local_to_world
from step4_connect_support.baseline_current.codesign_port.refresh_shared_geometry_view import write_viewer
from step4_connect_support.baseline_current import zero_thickness_heads as Z
from step2_local_support import geometry as G, withdrawal as W

PALETTE = ['#dc9d47', '#ac7098', '#7196c0', '#50a59b', '#77a76a']


def read_case(output):
    poses = ['pose_'+p for p in output.parent.name.removeprefix('pose').split('+')]
    filtered = output.parent/'step3_scheculer/independent_poses_floor2mm'
    if not filtered.exists():
        raise ValueError('Run the group-specific 2 mm contact-floor Step3 search before the surface-head constructor')
    case = read_independent(output.parent.parent.name, poses, output/'data/codesign_inputs_surface',
        independent_root=filtered)
    for report in case.source_reports:
        if (report.get('head_model') != Z.MODEL or report.get('floor_poses') != poses
                or report.get('floor_clearance_m') != .002
                or not report.get('all_pose_contact_floor_margin', {}).get('passed')):
            raise ValueError('Every source Step3 report must use this exact pose group and 2 mm surface-floor margin')
    case.pair = output.parent
    Z.prepare(case, S.RELIEF)
    I.save(case.source/'schedule.json',case.schedule)
    return case


def registration(case):
    bases, offsets = H.fixed_placements(case.tasks)
    heads, check = H.register(case.groups, case.heads, bases, offsets, general_layout=True)
    alignment = []
    reference = case.tasks[0].domain.mesh.vertices
    for task, basis, offset in zip(case.tasks, bases, offsets):
        error = float(np.max(np.abs(task.domain.mesh.vertices@basis+offset-reference)))
        assert error < 1e-10, 'All object copies must coincide in the common reference'
        alignment.append(dict(pose=task.pose, maximum_object_vertex_alignment_error_m=error))
    heights = []
    for head in heads:
        points = np.vstack(head.cells)
        for task, basis, offset in zip(case.tasks, bases, offsets):
            world = local_to_world(points, basis, offset)
            heights.append(dict(head=head.ident, owner_pose=case.poses[head.pose], pose=task.pose,
                minimum_height_m=float(world[:, 2].min()), passed=bool(world[:, 2].min() >= -H.FLOOR_TOL)))
    check.update(object_alignment=alignment, per_head_floor_checks=heights,
        head_model=case.head_model, mandatory_head_thickness_m=0.,
        contact_floor_margin_m=.002, floor_poses=list(case.poses),
        generated_support_roots=case.support_seed_records)
    if any(row['minimum_height_m'] < .002-1e-9 for row in heights):
        raise ValueError('Independent replay found a contact patch below the 2 mm all-pose margin')
    return bases, offsets, heads, check


def head_meshes(heads):
    # Zero-thickness heads remain their original saved patch triangles. Never
    # turn a convex hull spanning a curved patch into the contact surface.
    return [Z.patch_mesh(h.contact_points) for h in heads]


def png_scene(obj, meshes, colors, size, direction, translucent=False):
    from PIL import ImageColor
    points = np.vstack([obj.vertices]+[m.vertices for m in meshes])
    view = V.fit(points, V.R.axes(direction), margin=1.16)
    triangles = np.vstack([m.triangles for m in meshes])
    palette = np.vstack([np.tile(ImageColor.getrgb(c), (len(m.faces), 1)) for m, c in zip(meshes, colors)])
    if translucent:
        back, _ = V.render((obj.triangles, np.tile(V.R.GREY, (len(obj.faces), 1)), [], []), view, size)
        back = Image.blend(Image.new('RGB', back.size, 'white'), back, .28)
        front, ids = V.render((triangles, palette, [], []), view, size)
        back.paste(front, mask=Image.fromarray(np.uint8(ids >= 0)*255))
        return back
    all_triangles = np.vstack([obj.triangles, triangles])
    all_colors = np.vstack([np.tile(V.R.GREY, (len(obj.faces), 1)), palette])
    return V.render((all_triangles, all_colors, [], []), view, size)[0]


def retire_stations(output):
    archive = output/'data/rejected_stations'
    previous = output/'data/report.json'
    if previous.exists() and json.loads(previous.read_text()).get('schema') == 'independent_pose_connected_shape_v1':
        (archive/'data').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous, archive/'data/report.json')
        for path in output.glob('shape*'):
            if path.is_file():
                path.replace(archive/path.name)
        I.save(archive/'rejection.json', dict(reason='User rejected the separated-station model',
            replacement='Co-design fixed common-object registration; original Step3 heads retained',
            historical_body_data='../independent_body', corrected_shape=False))


def preview(output):
    case = read_case(output)
    bases, offsets, heads, check = registration(case)
    meshes = head_meshes(heads)
    colors = [PALETTE[h.pose % len(PALETTE)] for h in heads]
    size = 850
    page = Image.new('RGB', (2*size, 1050), 'white')
    for k, direction in enumerate(([-.8, -1., .65], [.8, 1., .65])):
        page.paste(png_scene(case.tasks[0].domain.mesh, meshes, colors, size, direction, True), (k*size, 100))
    ink = ImageDraw.Draw(page)
    ink.text((28, 18), f'{case.pair.name} | All {len(heads)} heads on ONE object', font=V.R.font(32), fill=V.R.INK)
    ink.text((28, 64), f'Same {case.poses[0]} placement, two camera angles; translucent object; bodies not built yet', font=V.R.font(21), fill=V.MUTED)
    for k, (pose, group) in enumerate(zip(case.poses, case.groups)):
        x = 28+k*330
        ink.rectangle((x, 981, x+20, 1001), fill=PALETTE[k])
        ink.text((x+30, 976), f'{pose}: {len(group)} heads', font=V.R.font(23), fill=V.R.INK)
    # Complete the replacement before removing the rejected public pictures.
    path = output/'all_heads.png'; page.save(path)
    mesh = trimesh.util.concatenate(meshes)
    work = output/'data/codesign_preview'; work.mkdir(parents=True, exist_ok=True)
    report = dict(object=case.name, poses=case.poses, source_schedule=str((case.source/'schedule.json').relative_to(I.ROOT)),
        complete=True, passed=False, constructed=False, diagnostic_only=True,
        physical_head_definition=Z.MODEL,
        presentation_description='所有零厚度接触面已放回同一个物体参照；颜色区分来源 pose。这里尚未构造身体。')
    export_viewer(work, mesh, [(h.ident,m) for h,m in zip(heads,meshes)], report, case.tasks, bases, offsets,
                  {h.ident:c for h,c in zip(heads,colors)})
    shutil.copyfile(work/'index.html', output/'all_heads.html')
    retire_stations(output)
    record = dict(complete=True, object=case.name, poses=case.poses, physical_head_count=len(heads),
        registration=check, placement=plain(dict(bases=bases, offsets=offsets)),
        all_heads_on_one_object=True, contact_surfaces_and_loads_unchanged=True,
        original_probe_volumes_required=False, head_model=Z.MODEL, constructed=False,
        provenance=dict(inputs=I.hashes(case.paths), code=I.hashes([Path(__file__), Path(H.__file__), Path(V.__file__),Path(Z.__file__)])),
        artifacts={'../all_heads.png':I.sha256(path), '../all_heads.html':I.sha256(output/'all_heads.html')})
    I.save(output/'data/common_object_heads.json', record)
    print('HEADS READY', case.pair.name, len(heads), 'all floors', check['all_heads_above_all_floors'], flush=True)
    return record


def direction_options(case, heads, bases, offsets):
    menus, checks = [], []
    # This finite material is generated by the constructor; degenerate surface
    # triangles must not be passed to a volume-based convex sweep test.
    root_heads, _ = H.register(case.groups,case.support_seeds,bases,offsets,general_layout=True)
    for k, task in enumerate(case.tasks):
        analyzer = W.Analyzer(task.domain.mesh, .01*task.domain.mesh.extents.max(), dict(vectors=case.catalogues[k].tolist()))
        cells = [SimpleNamespace(vertices=local_to_world(v,bases[k],offsets[k])) for h in root_heads for v in h.cells]
        rows = [dict(direction_id=int(i), **analyzer.test(cells, case.catalogues[k][i])) for i in case.menus[k]]
        valid = [r['direction_id'] for r in rows if r['clear']]
        menus.append(valid); checks.append(dict(pose=task.pose, tested=rows, common_direction_ids=valid,
            geometry='constructor_generated_support_roots', zero_surface_infeasibility_claim=False))
    return menus, checks


def run(output):
    case = read_case(output)
    bases, offsets, heads, check = registration(case)
    floor = H.floor_compatibility(case.tasks, case.demands, bases, offsets)
    data = output/'data'; data.mkdir(exist_ok=True)
    report = dict(schema='codesign_common_object_step4_v1', complete=False, object=case.name, poses=case.poses,
        constructed=False, passed=False, step3_passed=case.schedule['passed'],
        covered_counts=case.schedule['covered_counts'], physical_head_count=len(heads), shared_head_count=0,
        registration=check, floor_compatibility=floor, attempts=[],
        placement=plain(dict(bases=bases, offsets=offsets)), copied_algorithm='codesign_port/build_local_bodies.py',
        head_model=Z.MODEL, mandatory_head_thickness_m=0., generated_support_roots=case.support_seed_records,
        contact_floor_margin_m=.002, floor_poses=list(case.poses),
        model='Exact unshared contact surfaces on one object; bounded support roots then co-design nearest-floor bodies and connections')
    feet_menu = list(foot_menu(case, bases, offsets))
    if not check['all_heads_above_all_floors'] or not floor['passed']:
        # Call the copied constructor itself: its unchanged preconditions must
        # reject this input before caching sweeps or manufacturing a body.
        ids = [menu[0] for menu in case.menus]
        placement = dict(bases=bases, offsets=offsets, directions=np.array([c[i] for c,i in zip(case.catalogues,ids)]))
        try:
            with tempfile.TemporaryDirectory(prefix='codesign-precondition-') as tmp:
                L.build(Path(tmp), feet_menu[0] if feet_menu else [[] for _ in case.poses], case=case,
                    placement=placement, verify=True, allow_failed=True, skip_unreachable=True, export_stl=False)
        except ValueError as error:
            report['attempts'].append(dict(constructed=False, constructor_called=True, error=str(error)))
        else:
            raise AssertionError('Co-design constructor did not enforce its registration preconditions')
        report['status'] = 'original_head_crosses_floor' if not check['all_heads_above_all_floors'] else 'floor_demand_conflict'
    else:
        menus, checks = direction_options(case, heads, bases, offsets)
        report['whole_head_withdrawal'] = checks
        report['whole_head_exit_available'] = bool(all(menus))
        report['status'] = 'body_construction_failed'
        # A failed exit check is reported, but do not substitute that check for
        # the user's requested construction attempt. Run the copied builder
        # with original active-group directions and retain any failed candidate
        # explicitly as geometry only; the original full audit still applies.
        construction_menus = menus if all(menus) else case.menus
        for ids in itertools.islice(itertools.product(*construction_menus), 8):
            placement = dict(bases=bases, offsets=offsets, directions=np.array([c[i] for c,i in zip(case.catalogues,ids)]), direction_ids=list(ids))
            for fi, feet in enumerate(feet_menu):
                try:
                    with tempfile.TemporaryDirectory(prefix='codesign-common-body-') as tmp:
                        work = Path(tmp)
                        L.build(work, feet, case=case, placement=placement, verify=True, allow_failed=True,
                            skip_unreachable=True, floor_policy='nearest', cache_dir=data/'codesign_cache', export_stl=False)
                        built = json.loads((work/'report.json').read_text())
                        report['attempts'].append(dict(direction_ids=list(ids), feet_index=fi, constructed=True, passed=built['passed']))
                        if not report['constructed'] or built['passed']:
                            destination = data/'codesign_body'
                            if destination.exists():shutil.rmtree(destination)
                            shutil.copytree(work,destination)
                            shutil.copyfile(work/'fixture.obj',output/'shape.obj')
                            page=(work/'index.html').read_text().replace('href="fixture.obj"','href="shape.obj"').replace('<a href="fixture_mm.stl" download>STL · mm</a>','')
                            (output/'shape.html').write_text(page)
                            report.update(constructed=True, passed=bool(built['passed']), construction=built,
                                status='verified_connected_shape' if built['passed'] else 'connected_shape_failed_acceptance',
                                placement=plain(placement), volume_cm3=built['volume_cm3'])
                except (RuntimeError,ValueError) as error:
                    report['attempts'].append(dict(direction_ids=list(ids), feet_index=fi, constructed=False, error=str(error)))
                I.save(data/'codesign_report.json',plain(report))
                if report['passed'] or (report['constructed'] and (not case.schedule['passed'] or not all(menus))):break
            if report['passed'] or (report['constructed'] and (not case.schedule['passed'] or not all(menus))):break
    report.update(complete=True, verification=dict(performed=report['constructed']),
        provenance=dict(inputs=I.hashes(case.paths), code=I.hashes([Path(__file__),Path(H.__file__),Path(L.__file__),Path(S.__file__),Path(Z.__file__)])),
        artifacts={'../all_heads.png':I.sha256(output/'all_heads.png'), '../all_heads.html':I.sha256(output/'all_heads.html')})
    if report['constructed']:
        report['artifacts'].update({f'../{n}':I.sha256(output/n) for n in ['shape.obj','shape.html']})
    I.save(data/'codesign_report.json',plain(report)); I.save(data/'report.json',plain(report))
    print('CODESIGN COMPLETE',case.pair.name,report['status'],flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs',type=Path,nargs='+')
    parser.add_argument('--preview-only',action='store_true')
    args = parser.parse_args()
    for path in args.outputs:
        preview(path.resolve())
        if not args.preview_only:run(path.resolve())
