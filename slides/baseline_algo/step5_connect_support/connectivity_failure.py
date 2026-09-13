"""Show the actual Boolean boundaries behind a failed one-solid check."""
import argparse
from pathlib import Path
import sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.cases import selected_pose, pose_name
from step5_connect_support import belt_assembly as A, failure_visuals as F, visual_details as V
from step3_scheculer import contacts as I


def run(name):
    out = A.OUTPUTS/name/pose_name()/A.STAGE
    report = I.check_report(out/'connection.json')
    if report['geometry_constructed'] or not any(a.get('status') == 'loose_frame_union_disconnected'
                                                for a in report['geometry'].get('attempts', [])):
        return None
    source = out/'failed_attempt.npz'
    arrays = I.load_npz(source)
    domain, contacts, _, _, _, _, inputs = A.A.read_inputs(name)
    scale = float(domain.mesh.extents.max())
    joined, solid = A.B.union_parts(A.S.unpack_parts(arrays), scale)
    shells = list(joined.split(only_watertight=False))
    main = max(range(len(shells)), key=lambda i: shells[i].volume)
    extra = [shell for i, shell in enumerate(shells) if i != main and shell.volume >= 0]
    rows = [dict(index=i, signed_volume_mm3=float(s.volume)*1e9,
                 area_mm2=float(s.area)*1e6, bounds_mm=(s.bounds*1000).tolist(),
                 smallest_singular_length_mm=float(np.linalg.svd(s.vertices-s.vertices.mean(axis=0),
                                                                 compute_uv=False)[-1])*1000)
            for i, s in enumerate(shells)]
    copy = joined.copy(); copy.apply_scale(1000); copy.export(out/'connectivity_union_mm.stl')
    if extra:
        copy = trimesh.util.concatenate(extra); copy.apply_scale(1000)
        copy.export(out/'connectivity_extra_shells_mm.stl')
    view = V.camera(domain, [(dict(candidate_id='assembly'), arrays)])
    size = 790
    picture = V.scene(domain, contacts, [(dict(candidate_id='assembly'), arrays)], size, view=view, xray=True)
    for shell in extra:
        F.overlay_mesh(picture, shell, view, size, np.array([204., 55., 47.]))
    focus = max(extra, key=lambda s: s.area) if extra else shells[main]
    close = V.fit(focus.vertices, view[2], margin=2.2)
    detail = V.scene(domain, [], [(dict(candidate_id='assembly'), arrays)], size, view=close, xray=True)
    for shell in extra:
        F.overlay_mesh(detail, shell, close, size, np.array([204., 55., 47.]))
    page = Image.new('RGB', (1640, 1090), V.R.PAPER)
    page.paste(picture, (15, 100)); page.paste(detail, (835, 100))
    ink = ImageDraw.Draw(page)
    ink.text((22, 16), f'{name} / {pose_name()} / BOOLEAN CONNECTIVITY NOT VERIFIED', font=V.R.font(30), fill=V.R.INK)
    ink.text((22, 65), 'Actual attempted frame; red = extra nonnegative boundary shells', font=V.R.font(22), fill=V.MUTED)
    ink.text((840, 65), 'Close-up of the largest-area extra shell', font=V.R.font(22), fill=V.MUTED)
    largest = max((s.volume for s in extra), default=0.)*1e9
    lines = [f"Boundary shells: {len(shells)}; positive: {solid['positive_boundary_shells']}; cavities: {solid['cavity_count']}; zero: {solid['zero_boundary_shells']}",
             f'Largest extra positive signed volume: {largest:.6g} mm^3. No shells have been deleted.',
             'These counts describe Boolean boundary topology, not a measured macroscopic break in the frame.',
             'The retained head directions pass; one connected printed body has not been certified.']
    for i, line in enumerate(lines):
        ink.text((22, 910+39*i), line, font=V.R.font(23), fill='#a33336' if i in (1, 3) else V.R.INK)
    page.save(out/'connectivity_failure.png')
    files = ['connectivity_failure.png', 'connectivity_union_mm.stl']
    if extra: files.append('connectivity_extra_shells_mm.stl')
    result = dict(object=name, pose=pose_name(), complete=True, solid_check=solid, boundary_shells=rows,
                  largest_extra_positive_signed_volume_mm3=largest,
                  removed_shells=False, macroscopic_disconnection_proved=False,
                  scope='Diagnostic replay of the actual Boolean mesh; no material deletion, trajectory or acceptance change.',
                  provenance=dict(inputs=I.hashes(inputs+[out/'connection.json', source]),
                                  code=I.hashes([Path(__file__), Path(A.B.__file__), Path(A.S.__file__), Path(F.__file__), Path(V.__file__)])),
                  artifacts={file: I.sha256(out/file) for file in files})
    I.save(out/'connectivity_failure.json', result)
    print(name, pose_name(), 'connectivity diagnostic saved', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name'); parser.add_argument('--pose')
    args = parser.parse_args()
    with selected_pose(args.pose): run(args.name)
