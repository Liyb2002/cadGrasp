"""Measure and render the fixed current process bounding box."""
from __future__ import annotations

import html
import importlib.util
from pathlib import Path
import sys
import time

import stage5_common as C
_spec=importlib.util.spec_from_file_location('coopt_step52_render',Path(__file__).with_name('render.py'))
R=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=R
_spec.loader.exec_module(R)


def main():
    args, inputs, cases, directory = C.arguments('step5.2', 'step52_occupied_volume_v2')
    code = C.begin(directory, 'step5.2', inputs, cases)
    renderer = C.V.Renderer()
    rows = []
    for case in cases:
        started = time.perf_counter()
        data = inputs.load(case)
        out = inputs.base/case['id']/'step5/step5.2'
        out.mkdir(parents=True, exist_ok=False)
        camera = R.envelope(renderer, data, out/'occupied_volume.png')
        bounds = data['bounds']; extent = bounds[1]-bounds[0]
        record = dict(complete=True, object=inputs.name, id=case['id'], pose_count=len(data['rows']),
            bounding_box_min_m=bounds[0].tolist(), bounding_box_max_m=bounds[1].tolist(),
            bounding_box_dimensions_mm=(extent*1000).tolist(), bounding_box_volume_cm3=C.volume(bounds)*1e6,
            geometry='union of existing object and fixture vertices in every saved task-world placement',
            coordinate_frame='common task-world Cartesian frame', reference_envelope_fixed=True,
            extra_allowance_m=0, work_or_exit_sweeps_included=False, tool_or_robot_motion_included=False,
            lower_bound_scope='adding geometry while preserving all saved placements and existing material',
            source_step51=C.relative(inputs.base/case['id']/'step5/step5.1/data/report.json'),
            source_step4=C.relative(data['selected']),
            camera=dict(focus_m=camera[0].tolist(),basis=camera[1].tolist(),span_m=camera[2]),
            seconds=time.perf_counter()-started, upstream_files_changed=False)
        C.save(out/'data/report.json', record)
        dims=' × '.join(f'{v:.2f}' for v in record['bounding_box_dimensions_mm'])
        (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>{html.escape(case["id"])} · Step5.2</h1>'
            f'<p>Current task-world envelope: {dims} mm; {record["bounding_box_volume_cm3"]:.2f} cm³. '
            'All saved object and fixture states; the box remains fixed during base growth.</p>'
            '<img style="max-width:100%" src="occupied_volume.png">\n')
        rows.append(record)
        print('STEP5.2 BOX',case['id'],round(record['bounding_box_volume_cm3'],3),'cm3',flush=True)
    C.finish(directory, inputs, code, dict(step='step5.2',cases=rows,group_count=len(rows)))
    cards=''.join(f'<section><h2>{html.escape(r["id"])}</h2><p>{r["bounding_box_volume_cm3"]:.2f} cm³</p>'
        f'<img src="{r["id"]}/step5/step5.2/occupied_volume.png"></section>' for r in rows)
    gallery=inputs.base/'step5.2_index.html'
    gallery.write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1200px;'
        'margin:24px auto}img{width:100%}section{margin:30px 0}</style><h1>Step5.2 · Current occupied volume</h1>'+cards+'\n')
    print('GALLERY',C.relative(gallery),flush=True)


if __name__ == '__main__':
    main()
