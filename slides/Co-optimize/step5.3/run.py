"""Grow from saved supports, or emit a rigorous fixed-layout floor conflict."""
from __future__ import annotations

import html
import importlib.util
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'step5.2'))
import stage5_common as C
sys.path.insert(0,str(HERE.parent/'step5.2'))
_spec=importlib.util.spec_from_file_location('coopt_step52_render',HERE.parent/'step5.2/render.py')
R=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=R
_spec.loader.exec_module(R)
sys.path.insert(0,str(HERE))
import growth as G
from access import AddedMaterialAccess


def diagnostics(data, checked, renderer, out):
    before=R.reference(renderer,data,out/'existing_support_and_demands.png')
    cert=checked['fixed_layout_floor_certificates'][0] if checked['fixed_layout_floor_certificates'] else None
    k=cert['task_index'] if cert else int(C.np.argmin(checked['original_support_minimum_floor_height_m']))
    row=data['rows'][k]
    region=G.floor_region(data['states'],k,data['bounds'])
    own=G.polygon(row['object_ground'])
    available=G.polygon(C.np.vstack([region,own]))
    required=row['hull'][:,:2]
    polygons=[(available,'#f4f0df','#b89a43'),(region,'#e6e8ca','#8d9637'),
              (own,'#dce9ef','#517c93'),(required,None,'#c34656')]
    marked=([(C.np.asarray([cert['demand_xy_m']]),'#d92842')] if cert else [])
    line=(C.np.asarray(cert['inequality_a_xy']),cert['inequality_offset_m']-cert['attainable_lower_bound_m']) if cert else None
    image=R.polygon_image(polygons,marked,line)
    image.save(out/'floor_compatibility.png')
    C.V.sheet([before,image],out/'construction_process.png')
    return dict(shown_task=row['pose'],diagnostic='maximum floor-compatible ground coverage and separating demand',
                other_floor_task=cert['other_task'] if cert else None,
                maximum_floor_compatible_region_world_xy_m=region.tolist(),
                maximum_system_pressure_center_region_within_bbx_xy_m=available.tolist())


def main():
    args,inputs,cases,directory=C.arguments('step5.3','step53_deterministic_growth_v1')
    code=C.begin(directory,'step5.3',inputs,cases)
    renderer=C.V.Renderer()
    rows=[]
    for case in cases:
        started=time.perf_counter()
        data=inputs.load(case)
        box_report=inputs.base/case['id']/'step5/step5.2/data/report.json'
        if not box_report.is_file(): raise ValueError('Step5.2 must provide the frozen reference BBX first')
        inputs.protected[C.relative(box_report)]=C.digest(box_report)
        saved=C.np.asarray(C.json.loads(box_report.read_text())['bounding_box_min_m'])
        saved_high=C.np.asarray(C.json.loads(box_report.read_text())['bounding_box_max_m'])
        C.np.testing.assert_array_equal(data['bounds'],C.np.array([saved,saved_high]))
        out=inputs.base/case['id']/'step5/step5.3'
        out.mkdir(parents=True,exist_ok=False)
        checked=G.preflight(data)
        if checked['fixed_layout_floor_certificates']:
            result=dict(passed=False,status='fixed_layout_floor_hull_incompatible',base_constructed=False,
                        bounding_box_expansion_cannot_remove_this_conflict=True,growth_attempted=False)
            pictures=diagnostics(data,checked,renderer,out)
        elif checked['original_support_penetrates_a_floor']:
            result=dict(passed=False,status='original_nominal_mesh_penetrates_floor',base_constructed=False,
                        growth_attempted=False)
            pictures=diagnostics(data,checked,renderer,out)
        else:
            access=AddedMaterialAccess(data,inputs)
            outcome=G.Grow(data['fixture'],data['rows'],data['bounds'],legal=access).run()
            mesh,partial=outcome.pop('mesh'),outcome.pop('partial')
            result=dict(outcome,base_constructed=outcome['passed'],growth_attempted=True)
            before=R.reference(renderer,data,out/'existing_support_and_demands.png')
            middle=R.reference(renderer,data,out/'partial_growth.png',fixture=partial)
            final=R.reference(renderer,data,out/'grown_support.png',fixture=mesh)
            C.V.sheet([before,middle,final],out/'construction_process.png')
            if outcome['passed']: mesh.export(out/'support_with_base.obj')
            pictures=dict(shown_task='all tasks in their common fixture frame')
        report=dict(complete=True,object=inputs.name,id=case['id'],pose_count=len(data['rows']),
            step='step5.3',reference_bounds_m=data['bounds'].tolist(),reference_volume_cm3=C.volume(data['bounds'])*1e6,
            preserve_previous_step4_pass=True,step4_or_step51_recomputed=False,
            original_inputs_modified=False,full_ground_force_equilibrium_solved=False,
            geometric_ground_pressure_coverage_scope=True,checks=checked,pictures=pictures,
            result=result,seconds=time.perf_counter()-started)
        C.save(out/'data/report.json',report)
        if result['passed']:
            description='Deterministic geometric base growth completed; original material retained.'
        else:
            description=('Fixed current layout cannot supply its required ground convex hull. '
                'No valid new base is claimed. Enlarging the BBX does not remove the recorded separating-floor conflict.'
                if checked['fixed_layout_floor_certificates'] else 'The existing nominal mesh penetrates a task floor; adding material cannot correct it.')
        if checked['fixed_layout_floor_certificates']:
            worst=checked['fixed_layout_floor_certificates'][0]
            caption=(f'Left: current support and demand hulls. Right: task {worst["task"]}; '
                f'its limiting floor is task {worst["other_task"]}. The maximum floor-compatible ground region '
                'inside the BBX is olive, object-ground contacts are blue, and their combined envelope is cream. '
                'The required hull is red and the separating boundary is dark. The cross marks an impossible demand vertex.')
        elif result['passed'] or result.get('growth_attempted'):
            caption='Left to right: current support, first growth action, and the final construction candidate.'
        else:
            caption='Current support and demand hulls, alongside the floor-compatible ground region in the BBX.'
        (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>{html.escape(case["id"])} · Step5.3</h1>'
            f'<p>{description}</p><p>{caption}</p>'
            '<img style="max-width:100%" src="construction_process.png">\n')
        rows.append(dict(id=case['id'],pose_count=len(data['rows']),passed=result['passed'],status=result['status'],
            floor_certificate_count=len(checked['fixed_layout_floor_certificates']),
            worst_separation_gap_mm=(checked['fixed_layout_floor_certificates'][0]['separation_gap_m']*1000
                                     if checked['fixed_layout_floor_certificates'] else 0.),
            worst_task=(checked['fixed_layout_floor_certificates'][0]['task']
                        if checked['fixed_layout_floor_certificates'] else None),
            limiting_floor_task=(checked['fixed_layout_floor_certificates'][0]['other_task']
                                 if checked['fixed_layout_floor_certificates'] else None),
            seconds=report['seconds'],report=C.relative(out/'data/report.json')))
        print('STEP5.3',case['id'],result['status'],flush=True)
    C.finish(directory,inputs,code,dict(step='step5.3',cases=rows,group_count=len(rows),
        geometric_pass_count=sum(r['passed'] for r in rows),fixed_layout_incompatible_count=sum(r['floor_certificate_count']>0 for r in rows),
        original_step4_force_passes_unchanged=True,full_ground_force_equilibrium_solved=False))
    cards=''.join(f'<section><h2><a href="{r["id"]}/step5/step5.3/index.html">{html.escape(r["id"])}</a></h2><p>{r["status"]}; '
        f'task {r["worst_task"]}, limiting floor {r["limiting_floor_task"]}; '
        f'maximum separating-floor gap {r["worst_separation_gap_mm"]:.2f} mm</p>'
        f'<img src="{r["id"]}/step5/step5.3/construction_process.png"></section>' for r in rows)
    gallery=inputs.base/'step5.3_index.html'
    gallery.write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1700px;'
        'margin:24px auto}img{width:100%}section{margin:30px 0}</style><h1>Step5.3 · Deterministic growth and floor compatibility</h1>'
        '<p>Fixed current placements. Only actual ground contact contributes to the pressure-center hull. '
        'Rigorous floor conflicts are reported before adding material. Previous Step4 force PASS results remain unchanged.</p>'
        '<p>Left: current support and all demand hulls. Right: olive = maximum legal ground region inside the BBX; '
        'blue = object-ground contacts; cream = their combined envelope; red = required hull; '
        'dark line = separating boundary; red cross = impossible demand vertex. '
        'The legal region deliberately ignores object and access obstacles, so it is a generous upper bound.</p>'+cards+'\n')
    print('GALLERY',C.relative(gallery),flush=True)


if __name__ == '__main__':
    main()
