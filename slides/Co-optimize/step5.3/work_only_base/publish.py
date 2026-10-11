"""Read saved Step5.4 candidates and render actual contact-hull diagnostics.

No Boolean construction, new demands or force solve. Fresh output only.
"""
from pathlib import Path
import csv
import html
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
COOPT = HERE.parents[1]
sys.path.insert(0, str(COOPT/'step5.2'))
import stage5_common as C
from scipy.spatial import ConvexHull

spec = importlib.util.spec_from_file_location('work_only_floor_diagnostic_render', COOPT/'step5.2/render.py')
R = importlib.util.module_from_spec(spec); sys.modules[spec.name] = R; spec.loader.exec_module(R)
STAGE = 'step5.4/work_only_v2_views'


def main():
    args, inputs, cases, directory = C.arguments(STAGE, 'step54_work_only_v2_views_v1')
    numerical = inputs.base/'data/step53_work_only_then_z_v2'
    summary_file = numerical/'summary.json'; summary = json.loads(summary_file.read_text())
    if not summary['complete']:
        raise ValueError('Numerical batch must finish before presentation')
    inputs.protected[C.relative(summary_file)] = C.digest(summary_file)
    prepared = []
    for report in summary['cases']:
        src = inputs.base/report['group']/'step5/step5.4/work_only_v2'
        report_file = src/'report.json'; inputs.protected[C.relative(report_file)] = C.digest(report_file)
        trial = report['selected_candidate']
        if trial is None:
            prepared.append((report, None, [])); continue
        mesh_file = src/('support_with_base.obj' if report['passed'] else 'failed_candidate.obj')
        inputs.protected[C.relative(mesh_file)] = C.digest(mesh_file)
        mesh = C.trimesh.load(mesh_file, force='mesh', process=False)
        rows = []
        for old in trial['step54']['updated_ground_coverage']['per_pose']:
            path = src/'demands'/f"{old['pose']}.npz"; inputs.protected[C.relative(path)] = C.digest(path)
            with C.np.load(path, allow_pickle=False) as saved:
                fixture = saved['T_world_fixture'].copy(); obj = saved['T_world_object'].copy()
                target = saved['floor_demands_world_m'][saved['hull_indices'], :2].copy()
            world = C.F.transform_points(mesh.vertices, fixture)
            ground = world[C.np.abs(world[:, 2]) <= 1e-9, :2]
            object_world = C.F.transform_points(inputs.mesh.vertices, obj)
            own = object_world[C.np.abs(object_world[:, 2]) <= 1e-9, :2]
            points = C.np.vstack([ground, own])
            available = C.np.empty((0, 2))
            if len(points) >= 3 and C.np.linalg.matrix_rank(points-points.mean(0), tol=1e-10) == 2:
                available = points[ConvexHull(points).vertices]
            rows.append(dict(pose=old['pose'], passed=old['passed'], demand=target,
                             available=available, contacts=points))
        prepared.append((report, mesh, rows))
    code = C.begin(directory, STAGE, inputs, cases)
    sections = []; table = []; volumes = []; count = 0
    for index, (report, mesh, rows) in enumerate(prepared, 1):
        out = inputs.base/report['group']/'step5'/STAGE; out.mkdir(parents=True, exist_ok=False)
        tiles = []; prefix = f"../../{report['group']}/step5/{STAGE}"
        for row in rows:
            polygons = [(row['available'], '#d9eadc', '#25743a'), (row['demand'], None, '#a42e2e')]
            points = [(row['contacts'], '#25743a')] if not len(row['available']) else []
            R.polygon_image(polygons, points=points, size=640).save(out/f"{row['pose']}.png")
            status = '通过' if row['passed'] else '覆盖不足'
            tiles.append(f"<figure><img src='{prefix}/{row['pose']}.png'><figcaption>{row['pose']}：{status}</figcaption></figure>")
            count += 1
        original = f"../../{report['group']}/step5/step5.4/work_only_v2"
        sections.append(f"<section><h2>{index}. {html.escape(report['group'])}</h2>"
            f"<p><a href='{original}/report.json'>检查结果及全部尝试</a></p><div class='grid'>{''.join(tiles)}</div></section>")
        accepted = report['accepted_step54_bbx']; base_volume = f"{accepted['volume_cm3']:.1f}" if accepted else '未找到'
        table.append(f"<tr><td>{index}</td><td>{sum(r['passed'] for r in rows)}/{report['pose_count']}</td>"
            f"<td>{report['object_only_bbx']['volume_cm3']:.1f}</td><td>{report['step4_bbx']['volume_cm3']:.1f}</td><td>{base_volume}</td></tr>")
        volumes.append(dict(group=report['group'], object_only_bbx_cm3=report['object_only_bbx']['volume_cm3'],
            step4_bbx_cm3=report['step4_bbx']['volume_cm3'], accepted_base_bbx_cm3=accepted['volume_cm3'] if accepted else ''))
    with (directory/'volumes_cm3.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(volumes[0])); writer.writeheader(); writer.writerows(volumes)
    document = f"""<!doctype html><meta charset='utf-8'><title>Step5.3 / Step5.4 work-only results</title>
<style>body{{font:16px system-ui;max-width:1300px;margin:30px auto}}td,th{{padding:8px;border:1px solid #ccc}}
table{{border-collapse:collapse}}.grid{{display:grid;grid-template-columns:repeat(3,1fr)}}figure{{margin:10px}}img{{max-width:100%}}</style>
<h1>允许底座碰退出路径：实际接地结果</h1><p>Step5.3 始终排除全部工作锥和静态物体内部，退出路径允许相交。
Step5.4 按各任务世界 +Z 同时上移工件和底座，保留相对位置；上移后更新全部原始需求。
这轮 {summary['base_passed_count']}/{summary['group_count']} 组找到可用底座。有限候选失败不等于全局无解。</p>
<p>红线：上移后需求凸包；绿色区域：所示候选的真实接地点凸包，包含确实着地的工件。
绿色只有点时没有足够面积；大块材料的投影不能替代接地。这些均是失败候选，不能当作成功底座。</p>
<p><a href='../step53_work_only_then_z_v2/report.html'>算法、位移与详细数值报告</a> ·
<a href='../step53_work_only_then_z_v2/movements_mm.csv'>逐 pose 位移</a> · <a href='volumes_cm3.csv'>BBX 体积</a></p>
<table><tr><th>组</th><th>最佳候选接地覆盖</th><th>无支撑 BBX cm³</th><th>Step4 BBX cm³</th><th>可用底座 BBX cm³</th></tr>{''.join(table)}</table>
<p>无支撑体积保留 Step4 的工件位置；过程运输空间与工作状态 BBX 分开。候选用0.1 mm工作锥避让余量。</p>{''.join(sections)}"""
    (directory/'report.html').write_text(document)
    # A new gallery name, preserving all previous index files.
    gallery = inputs.base/'step5.3_work_only_then_z_index.html'
    if gallery.exists():
        raise ValueError('Existing gallery is protected')
    gallery.write_text("<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='0; url=data/step54_work_only_v2_views_v1/report.html'><a href='data/step54_work_only_v2_views_v1/report.html'>Step5.3 / Step5.4</a>")
    C.finish(directory, inputs, code, dict(group_count=len(prepared), rendered_ground_views=count,
        numerical_batch=C.relative(summary_file), demand_regenerated=False, force_solver_run=False,
        geometry_constructed=False))
    print('PRESENTATION COMPLETE', count, flush=True)


if __name__ == '__main__':
    main()
