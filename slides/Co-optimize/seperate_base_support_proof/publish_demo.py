"""Publish two geometric obstructions for the same six-pose fresh result."""
from __future__ import annotations

import html
import json
from pathlib import Path
import sys

PROOF = Path(__file__).resolve().parent
sys.path.insert(0,str(PROOF))
import diagnose as D
C, np = D.C, D.np
from shapely import wkt
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist
import base_geometry as G

spec = D.importlib.util.spec_from_file_location('proof_presentation_render', D.COOPT/'step5.2/render.py')
R = D.importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)
LABEL = 'pose1+2+3+4+5+6'


def relative_link(path, base):
    import os
    return os.path.relpath(path,base)


def draw_mesh(renderer,data,base_value,index,path):
    row = data['rows'][index]
    value = G.transform(base_value,row['fixture_transform'])
    mesh = G.unpack(value)
    fixture = C.posed(data['fixture'],row['fixture_transform'])
    body = C.posed(data['body'],row['object_transform'])
    low, high = mesh.bounds
    low = low-np.array([.01,.01,.02])
    high = high+np.array([.01,.01,.001])
    high[2] = 0.
    below = G.md.Manifold.cube(tuple((high-low)*1000)).translate(tuple(low*1000))
    collision = value ^ below
    collision_mesh = G.unpack(collision)
    above_mesh = G.unpack(value-below)
    joined = np.vstack([mesh.vertices,body.vertices,fixture.vertices,row['hull']])
    camera = C.V.camera_for(joined)
    parts = [C.V.floor_piece(joined,opacity=.18),
             C.V.piece(fixture,C.V.BLUE,.72),C.V.piece(body,C.V.GRAY,.28),
             C.V.piece(above_mesh,'#df9b3e')]
    if len(collision_mesh.faces):
        parts.append(C.V.piece(collision_mesh,'#dc344e'))
    parts.append(C.H.outline_piece(row['hull'],'#e97c29',camera,1000))
    renderer.render([C.V.CPU.placed(piece) for piece in parts],camera,1000).save(path)
    return dict(pose=row['pose'],base_minimum_z_mm=float(mesh.vertices[:,2].min()*1000),
                base_below_ground_cm3=G.volume(collision)*1e6,
                nominal_step4_support_minimum_z_mm=float(fixture.vertices[:,2].min()*1000))


def ground_plot(data,certificate,path):
    row = data['rows'][certificate['task_index']]
    hull = row['hull'][:,:2]
    own = row['object_ground']
    low, high = np.vstack([hull,own]).min(0)-.02,np.vstack([hull,own]).max(0)+.02
    rectangle = np.array([[*low],[high[0],low[1]],[*high],[low[0],high[1]]])
    a = np.asarray(certificate['inequality_a_xy'])
    b = certificate['inequality_offset_m']-certificate['attainable_lower_bound_m']
    legal = D.growth.clip_polygon(rectangle,a,b)
    own_hull = D.geometry_vertices(D.MultiPoint(own).convex_hull)
    R.polygon_image([(rectangle,'#fff4f3','#d7dddf'),(legal,'#e2f3eb','#31976a'),
                     (hull,None,'#de7832'),(own_hull,None,'#31866b')],
        points=[([certificate['demand_xy_m']],'#d02e46')],line=(a,b)).save(path)


def work_plot(data,certificate,out,path):
    pose = certificate['pose']
    row = next(r for r in data['rows'] if r['pose']==pose)
    free = wkt.loads((out/f'{pose}_free.wkt').read_text())
    work = wkt.loads((out/f'{pose}_work.wkt').read_text())
    attainable = np.asarray(certificate['attainable_convex_hull_xy_m'])
    hull = row['hull'][:,:2]
    points = np.vstack([attainable,hull])
    low, high = points.min(0)-.02,points.max(0)+.02
    view = Polygon([[*low],[high[0],low[1]],[*high],[low[0],high[1]]])
    forbidden = work.intersection(view)
    pieces = [forbidden] if forbidden.geom_type=='Polygon' else list(forbidden.geoms)
    polygons = [(np.asarray(p.exterior.coords)[:-1],'#f7dfe3','#d899a3')
                for p in pieces if p.geom_type=='Polygon']
    polygons += [(attainable,'#e0f0e7','#309268'),(hull,None,'#de7832')]
    sep = certificate['separator']
    R.polygon_image(polygons,points=[([sep['demand_xy_m']],'#c72742')],
        line=(np.asarray(sep['normal_xy']),sep['offset_m'])).save(path)


def main():
    out = PROOF/'demo'
    out.mkdir(exist_ok=False)
    snapshot_dir = out/'execution'
    snapshot_dir.mkdir()
    code = C.freeze(snapshot_dir)
    inputs = C.Inputs('B',D.BASE/LABEL/'data/selection.json')
    inputs.base = D.BASE
    case = inputs.cases[0]
    data = inputs.load(case)
    certificate_path = PROOF/'certificates'/LABEL/'report.json'
    cert = json.loads(certificate_path.read_text())
    floor = cert['ground_certificates'][0]
    work = max((r for r in cert['work_per_pose'] if r['unbounded_proof_found'] and not r['passed']),
               key=lambda r:r['separator']['separation_gap_m'])
    row = data['rows'][floor['task_index']]
    inner = Polygon(row['hull'][:,:2])
    outer = inner.buffer(.002,join_style=2)
    contours = [np.asarray(outer.exterior.coords)[:-1]*1000,
                np.asarray(inner.exterior.coords)[:-1]*1000]
    ring = G.md.CrossSection(contours,G.md.FillRule.EvenOdd).extrude(3.)
    base = G.transform(ring,np.linalg.inv(row['fixture_transform']))
    # One short connector fixes the illustrated ring to the saved support.
    top = G.unpack(base).vertices
    world = C.F.transform_points(top,row['fixture_transform'])
    top = top[world[:,2]>.0029]
    support = data['fixture'].vertices
    support = support[C.F.transform_points(support,row['fixture_transform'])[:,2]>.003]
    a,b = np.unravel_index(np.argmin(cdist(top,support)),(len(top),len(support)))
    corners = np.array([[x,y,z] for x in [-.00075,.00075] for y in [-.00075,.00075] for z in [-.00075,.00075]])
    connector = G.solid(C.trimesh.convex.convex_hull(np.vstack([top[a]+corners,support[b]+corners])))
    base = base+connector
    G.unpack(base).export(out/'illustrative_base_common.obj')
    data['fixture'].export(out/'saved_step4_support.obj')
    renderer = C.V.Renderer()
    before = draw_mesh(renderer,data,base,floor['task_index'],out/'ground_before.png')
    after = draw_mesh(renderer,data,base,floor['other_index'],out/'ground_after.png')
    if before['base_minimum_z_mm'] < -1e-6 or after['base_minimum_z_mm'] >= -1e-6:
        raise RuntimeError('The illustrated base must clear the first floor and penetrate the second')
    ground_plot(data,floor,out/'ground_certificate.png')
    work_plot(data,work,certificate_path.parent,out/'work_certificate.png')
    body_points = np.vstack([C.F.transform_points(data['body'].vertices,r['object_transform']) for r in data['rows']])
    body_bounds = np.array([body_points.min(0),body_points.max(0)])
    record = dict(id=LABEL,poses=data['record']['poses'],pose_count=6,
        step4_passed=True,original_demands_per_pose=32768,total_original_demands=6*32768,
        selected_fresh_step4=case['experiment'],ground_certificate=floor,work_certificate=work,
        illustrated_base_before=before,illustrated_base_after=after,
        illustrated_base_scope='one pose ground-covering ring and connector, fixed to fixture; not a feasible all-pose base',
        illustrative_base_height_mm=3.,illustrative_base_rim_width_mm=2.,
        base_material_volume_cm3=G.volume(base)*1e6,
        work_collisions_allowed_only_in_ground_illustration=True,
        transfer_scope='different installed states; empty fixture can be lifted during transfer',
        floor_proof_bounding_box_independent=True,work_proof_bounding_box_independent=True,
        body_only_process_bounding_box_cm3=C.volume(body_bounds)*1e6,
        step4_process_bounding_box_cm3=C.volume(data['bounds'])*1e6,
        saved_step4_support_volume_cm3=case['volume_cm3'],
        feasible_base_bounding_box_cm3=None,
        proof_scope='fixed saved layouts; does not reject other jointly optimized placements',
        source_certificate=C.relative(certificate_path),native_and_step4_inputs_unchanged=True)
    C.save(out/'report.json',record)
    inputs.verify()
    for p,h in code.items():
        assert C.digest(C.ROOT/p)==h and C.digest(snapshot_dir/'executed_sources'/p)==h,p
    C.save(snapshot_dir/'verification.json',dict(code=code,inputs=inputs.protected,
             source_snapshots_match=True,protected_inputs_unchanged=True))
    floor_gap = floor['separation_gap_m']*1000
    work_gap = work['separator']['separation_gap_m']*1000
    certificate_file = relative_link(certificate_path,out)
    step51 = relative_link(D.BASE/LABEL/'step5/step5.1/index.html',out)
    step52 = relative_link(D.BASE/LABEL/'step5/step5.2/index.html',out)
    sheet = relative_link(D.BASE/LABEL/'step5/step5.1/floor_demands.png',out)
    text = f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>共享底座与支撑分开优化的两个失败：同一组六个 pose</title>
<style>body{{font:16px/1.65 system-ui;max-width:1100px;margin:28px auto;padding:0 20px;color:#243e4b}}
img{{max-width:100%}}.pair{{display:flex;gap:12px}}.pair>div{{width:50%}}code{{background:#eef2f3}}a{{color:#1674aa}}
table{{border-collapse:collapse}}td,th{{padding:8px 16px;border:1px solid #d8e1e5}}h2{{margin-top:38px}}</style>
<h1>同一组 6 pose，两个独立的底座失败</h1>
<p><strong>{LABEL}</strong>。本轮重新从原 Step4.1 冷启动；Step4 的六个 pose 各自通过全部 32,768 个原始需求，共 196,608 个。
然后直接运行原 Step5.1、5.2。以下诊断没有改变支撑、落座布局、需求或正式算法。</p>
<p>Step4 的 PASS 是搜索接触上的承载通过。它尚未要求共享实体底座完成接地覆盖、避让工作锥和所有落座状态的地面。</p>
<p><a href="{step51}">Step5.1：所有原始点和最小凸包</a> · <a href="{step52}">Step5.2：过程包围盒</a>
 · <a href="{certificate_file}">数值证书</a> · <a href="report.json">示例数据</a></p>
<img src="{sheet}" alt="六个 pose 的全部需求点和凸包">
<h2>错误 1：Pose 4 的底座，换到 Pose 5 后穿过地面</h2>
<p>图中用 2 mm 宽、3 mm 高的外圈和一条短连接，包住 Pose 4 的需求凸包；底座在 Pose 4 中最低高度为 {before['base_minimum_z_mm']:.6f} mm。
随同一支撑换到 Pose 5 的安装状态后，底座最低高度变成 {after['base_minimum_z_mm']:.3f} mm。红色部分为底座穿过地面的实体。</p>
<div class="pair"><div><p>Pose 4：底座接地</p><img src="ground_before.png"></div>
<div><p>Pose 5：同一底座穿地</p><img src="ground_after.png"></div></div>
<p>上图是直观失败示例。下面的证书更强：不要求外圈、不限制底座尺寸，也包含物体自身的接地点。</p>
<img src="ground_certificate.png" alt="地面约束分离直线">
<p>绿色表示所有合法系统接地点必须满足的半平面；橙色是 Pose 4 的需求凸包；红叉是无法覆盖的需求顶点。</p>
<p>Pose 4 的底座接地点若在 Pose 5 不穿地，就必须满足（x、y 单位为米）：</p>
<p><code>0.839137082 x − 0.232019450 y + 0.034419200 ≥ 0</code>。</p>
<p>物体自身的接地点也都在这一半平面。任何数量的接地点，其凸包仍在这一半平面。
但是一个原始需求顶点的左端值为 −{floor_gap:.6f} mm。
因此这个固定布局无论换什么底座形状或扩大多少，都无法同时满足接地覆盖和所有落座地面。</p>
<p>这是最终落座状态的冲突。换姿态时先移走物体，并允许抬起空支撑，不会消除落座后的矛盾。
现有 Step4 名义支撑本身也有部分穿地；数据中单独记录了它与新增底座的高度，避免混在一起。</p>
<h2>错误 2：即使忽略其他地面，工作锥也阻止 Pose 1 的覆盖</h2>
<img src="work_certificate.png" alt="工作锥与接地覆盖的分离证书">
<p>红色为六个 pose 工作锥在 Pose 1 地面上的禁止区域；绿色为剩余全部可用接地点和物体自身接地点的凸包。
橙色是需求凸包；黑线把一个需求顶点与绿色凸包分开，缺口为 <strong>{work_gap:.6f} mm</strong>。</p>
<p>这里故意允许接地材料不连续、数量无限、尺寸任意；忽略连接、实体厚度、其他 pose 的地面、物体碰撞和退出路径。
即使在这些放宽条件下，全部可用接地点的凸包仍不能覆盖需求。</p>
<p>这不是有限包围盒的搜索失败。数值证书给出完整的方向圆覆盖：以 Pose 1 地面坐标原点为圆心，
半径 {work['radial_certificate']['radius_m']*1000:.0f} mm 之外的每个点都在至少一个工作锥内。
因此圆内计算已经包含所有可能的自由接地点。证明使用原始完整工作三角形和 30° 半角，
采用严格内接的 32 边锥，并向内缩 0.1 μm；它比真实圆锥禁止的空间更少，拒绝结论也对真实圆锥成立。</p>
<p>所有允许的系统接地点满足：<code>0.656740739 x − 0.754116438 y − 0.041430204 ≤ 0</code>，
而所示原始需求顶点的值是 +{work_gap:.6f} mm。</p>
<h2>问题应该反馈到哪里</h2>
<p>这组示例保留原来的目标：系统接地点的凸包覆盖需求，材料可以稀疏分布；不要求铺满凸包或做完整外圈。
失败发生在当前 Step4 已经选定的共享布局。将底座可行性反馈给 Step4，重新选择相对布局或高度，仍可能找到结果。
这些证书没有否定同一组 pose 的其他布局。</p>
<table><tr><th>量</th><th>结果</th></tr>
<tr><td>无支撑的物体过程 BBX</td><td>{record['body_only_process_bounding_box_cm3']:.3f} cm³</td></tr>
<tr><td>Step4 结尾的过程 BBX</td><td>{record['step4_process_bounding_box_cm3']:.3f} cm³</td></tr>
<tr><td>Step4 名义支撑材料</td><td>{record['saved_step4_support_volume_cm3']:.3f} cm³</td></tr>
<tr><td>满足全部条件的底座过程 BBX</td><td>当前固定布局无可行值</td></tr></table>
<p><a href="../index.html">返回本轮全部实验</a></p></html>'''
    (out/'index.html').write_text(text)
    print('DEMO COMPLETE',LABEL,'ground gap',floor_gap,'mm','work gap',work_gap,'mm',flush=True)


if __name__=='__main__':
    main()
