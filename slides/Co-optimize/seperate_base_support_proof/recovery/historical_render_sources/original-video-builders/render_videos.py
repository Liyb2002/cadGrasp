"""Four narrated CAD demonstrations, rendered from saved Step4/5 geometry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import time

import prepare as Q
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

C, np, G = Q.C, Q.np, Q.G
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, MUTED = '#253b4b', '#607684'
BLUE, GRAY, RED, GREEN = '#2a91d2', '#a5adb5', '#dc344e', '#23865e'
ORANGE, PURPLE, TEAL = '#df8236', '#895eab', '#299584'
W, H = 1280, 720
FPS = 24


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def smooth(v):
    v = float(np.clip(v, 0., 1.))
    return v*v*(3-2*v)


def progress(t, start, end):
    return smooth((t-start)/(end-start))


def xyz(mesh, transform):
    return C.posed(mesh, transform)


def hull_piece(vertices, color, opacity=.07):
    vertices = np.asarray(vertices)
    faces = [[0, i, i+1] for i in range(1, len(vertices)-1)]
    return C.V.piece(C.trimesh.Trimesh(vertices, faces, process=False), color, opacity)


def line_piece(points, color, camera, size, fraction=1., pixels=1.5):
    points = np.asarray(points)
    ends = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(ends-points, axis=1)
    remaining = lengths.sum()*float(np.clip(fraction, 0, 1))
    segments = []
    for a, b, length in zip(points, ends, lengths):
        if remaining <= 1e-15:
            break
        segment = min(remaining, length)
        segments.append([a, a+(b-a)*segment/max(length, 1e-20)])
        remaining -= segment
    if not segments:
        return None
    segments = np.asarray(segments)
    vector = (segments[:, 1]-segments[:, 0])@camera[1][:2].T
    length = np.linalg.norm(vector, axis=1)
    side = np.c_[-vector[:, 1], vector[:, 0]]/np.maximum(length[:, None], 1e-20)
    offset = side@camera[1][:2]*(pixels*camera[2]/size)
    corners = np.stack([segments[:, 0]+offset, segments[:, 1]+offset,
                        segments[:, 1]-offset, segments[:, 0]-offset], axis=1)
    triangles = corners[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3, 3)
    return dict(triangles=triangles, normals=np.broadcast_to(camera[1][2], triangles.shape),
        colors=np.broadcast_to(C.V.CPU.rgb(color), triangles.shape), bias=8e-6, unlit=True)


def floor(points, camera=None):
    return C.V.floor_piece(points, color='#e5edf1', opacity=.24)


def ground_grid(points, view, size):
    xy = np.asarray(points)[:, :2]
    low, high = xy.min(0)-.007, xy.max(0)+.007
    segments = []
    for u in np.linspace(0., 1., 5):
        x, y = low+u*(high-low)
        segments += [np.array([[x, low[1], 0.], [x, high[1], 0.]]),
                     np.array([[low[0], y, 0.], [high[0], y, 0.]])]
    pieces = [line_piece(s, '#cbd8e0', view, size, pixels=.35) for s in segments]
    return dict(triangles=np.concatenate([p['triangles'] for p in pieces]),
        normals=np.concatenate([p['normals'] for p in pieces]),
        colors=np.concatenate([p['colors'] for p in pieces]), bias=1e-6, unlit=True)


def arrow_mesh(arrows, length=.080, radius=.00035, head=.0035, head_radius=.0011):
    vertices, faces = [], []
    for origin, direction, _ in arrows:
        if length < head*.7:
            continue
        direction = np.asarray(direction)
        tangent = Q.tangent_frame(direction)
        ring = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])@tangent.T/np.sqrt(2)
        origin = np.asarray(origin)
        end = origin+length*direction
        neck = end-head*direction
        v = np.vstack([origin+radius*ring, neck+radius*ring, neck+head_radius*ring, end])
        f = [[0, 2, 1], [0, 3, 2]]
        for i in range(4):
            j = (i+1) % 4
            f += [[i, j, 4+j], [i, 4+j, 4+i], [4+i, 4+j, 8+j], [4+i, 8+j, 8+i], [8+i, 8+j, 12]]
        shift = len(vertices)*13
        vertices.append(v)
        faces.extend(np.asarray(f)+shift)
    if not vertices:
        return C.trimesh.Trimesh()
    return C.trimesh.Trimesh(np.vstack(vertices), np.asarray(faces), process=False)


def point_mesh(point, radius=.002):
    sphere = C.trimesh.creation.icosphere(subdivisions=1, radius=radius)
    sphere.apply_translation(point)
    return sphere


def interpolate(a, b, fraction):
    rotations = Rotation.from_matrix(np.array([a[:3, :3], b[:3, :3]]))
    result = np.eye(4)
    result[:3, :3] = Slerp([0., 1.], rotations)(float(fraction)).as_matrix()
    result[:3, 3] = (1-fraction)*a[:3, 3]+fraction*b[:3, 3]
    return result


class Artist:
    def __init__(self):
        self.cpu = C.V.Renderer()
        self.cache = {}

    def draw(self, parts, camera, size):
        parts = [C.V.CPU.placed(part) for part in parts if part is not None and len(part['triangles'])]
        return self.cpu.render(parts, camera, size)

    def split_ground(self, value, transform):
        posed = G.transform(value, transform)
        top = posed.trim_by_plane((0., 0., 1.), 0.)
        below = posed-top
        return G.unpack(top), G.unpack(below)

    def view(self, data, row, camera, size, *, base=None, object_alpha=.16,
             support_alpha=.74, hull=True, extra_hulls=(), transform=None, object_transform=None,
             reveal=1., red_base=True, base_alpha=1.):
        transform = row['fixture_transform'] if transform is None else transform
        object_transform = row['object_transform'] if object_transform is None else object_transform
        fixture = xyz(data['fixture'], transform)
        body = xyz(data['body'], object_transform)
        reference = np.vstack([fixture.vertices, body.vertices, row['hull']])
        parts = [floor(reference), ground_grid(reference, camera, size)]
        if reveal > 0:
            mesh = fixture
            if reveal < 1:
                centers = fixture.triangles_center
                distances = np.linalg.norm(centers-body.vertices.mean(0), axis=1)
                cutoff = float(np.quantile(distances, reveal))
                mesh = C.trimesh.Trimesh(fixture.vertices, fixture.faces[distances <= cutoff], process=False)
            parts.append(C.V.piece(mesh, BLUE, support_alpha))
        if object_alpha > 0:
            parts.append(C.V.piece(body, GRAY, object_alpha))
        if base is not None:
            if red_base:
                above, below = self.split_ground(base, transform)
                parts += [C.V.piece(above, '#dfa44d', base_alpha), C.V.piece(below, RED, base_alpha)]
            else:
                parts.append(C.V.piece(xyz(G.unpack(base), transform), '#dfa44d'))
        if hull:
            parts += [hull_piece(row['hull'], row['color']), line_piece(row['hull'], row['color'], camera, size, pixels=2.)]
        for vertices, color, alpha in extra_hulls:
            if alpha:
                parts.append(hull_piece(vertices, color, alpha))
            parts.append(line_piece(vertices, color, camera, size))
        return self.draw(parts, camera, size)


def canvas(number, title, case, subtitle, detail=''):
    frame = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(frame)
    d.rounded_rectangle((28, 22, 82, 70), radius=13, fill=INK)
    d.text((42, 25), f'{number:02}', font=font(26, True), fill='white')
    d.text((99, 22), title, font=font(28, True), fill=INK)
    d.text((101, 58), case, font=font(17), fill=MUTED)
    d.line((28, 92, W-28, 92), fill='#e1e9ed', width=2)
    d.rounded_rectangle((28, 630, W-28, 703), radius=13, fill='#f0f5f8')
    d.text((47, 638), subtitle, font=font(23, True), fill=INK)
    if detail:
        d.text((47, 676), detail, font=font(15), fill=MUTED)
    return frame


def paste_tile(frame, image, x, y, width, height, label=None, color=INK):
    image = image.copy()
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    frame.paste(image, (x+(width-image.width)//2, y+(height-image.height)//2))
    d = ImageDraw.Draw(frame)
    if label:
        d.text((x+8, y+4), label, font=font(17, True), fill=color)


def polygon_chart(polygons, point=None, caption=None, line=None, size=(400, 360), limits=None):
    image = Image.new('RGB', size, 'white')
    d = ImageDraw.Draw(image)
    points = np.vstack([vertices for vertices, _, _ in polygons if len(vertices)])
    low, high = (points.min(0), points.max(0)) if limits is None else limits
    center = (low+high)/2
    span = max((high-low)[0], (high-low)[1], .001)*1.2
    scale = min(size[0]-35, size[1]-60)/span
    def screen(vertices):
        v = (np.asarray(vertices)-center)*scale
        return [(float(x+size[0]/2), float(size[1]/2+8-y)) for x, y in v]
    d.rounded_rectangle((3, 3, size[0]-3, size[1]-3), radius=12, fill='#fafcfd', outline='#e1e8ec')
    if caption:
        d.text((16, 12), caption, font=font(16, True), fill=INK)
    for vertices, fill_color, outline_color in polygons:
        if len(vertices) < 3:
            continue
        pixels = screen(vertices)
        if fill_color:
            d.polygon(pixels, fill=fill_color)
        if outline_color:
            d.line(pixels+[pixels[0]], fill=outline_color, width=3, joint='curve')
    if line is not None:
        a, b = line
        boundary = np.array([low, [high[0], low[1]], high, [low[0], high[1]]])
        intersections = []
        for first, last in zip(boundary, np.roll(boundary, -1, axis=0)):
            first_value, last_value = first@a+b, last@a+b
            if first_value*last_value <= 0 and abs(first_value-last_value) > 1e-12:
                intersections.append(first+(last-first)*first_value/(first_value-last_value))
        if len(intersections) >= 2:
            d.line(screen(intersections[:2]), fill=RED, width=2)
        if point is not None:
            foot = np.asarray(point)-(np.asarray(point)@a+b)*np.asarray(a)/(a@a)
            d.line(screen([point, foot]), fill=RED, width=3)
    if point is not None:
        x, y = screen([point])[0]
        d.ellipse((x-5, y-5, x+5, y+5), fill=RED, outline='white', width=1)
    return image


def camera(points, span_scale=1., sight=None):
    if sight is None:
        focus, basis, span = C.V.camera_for(points)
    else:
        sight = np.asarray(sight)/np.linalg.norm(sight)
        right = np.cross([0., 0., 1.], sight); right /= np.linalg.norm(right)
        basis = np.array([right, np.cross(sight, right), sight])
        projected = np.asarray(points)@basis.T
        low, high = projected.min(0), projected.max(0)
        focus = ((low+high)/2)@basis
        span = max(float(np.max(high[:2]-low[:2]))*1.16, .001)
    return focus, basis, span*span_scale


def labels(frame, rows, y=594):
    d = ImageDraw.Draw(frame)
    for i, row in enumerate(rows):
        x = 53+i*155
        d.rounded_rectangle((x, y+6, x+20, y+13), radius=3, fill=row['color'])
        d.text((x+28, y), row['pose'].replace('_', ' '), font=font(16), fill=INK)


def growth_video(art, data):
    reference = data['rows'][0]['fixture_transform']
    common_hulls = [Q.hull_common(row) for row in data['rows']]
    hulls = [C.F.transform_points(points, reference) for points in common_hulls]
    body_meshes = [xyz(data['body'], reference@np.linalg.inv(row['fixture_transform'])@row['object_transform']) for row in data['rows']]
    support = xyz(data['fixture'], reference)
    cam = camera(np.vstack([support.vertices, *hulls, *[body.vertices for body in body_meshes]]))
    cameras = [camera(np.vstack([C.F.transform_points(data['fixture'].vertices, row['fixture_transform']),
                C.F.transform_points(data['body'].vertices, row['object_transform']), row['hull']])) for row in data['rows']]
    distances = np.linalg.norm(support.triangles_center-body_meshes[0].vertices.mean(0), axis=1)
    side_cache = []
    for row, view in zip(data['rows'], cameras):
        side_cache.append(art.view(data, row, view, 370, reveal=0, hull=False))
    def frame(t):
        fractions = [progress(t, .35, 2.25), progress(t, 2.3, 4.2)]
        grow = progress(t, 4.4, 7.7)
        if t < 2.3:
            subtitle = '先画 Pose11 的最小凸包'
        elif t < 4.4:
            subtitle = '再画 Pose19 的最小凸包'
        elif t < 7.7:
            subtitle = '在同一个支撑坐标系中，长出共享支撑'
        else:
            subtitle = '两个接地需求凸包，对应同一件共享支撑'
        out = canvas(1, '两个凸包 → 共享支撑', 'pose11 + pose19', subtitle,
            '最小凸包来自各 pose 的全部 32,768 个原始需求；支撑生长为保存模型的显现示意。')
        parts = [C.V.piece(body, GRAY, .09) for body in body_meshes]
        for hull, row, fraction in zip(hulls, data['rows'], fractions):
            if fraction > 0:
                parts += [hull_piece(hull, row['color'], .06*fraction), line_piece(hull, row['color'], cam, 690, fraction, 2.)]
        if grow > 0:
            mesh = support if grow >= .9999 else C.trimesh.Trimesh(support.vertices, support.faces[distances <= np.quantile(distances, grow)], process=False)
            parts.append(C.V.piece(mesh, BLUE, .82))
        parts.append(floor(np.vstack(hulls)))
        paste_tile(out, art.draw(parts, cam, 690), 28, 103, 757, 496, '共享支撑坐标系')
        for i, (row, view, fraction) in enumerate(zip(data['rows'], cameras, fractions)):
            if grow == 0:
                image = side_cache[i].copy()
            else:
                image = art.view(data, row, view, 340, reveal=grow)
            if fraction > 0 and grow == 0:
                body = xyz(data['body'], row['object_transform'])
                parts = [floor(row['hull']), C.V.piece(body, GRAY, .16),
                         hull_piece(row['hull'], row['color'], .07*fraction), line_piece(row['hull'], row['color'], view, 340, fraction)]
                image = art.draw(parts, view, 340)
            paste_tile(out, image, 815, 105+i*242, 405, 234, row['pose'].replace('_', ' '), row['color'])
        labels(out, data['rows'])
        return out
    return frame, 10.


def ground_video(art, data, small, large, report):
    source, target = data['rows'][1], data['rows'][0]
    transforms = [interpolate(source['fixture_transform'], target['fixture_transform'], x) for x in np.linspace(0, 1, 33)]
    reference_points = [source['hull'], target['hull']]
    for t in transforms:
        reference_points += [C.F.transform_points(data['fixture'].vertices, t), C.F.transform_points(G.unpack(large).vertices, t)]
    reference_points += [C.F.transform_points(data['body'].vertices, row['object_transform']) for row in data['rows']]
    cam = camera(np.vstack(reference_points))
    cert = report['original_ground_separator']
    a, b = np.asarray(cert['inequality_a_xy']), cert['inequality_offset_m']
    points = np.vstack([source['hull'][:, :2], source['object_ground'], -b*a/(a@a)])
    low, high = points.min(0)-.012, points.max(0)+.012
    rectangle = np.array([low, [high[0], low[1]], high, [low[0], high[1]]])
    allowed = Q.D.growth.clip_polygon(rectangle, a, b)
    chart = polygon_chart([(rectangle, '#fbedef', '#e5e9ec'), (allowed, '#dff2e9', GREEN),
                          (source['hull'][:, :2], None, PURPLE)], point=cert['demand_xy_m'],
        caption='Pose19：接地平面上的合法区域', line=(a, b), limits=(low, high), size=(408, 365))
    def frame(t):
        second = t >= 8.
        local_t = t-8 if second else t
        base = large if second else small
        turn = progress(local_t, 2.6, 5.7)
        state = interpolate(source['fixture_transform'], target['fixture_transform'], turn)
        object_alpha = .16*(1-progress(local_t, 2., 2.6)) if local_t < 4 else .16*progress(local_t, 5.7, 6.3)
        row = source if local_t < 4 else target
        hull_common = Q.hull_common(source)
        moved_hull = C.F.transform_points(hull_common, state)
        plotted = dict(row, hull=moved_hull, color=PURPLE)
        points = C.F.transform_points(G.unpack(base).vertices, state)
        depth = max(0., -float(points[:, 2].min())*1000)
        if local_t < 2.6:
            subtitle = '扩大外圈：仍先包住 Pose19 的凸包' if second else '最小外圈：先包住 Pose19 的凸包'
        elif local_t < 5.7:
            subtitle = '取出物体，改变空支撑的摆放'
        else:
            subtitle = '扩大仍然撞地：固定布局下无法完成覆盖' if second else '落到 Pose11 状态后，底座穿过地面'
        out = canvas(2, '扩大底座仍然撞地', 'pose11 + pose19', subtitle,
            '红色是实际穿地的底座实体。转动路径为示意；无解证据来自最终落座状态的接地约束。')
        image = art.view(data, plotted, cam, 704, base=base, transform=state,
            object_transform=row['object_transform'], object_alpha=object_alpha)
        paste_tile(out, image, 28, 104, 764, 504, '扩大外圈 +24 mm' if second else '最小外圈 · 2 mm 宽', ORANGE)
        paste_tile(out, chart, 827, 113, 408, 365)
        d = ImageDraw.Draw(out)
        d.text((838, 492), f'实际穿地深度  {depth:5.2f} mm', font=font(22, True), fill=RED if depth > .01 else GREEN)
        d.text((838, 532), '绿色：底座可用接地点', font=font(17), fill=GREEN)
        d.text((838, 561), '紫色：必须覆盖的需求凸包', font=font(17), fill=PURPLE)
        d.text((838, 590), '分离证书：缺口 50.11 mm', font=font(17, True), fill=RED)
        return out
    return frame, 16.


def seating_video(art, data, solution, moved, base, report):
    lift_end = Q.lift(data, [.051, 0.])
    cameras = []
    for i in range(2):
        points = []
        for state in [data, lift_end, moved]:
            row = state['rows'][i]
            points += [C.F.transform_points(data['body'].vertices, row['object_transform']),
                       C.F.transform_points(data['fixture'].vertices, row['fixture_transform']), row['hull']]
        points.append(C.F.transform_points(G.unpack(base).vertices, moved['rows'][i]['fixture_transform']))
        cameras.append(camera(np.vstack(points), 1.04))
    def frame(t):
        if t < 6.6:
            height = .051*progress(t, .7, 5.3)
            current = Q.lift(data, [height, 0.])
            subtitle = '只抬 Pose11：凸包会变化，但另一个地面约束开始失败'
            detail = '只抬 Pose11 需要 ≥50.11 mm，同时又要求 ≤16.58 mm；不存在共同可行高度。'
        elif t < 9.:
            height = .051*(1-progress(t, 7.2, 8.8))
            current = Q.lift(data, [height, 0.])
            subtitle = '单独 +Z 无解；需要一起调整落座布局'
            detail = '两个 pose 都只做 +Z 也无解。下面加入横向调整，展示接地几何的一组可行结果。'
        else:
            fraction = progress(t, 9., 13.6)
            current = Q.blend_seating(data, solution, fraction)
            subtitle = '联合调整 XYZ：重新计算凸包，长出接地外圈' if t < 15.4 else '接地几何通过：两个 pose 的全部需求均被包围'
            detail = '这一段仅验证底座接地与凸包覆盖；新布局的承载、工作锥和退出路径尚未复核。'
        out = canvas(3, '抬升改变凸包；单独 +Z 不足', 'pose11 + pose19', subtitle, detail)
        values = Q.floor_values(current)
        for i, (row, cam) in enumerate(zip(current['rows'], cameras)):
            original = data['rows'][i]
            present_base = base if t >= 14.2 else None
            image = art.view(current, row, cam, 550, base=present_base,
                base_alpha=progress(t, 14.2, 15.4), extra_hulls=[(original['hull'], '#b5c2ca', 0.)])
            paste_tile(out, image, 31+i*618, 113, 592, 459, row['pose'].replace('_', ' '), row['color'])
            d = ImageDraw.Draw(out)
            delta = row['object_transform'][:3, 3]-original['object_transform'][:3, 3]
            y = 555
            d.text((52+i*618, y), f'物体位移  X {delta[0]*1000:+.1f} / Y {delta[1]*1000:+.1f} / Z {delta[2]*1000:+.1f} mm', font=font(17), fill=INK)
            gap = next(v['min_original_demand_height_mm'] for v in values if v['demand_pose'] == row['pose'])
            text = f'另一地面约束的最小余量  {gap:+.2f} mm'
            if t >= 15.4:
                text = '实际接地覆盖  32,768 / 32,768'
            d.text((52+i*618, 590), text, font=font(18, True), fill=GREEN if gap >= -1e-6 else RED)
        return out
    return frame, 18.


def work_video(art, data, work, report):
    target_row = work['target_row']
    reference = target_row['fixture_transform']
    objects = [xyz(data['body'], reference@np.linalg.inv(row['fixture_transform'])@row['object_transform']) for row in data['rows']]
    support = xyz(data['fixture'], reference)
    patches = []
    for triangles, color in work['patches']:
        points = C.F.transform_points(triangles.reshape(-1, 3), reference)
        patches.append((C.trimesh.Trimesh(points, np.arange(len(points)).reshape(-1, 3), process=False), color))
    forest = []
    for row in data['rows']:
        arrows = [a for a in work['arrows'] if a[2] == row['color']]
        mesh = arrow_mesh(arrows)
        forest.append((xyz(mesh, reference), row['color']))
    witness = work['witness']
    origin = np.asarray(witness['origin_m'])
    direction = np.asarray(witness['direction'])
    witness_length = witness['target_distance_m']+.045
    target = C.F.transform_points([witness['target_m']], reference)[0]
    hulls = [(C.F.transform_points(Q.hull_common(row), reference), row['color']) for row in data['rows']]
    points = np.vstack([support.vertices, *[m.vertices for m in objects],
                        *[v for v, _ in hulls], *[m.vertices for m, _ in forest],
                        C.F.transform_points(G.unpack(work['candidates'][-1][0]).vertices, reference)])
    cam = camera(points, .93, sight=[-1., 1., .8])
    p = work['proof']
    attainable = np.asarray(p['attainable_convex_hull_xy_m'])
    hull = target_row['hull'][:, :2]
    sep = p['separator']
    # Display true forbidden regions in the same close-up, plus the global certificate's hull.
    from shapely import wkt
    forbidden = wkt.loads((Q.PROOF/'certificates'/data['case']['id']/f'{target_row["pose"]}_work.wkt').read_text())
    points_xy = np.vstack([attainable, hull])
    low, high = points_xy.min(0)-.01, points_xy.max(0)+.01
    view = Q.Polygon([low, [high[0], low[1]], high, [low[0], high[1]]])
    forbidden = forbidden.intersection(view)
    regions = [forbidden] if forbidden.geom_type == 'Polygon' else list(forbidden.geoms)
    polygons = [(np.asarray(r.exterior.coords)[:-1], '#f8e0e4', None) for r in regions if r.geom_type == 'Polygon']
    polygons += [(attainable, '#dff1e8', GREEN), (hull, None, TEAL)]
    chart = polygon_chart(polygons, point=sep['demand_xy_m'], caption='Pose14：所有可用接地点的凸包',
        line=(np.asarray(sep['normal_xy']), sep['offset_m']), size=(406, 358), limits=(low, high))
    candidates = []
    for value, cut, _ in work['candidates']:
        candidates.append((xyz(G.unpack(value-cut), reference), xyz(G.unpack(cut), reference)))
    def frame(t):
        ray_progress = progress(t, 1., 2.1)
        hull_progress = progress(t, 3., 5.)
        dense = progress(t, 5.2, 7.)
        index = None if t < 8 else min(2, int((t-8)/2.3))
        if t < 1.:
            subtitle = '先画四个 pose 的原始工作面积'
        elif t < 3.:
            subtitle = '一秒后出现真实工作方向；它将穿过所需底座'
        elif t < 5.2:
            subtitle = '保留这根 3D 箭头，画出各 pose 的接地需求凸包'
        elif t < 8.:
            subtitle = '密集箭头表示完整工作锥的方向与工作面积'
        elif index == 0:
            subtitle = '最小包裹外圈占用工作锥：碰撞部分为红色'
        elif index == 1:
            subtitle = '扩大外圈，仍然占用工作锥'
        else:
            subtitle = '扩大并偏移仍冲突；固定布局下，任意底座都缺少可用接地范围'
        out = canvas(4, '工作锥阻断接地覆盖', 'pose1 + pose11 + pose14 + pose27', subtitle,
            '箭头是工作锥的可视化；无解来自任意范围、任意拓扑的分离证书，限定于当前固定布局。')
        parts = [floor(target_row['hull']), ground_grid(target_row['hull'], cam, 714), C.V.piece(support, BLUE, .60)]
        parts += [C.V.piece(m, GRAY, .07) for m in objects]
        for mesh, color in patches:
            patch = C.V.piece(mesh, color, .96)
            patch['bias'] = .00005
            parts.append(patch)
        if dense > 0:
            parts += [C.V.piece(m, color, .56*dense) for m, color in forest]
        if ray_progress > 0:
            main_arrow = xyz(arrow_mesh([(origin, direction, RED)], length=witness_length*ray_progress,
                radius=.0017, head=.013, head_radius=.0055), reference)
            if len(main_arrow.faces):
                parts.append(C.V.piece(main_arrow, '#f2b633'))
        if hull_progress > 0:
            for vertices, color in hulls:
                parts += [hull_piece(vertices, color, .045*hull_progress), line_piece(vertices, color, cam, 714, hull_progress, 1.7)]
        if index is not None:
            above, collision = candidates[index]
            parts += [C.V.piece(above, '#dfa44d'), C.V.piece(collision, RED)]
            parts.append(C.V.piece(point_mesh(target, .0025), RED))
        paste_tile(out, art.draw(parts, cam, 714), 22, 109, 772, 493, '共同支撑坐标系 · Pose14 落座视角')
        d = ImageDraw.Draw(out)
        if t >= 5.:
            paste_tile(out, chart, 831, 114, 406, 358)
            d.text((843, 486), '覆盖仍缺  28.00 mm', font=font(22, True), fill=RED)
            d.text((843, 529), '绿色：可用接地点的最大凸包', font=font(16), fill=GREEN)
            d.text((843, 559), '青色：Pose14 的需求凸包', font=font(16), fill=TEAL)
            d.text((843, 589), '远处也受阻：不限底座尺寸', font=font(16, True), fill=INK)
        else:
            d.rounded_rectangle((831, 115, 1238, 589), radius=14, fill='#f7fafc', outline='#e1e8ed')
            d.text((852, 132), '原始工作面积', font=font(21, True), fill=INK)
            for i, (row, task) in enumerate(zip(data['rows'], work['tasks'])):
                d.text((852, 183+i*40), row['pose'].replace('_', ' '), font=font(19, True), fill=row['color'])
                d.text((1032, 183+i*40), f'{len(task.domain.work_ids)} 个三角片', font=font(17), fill=MUTED)
            d.text((852, 364), '每个工作面上的所有点', font=font(18), fill=INK)
            d.text((852, 396), '对应 30° 半角的连续工作锥', font=font(18), fill=INK)
            if t >= 1.:
                d.text((852, 462), '黄色箭头来自 Pose27', font=font(20, True), fill='#ba8616')
                d.text((852, 504), f'与原法向夹角  {witness["angle_deg"]:.2f}°', font=font(18), fill=INK)
                d.text((852, 541), '将穿过 Pose14 所需覆盖的位置', font=font(17), fill=RED)
        labels(out, data['rows'], y=599)
        return out
    return frame, 16.


def encode(output, stem, draw, duration, review=False):
    frame_count = round(duration*FPS)
    poster_times = [0.5, duration*.32, duration*.66, duration-1.]
    if review:
        sheet = Image.new('RGB', (W*2, H*2), 'white')
        for i, t in enumerate(poster_times):
            frame = draw(t)
            frame.save(output/f'{stem}_review_{i}.png')
            sheet.paste(frame, ((i % 2)*W, (i//2)*H))
        sheet.save(output/f'{stem}_review.png')
        return
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-profile:v', 'main',
        '-pix_fmt', 'yuv420p', '-tag:v', 'avc1', '-movflags', '+faststart', str(output/f'{stem}.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    began = time.monotonic()
    try:
        for i in range(frame_count):
            frame = draw(i/FPS)
            encoder.stdin.write(np.asarray(frame, np.uint8).tobytes())
            if i == frame_count-24:
                frame.save(output/f'{stem}.png')
            if i % (FPS*2) == 0:
                print('FRAME', stem, i, '/', frame_count, f'{time.monotonic()-began:.1f}s', flush=True)
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,profile,width,height,pix_fmt,nb_frames,r_frame_rate,duration',
        '-of', 'json', str(output/f'{stem}.mp4')]))['streams'][0]
    assert int(metadata['nb_frames']) == frame_count and abs(float(metadata['duration'])-duration) < 1/FPS
    C.save(output/f'{stem}_metadata.json', dict(metadata, actual_frame_count=frame_count,
        render_seconds=time.monotonic()-began, explanatory_geometry_animation=True, optimization_trajectory=False))
    print('VIDEO COMPLETE', stem, metadata, flush=True)


def publish(output, report):
    titles = ['两个凸包 → 共享支撑', '最小外圈、扩大外圈仍撞地', '凸包随抬升变化；单独 +Z 不足', '工作锥使接地覆盖不可能']
    stems = ['01_hulls_and_support', '02_ground_collision', '03_lift_and_layout', '04_work_cones']
    descriptions = [
        'pose11+19：先画各自的最小凸包，再显现同一件保存的 Step4 支撑。生长顺序是演示，不是优化日志。',
        'pose11+19：先包围 Pose19，再取出物体、转动空支撑、落到 Pose11。最小外圈和扩大24mm的外圈都穿地；最终落座约束给出不限尺寸的反证。',
        '这个 set 不能仅靠抬一个 pose 解决：Pose11 要抬至少50.11mm，又最多只能抬16.58mm。视频先画这个真实变化，再展示加入横向落座调整的一组接地几何结果。新布局尚未通过承载、工作锥与退出验证。',
        '四 pose：工作面积先出现；第1秒开始长出一根来自 Pose27 原始工作面的3D箭头。接着画凸包与密集工作锥箭头。三种外圈都有实体工作锥碰撞；所有合法接地点的凸包仍离 Pose14 需求28.00mm。结论限定当前固定布局，允许任意底座大小和拓扑。'
    ]
    cards = []
    for title, stem, description in zip(titles, stems, descriptions):
        cards.append(f'<section><h2>{html.escape(title)}</h2><video controls preload="metadata" poster="{stem}.png" src="{stem}.mp4"></video><p>{html.escape(description)}</p><p><a href="{stem}.mp4" download>下载 MP4</a></p></section>')
    text = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>底座与支撑分开优化：四段视频</title><style>body{font:16px/1.7 system-ui;color:#253b4b;max-width:1140px;margin:28px auto;padding:0 20px}video{width:100%;background:white;border:1px solid #e0e8ed;border-radius:12px}section{margin:36px 0}a{color:#1674aa}h1{font-size:30px}h2{font-size:23px}</style><h1>底座与支撑分开优化：四段视频</h1><p>来自两组保存的 Step4、Step5.1 数据。视频为1280×720、24fps、H.264 MP4，带中文说明，无配音。原objects、正式算法和历史结果保持不变。</p>'+''.join(cards)+'<p><a href="geometry/geometry_report.json">几何证据、位移和实际接地覆盖</a> · <a href="geometry/forward_hull_data.npz">原需求正向投影后的凸包数据</a> · <a href="verification.json">输入和执行源码核验</a></p></html>'
    (output/'index.html').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pair, four, small, large, solution, moved, base, work, report, inputs = Q.prepare(output/'geometry')
    freeze = output/'execution'
    freeze.mkdir()
    code = C.freeze(freeze)
    C.save(output/'execution.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        code=code, inputs={key:value for inp in inputs for key,value in inp.protected.items()},
        step4_rerun=False, original_step51_unchanged=True, force_solver_run=False,
        work_overlays_explicitly_requested=True, review_only=args.review))
    art = Artist()
    jobs = [('01_hulls_and_support', growth_video(art, pair)),
            ('02_ground_collision', ground_video(art, pair, small, large, report)),
            ('03_lift_and_layout', seating_video(art, pair, solution, moved, base, report)),
            ('04_work_cones', work_video(art, four, work, report))]
    for stem, (draw, duration) in jobs:
        print('START', stem, flush=True)
        encode(output, stem, draw, duration, args.review)
    for inp in inputs:
        inp.verify()
    for path, expected in code.items():
        assert C.digest(C.ROOT/path) == expected and C.digest(freeze/'executed_sources'/path) == expected, path
    protected = {key:value for inp in inputs for key,value in inp.protected.items()}
    C.save(output/'verification.json', dict(complete=True, protected_inputs=protected,
        executed_sources=code, protected_inputs_unchanged=True, source_snapshots_match=True,
        original_force_demand_count=6*32768, forward_only_hull_update=True,
        actual_xyz_base_ground_coverage=report['xyz_actual_base_checks'],
        not_a_force_or_exit_acceptance=True))
    if not args.review:
        publish(output, report)
    print('ALL COMPLETE', str(output), flush=True)


if __name__ == '__main__':
    main()
