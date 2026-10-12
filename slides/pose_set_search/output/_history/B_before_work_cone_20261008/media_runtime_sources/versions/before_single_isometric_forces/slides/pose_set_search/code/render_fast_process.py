"""Show selected layouts with a sampled occupancy surface, without Booleans.

Preview geometry is a 1.25 mm grid of fitted material minus current body,
working bands and full nominal exits. It is not a final accepted solid. The
optimizer's Sobol volume estimate is kept distinct from this display grid.
"""
import argparse
import hashlib
from collections import OrderedDict
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/pose_set_search_numba')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pose_set_search_matplotlib')
from common import *
from model import Model, Layout
from case_sets import CASES
from reuse_first import registered
from volume_guidance import VolumeGuidance
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import marching_cubes
from solid_render import arrow_mesh


def original_raster():
    # Import the existing video's renderer without loading/running its local
    # geometry example. Its rasterizer is independent of that geometry module.
    path = CO/'operation_demo/juxtapose/code/render.py'
    # Keep the module name used by the original demo's Numba disk cache.
    spec = importlib.util.spec_from_file_location('juxtapose_demo_render', path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get('geometry')
    sys.modules['geometry'] = SimpleNamespace()
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules['geometry']
        else:
            sys.modules['geometry'] = previous
    return module.raster


RASTER = original_raster()


def load_layout(path):
    with np.load(path) as data:
        return Layout(data['placements'].copy(), data['directions'].copy(),
                      data['hosts'].copy(), tuple(map(int,data['active'])))


class GridShapes:
    def __init__(self, model, layouts, pitch=.00125, cache_path=None):
        self.model, self.pitch = model, pitch
        signature = hashlib.sha256(repr([layout.key() for layout in layouts]).encode()).hexdigest()
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as saved:
                if str(saved['signature']) == signature and float(saved['pitch']) == pitch:
                    self.low = saved['low'].copy()
                    self.shape = tuple(map(int,saved['shape']))
                    self.ids = saved['ids'].copy()
                    self.points = saved['points'].copy()
                    keys = json.loads(str(saved['keys']))
                    self.wraps = {(kind,k,bytes.fromhex(q)):saved[f'wrap_{i}'].copy()
                                  for i,(kind,k,q) in enumerate(keys)}
                    self.cache = OrderedDict()
                    print('REUSE PREVIEW GRID',self.shape,'shell samples',len(self.ids),flush=True)
                    return
        bounds = []
        for layout in layouts:
            for k in layout.active:
                points = C.transform_points(model.mesh.vertices, layout.placements[k])
                bounds.extend([points.min(0)-model.thickness-2*pitch,
                               points.max(0)+model.thickness+2*pitch])
        bounds = np.asarray(bounds)
        self.low = np.floor(bounds.min(0)/pitch)*pitch
        self.shape = tuple((np.ceil((bounds.max(0)-self.low)/pitch).astype(int)+1).tolist())
        indices = np.indices(self.shape, dtype=np.int32).reshape(3,-1).T
        dense = self.low+indices*pitch
        dense_wraps = {}
        union = np.zeros(len(dense),bool)
        for layout in layouts:
            for k in layout.active:
                key = self.key('wrap',k,layout.placements[k])
                if key not in dense_wraps:
                    mask = self.query(model.wrap_rays[k], dense, layout.placements[k])
                    dense_wraps[key] = mask
                    union |= mask
        self.ids = np.flatnonzero(union)
        self.points = dense[self.ids]
        self.wraps = {key:mask[self.ids] for key,mask in dense_wraps.items()}
        self.cache = OrderedDict()
        if cache_path is not None:
            keys = [[kind,k,q.hex()] for kind,k,q in self.wraps]
            np.savez_compressed(cache_path,signature=signature,pitch=pitch,low=self.low,
                shape=self.shape,ids=self.ids,points=self.points,keys=json.dumps(keys),
                **{f'wrap_{i}':mask for i,mask in enumerate(self.wraps.values())})
        print('PREVIEW GRID', self.shape, 'shell samples',len(self.ids),'pitch mm',pitch*1000, flush=True)

    def key(self, kind, k, q):
        return kind, (None if kind == 'body' else k), np.round(q,11).tobytes()

    def query(self, ray, points, q):
        result = np.zeros(len(points),bool)
        low,high = ray.mesh.bounds
        for start in range(0,len(points),200000):
            local = C.transform_points(points[start:start+200000],np.linalg.inv(q))
            candidates = np.flatnonzero(np.all((local>=low-1e-12)&(local<=high+1e-12),axis=1))
            if len(candidates):
                result[start+candidates] = ray.contains_points(local[candidates])
        return result

    def contains(self, layout, k, kind):
        q = layout.placements[k]
        key = self.key(kind,k,q)
        if kind == 'wrap':
            return self.wraps[key]
        if key not in self.cache:
            ray = self.model.ray if kind == 'body' else self.model.work_rays[k]
            self.cache[key] = self.query(ray,self.points,q)
        self.cache.move_to_end(key)
        while len(self.cache) > 128:
            self.cache.popitem(last=False)
        return self.cache[key]

    def sweep(self, layout, k):
        q = layout.placements[k]
        direction = q[:3,:3].T @ layout.directions[k]
        key = 'sweep', np.round(q,11).tobytes(), np.round(direction,11).tobytes()
        if key not in self.cache:
            local = C.transform_points(self.points,np.linalg.inv(q))
            displacement = self.model.length*direction
            low = self.model.mesh.bounds[0]+np.minimum(displacement,0)
            high = self.model.mesh.bounds[1]+np.maximum(displacement,0)
            candidates = np.flatnonzero(np.all((local>=low)&(local<=high),axis=1))
            locked = np.zeros(len(self.points),bool)
            for start in range(0,len(candidates),200000):
                ids = candidates[start:start+200000]
                locations, rays, _ = self.model.ray.intersects_location(
                    local[ids],np.tile(-direction,(len(ids),1)),multiple_hits=False)
                distances = (local[ids[rays]]-locations) @ direction
                locked[ids[rays[(distances>=0)&(distances<=self.model.length)]]] = True
            self.cache[key] = locked
        self.cache.move_to_end(key)
        return self.cache[key]

    def mask(self, layout):
        alive = np.zeros(len(self.points),bool)
        for k in layout.active:
            alive |= self.contains(layout,k,'wrap')
        for k in layout.active:
            alive &= ~self.contains(layout,k,'body')
            alive &= ~self.contains(layout,k,'work')
            alive &= ~self.sweep(layout,k)
        return alive

    def mesh(self, mask):
        if not mask.any():
            return C.trimesh.Trimesh()
        dense = np.zeros(int(np.prod(self.shape)),np.uint8)
        dense[self.ids[mask]] = 1
        field = np.pad(dense.reshape(self.shape),1)
        vertices, faces, _, _ = marching_cubes(field,level=.5,spacing=(self.pitch,)*3,
                                              allow_degenerate=False,gradient_direction='ascent')
        # ascent faces bound occupied cells with outward orientation.
        return C.trimesh.Trimesh(vertices+self.low-self.pitch,faces,process=False)


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+name,size)


class Panel:
    def __init__(self, bounds, size=(360,320), elevation=35.26438968, azimuth=-45, points=None):
        self.width,self.height = size
        elevation,azimuth = np.radians([elevation,azimuth])
        self.camera = np.array([np.cos(elevation)*np.cos(azimuth),
                               np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
        self.right = np.array([-np.sin(azimuth),np.cos(azimuth),0.])
        self.up = np.cross(self.camera,self.right)
        self.light = np.array([1.,-1.,1.8]); self.light /= np.linalg.norm(self.light)
        if points is None:
            points = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
        basis = np.column_stack([self.right,self.up,self.camera])
        projected = points @ basis
        self.center = ((projected.min(0)+projected.max(0))/2) @ basis.T
        self.scale = .91*min(self.width/np.ptp(projected[:,0]),
                             self.height/np.ptp(projected[:,1]))

    def render(self, meshes):
        image = np.full((self.height,self.width,3),255,np.uint8)
        depth = np.full((self.height,self.width),-np.inf)
        triangles,colors = [],[]
        for mesh,color in meshes:
            if not len(mesh.faces):
                continue
            normals = mesh.face_normals
            keep = normals @ self.camera > 0
            ts,ns = mesh.triangles[keep],normals[keep]
            q = ts-self.center
            projected = np.stack([q @ self.right*self.scale+self.width/2,
                                   -q @ self.up*self.scale+self.height/2,
                                   ts @ self.camera],axis=2)
            shade = .60+.40*np.maximum(ns @ self.light,0)
            rgb = np.array(ImageDraw.ImageColor.getrgb(color))
            triangles.append(projected);colors.append((shade[:,None]*rgb).astype(np.uint8))
        if triangles:
            RASTER(np.concatenate(triangles),np.concatenate(colors),image,depth,0,0,self.width,self.height)
        return Image.fromarray(image)


def winner(row):
    event = row.get('decision',{})
    return next((trial for trial in event.get('trials',[]) if trial.get('accepted')),{})


def operation_text(trial):
    operation = trial.get('operation','restore rotating reuse')
    detail = trial.get('detail',{})
    if operation == 'juxtapose':
        shift = np.linalg.norm(detail.get('horizontal_offset_fixture_m',[0,0,0]))*1000
        return f"Juxtapose {detail.get('guest','')} -> {detail.get('host','')}, offset {shift:.1f} mm"
    if 'translation' in operation:
        return f"Translation {detail.get('pose','')}, step {detail.get('step_m',0)*1000:.2f} mm"
    if 'direction' in operation:
        return f"{operation.replace('direction','Direction')}, step {detail.get('step_degrees',0):g} deg"
    return operation


def heading(row):
    trial = winner(row)
    if trial:
        return operation_text(trial)
    phase = row['phase']
    if phase.startswith('insert_'):
        return 'Insert '+phase[len('insert_'):].replace('_registered','')+' at native seating'
    if phase.endswith('initial'):
        return 'Initialize similar exits; rotate fixture reuse'
    return phase.replace('_',' ')


def chosen_owner(row,layout,poses):
    detail = winner(row).get('detail',{})
    if detail.get('guest') in poses:
        return poses.index(detail['guest'])
    if detail.get('pose') in poses:
        return poses.index(detail['pose'])
    inserted = [change['pose'] for change in row.get('changes',[]) if change.get('inserted')]
    if inserted:
        return poses.index(inserted[-1])
    worst = row.get('decision',{}).get('hardest')
    return int(worst['pose_index']) if worst else layout.active[-1]


def card(row,layout,mesh,model,panel,added,removed):
    image = Image.new('RGB',(780,600),'white')
    draw = ImageDraw.Draw(image)
    index = row['index']
    draw.rounded_rectangle((4,4,775,595),radius=14,outline='#dbe3e9',width=2)
    draw.text((22,18),f"{index:02d}  {heading(row)}",font=font(20,True),fill='#243340')
    fails = row['failed_load_count']
    draw.text((22,52),f"{len(layout.active)} poses | unmet sampled demands: {fails:,}",font=font(19),fill='#247f50' if fails==0 else '#b85835')
    draw.text((22,83),f"Search V~ {row['estimated_volume_cm3']:.1f} cm3  |  rotate {row['rotating_reuse_pose_count']} / Juxtapose {row['juxtaposed_pose_count']}",font=font(19),fill='#475660')
    owner = chosen_owner(row,layout,model.poses)
    body = transform_mesh(model.mesh,layout.placements[owner])
    direction = layout.directions[owner]
    origin = body.vertices.mean(0)
    origin[2] = body.bounds[1,2]+.007
    arrow = arrow_mesh(origin,direction,.040)
    image.paste(panel.render([(mesh,'#319cd7')]),(22,141))
    image.paste(panel.render([(body,'#a4a8ac'),(mesh,'#319cd7'),(arrow,'#ffe000')]),(398,141))
    draw.text((33,117),'New shared support',font=font(16),fill='#517184')
    draw.text((408,117),model.poses[owner]+' in fixture coordinates',font=font(16),fill='#517184')
    draw.text((22,467),f"Grid material change: +{added:.1f} / -{removed:.1f} cm3",font=font(18),fill='#475660')
    event = row.get('decision',{})
    trials = event.get('trials',[])
    if trials:
        draw.text((22,495),f"Screened {event.get('screened_candidates',event.get('sampled_layouts','?'))}; checked {len(trials)} finalists; choose best protected rank",font=font(15),fill='#475660')
        hardest = event.get('hardest')
        if hardest:
            draw.text((22,517),f"Hardest: {model.poses[hardest['pose_index']]} / load {hardest['load_index']} | cone loss {hardest.get('loss',0):.5g}",font=font(14),fill='#657581')
        else:
            draw.text((22,517),'Preserve all sampled demands, then reduce volume.',font=font(14),fill='#657581')
        for number,trial in enumerate(trials[:3]):
            text = ('SELECT ' if trial.get('accepted') else 'reject ')+operation_text(trial)
            text = text[:85]
            draw.text((22,537+number*18),text,font=font(14),fill='#247f50' if trial.get('accepted') else '#7c8389')
    else:
        annotations = row.get('annotations',[])
        insertion = next((item for item in reversed(annotations) if item['phase'].startswith('insertion_')),None)
        if insertion:
            stage = insertion['decision']
            draw.text((22,501),f"Insertion {'PASS' if stage['passed'] else 'FAIL'}: {stage['added_pose']}; original load count checked at sampled contacts",font=font(15),fill='#475660')
    return image


def render_method(out,model,grid):
    began = time.monotonic()
    rows = json.loads((out/'process.json').read_text())
    report = json.loads((out/'search_report.json').read_text())
    layouts = [load_layout(out/row['layout']) for row in rows]
    # Common scale/fixture-coordinate camera throughout this process.
    masks = [grid.mask(layout) for layout in layouts]
    visible_union = np.logical_or.reduce(masks)
    scene = [grid.points[visible_union]]
    for row,layout in zip(rows,layouts):
        owner = chosen_owner(row,layout,model.poses)
        body = transform_mesh(model.mesh,layout.placements[owner])
        scene.append(body.vertices)
        origin = body.vertices.mean(0); origin[2] = body.bounds[1,2]+.007
        scene.append(arrow_mesh(origin,layout.directions[owner],.040).vertices)
    panel = Panel(None,points=np.vstack(scene))
    columns = 3
    canvas = Image.new('RGB',(columns*780,140+math.ceil(len(rows)/columns)*600),'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((25,15),out.parent.name+' / '+out.name+'  |  selected search states',font=font(33,True),fill='#243340')
    draw.text((25,62),'Blue: discretized material; gray: object; yellow: exit. Full nominal exits cut; no lateral sliding tunnel.',font=font(21),fill='#475660')
    draw.text((25,98),'Quick comparison: sampled contacts + original 32,768 demands/pose. No final acceptance. Grid 1.25 mm; V~ is Sobol guidance.',font=font(19),fill='#657581')
    previous = np.zeros(len(grid.points),bool)
    render_rows = []
    final_mesh = None
    for index,(row,layout,mask) in enumerate(zip(rows,layouts,masks)):
        mesh = grid.mesh(mask)
        added = float(np.count_nonzero(mask & ~previous)*grid.pitch**3*1e6)
        removed = float(np.count_nonzero(previous & ~mask)*grid.pitch**3*1e6)
        tile = card(row,layout,mesh,model,panel,added,removed)
        canvas.paste(tile,((index%columns)*780,140+(index//columns)*600))
        render_rows.append(dict(index=index,phase=row['phase'],grid_added_cm3=added,
            grid_removed_cm3=removed,grid_volume_cm3=float(mask.sum()*grid.pitch**3*1e6),
            preview_triangle_count=len(mesh.faces)))
        previous = mask
        final_mesh = mesh
        print('DRAW STEP',out.parent.name,out.name,index,'/',len(rows)-1,'faces',len(mesh.faces),flush=True)
    canvas.save(out/'process.png')
    final = load_layout(out/'sampled_layout.npz')
    assert final.key() == layouts[-1].key()
    # Exactly the same display support, rigidly moved for every saved pose.
    final_canvas = Image.new('RGB',(3*610,110+math.ceil(len(final.active)/3)*560),'white')
    text = ImageDraw.Draw(final_canvas)
    text.text((23,15),out.parent.name+' / '+out.name+'  |  final sampled configuration',font=font(30,True),fill='#243340')
    text.text((23,61),f"V~ {report['estimated_volume_cm3']:.1f} cm3 | rotate {report['rotating_reuse_pose_count']} / Juxtapose {report['juxtaposed_pose_count']} | {'sampled PASS' if report['sampled_force_passed'] else 'sampled FAIL'}",font=font(24),fill='#475660')
    final_states = []
    for index,k in enumerate(final.active):
        frame = model.native[final.hosts[k]]
        support = transform_mesh(final_mesh,frame)
        body = transform_mesh(model.mesh,frame @ final.placements[k])
        direction = frame[:3,:3] @ final.directions[k]
        origin = body.vertices.mean(0);origin[2] = body.bounds[1,2]+.007
        arrow = arrow_mesh(origin,direction,.048)
        bounds = np.array([np.vstack([support.vertices,body.vertices,arrow.vertices]).min(0),
                           np.vstack([support.vertices,body.vertices,arrow.vertices]).max(0)])
        pose_panel = Panel(bounds,size=(585,480),points=np.vstack([support.vertices,body.vertices,arrow.vertices]))
        x,y = (index%3)*610,110+(index//3)*560
        final_canvas.paste(pose_panel.render([(body,'#a4a8ac'),(support,'#319cd7'),(arrow,'#ffe000')]),(x+12,y+43))
        mode = 'rotate fixture' if registered(final,k) else 'Juxtapose -> '+model.poses[final.hosts[k]]
        text.text((x+23,y+8),model.poses[k]+' | '+mode,font=font(21,True),fill='#243340')
        text.text((x+23,y+525),f"sampled loads {report['counts'][model.poses[k]]:,}/32,768",font=font(19),fill='#475660')
        final_states.append(dict(pose=model.poses[k],host=model.poses[final.hosts[k]],
            object_to_fixture=final.placements[k].tolist(),fixture_to_world=frame.tolist(),
            direction_world=direction.tolist()))
    final_canvas.save(out/'final_result.png')
    np.savez_compressed(out/'preview_support.npz',vertices=final_mesh.vertices,faces=final_mesh.faces)
    C.save(out/'render.json',dict(complete=True,preview_only=True,geometry_verified=False,
        final_acceptance_run=False,grid_pitch_m=grid.pitch,grid_shape=grid.shape,
        display_material='union of fitted wraps minus every current body/work band/full nominal exit',
        volume_label='Search Sobol estimate; separate grid material deltas are preview-only',
        same_rigid_support_in_every_final_pose=True,steps=render_rows,states=final_states,
        raster_source='slides/Co-optimize/operation_demo/juxtapose/code/render.py:raster',
        seconds=time.monotonic()-began))
    write_details(out,rows,report,render_rows)


def write_details(out,rows,report,render_rows):
    lines = [f'# {out.parent.name} / {out.name}', '',
        '快速采样搜索；这次按要求不做最终实体、退出和工作面验收。', '',
        '[每步选择与新支撑](process.png) · [最终每个 pose](final_result.png) · '
        '[全部候选记录](trace.json) · [布局与过程](process.json) · [搜索统计](search_report.json)', '',
        f'当前加入 {report["pose_count"]}/{report["requested_pose_count"]} 个 pose；'
        f'采样力／力矩 {"通过" if report["sampled_force_passed"] else "尚未通过"}；'
        f'体积估计 {report["estimated_volume_cm3"]:.2f} cm³；'
        f'搜索 {report["search_seconds"]:.1f} s。', '',
        '图里蓝色是同一采样材料模型的 1.25 mm 网格边界：当前贴合壳并集，扣除所有当前物体、工作带和完整退出路径。'
        '没有挖平移轨迹，也没有造一个供物体横向滑动的通道。'
        '过程图左侧只看支撑，右侧加入该步相关物体；最终图用同一蓝色形状随支撑摆放做刚体变换。', '',
        'V~ 是搜索时的 Sobol 体积估计。搜索候选共同扩张采样框时 epoch 会变化，'
        '不同 epoch 的原始 V~ 不宜直接比较。下表增加／删除使用统一显示网格，可比较相邻形状。', '',
        '| 步 | 选择 | 未满足载荷数 | V~ cm³ / epoch | 显示网格 + / − cm³ |',
        '|---|---|---:|---:|---:|']
    for row,render in zip(rows,render_rows):
        lines.append(f'| {row["index"]} | {heading(row)} | {row["failed_load_count"]} | '
                     f'{row["estimated_volume_cm3"]:.2f} / {row.get("volume_epoch")} | '
                     f'+{render["grid_added_cm3"]:.2f} / −{render["grid_removed_cm3"]:.2f} |')
    lines += ['', '下面包括未更新形状的拒绝回合。Juxtapose 内部的临时调整在分支被选中后才成为过程步骤。', '']
    for index,event in enumerate(report.get('events',[])):
        lines += [f'## 候选回合 {index}: {event["phase"]}', '',
                  f'选择结果：{event.get("accepted","初始化 / 插入")}; '
                  f'筛选数：{event.get("screened_candidates",event.get("sampled_layouts","—"))}。', '']
        hardest = event.get('hardest')
        if hardest:
            lines += ['最难原始载荷：`'+json.dumps(hardest,ensure_ascii=False)+'`。', '']
        for trial in event.get('trials',[]):
            lines += ['- '+('选中' if trial.get('accepted') else '未选')+'：'+operation_text(trial)+
                      '；结果 '+json.dumps({key:trial[key] for key in ['counts','lost_protected_loads','rank','error'] if key in trial},ensure_ascii=False)+'。']
        lines.append('')
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=HERE/'output/B')
    parser.add_argument('--case',choices=['all']+list(CASES),default='all')
    parser.add_argument('--cases',choices=list(CASES),nargs='+')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--redraw',action='store_true')
    args = parser.parse_args()
    cases = args.cases or (list(CASES) if args.case=='all' else [args.case])
    visited = set()
    while True:
        did_work = False
        for case in cases:
            methods = [args.root/case/method for method in ['whole','incremental']]
            if not all((out/'search_report.json').exists() for out in methods):
                continue
            todo = [out for out in methods if not (out/'render.json').exists() or (args.redraw and case not in visited)]
            if not todo and (args.root/case/'comparison.json').exists():
                continue
            model = Model([f'pose_{k}' for k in CASES[case]])
            layouts = []
            for out in methods:
                layouts.extend(load_layout(out/row['layout']) for row in json.loads((out/'process.json').read_text()))
            if todo:
                grid = GridShapes(model,layouts,cache_path=args.root/case/'preview_grid.npz')
                for out in todo:
                    render_method(out,model,grid)
            finals = [load_layout(out/'sampled_layout.npz') for out in methods]
            common_volume = VolumeGuidance(model,finals,power=17)
            C.save(args.root/case/'comparison.json',dict(
                evaluation='common-frame 131072-point Sobol material occupancy; not exact acceptance',
                final_acceptance_run=False,poses=model.poses,
                volumes_cm3={out.name:common_volume.estimate(layout) for out,layout in zip(methods,finals)},
                grid_volumes_cm3={out.name:json.loads((out/'render.json').read_text())['steps'][-1]['grid_volume_cm3'] for out in methods},
                common_frame_low=common_volume.low.tolist(),common_frame_high=common_volume.high.tolist(),
                common_sample_count=len(common_volume.points)))
            did_work = True
            visited.add(case)
        if not args.watch or all((args.root/case/method/'render.json').exists()
                                 for case in cases for method in ['whole','incremental']):
            break
        batch = args.root/'batch.json'
        if batch.exists() and json.loads(batch.read_text()).get('complete') and not did_work:
            raise RuntimeError('Search finished but some result figures could not be generated')
        time.sleep(5)


if __name__ == '__main__':
    main()
