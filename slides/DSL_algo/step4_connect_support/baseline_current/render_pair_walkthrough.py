"""Render actual construction intermediates with the established CAD renderer."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step2_local_support import geometry as G
from step4_connect_support.baseline_current import reseating as R,outer_feet as O,convex_foot as F
from step4_connect_support.baseline_current import build_coupled_saddle as S,clean_render as V,publish_compact as P
from step4_connect_support.baseline_current import acceptance as ACCEPT,render_floor_demands as D

VIEW=np.array([.68,-1.,.8])
GREY='#cbd2ce'
POSE_COLORS=['#5298bd','#d89a59']


def piece(mesh,color,alpha=1.,smooth=False):
    item=P.CPU.piece(dict(v=mesh.vertices.ravel(),f=mesh.faces.ravel()),color,smooth=smooth)
    item['opacity']=alpha
    return P.CPU.placed(item)


def grid(pictures,columns,size):
    canvas=Image.new('RGB',(columns*size,((len(pictures)+columns-1)//columns)*size),'white')
    for i,p in enumerate(pictures):canvas.paste(p.resize((size,size),Image.Resampling.LANCZOS),((i%columns)*size,(i//columns)*size))
    return canvas


def render(group):
    out=group/'step4';data=out/'data';report=I.check_report(data/'report.json');case=R.load_case(out)
    work=out/report['body_directory'];design=json.loads((work/'design.json').read_text())
    mesh=trimesh.load(out/'shape.obj',force='mesh',process=False)
    unchanged=I.sha256(out/'shape.obj')
    b=np.asarray(report['placement']['bases']);o=np.asarray(report['placement']['offsets'])
    designer=O.Designer(case,report['placement'],data/'construction_cache')
    colors=V.head_colors([p['id'] for p in designer.patches])
    roots=[S.unpack(p['original']) for p in designer.patches]
    heads=[piece(m,colors[p['id']]) for p,m in zip(designer.patches,roots)]
    bodies=[trimesh.load(work/r['file'],force='mesh',process=False) for r in design['local_bodies']]
    legs=O.union([S.solid(m) for m in bodies]);blanks=[];pads=[];replayed=[[] for _ in roots]
    for k,floor in enumerate(design['feet_xy_m']):
        for xy in floor:
            xy=np.asarray(xy)
            v=np.vstack([np.c_[xy,np.full(len(xy),z)] for z in (0.,.003)])@b[k]+o[k]
            pads.append(S.solid(G.hull_mesh(v)))
    for landing in design['consolidated_landings']:
        h,k=landing['head_body'],landing['floor_index'];patch=designer.patches[h]
        xy=np.asarray(landing['polygon_xy_m'])
        terminal=np.vstack([np.c_[xy,np.full(len(xy),z)] for z in (0.,.003)])@b[k]+o[k]
        blanks.append(S.solid(G.hull_mesh(np.vstack([patch['v'],patch['root'],terminal]))))
        rebuilt=F.loft(patch,xy,b[k],o[k],designer.carve,O.component)
        if rebuilt is None:raise RuntimeError('Saved loft cannot be reconstructed')
        replayed[h].append(rebuilt)
    replay=[]
    for h,(parts,saved) in enumerate(zip(replayed,bodies)):
        a=O.union(parts);z=S.solid(saved)
        difference=(abs(float((a-z).volume()))+abs(float((z-a).volume())))*S.SCALE**3
        if difference>8e-14:raise RuntimeError(f'Construction replay differs for head {h}: {difference}')
        replay.append(dict(head=h,symmetric_difference_volume_m3=difference))
    blank=O.union(blanks);removed=blank-legs
    foot=O.union(pads);bridges=S.solid(mesh)-legs
    # Only the displayed forbidden volume is cropped. Full 500 mm sweeps are
    # rendered separately and remain the unmodified collision-check input.
    bound=mesh.bounds+np.array([[-.015]*3,[.015]*3])
    box=trimesh.creation.box(bound[1]-bound[0]);box.apply_translation(bound.mean(0))
    near=designer.forbidden^S.solid(box)
    saved_meshes={
        'heads':S.unpack(O.union([p['original'] for p in designer.patches])),
        'forbidden_near':S.unpack(near),'feet':S.unpack(foot),
        'uncut_bodies':S.unpack(blank),'removed_material':S.unpack(removed),
        'carved_bodies':S.unpack(legs),'added_connections':S.unpack(bridges)}
    for name,m in saved_meshes.items():m.export(data/f'walkthrough_{name}.obj',digits=17,include_normals=False)
    camera=P.CPU.fit(np.vstack([mesh.vertices,S.unpack(blank).vertices]),VIEW,1.,padding=1.22)
    renderer=V.Renderer();pictures=[];images=[]
    def draw(filename,parts,cam=camera):
        picture=renderer.render(parts,cam,1200);picture.save(out/filename);images.append(filename)
        return picture
    pictures.append(draw('01_heads.png',heads))
    near_sweeps=[S.unpack(S.solid(s)^S.solid(box)) for s in designer.sweeps]
    pictures.append(draw('02_exit_spaces.png',heads+[piece(s,POSE_COLORS[k],.25) for k,s in enumerate(near_sweeps)]))
    pictures.append(draw('03_forbidden_space.png',heads+[piece(saved_meshes['forbidden_near'],'#a18db8',.32)]))
    demands=[]
    for k,xy in enumerate(case.demands):
        hull=xy[ConvexHull(xy).vertices];points=np.c_[hull,np.zeros(len(hull))]@b[k]+o[k]
        face=D.polygon_piece(points,POSE_COLORS[k]);face['opacity']=.25
        demands.append(P.CPU.placed(face));demands.append(P.CPU.placed(D.boundary_piece(points,POSE_COLORS[k])))
    pictures.append(draw('04_feet.png',heads+demands+[piece(saved_meshes['feet'],GREY)]))
    pictures.append(draw('05_uncut_bodies.png',[piece(saved_meshes['uncut_bodies'],GREY)]+heads))
    pictures.append(draw('06_removed_material.png',[piece(saved_meshes['carved_bodies'],GREY),
        piece(saved_meshes['removed_material'],'#d47c74',.65)]+heads))
    pictures.append(draw('07_carved_bodies.png',[piece(saved_meshes['carved_bodies'],GREY)]+heads))
    pictures.append(draw('08_connections.png',[piece(saved_meshes['carved_bodies'],GREY),
        piece(saved_meshes['added_connections'],'#7c8984')]+heads))
    pictures.append(draw('09_final_support.png',[piece(mesh,GREY)]+heads))
    grid(pictures,3,850).save(out/'construction_steps.png');images.append('construction_steps.png')
    # Complete per-pose sweep figures: one actual object and its exit envelope.
    sweep_pictures=[];initial_pictures=[];rows=[]
    for k,task in enumerate(case.tasks):
        obj=trimesh.Trimesh(task.domain.mesh.vertices@b[k]+o[k],task.domain.mesh.faces,process=False)
        direction=-np.asarray(report['placement']['directions'][k])@b[k]
        sweep=designer.sweeps[k]
        side=np.cross(direction,b[k][2]);view=side*.8+b[k][2]*.65-direction*.15
        cam=P.CPU.fit(np.vstack([sweep.vertices,mesh.vertices]),view,1.,padding=1.15)
        parts=[piece(sweep,POSE_COLORS[k],.18),piece(obj,'#a6b2ba',.60)]+heads
        sweep_pictures.append(draw(f'exit_space_{task.pose}.png',parts,cam))
        own=[piece(roots[h],colors[p['id']]) for h,p in enumerate(designer.patches) if p['pose']==k]
        cam0=P.CPU.fit(obj.vertices,VIEW@b[k],1.,padding=1.20)
        initial_pictures.append(draw(f'step3_{task.pose}.png',own+[piece(obj,'#a6b2ba',.40)],cam0))
        hull=F.landing(S.solid(mesh),b[k],o[k]).convex_hull
        required=MultiPoint(case.demands[k]).convex_hull
        rows.append(dict(pose=task.pose,actual_ground_hull_xy_m=np.asarray(hull.exterior.coords).tolist(),
            uncovered_demand_area_m2=float(required.difference(hull.buffer(1e-10)).area)))
    grid(sweep_pictures,2,1200).save(out/'exit_spaces.png');images.append('exit_spaces.png')
    grid(initial_pictures,2,1000).save(out/'step3_heads.png');images.append('step3_heads.png')
    working=designer.access.verify(mesh)
    geometry=ACCEPT.evaluate(case,report,mesh,working)
    assert geometry['passed'] and all(r['uncovered_demand_area_m2']<=1e-12 for r in rows)
    assert I.sha256(out/'shape.obj')==unchanged
    evidence=dict(complete=True,passed=True,object=case.name,poses=case.poses,
        step3_covered_counts=case.schedule['covered_counts'],step3_passed=case.schedule['passed'],
        step4_geometry=geometry,working_surface=working,ground_hulls=rows,
        intermediate_loft_replay=replay,geometry_unchanged=True,
        full_sweep_length_m=S.SWEEP_LENGTH,near_forbidden_display_bounds_m=bound.tolist(),
        sweep_display_crop_is_not_a_collision_bound=True,
        image_order=images,text_in_images=False,arrows_in_images=False,axes_in_images=False,
        html_generated=False,videos_generated=False,
        provenance=dict(inputs=I.hashes(case.paths+[data/'report.json',work/'design.json',out/'shape.obj']),
            code=I.hashes([Path(__file__),Path(O.__file__),Path(F.__file__),Path(ACCEPT.__file__),Path(V.__file__)])),
        artifacts={**{'../'+n:I.sha256(out/n) for n in images},
            **{f'walkthrough_{name}.obj':I.sha256(data/f'walkthrough_{name}.obj') for name in saved_meshes}})
    I.save(data/'walkthrough.json',evidence)
    text='''# pose1+3：完整运行与实体构造过程

Step0 地面兼容通过。重新生成每 pose 32,768 个载荷和 200 个候选；Step3 各选 4 个头，两组均覆盖全部载荷。Step4 构造一件刚性支撑，两个摆放的完整 500 mm 退出、地面、接触、连通、凸包覆盖及工作面检查通过。力/力矩以 Step3 为准，无尺寸上限。

原 Step5 已改名 Step4，所有图片都在本目录。图片没有文字、箭头或坐标轴。灰色为支撑，各头颜色保持一致。

[最终两个 pose 与支撑](overview.png) · [构造九宫格](construction_steps.png) · [完整退出空间](exit_spaces.png) · [Step3 两组原头](step3_heads.png) · [原始需求统一到一个物体坐标](data/floor_demands_common_frame.png)

九宫格从左到右、从上到下：

1. `01_heads.png`：选定摆放后，八个头在同一支撑坐标中的位置。
2. `02_exit_spaces.png`：两种 pose 的退出扫掠空间，蓝色为 pose1、橙色为 pose3。
3. `03_forbidden_space.png`：合并后带构造间隙的禁区，紫色。
4. `04_feet.png`：外侧脚垫；蓝/橙面是需要被实际接地凸包包住的需求。
5. `05_uncut_bodies.png`：头到脚垫的原始凸包脚体，尚未裁剪。
6. `06_removed_material.png`：红色为裁剪丢弃的材料，灰色为保留的脚体。
7. `07_carved_bodies.png`：裁剪后的独立局部身体。
8. `08_connections.png`：添加短连接，深灰色突出新增材料。
9. `09_final_support.png`：最终同一件连通支撑。小脚垫不足以保留规定边框和中筋，因此末尾检查掏空后保持实心。

九宫格的头、脚和身体使用相同相机。第 2、3 张仅显示支撑附近的禁区，方便看清裁剪位置；`exit_spaces.png` 展示两个 pose 的完整扫掠，实际碰撞检查始终用完整 500 mm 空间并确认终点完全分离。

原始撒点图使用一个固定物体坐标。实体构造图使用搜索出的独立摆放，因此不同 pose 的物体相对支撑位置可以不同。各组四个头的内部接触关系保持不变。

`data/walkthrough.json` 保存最终几何验证、实际地面凸包和中间脚体重放误差。裁剪前几何由选中的原始头/脚参数重建；裁剪后身体与实际构造输出逐体比较，未用示意形状替换真实结果。只运行本次 pose1+3，其他组合未改。
'''
    (out/'README.md').write_text(text)
    print('WALKTHROUGH FIGURES',out/'construction_steps.png',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',default='pose1+3')
    args=p.parse_args();render(I.OUTPUTS/'B'/args.group)
