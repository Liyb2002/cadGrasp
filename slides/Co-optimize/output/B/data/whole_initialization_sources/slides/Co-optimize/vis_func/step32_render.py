"""Eight isometric orbit views of all registered working exclusions together."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import argparse,time
from co_common import *
from work_access import RegisteredWorkVolumes
from work_access_render import Panel,render_scene,Image


def render(name,group,base=None):
    began=time.monotonic()
    base=Path(base) if base is not None else (HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']/'step3'
    out=base/'step3.2';(out/'data').mkdir(parents=True,exist_ok=True)
    source=base/'step3.1';report=json.loads((source/'data/report.json').read_text())
    if report.get('schema')!='whole_registered_work_cones_v1':
        raise RuntimeError('Generate the new merged Step3.1 initialization first')
    body=trimesh.load(source/'registered_object.obj',force='mesh',process=False)
    support=trimesh.load(source/'wrapped_support.obj',force='mesh',process=False)
    states={pose:state(name,pose) for pose in group['poses']}
    access=RegisteredWorkVolumes(body,states);region,definition=access.rounded_excerpt()
    transform=np.asarray(report['T_fixture_to_reference_world'])
    posed_body=body.copy();posed_body.apply_transform(transform)
    posed_support=support.copy();posed_support.apply_transform(transform)
    region.apply_transform(transform)
    region.metadata.update(uniform_display_color=True,work_region_key=(group['id'],tuple(access.ids),transform.tobytes()))
    working=trimesh.Trimesh(body.vertices.copy(),body.faces[access.ids].copy(),process=False)
    working.apply_transform(transform)
    points=np.vstack([posed_body.vertices,posed_support.vertices,region.vertices])
    elevation=35.26438968;angles=[-45+45*i for i in range(8)];size=(800,700)
    panels=[Panel(None,size=size,points=points,elevation=elevation,azimuth=angle) for angle in angles]
    scale=min(panel.scale for panel in panels)
    sheet=Image.new('RGB',(4*size[0],2*size[1]),'white')
    for i,panel in enumerate(panels):
        panel.scale=scale
        layers=[(working,'#FFBD66'),(posed_body,'#A5ADB5'),(posed_support,'#2A91D2'),(region,'#F0A33E',.28)]
        sheet.paste(render_scene(panel,layers),((i%4)*size[0],(i//4)*size[1]))
    path=out/'overview.png';sheet.save(path)
    inputs=[source/'registered_object.obj',source/'wrapped_support.obj',source/'data/report.json']
    inputs+=[p for task,T,mesh in states.values() for p in task.inputs]
    record=dict(complete=True,passed=True,status='visualized',stage='step3.2',presentation_only=True,
        object=name,pose_set=group['id'],poses=group['poses'],reference_pose=report['reference_pose'],
        registered_union_of_all_working_areas=True,view_count=8,
        view_order='left to right, top then bottom;45-degree orbit steps',
        elevation_degrees=elevation,azimuth_degrees=angles,onscreen_text=False,
        image_size_px=list(sheet.size),work_access_definition=access.definition(),display_region=definition,
        support_source='step3.1/wrapped_support.obj',support_sha256=I.sha256(source/'wrapped_support.obj'),
        source_initialization_sha256=I.sha256(source/'data/report.json'),
        geometry_changed=False,mechanics_rerun=False,full_fixture_accepted=False,
        wheel_source='slides/pose_set_search/code/render_work_access_figure.py and render_mesh_media.py',
        seconds=time.monotonic()-began,
        provenance=provenance(inputs,[Path(__file__),HERE/'vis_func/work_access_render.py',HERE/'helper_func/work_access.py']),
        artifacts={'../overview.png':I.sha256(path)})
    save(out/'data/report.json',record);I.check_report(out/'data/report.json')
    (out/'README.md').write_text('# Step3.2：整组工作禁区\n\n[overview.png](overview.png) 为所有 pose 注册到首个 pose 后的工作禁区并集。'
        '左到右、上到下，每次绕45°，共8个等轴测视角。灰色物体、蓝色真实支撑、橙色工作面、半透明琥珀色禁区。\n\n'
        '圆弧外侧来自同一个球形显示截断面；实际禁区为原工作面向外完整角度范围形成的半无限区域。'
        '本步骤只画图，不修改 Step3.1 的支撑，也不重新求力／力矩。\n')
    print('STEP3.2 WORK UNION',group['id'],len(access.ids),'working faces',flush=True)
    return record


def main():
    from run_all import saved_groups
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B');parser.add_argument('--sets',nargs='+')
    parser.add_argument('--scope',choices=['existing','selected'],default='existing');args=parser.parse_args()
    groups=saved_groups(args.object,args.scope)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
    if not groups:parser.error('No matching existing pose sets')
    for group in groups:render(args.object,group)
    print('STEP3.2 TOTAL',len(groups),'orbit figures',flush=True)


if __name__=='__main__':main()
