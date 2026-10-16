"""Blender entry point for ten minimal robot illustrations."""
import argparse
import json
from pathlib import Path
import sys
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from arm_visuals import camera,cone,illustrated_arm,light,line,material,setup


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    for path in sorted(args.directory.glob('[0-9][0-9]_*.json')):
        config=json.loads(path.read_text()); scene=setup()
        points,tip,target,horizontal,lateral=illustrated_arm(config)
        view=(-lateral+.36*Vector((0,0,1))-.42*horizontal).normalized()
        orange=material('direction',(.94,.39,.13)); haze=material('allowed directions',(.98,.59,.28),alpha=.10)
        label=cone(points[-1],Vector(config['task_constraint']['working_direction_world']),
                   (target-points[-1]).length,
                   config['task_constraint']['cone_half_angle_deg'],view,orange,haze)
        blue=material('station distance',(.23,.47,.67))
        root=Vector(config['feasibility']['base_pose_world']['position_m'])
        scale=config['robot_specification']['reach_m']/.855
        offset=-lateral*.19*scale
        start,finish=root+offset,target+offset
        line('distance extension',[root,start],.0015*scale,blue)
        line('distance extension',[target,finish],.0015*scale,blue)
        for i in range(16):
            line('base to surface distance',[start+(finish-start)*(i/16),
                 start+(finish-start)*((i+.6)/16)],.002*scale,blue)
        tick=(Vector((0,0,1))-(finish-start).normalized()*.3).normalized()*.018*scale
        for endpoint in (start,finish):
            line('distance tick',[endpoint-tick,endpoint+tick],.002*scale,blue)
        distance_label=(start+finish)/2-Vector((0,0,.055*scale))
        light('key',(-1,-2,3),(-.3,0,.5),180,3); light('fill',(2,1,2),(-.3,0,.5),110,2.5)
        bpy.context.view_layer.update()
        extent=[obj.matrix_world@Vector(corner) for obj in scene.objects
                if obj.type in ('MESH','CURVE') for corner in obj.bound_box]
        extent += [label+Vector((0,0,.06)),label-Vector((0,0,.06)),distance_label]
        cam=camera(extent,view,1.12); bpy.context.view_layer.update()
        coordinate=world_to_camera_view(scene,cam,label)
        distance_coordinate=world_to_camera_view(scene,cam,distance_label)
        (args.output/(config['id']+'.json')).write_text(json.dumps(dict(
            angle_label_px=[coordinate.x*1500,(1-coordinate.y)*1000],
            distance_label_px=[distance_coordinate.x*1500,(1-distance_coordinate.y)*1000])))
        scene.render.filepath=str(args.output/(config['id']+'.png'))
        bpy.ops.render.render(write_still=True)
        print('ROBOT FIGURE',config['id'],flush=True)


if __name__=='__main__': main()
