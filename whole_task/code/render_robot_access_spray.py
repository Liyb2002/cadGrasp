"""Render the native Panda spray motion in the unchanged robot_access workstation.

No font objects, overlays, numbers, titles or subtitles are added.
"""
from pathlib import Path
import argparse
import json
import math
import sys

import bpy
from mathutils import Matrix,Vector

ROOT=Path(__file__).resolve().parents[2]
for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
    sys.path.insert(0,str(site))
for library in list(sys.path):
    if Path(library).name==f'python{sys.version_info.major}.{sys.version_info.minor}':
        sys.path.insert(0,str(Path(library)/'site-packages'))
import numpy as np
DATA=ROOT/'whole_task/assets/robot_access_spray'
SOURCE=ROOT/'slides/idea/data/robot_access'
sys.path.insert(0,str(ROOT/'robots_config'))
from arm_visuals import cylinder,line,material,sphere


def mesh_object(name,vertices,faces,mat):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices.tolist(),[],faces.tolist());mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj);mesh.materials.append(mat)
    for face in mesh.polygons:face.use_smooth=True
    return obj


def set_camera(scene,center,view,width):
    obj=scene.camera; obj.location=Vector(center)+Vector(view).normalized()*3
    obj.rotation_euler=(Vector(center)-obj.location).to_track_quat('-Z','Y').to_euler()
    obj.data.type='ORTHO';obj.data.ortho_scale=width


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',type=int)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((DATA/'motion.json').read_text())
    trajectory=np.load(DATA/'trajectory.npz'); meshes=np.load(DATA/'robot_meshes.npz')
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE/'robot_access.blend'))
    scene=bpy.context.scene
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24
    scene.cycles.use_denoising=False;scene.render.resolution_x=1280;scene.render.resolution_y=800
    scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    remove=('robot_white','robot_black','Spindle mount','Processing spindle','Dark spindle collar',
            'Tool collet','Fine processing bit','Dashed rear access','Blocked cross')
    for obj in list(bpy.data.objects):
        if obj.name.startswith(remove) or obj.type=='FONT':bpy.data.objects.remove(obj,do_unlink=True)
    white=material('Spray Panda white',(.88,.90,.92),.08)
    black=material('Spray Panda charcoal',(.16,.19,.22),.3)
    silver=material('Spray nozzle alloy',(.60,.66,.72),.72)
    green=material('Atomized paint',(.05,.65,.67),.1)
    orange=material('Small allowed direction boundary',(.98,.53,.16))
    haze=material('Fine spray haze',(.2,.75,.78),alpha=.07)
    haze.blend_method='HASHED'
    parts=[]
    for item in metadata['robot_parts']:
        name=item['name']
        obj=mesh_object(name,meshes[name+'_v'],meshes[name+'_f'],white if item['color']=='white' else black)
        parts.append((obj,item['body']))
    tool=bpy.data.objects.new('Spray tool',None);bpy.context.collection.objects.link(tool)
    for name,a,b,r0,r1,mat in [
        ('Nozzle mount',-.018,.006,.030,.030,black),
        ('Nozzle body',.006,.060,.022,.022,silver),
        ('Nozzle collar',.060,.073,.016,.016,black),
        ('Nozzle taper',.073,.100,.013,.004,silver),
        ('Nozzle opening',.100,.103,.004,.004,black)]:
        obj=cylinder(name,(0,0,a),(0,0,b),r0,mat,r1);obj.parent=tool
    particles=[]
    seed=sphere('Spray particle',Vector((0,0,0)),1,green)
    for i in range(len(trajectory['particle_phases'])):
        obj=seed if i==0 else bpy.data.objects.new(f'Spray particle {i}',seed.data)
        if i:bpy.context.collection.objects.link(obj)
        particles.append(obj)
    nominal=Vector(metadata['nominal_axis_world'])
    view=Vector((-.38,-1.55,.83)).normalized()
    tangent=view.cross(nominal).normalized();other=nominal.cross(tangent)
    cone_lines=[]
    for sign in [-1,1]:
        pts=[Vector((0,0,0)),nominal*.047+tangent*(sign*.047*math.tan(math.radians(10)))]
        cone_lines.append(line('Allowed direction edge',pts,.00038,orange))
    circle=[nominal*.047+math.tan(math.radians(10))*.047*(math.cos(t)*tangent+math.sin(t)*other)
            for t in np.linspace(0,2*math.pi,65)]
    cone_lines.append(line('Allowed direction boundary',circle,.00028,orange))
    arc=[.026*(math.cos(t)*nominal+math.sin(t)*tangent) for t in np.linspace(-math.radians(10),math.radians(10),33)]
    cone_lines.append(line('Angle arc without label',arc,.0004,orange))
    # A thin, translucent spray frustum follows the moving nozzle, inside the fixed envelope.
    count=40
    plume_mesh=bpy.data.meshes.new('spray plume')
    plume_mesh.from_pydata([(0,0,0)]+[(.052*math.tan(math.radians(4))*math.cos(i*2*math.pi/count),
        .052*math.tan(math.radians(4))*math.sin(i*2*math.pi/count),.052) for i in range(count)],[],
        [(0,i+1,(i+1)%count+1) for i in range(count)])
    plume=bpy.data.objects.new('spray plume',plume_mesh);bpy.context.collection.objects.link(plume);plume_mesh.materials.append(haze)
    assert not any(obj.type=='FONT' for obj in scene.objects)
    camera_info=json.loads((SOURCE/'cameras.json').read_text())['01_robot_working']
    wide_center=np.array(camera_info['target']);wide_width=camera_info['width_m']
    close_center=np.array([.642,0,.705]);close_width=.40
    frames=[args.preview] if args.preview is not None else range(metadata['frames'])
    for frame in frames:
        for obj,body in parts:obj.matrix_world=Matrix(trajectory['link_transforms'][frame,body])
        H=trajectory['hand_transforms'][frame];tool.matrix_world=Matrix(H)
        nozzle=trajectory['nozzles'][frame]
        for obj in cone_lines:obj.location=nozzle
        plume_transform=H.copy();plume_transform[:3,3]=nozzle;plume.matrix_world=Matrix(plume_transform)
        for i,obj in enumerate(particles):
            phase=(frame/metadata['fps']*6+trajectory['particle_phases'][i])%1
            obj.location=nozzle+(trajectory['spray_endpoints'][frame,i]-nozzle)*phase
            obj.scale=(.00042+.00038*phase,)*3
        t=frame/metadata['fps']
        fraction=np.clip((t-1.6)/1.7,0,1);fraction=fraction*fraction*(3-2*fraction)
        center=(1-fraction)*wide_center+fraction*close_center
        width=(1-fraction)*wide_width+fraction*close_width
        set_camera(scene,center,view,width)
        scene.render.filepath=str(args.output/f'{frame:04d}.png')
        bpy.context.view_layer.update();bpy.ops.render.render(write_still=True)
        print('SPRAY FRAME',frame,flush=True)


if __name__=='__main__': main()
