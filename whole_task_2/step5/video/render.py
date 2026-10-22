"""White studio two-robot rendering, styled after slides/idea/vis/robot_access."""
from pathlib import Path
import argparse,json,sys,math
import bpy
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[3]
for site in (ROOT/'.venv/lib').glob('python*/site-packages'):sys.path.insert(0,str(site))
for library in list(sys.path):
    if Path(library).name==f'python{sys.version_info.major}.{sys.version_info.minor}':
        sys.path.insert(0,str(Path(library)/'site-packages'))
import numpy as np
sys.path.insert(0,str(ROOT/'robots_config'))
from arm_visuals import material,cylinder,box,line,light


def mesh(name,v,f,mat,smooth=False):
    data=bpy.data.meshes.new(name);data.from_pydata(v.tolist(),[],f.tolist());data.update()
    obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj);data.materials.append(mat)
    for p in data.polygons:p.use_smooth=smooth
    return obj


def camera(scene,target,view,width):
    scene.camera.location=Vector(target)+Vector(view).normalized()*4
    scene.camera.rotation_euler=(Vector(target)-scene.camera.location).to_track_quat('-Z','Y').to_euler()
    scene.camera.data.ortho_scale=width


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='*');p.add_argument('--samples',type=int,default=16)
    p.add_argument('--engine',default='BLENDER_WORKBENCH');a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    a.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((a.data/'plan.json').read_text());t=np.load(a.data/'trajectory.npz');geometry=np.load(a.data/'scene_meshes.npz');robots=np.load(a.data/'robot_meshes.npz')
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine=a.engine
    scene.eevee.taa_render_samples=a.samples;scene.eevee.use_gtao=True;scene.eevee.gtao_distance=.035;scene.eevee.gtao_factor=1.15
    scene.eevee.use_soft_shadows=True
    scene.cycles.samples=a.samples;scene.cycles.use_denoising=False
    scene.render.use_persistent_data=True
    scene.render.resolution_x=1280;scene.render.resolution_y=800;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=True;scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(1,1,1,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.8
    scene.view_settings.view_transform='Standard';scene.view_settings.look='Medium High Contrast';scene.view_settings.exposure=0
    scene.use_nodes=True;nodes=scene.node_tree.nodes;nodes.clear()
    layers=nodes.new('CompositorNodeRLayers');exposure=nodes.new('CompositorNodeExposure');exposure.inputs[1].default_value=-1.35
    over=nodes.new('CompositorNodeAlphaOver');over.inputs[1].default_value=(1,1,1,1);output=nodes.new('CompositorNodeComposite')
    scene.node_tree.links.new(layers.outputs['Image'],exposure.inputs[0]);scene.node_tree.links.new(exposure.outputs[0],over.inputs[2]);scene.node_tree.links.new(over.outputs[0],output.inputs[0])
    if a.engine=='BLENDER_WORKBENCH':
        # Fast solid viewport rendering: no ray tracing, scene lights or compositor.
        scene.render.film_transparent=False;scene.render.use_compositing=False
        scene.display.render_aa='FXAA'
        shading=scene.display.shading;shading.light='STUDIO';shading.studio_light='paint.sl'
        shading.color_type='MATERIAL';shading.show_shadows=False;shading.show_cavity=False
        shading.show_specular_highlight=True;shading.show_object_outline=False
        shading.background_type='WORLD';scene.world.color=(1,1,1)
        scene.view_settings.exposure=0;scene.view_settings.look='Medium High Contrast'
    white=material('Ceramic robot housing',(.88,.90,.92),.08,roughness=.38)
    black=material('Charcoal joints',(.16,.19,.22),.3)
    gray=material('Gray workpiece',(.66,.69,.72),.13)
    blue=material('Blue final fixture',(.16,.43,.64),.12)
    silver=material('Tool alloy',(.60,.66,.72),.72,roughness=.28)
    pedestal=material('White work table',(.91,.92,.93),.08)
    pale_pink=material('Pink working area',(1,.52,.74),.05)
    pink=material('Pink paint',(1,.52,.74),.05)
    ink=material('Fresh pink brush stroke',(.78,.22,.49),.05)
    obj=mesh('Bunny',geometry['object_v'],geometry['object_f'],gray,True)
    fixture=mesh('Final sparse foot fixture',geometry['fixture_v'],geometry['fixture_f'],blue)
    obj.data.materials.append(pale_pink);obj.data.materials.append(pink)
    patch_count=len(plan['object_poses']);strokes=[]
    for i in range(patch_count):
        curve=bpy.data.curves.new('Pink strokes '+str(i),'CURVE');curve.dimensions='3D'
        curve.resolution_u=1;curve.bevel_depth=.0005;curve.bevel_resolution=2
        curve.materials.append(ink)
        trace=bpy.data.objects.new('Pink strokes '+str(i),curve);bpy.context.collection.objects.link(trace)
        strokes.append(trace)

    robot_parts=[]
    for part in plan['robot_parts']:
        color=part['color']
        if isinstance(color,str):mat=white if color=='white' else black
        else:mat=material(part['name']+' surface',tuple(color),.12)
        robot_parts.append((mesh(part['name'],robots[part['name']+'_v'],robots[part['name']+'_f'],mat,True),part))
    tool=bpy.data.objects.new('Fixed direction needle',None);bpy.context.collection.objects.link(tool)
    for name,start,end,r0,r1,mat in [('Tool flange',.030,.050,.018,.018,black),('Tool barrel',.050,.105,.010,.010,silver),
        ('Tool collet',.105,.123,.008,.003,silver),('Fine needle',.123,.155,.0012,.0006,silver)]:
        piece=cylinder(name,(0,0,start),(0,0,end),r0,mat,r1);piece.parent=tool
    z=plan['table_height_m'];base_z=plan.get('work_robot_base_world_m',[0,0,z])[2]
    # Broad clean white bench and a lower fixed mount for the working arm.
    box('Main workbench',(.79,.03,z/2),(1.10,1.00,z),pedestal,.014)
    box('Work robot mount',(0,0,base_z/2),(.30,.34,base_z),pedestal,.014)
    floor=material('Studio floor',(1,1,1))
    studio=box('Infinite studio',(.5,0,-.04),(200,200,.06),floor,.001);studio.hide_render=True
    light('Large softbox',(-1.2,-2.8,4.3),(.5,0,.7),140,3)
    light('Soft fill',(2.6,1.3,3.0),(.5,0,.7),100,2.5)
    light('Top edge',(0,2.0,4.0),(.5,0,.7),90,2.5)
    cam=bpy.data.cameras.new('Camera');cam.type='ORTHO';scene.camera=bpy.data.objects.new('Camera',cam);bpy.context.collection.objects.link(scene.camera)
    for piece in scene.objects:
        if piece.type=='MESH':piece.data.use_auto_smooth=True
    assert not any(o.type=='FONT' for o in scene.objects)
    kernel=np.exp(-.5*(np.arange(-18,19)/6.)**2);kernel/=kernel.sum()
    push=np.convolve((t['work_task']>=0).astype(float),kernel,mode='same')
    # Fit the native visual bounds over every frame, including raised gripper paths.
    view=Vector((-.35,-1.75,.85)).normalized();rotation=(-view).to_track_quat('-Z','Y')
    right=np.array(rotation@Vector((1,0,0)));up=np.array(rotation@Vector((0,1,0)));points=[]
    for piece,part in robot_parts:
        v=np.array([p[:] for p in piece.bound_box]);index=part['body'] if part['robot']=='kuka' else plan['panda_bodies'].index(part['body'])
        transforms=t['kuka_links' if part['robot']=='kuka' else 'panda_links'][:,index]
        points.append(np.einsum('fij,nj->fni',transforms[:,:3,:3],v)+transforms[:,None,:3,3])
    points=np.concatenate([p.reshape(-1,3) for p in points]);x=points@right;y=points@up
    wide=right*((x.min()+x.max())/2)+up*((y.min()+y.max())/2)+np.array(view)*.1
    wide_width=max(x.max()-x.min(),(y.max()-y.min())*1.6)*1.12
    frames=a.frames if a.frames else range(plan['frames'])
    for f in frames:
        obj.matrix_world=Matrix(t['object_T'][f]);fixture.matrix_world=Matrix(t['fixture_T'][f])
        # Painted faces remain pink while the gripper moves the bunny.
        face_material=np.zeros(len(obj.data.polygons),dtype=np.int32)
        for i in range(patch_count):
            progress=float(t['paint_progress'][f,i]);active=int(t['work_task'][f])==i
            if progress>=1-1e-8:face_material[geometry[f'patch_{i}_ids']]=2
            elif active:
                ids=geometry[f'patch_{i}_ids'];face_material[ids]=np.maximum(face_material[ids],1)
            trace=strokes[i];trace.hide_render=progress<=0;trace.matrix_world=obj.matrix_world
            trace.data.splines.clear()
            if progress>0:
                # Progress switches from the completed outline to interior strokes.
                phase=.47/.80
                for name,u in [('outline',min(progress/phase,1.)),('interior',max(0.,(progress-phase)/(1-phase)))]:
                    if u<=0:continue
                    path=geometry[f'path_{i}_{name}'];arc=geometry[f'path_{i}_{name}_s']
                    j=min(int(np.searchsorted(arc,u,side='right')),len(arc)-1)
                    prev=max(0,j-1);v=(u-arc[prev])/max(arc[j]-arc[prev],1e-15)
                    tip=path[prev]*(1-v)+path[j]*v
                    points=np.concatenate([path[:j],tip[None]])
                    # Tiny outward lift makes the surface stroke visible.
                    local_axis=np.asarray(plan['work_axis_world'])@np.asarray(plan['object_poses'][i])[:3,:3]
                    points=points-local_axis*.00012
                    spline=trace.data.splines.new('POLY');spline.points.add(len(points)-1)
                    spline.points.foreach_set('co',np.c_[points,np.ones(len(points))].ravel())
        obj.data.polygons.foreach_set('material_index',face_material)
        obj.data.update()

        for piece,part in robot_parts:
            index=part['body'] if part['robot']=='kuka' else plan['panda_bodies'].index(part['body'])
            piece.matrix_world=Matrix(t['kuka_links' if part['robot']=='kuka' else 'panda_links'][f,index])
        tool.matrix_world=Matrix(t['kuka_links'][f,-1])
        # Hold a common overview during handling; a gentle push during work
        # keeps both robots visible while making the small working region clear.
        task=int(t['task'][f]);close=np.array([.50,0,.74])
        camera(scene,wide+(close-wide)*push[f],view,wide_width+(.72-wide_width)*push[f])
        bpy.context.view_layer.update();scene.render.filepath=str(a.output/f'{f:04d}.png')
        bpy.ops.render.render(write_still=True)
        if f%12==0:print('STEP5 FRAME',f,'/',plan['frames'],flush=True)


if __name__=='__main__':main()
