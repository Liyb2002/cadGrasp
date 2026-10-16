"""Blender primitives for robot and task-direction illustrations."""
import math
import bpy
from mathutils import Vector


def material(name, color, metallic=0, alpha=1, roughness=.38):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    rgb = tuple(c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in color)
    rgba = rgb + (alpha,)
    result.diffuse_color = rgba
    node = result.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value = rgba
    node.inputs['Metallic'].default_value = metallic
    node.inputs['Roughness'].default_value = roughness
    node.inputs['Alpha'].default_value = alpha
    return result


def cylinder(name, first, last, radius, mat, radius2=None, bevel=True):
    first, last = Vector(first), Vector(last)
    delta = last-first
    bpy.ops.mesh.primitive_cone_add(vertices=40, radius1=radius,
                                   radius2=radius if radius2 is None else radius2, depth=delta.length)
    obj = bpy.context.object
    obj.name = name
    obj.location = (first+last)/2
    obj.rotation_euler = delta.to_track_quat('Z', 'Y').to_euler()
    obj.data.materials.append(mat)
    for face in obj.data.polygons: face.use_smooth = len(face.vertices)==4
    if bevel:
        mod=obj.modifiers.new('rounded edges','BEVEL'); mod.width=radius*.12; mod.segments=3
        obj.modifiers.new('normals','WEIGHTED_NORMAL')
    return obj


def sphere(name, center, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=radius, location=center)
    obj=bpy.context.object; obj.name=name; obj.data.materials.append(mat)
    for face in obj.data.polygons: face.use_smooth=True
    return obj


def box(name, center, size, mat, radius=.01):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj=bpy.context.object; obj.name=name; obj.dimensions=size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    mod=obj.modifiers.new('soft edges','BEVEL'); mod.width=radius; mod.segments=4
    obj.modifiers.new('normals','WEIGHTED_NORMAL')
    return obj


def line(name, points, radius, mat):
    curve=bpy.data.curves.new(name,'CURVE'); curve.dimensions='3D'
    curve.bevel_depth=radius; curve.bevel_resolution=3
    poly=curve.splines.new('POLY'); poly.points.add(len(points)-1)
    for item,point in zip(poly.points,points): item.co=(*point,1)
    obj=bpy.data.objects.new(name,curve); bpy.context.collection.objects.link(obj)
    curve.materials.append(mat)
    return obj


def direction_arrow(start, end, mat, radius=.005):
    start,end=Vector(start),Vector(end); delta=(end-start).normalized()
    cylinder('direction arrow',start,end-delta*.025,radius,mat,bevel=False)
    cylinder('direction arrowhead',end-delta*.03,end,radius*3,mat,0,bevel=False)


def cone(pivot, axis, length, half_angle_deg, view, mat, haze):
    pivot,axis,view=Vector(pivot),Vector(axis).normalized(),Vector(view).normalized()
    first=view.cross(axis).normalized(); second=axis.cross(first)
    alpha=math.radians(half_angle_deg)
    center=pivot+axis*length
    radius=length*math.tan(alpha)
    ring=[center+radius*(math.cos(i*2*math.pi/80)*first+math.sin(i*2*math.pi/80)*second) for i in range(81)]
    mesh=bpy.data.meshes.new('allowed direction cone')
    mesh.from_pydata([pivot]+ring[:-1],[],[(0,i+1,(i+1)%80+1) for i in range(80)])
    obj=bpy.data.objects.new('allowed direction cone',mesh); bpy.context.collection.objects.link(obj)
    mesh.materials.append(haze)
    line('cone boundary',ring,.0012,mat)
    for sign in (-1,1):
        end=center+sign*radius*first
        for i in range(9):
            line('boundary dash',[pivot+(end-pivot)*(i/9),pivot+(end-pivot)*((i+.55)/9)],.0013,mat)
    direction_arrow(pivot,center,mat,.0036)
    distance=min(.17,length*.58)
    arc=[pivot+distance*(math.cos(-alpha+2*alpha*i/40)*axis+math.sin(-alpha+2*alpha*i/40)*first) for i in range(41)]
    line('angle arc',arc,.0025,mat)
    return pivot+axis*distance+first*(distance*math.sin(alpha)+.075)


def light(name, position, target, energy, size):
    data=bpy.data.lights.new(name,'AREA'); data.energy=energy; data.shape='DISK'; data.size=size
    obj=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(obj)
    obj.location=position; obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def setup(width=1500,height=1000,samples=48):
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene; scene.render.engine='CYCLES'
    scene.cycles.device='CPU'; scene.cycles.samples=samples; scene.cycles.use_denoising=False
    scene.render.resolution_x=width; scene.render.resolution_y=height; scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=True
    scene.view_settings.view_transform='Standard'; scene.view_settings.look='Medium High Contrast'
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(1,1,1,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.65
    scene.use_nodes=True
    nodes=scene.node_tree.nodes; nodes.clear()
    layers=nodes.new('CompositorNodeRLayers')
    exposure=nodes.new('CompositorNodeExposure'); exposure.inputs[1].default_value=-1.2
    over=nodes.new('CompositorNodeAlphaOver'); over.inputs[1].default_value=(1,1,1,1)
    output=nodes.new('CompositorNodeComposite')
    scene.node_tree.links.new(layers.outputs['Image'],exposure.inputs[0])
    scene.node_tree.links.new(exposure.outputs[0],over.inputs[2])
    scene.node_tree.links.new(over.outputs[0],output.inputs[0])
    return scene


def camera(points, view, margin=1.15):
    scene=bpy.context.scene
    view=Vector(view).normalized(); rotation=(-view).to_track_quat('-Z','Y')
    right=rotation@Vector((1,0,0)); up=rotation@Vector((0,1,0))
    x=[Vector(p).dot(right) for p in points]; y=[Vector(p).dot(up) for p in points]
    target=right*((min(x)+max(x))/2)+up*((min(y)+max(y))/2)
    target+=view*(sum(Vector(p).dot(view) for p in points)/len(points))
    width=max(max(x)-min(x),(max(y)-min(y))*scene.render.resolution_x/scene.render.resolution_y)*margin
    data=bpy.data.cameras.new('Camera'); data.type='ORTHO'; data.ortho_scale=width
    obj=bpy.data.objects.new('Camera',data); bpy.context.collection.objects.link(obj)
    obj.location=target+view*4; obj.rotation_euler=rotation.to_euler(); scene.camera=obj
    return obj


def illustrated_arm(config):
    """A 6/7-joint silhouette, not an exact manufacturer CAD or IK model."""
    axis=Vector(config['task_constraint']['working_direction_world']).normalized()
    horizontal=Vector((axis.x,axis.y,0))
    if horizontal.length<.01: horizontal=Vector((1,0,0))
    horizontal.normalize(); lateral=Vector((0,0,1)).cross(horizontal); up=Vector((0,0,1))
    task=config['task_constraint']; patch=task['world_position_constraint']
    target=Vector(patch['center_world_m'])
    root=Vector(config['feasibility']['base_pose_world']['position_m'])
    scale=config['robot_specification']['reach_m']/.855
    gap=task['tcp_to_surface_distance_m']['nominal']
    tool_length=config['feasibility']['tcp_transform']['translation_flange_m'][2]
    tip=target-axis*gap; end=tip-axis*tool_length
    shoulder=root+up*.23*scale; elbow=root+horizontal*.17*scale+up*.65*scale
    wrist=end-axis*.16*scale+up*.04*scale
    points=[root+up*.11*scale,shoulder,elbow,wrist,end-axis*.075*scale,end]
    if config['robot_specification']['dof']==7:
        points.insert(2,shoulder*.5+elbow*.5+lateral*.025*scale)
    family=config['robot_specification']['manufacturer']
    body_rgb=(.9,.92,.94); accent_rgb=(.19,.22,.26)
    if family=='KUKA': body_rgb=(.98,.40,.09); accent_rgb=(.25,.27,.29)
    if 'med14' in config['id']: body_rgb=(.93,.94,.95); accent_rgb=(.55,.30,.62)
    if family=='Universal Robots': body_rgb=(.69,.74,.78); accent_rgb=(.32,.64,.83)
    if family=='ABB': body_rgb=(.94,.94,.94); accent_rgb=(.84,.14,.13)
    if family=='FANUC': body_rgb=(.92,.94,.91); accent_rgb=(.36,.68,.36)
    body=material('arm housing',body_rgb,.18); joints=material('joint housings',(.16,.19,.22),.25)
    accent=material('joint caps',accent_rgb,.15); metal=material('tool metal',(.63,.69,.74),.65)
    base=material('base',(.65,.69,.73),.25)
    box('base flange',root+up*.025*scale,tuple(v*scale for v in (.29,.27,.05)),base)
    cylinder('base column',root+up*.045*scale,points[0],.104*scale,body)
    slender=family=='Universal Robots'
    for i,(a,b) in enumerate(zip(points,points[1:])):
        radius=(.050 if slender else .068)*(1 if i<3 else .72)*scale; direction=(b-a).normalized()
        cylinder('dark link core',a,b,radius*.72,joints)
        cylinder('arm link',a+direction*.025*scale,b-direction*.025*scale,radius,body,radius*.84)
    for i,point in enumerate(points):
        radius=(.073 if slender else .084)*(1 if i<3 else .72)*scale
        sphere('joint',point,radius,joints)
        cylinder('joint axle',point-lateral*radius*.8,point+lateral*radius*.8,radius*.85,body)
        cylinder('joint cap',point-lateral*radius*.86,point-lateral*radius*.80,radius*.66,accent)
    cylinder('tool body',end,end+axis*tool_length*.45,.026*scale,metal,.021*scale)
    cylinder('tool nozzle',end+axis*tool_length*.45,tip,.018*scale,metal,.006*scale)
    if gap==0:
        cylinder('contact tool face',tip-axis*.012,tip,.028*scale,metal)
    radius=patch['radius_m']; thickness=.04*scale
    work=box('workpiece',target+axis*thickness/2,(radius*2.5,radius*2.5,thickness),
             material('workpiece',(.70,.75,.77)),.012*scale)
    work.rotation_euler=axis.to_track_quat('Z','Y').to_euler()
    cylinder('assumed work window',target-axis*.0015,target-axis*.0005,radius,
             material('surface',(.50,.79,.72)),bevel=False)
    return points,tip,target,horizontal,lateral
