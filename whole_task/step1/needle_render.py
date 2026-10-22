"""Rasterized fixed-needle spherical-sector figures; no text on Step1.3."""
import math
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
from whole_task.step5.video import raster as r
from .common import sha256,ROOT
from .work_disks import tangent_frame

GOLD=(.78,.55,.16)
LINE_VERTEX='''#version 330
in vec3 position;uniform mat4 vp;void main(){gl_Position=vp*vec4(position,1);}
'''
LINE_FRAGMENT='''#version 330
uniform vec3 ink;out vec4 pixel;void main(){pixel=vec4(ink,1);}
'''


def rod(a,b,radius):
    return trimesh.creation.cylinder(radius=radius,segment=np.asarray([a,b]),sections=16)


def sector_lines(ws):
    origin=np.asarray(ws['reference_head_front_world_m']);axis=np.asarray(ws['nominal_axis_world'])
    basis=tangent_frame(axis)[:,:2];alpha=np.deg2rad(ws['half_angle_deg']);length=ws['needle_length_m']
    phi=np.linspace(0,2*np.pi,161)
    directions=np.cos(alpha)*axis+np.sin(alpha)*(np.cos(phi)[:,None]*basis[:,0]+np.sin(phi)[:,None]*basis[:,1])
    circle=origin+length*directions;segments=[np.stack([circle[:-1],circle[1:]],axis=1).reshape(-1,3)]
    for f in np.arange(0,1,.125):
        d=np.cos(alpha)*axis+np.sin(alpha)*(np.cos(f*2*np.pi)*basis[:,0]+np.sin(f*2*np.pi)*basis[:,1])
        segments.append(np.array([origin,origin+length*d]))
    # Spherical cap meridians show that r is Euclidean radius, not axial depth.
    for phi in np.arange(0,2*np.pi,np.pi/2):
        t=np.linspace(0,alpha,35);d=np.cos(t)[:,None]*axis+np.sin(t)[:,None]*(np.cos(phi)*basis[:,0]+np.sin(phi)*basis[:,1])
        points=origin+length*d;segments.append(np.stack([points[:-1],points[1:]],1).reshape(-1,3))
    return np.vstack(segments)


class Renderer:
    def __init__(self,width=850,height=800):
        import moderngl
        self.gl=moderngl;self.ctx=r.context();self.width=width;self.height=height
        self.studio=r.Studio(self.ctx,width,height)
        self.shader=self.ctx.program(vertex_shader=r.VERTEX,fragment_shader=r.FRAGMENT)
        self.shadow=self.ctx.program(vertex_shader=r.SHADOW_VERTEX,fragment_shader=r.SHADOW_FRAGMENT)
        self.lines=self.ctx.program(vertex_shader=LINE_VERTEX,fragment_shader=LINE_FRAGMENT)
        self.shader['shadow_map'].value=3

    def draw(self,items,points,direction,lines=None):
        direction=np.asarray(direction,dtype=float);direction/=np.linalg.norm(direction)
        if np.linalg.norm(np.cross([0,0,1],direction))<1e-5:direction=direction+[.01,0,0];direction/=np.linalg.norm(direction)
        right=np.cross([0,0,1],direction);right/=np.linalg.norm(right);up=np.cross(direction,right)
        basis=np.array([right,up,direction]);projected=np.asarray(points)@basis.T
        low,high=projected.min(0),projected.max(0);target=((low+high)/2)@basis
        scale=max(high[0]-low[0],(high[1]-low[1])*self.width/self.height)*1.16
        vp=r.orthographic(scale,scale*self.height/self.width)@r.view_matrix(direction,target)
        self.shader['view_projection'].write(np.asarray(vp,dtype='f4').T.tobytes());self.shader['eye'].value=tuple(target+direction*4)
        key=np.array([-.65,-.9,1.45]);key/=np.linalg.norm(key)
        light=r.orthographic(scale*1.8,scale*1.8)@r.view_matrix(key,target)
        for shader in [self.shader,self.shadow]:shader['light_projection'].write(np.asarray(light,dtype='f4').T.tobytes())
        shapes=[]
        for mesh,color,mask in items:
            shape=r.Shape(self.ctx,self.shader,mesh.vertices,mesh.faces,color,dynamic=mask is not None,
                metallic=.04,roughness=.57)
            if mask is not None:
                colors=np.tile(color,(len(mesh.faces),1));colors[mask]=r.PINK;shape.paint(colors)
            shapes.append(shape)
        ctx=self.ctx;ctx.enable(self.gl.DEPTH_TEST)
        self.studio.shadow_buffer.use();ctx.viewport=(0,0,1536,1536);self.studio.shadow_buffer.clear(depth=1)
        for shape in shapes:shape.shadow(ctx,self.shadow,np.eye(4))
        self.studio.gbuffer.use();ctx.viewport=(0,0,*self.studio.size);self.studio.gbuffer.clear(1,1,1,1,depth=1);self.studio.shadow.use(3)
        for shape in shapes:shape.draw(self.shader,np.eye(4))
        if lines is not None:
            self.lines['vp'].write(np.asarray(vp,dtype='f4').T.tobytes());self.lines['ink'].value=GOLD
            buffer=ctx.buffer(np.asarray(lines,dtype='f4').tobytes());vao=ctx.vertex_array(self.lines,[(buffer,'3f','position')])
            ctx.line_width=1.6;vao.render(self.gl.LINES);vao.release();buffer.release()
        image=Image.frombytes('RGB',(self.width,self.height),self.studio.finish(vp,direction)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        for shape in shapes:
            shape.vao.release();shape.shadow_vao.release();shape.buffer.release();shape.colors.release()
        return image

    def close(self):self.ctx.release()


def draw_examples(mesh,masks,transforms,heads,views,ws,output):
    renderer=Renderer();canvas=Image.new('RGB',(3400,1600),'white');metadata=[]
    O=np.asarray(ws['reference_head_front_world_m']);sector=sector_lines(ws)
    try:
        for i,(mask,T,head,view) in enumerate(zip(masks,transforms,heads,views)):
            left=renderer.draw([(mesh,r.GRAY,mask)],mesh.vertices,view)
            world=mesh.copy();world.apply_transform(T)
            points=world.triangles_center[mask]
            center=np.average(points,weights=mesh.area_faces[mask],axis=0)
            # The rigid reference needle illustrates the sector radius. The
            # selected region is the entire sector intersection, not a path.
            aim=points[np.linalg.norm(points-center,axis=1).argmin()]
            axis=aim-O;axis/=np.linalg.norm(axis)
            root=O.copy();tcp=root+ws['needle_length_m']*axis
            flange=root-ws['head_front_to_flange_m']*axis
            headball=trimesh.creation.icosphere(subdivisions=3,radius=ws['head_proxy_radius_m']);headball.apply_translation(flange)
            shaft=rod(root,tcp-axis*.002,ws['needle_outer_radius_m'])
            tip=trimesh.creation.cone(radius=ws['needle_outer_radius_m'],height=.002,sections=16)
            A=trimesh.geometry.align_vectors([0,0,1],axis);A[:3,3]=tcp-axis*.002;tip.apply_transform(A)
            stem=rod(flange-axis*.025,flange-axis*.045,.011)
            floor=trimesh.creation.box([1.2,1.2,.008]);floor.apply_translation([0,-.5,-.004])
            items=[(floor,(.92,.93,.94),None),(world,r.GRAY,mask),(headball,(.73,.77,.8),None),
                   (shaft,(.48,.53,.58),None),(tip,(.48,.53,.58),None),(stem,(.65,.69,.73),None)]
            bounds=np.vstack([world.vertices,sector,headball.vertices,stem.vertices])
            right=renderer.draw(items,bounds,[1.1,1.4,.8],sector)
            row,col=divmod(i,2);canvas.paste(left,(col*1700,row*800));canvas.paste(right,(col*1700+850,row*800))
            metadata.append(dict(id=f'C{i+1:02}',illustrated_tip_world_m=tcp.tolist(),head_front_world_m=root.tolist(),
                flange_world_m=flange.tolist(),needle_axis_world=axis.tolist(),rendered_needle_length_m=float(np.linalg.norm(tcp-root)),
                needle_illustrates_sector_radius_not_contact_path=True,gold_sphere_radius_m=ws['needle_length_m'],
                head_is_spherical_proxy=True,native_robot_pose_certified=False))
        canvas.save(output/'working_areas_overview.png')
    finally:renderer.close()
    return dict(no_text=True,color='pink',overview_size=list(canvas.size),scenes=metadata,
        engine='OpenGL rasterization',spherical_cap_not_axial_plane=True,
        head_and_sphere_center_coincide=True,working_area_is_geometric_intersection=True,
        source_sha256={str(p.relative_to(ROOT)):sha256(p)for p in [__import__('pathlib').Path(__file__),__import__('pathlib').Path(r.__file__)]})


def draw_configuration(config,ws,output):
    renderer=Renderer(1100,900);O=np.asarray(ws['reference_head_front_world_m']);axis=np.asarray(ws['nominal_axis_world'])
    root=O.copy();point=root+ws['needle_length_m']*axis;flange=root-ws['head_front_to_flange_m']*axis
    head=trimesh.creation.icosphere(subdivisions=3,radius=ws['head_proxy_radius_m']);head.apply_translation(flange)
    shaft=rod(root,point,ws['needle_outer_radius_m']);floor=trimesh.creation.box([1.3,1.3,.008]);floor.apply_translation([0,-.45,-.004])
    items=[(floor,(.93,.94,.95),None),(head,(.73,.77,.80),None),(shaft,(.45,.5,.56),None)]
    # Native robot geometry is shown as context at one solved reference pose.
    native_ik=False
    try:
        from whole_task.step5.video.kuka import Arm,visual_parts,description
        arm=Arm(config['feasibility']['base_pose_world']['position_m'],np.asarray(config['feasibility']['base_pose_world']['rotation_matrix']),tool=.12)
        q=arm.solve(point,-axis);_,_,frames=arm.forward(q,geometry=True);links,_,_=description()
        for j,(rotation,position) in enumerate(frames):
            offset=np.fromstring(links[j].find('visual/origin').get('xyz'),sep=' ')
            for v,f,color in visual_parts(j):
                part=trimesh.Trimesh((v+offset)@rotation.T+position,f,process=False)
                items.append((part,tuple(np.asarray(color[:3])),None))
        native_ik=True
    except ValueError:
        # The diagram still faithfully represents the declared geometric model.
        pass
    bounds=np.vstack([m.vertices for m,c,k in items if m is not floor]+[sector_lines(ws)])
    try:picture=renderer.draw(items,bounds,[1.2,1.5,.8],sector_lines(ws))
    finally:renderer.close()
    canvas=Image.new('RGB',(1600,900),'white');canvas.paste(picture,(0,0));draw=ImageDraw.Draw(canvas)
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    for y,text,size in [(90,'STEP 1.1',26),(160,'Needle / sector radius',30),(220,f'{1000*ws["needle_length_m"]:g} mm',46),
        (330,f'Direction half-angle: {ws["half_angle_deg"]:g} deg',23),
        (390,'Sphere centre = needle root',23),(450,'All enclosed surface is selected',23),
        (510,'No normal or visibility filtering',22),(590,'Step1: geometric region model',21)]:
        draw.text((1095,y),text,font=ImageFont.truetype(font,size),fill='#344d60')
    canvas.save(output/'result.png')
    return dict(native_reference_pose_IK_solved=native_ik,full_arm_collision_certified=False,figure_size=list(canvas.size))
