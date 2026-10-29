"""Reference-style fixture orbit with all pressure hulls and four task views.

The blue triangles are the saved final feet and membrane when a certified
final design is supplied, or the membrane alone for a ground-plane preview.
Every ground plane stays tangent to the retained membrane. All original
paired loads are reprojected to that plane; previews carry no design certificate.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
import trimesh

from subdivide_task.operations.state import transform
from subdivide_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step3.run_membranes import load_model
from subdivide_task.step4.membrane_search import read_group
from .height_reference import lower_layout, model_for_report
from .membrane_movie import Scene
from .video import raster as r

COLORS = np.array([[.87,.44,.19], [.49,.32,.66], [.12,.55,.46],
                   [.78,.27,.44], [.66,.55,.14], [.29,.43,.71]])
BLUE = (.16,.53,.76)
VIEW = np.array([1., -1., .9]) / np.linalg.norm([1., -1., .9])

MARKER_VERTEX = '''#version 330
in vec3 position;uniform mat4 world;uniform mat4 vp;uniform float point_size;
void main(){gl_Position=vp*world*vec4(position,1);gl_Position.z-=.000001*gl_Position.w;
gl_PointSize=point_size;}
'''
MARKER_FRAGMENT = '''#version 330
uniform vec3 ink;uniform bool point_marker;
layout(location=0)out vec4 pixel;layout(location=1)out vec4 surface_normal;
void main(){if(point_marker && length(gl_PointCoord-vec2(.5))>.5)discard;
pixel=vec4(pow(ink,vec3(2.2)),1);surface_normal=vec4(.5,.5,1,1);}
'''


def selected_tasks(layout, ids, count=4):
    """Prefer distinct ground normals; include the W13/W12 limiting example."""
    count = min(count, len(ids))
    chosen = [ids.index(name) for name in ('W13','W12') if name in ids][:count]
    if not chosen: chosen = [0]
    normals = layout.material[:, :3, 2]
    angles = np.arccos(np.clip(normals @ normals.T, -1., 1.))
    while len(chosen) < count:
        remaining = [i for i in range(len(ids)) if i not in chosen]
        chosen.append(max(remaining, key=lambda i: (float(angles[i,chosen].min()), -i)))
    return chosen


def collect(model, groups):
    families = []
    for number, group in enumerate(groups, 1):
        q, final_mesh = read_group(group)
        original, core_mesh = read_group(group['core'])
        mesh=final_mesh if group.get('display_final_support',False) else core_mesh
        if group.get('grounding'):
            from .ground_contact import normalize_ground
            expected=normalize_ground(original,np.asarray(group['grounding']['global_floor_heights_m']))
        else:
            distance = (group.get('height_reference') or {}).get('lowering_m', 0.)
            expected = lower_layout(original, distance)
        for key, value in q.arrays().items():
            np.testing.assert_array_equal(value, expected.arrays()[key])
        part = model.subset(group['members'])
        tasks = part.prepare(q)
        normals = q.material[:,:3,2]
        installation_offsets = np.einsum('ij,ij->i',normals,q.material[:,:3,3])
        offsets = (core_mesh.vertices@normals.T).min(0)
        ground_heights = offsets-installation_offsets
        assigned = set(); faces = []
        for i in range(len(tasks)):
            if i in assigned: continue
            indices = [j for j in range(len(tasks)) if np.linalg.norm(normals[j]-normals[i])<1e-8
                       and abs(offsets[j]-offsets[i])<1e-8]
            assigned.update(indices)
            local_clouds = []; material_clouds = []; demands = []
            plane = np.linalg.inv(q.material[i])
            for j in indices:
                task = tasks[j]
                height = ground_heights[j]
                def on_ground(targets):
                    xy=G.pressure_points(targets,task['domain'].com)+height*targets[:,:2]/targets[:,2,None]
                    points=np.c_[xy,np.full(len(xy),height)]
                    # The horizontal moment equations must hold at the new
                    # height, not merely after moving old polygon vertices.
                    origin_moment=targets[:,3:]+np.cross(task['domain'].com,targets[:,:3])
                    residual=np.cross(points,targets[:,:3])[:,:2]-origin_moment[:,:2]
                    if np.abs(residual).max()>1e-10:raise ValueError('Incorrect pressure projection at contact plane')
                    return transform(points,q.material[j])
                material = on_ground(task['samples']['need_wrench'])
                boundary = on_ground(task['endpoints']['endpoint_need_wrench'])
                local = transform(np.vstack([material,boundary]), plane)
                if np.abs(local[:,2]-ground_heights[i]).max()>1e-8: raise ValueError('Cloud leaves its tangent ground plane')
                material_clouds.append(material); local_clouds.append(local[:,:2])
                demands.append(dict(id=task['id'],sample_count=len(material),endpoint_count=len(boundary),
                                    floor_world_z_m=float(height),shell_ground_gap_m=float(
                                        (mesh.vertices@normals[j]-offsets[j]).min())))
            points = np.vstack(local_clouds)
            hull = ConvexHull(points)
            if (points@hull.equations[:,:2].T+hull.equations[:,2]).max()>1e-9:
                raise ValueError('Drawn hull omits an original paired demand')
            boundary = transform(np.c_[points[hull.vertices],np.full(len(hull.vertices),ground_heights[i])], q.material[i])
            faces.append(dict(cloud=np.vstack(material_clouds),hull=boundary,
                              color=COLORS[len(faces)%len(COLORS)],tasks=demands,
                              normal=normals[i],offset=float(offsets[i])))
        families.append(dict(number=number,group=group,layout=q,mesh=mesh,ids=part.ids,
                             faces=faces,ground_heights=ground_heights,
                             selected=selected_tasks(q,part.ids)))
    return families


def camera(points, direction, width, height, *, orbit=False):
    points = np.asarray(points)
    center = (points.min(0)+points.max(0))/2
    relative = points-center
    directions = [direction]
    if orbit:
        azimuth = np.arctan2(direction[1],direction[0]); elevation = np.arcsin(direction[2])
        directions = [np.array([np.cos(elevation)*np.cos(a),np.cos(elevation)*np.sin(a),np.sin(elevation)])
                      for a in azimuth+np.linspace(0,2*np.pi,73)]
    scale = 0.
    for d in directions:
        right = np.cross([0,0,1],d);right/=np.linalg.norm(right)
        up = np.cross(d,right)
        scale = max(scale, 2*float(np.abs(relative@right).max()),
                    2*float(np.abs(relative@up).max())*width/height)
    return center, scale*1.08


class HullScene(Scene):
    def __init__(self, width, height):
        super().__init__(width,height)
        self.shader.release()
        fragment = r.FRAGMENT.replace('uniform sampler2DShadow shadow_map;',
                                      'uniform float opacity;uniform sampler2DShadow shadow_map;')
        fragment = fragment.replace('pixel=vec4(c,1);','pixel=vec4(c,opacity);')
        self.shader = self.ctx.program(vertex_shader=r.VERTEX,fragment_shader=fragment)
        self.shader['shadow_map'].value=3
        self.marker = self.ctx.program(vertex_shader=MARKER_VERTEX,fragment_shader=MARKER_FRAGMENT)

    def markers(self, faces):
        result = []
        for face in faces:
            row = dict(color=face['color'])
            for key in ('cloud','hull'):
                buffer = self.ctx.buffer(np.asarray(face[key],dtype='f4').tobytes())
                row[key] = self.ctx.vertex_array(self.marker,[(buffer,'3f','position')])
                row[key+'_buffer'] = buffer
            result.append(row)
        return result

    def picture(self, items, markers, world, direction, framing, *, transparent=()):
        target, scale = framing
        vp = r.orthographic(scale,scale*self.height/self.width) @ r.view_matrix(direction,target)
        self.shader['view_projection'].write(np.asarray(vp,dtype='f4').T.tobytes())
        self.shader['eye'].value=tuple(target+direction*4)
        key=np.array([-.65,-.9,1.45]);key/=np.linalg.norm(key)
        light=r.orthographic(scale*1.8,scale*1.8)@r.view_matrix(key,target)
        for shader in (self.shader,self.shadow):shader['light_projection'].write(np.asarray(light,dtype='f4').T.tobytes())
        ctx=self.ctx;ctx.enable(self.gl.DEPTH_TEST)
        self.studio.shadow_buffer.use();ctx.viewport=(0,0,1536,1536);self.studio.shadow_buffer.clear(depth=1)
        for shape,T in items:shape.shadow(ctx,self.shadow,T)
        self.studio.gbuffer.use();ctx.viewport=(0,0,*self.studio.size)
        self.studio.gbuffer.clear(1,1,1,1,depth=1);self.studio.shadow.use(3)
        self.shader['opacity'].value=1.
        for shape,T in items:shape.draw(self.shader,T)
        if transparent:
            ctx.enable(self.gl.BLEND);ctx.blend_func=self.gl.SRC_ALPHA,self.gl.ONE_MINUS_SRC_ALPHA
            self.studio.gbuffer.depth_mask=False;self.shader['opacity'].value=.42
            for shape,T in transparent:shape.draw(self.shader,T)
            self.studio.gbuffer.depth_mask=True;ctx.disable(self.gl.BLEND)
        self.marker['vp'].write(np.asarray(vp,dtype='f4').T.tobytes())
        self.marker['world'].write(np.asarray(world,dtype='f4').T.tobytes())
        self.marker['point_size'].value=1.05*self.studio.size[0]/self.width
        ctx.enable(self.gl.PROGRAM_POINT_SIZE)
        for row in markers:
            # Light point clouds, dark exact hull contours, with real occlusion.
            self.marker['ink'].value=tuple(row['color']*.58+.42)
            self.marker['point_marker'].value=True;row['cloud'].render(self.gl.POINTS)
            self.marker['ink'].value=tuple(row['color']*.8)
            self.marker['point_marker'].value=False
            ctx.line_width=1.3*self.studio.size[0]/self.width
            row['hull'].render(self.gl.LINE_LOOP)
        return Image.frombytes('RGB',(self.width,self.height),self.studio.finish(vp,direction)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def right_views(model, family, width, height):
    q=family['layout'];scene=HullScene(width,height)
    shape=scene.shape(family['mesh'],BLUE);markers=scene.markers(family['faces'])
    clouds=[face['hull'] for face in family['faces']]
    fit=[]
    for i in family['selected']:
        T=np.linalg.inv(q.material[i])
        T[2,3]-=family['ground_heights'][i]
        P=q.poses[i].copy();P[2,3]-=family['ground_heights'][i]
        fit.extend([transform(family['mesh'].vertices,T),transform(model.source.vertices,P),
                    *[transform(p,T) for p in clouds]])
    framing=camera(np.vstack(fit),VIEW,width,height)
    pictures=[]
    try:
        for i in family['selected']:
            mask=np.zeros(len(model.source.faces),bool)
            mask[model.records[family['group']['members'][i]]['working_face_ids']]=True
            body=scene.shape(model.source,r.GRAY,mask)
            T=np.linalg.inv(q.material[i])
            T[2,3]-=family['ground_heights'][i]
            P=q.poses[i].copy();P[2,3]-=family['ground_heights'][i]
            world=transform(family['mesh'].vertices,T)
            active=next(face for face in family['faces'] if family['ids'][i] in [t['id'] for t in face['tasks']])
            polygon=transform(active['hull'],T)
            if abs(world[:,2].min())>1e-8 or np.abs(polygon[:,2]).max()>1e-8:
                raise ValueError('Membrane and selected pressure hull must both meet z=0')
            xy=np.vstack([world[:,:2],polygon[:,:2]])
            low,high=xy.min(0)-.014,xy.max(0)+.014
            floor=trimesh.creation.box([*(high-low),.0003])
            floor.apply_translation([*(low+high)/2,-.00015])
            floor_shape=scene.shape(floor,(.965,.972,.977))
            pictures.append(scene.picture([(floor_shape,np.eye(4)),(shape,T)],markers,T,VIEW,framing,
                                           transparent=[(body,P)]))
    finally:scene.close()
    return pictures


def metadata(families):
    return [dict(set=f['number'],support_file=(f['group'] if f['group'].get('display_final_support') else f['group']['core'])['file'],
                 support_sha256=(f['group'] if f['group'].get('display_final_support') else f['group']['core'])['sha256'],
                 support_geometry=('saved_final_membrane_and_sparse_feet' if f['group'].get('display_final_support')
                                   else 'saved_step4_contact_membrane_without_added_legs'),
                 all_task_ids=f['ids'],right_task_ids=[f['ids'][i] for i in f['selected']],
                 task_ground_heights_world_m=f['ground_heights'].tolist(),
                 right_display_translation_z_m=[-float(f['ground_heights'][i]) for i in f['selected']],
                 right_world_from_support=[np.linalg.inv(f['layout'].material[i]).tolist() for i in f['selected']],
                 right_world_from_object=[f['layout'].poses[i].tolist() for i in f['selected']],
                 ground_faces=[dict(task_demands=p['tasks'],hull_support_xyz_m=p['hull'].tolist(),
                                    plane_normal_support=p['normal'].tolist(),plane_offset_support_m=p['offset'],
                                    drawn_original_sample_count=len(p['cloud'])) for p in f['faces']])
            for f in families]


def render_movie(model,groups,path,motion_path,size=(1600,900),fps=24,seconds_per_set=8.,*,poster_path=None):
    families=collect(model,groups)
    width,height=size;main_width=round(width*.5625);side_width=(width-main_width)//2;side_height=height//2
    path=Path(path);motion_path=Path(motion_path);path.parent.mkdir(parents=True,exist_ok=True)
    poster_path=Path(poster_path or path.with_name('hulls_and_support.png'))
    process=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','medium',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    posters=[];reviews=[];motion=[]
    frames_per_set=round(seconds_per_set*fps)
    scene=None
    try:
        for family in families:
            side=right_views(model,family,side_width,side_height)
            scene=HullScene(main_width,height)
            mesh=family['mesh'];shape=scene.shape(mesh,BLUE);markers=scene.markers(family['faces'])
            T=np.linalg.inv(family['layout'].material[0])
            points=np.vstack([transform(mesh.vertices,T),*[transform(f['hull'],T) for f in family['faces']]])
            framing=camera(points,VIEW,main_width,height,orbit=True)
            start=np.arctan2(VIEW[1],VIEW[0]);elevation=np.arcsin(VIEW[2])
            for k,angle in enumerate(np.linspace(0,2*np.pi,frames_per_set)):
                d=np.array([np.cos(elevation)*np.cos(start+angle),np.cos(elevation)*np.sin(start+angle),np.sin(elevation)])
                left=scene.picture([(shape,T)],markers,T,d,framing)
                frame=Image.new('RGB',size,'white');frame.paste(left,(0,0))
                for j,image in enumerate(side):frame.paste(image,(main_width+(j%2)*side_width,(j//2)*side_height))
                process.stdin.write(frame.tobytes());motion.append([family['number'],angle,*d])
                if k==0:posters.append(frame.copy())
                if k in (0,frames_per_set//4,frames_per_set//2,3*frames_per_set//4):reviews.append(frame.copy())
            print('HULL ORBIT',family['number'],'right tasks',[family['ids'][i] for i in family['selected']],flush=True)
            scene.close();scene=None
    finally:
        process.stdin.close()
        if scene is not None:scene.close()
    if process.wait()!=0:raise RuntimeError('Hull-orbit movie encoding failed')
    poster=Image.new('RGB',(width,height*len(posters)),'white')
    for k,frame in enumerate(posters):poster.paste(frame,(0,k*height))
    poster.save(poster_path)
    sheet=Image.new('RGB',(width*2,height*((len(reviews)+1)//2)),'white')
    for k,frame in enumerate(reviews):sheet.paste(frame,((k%2)*width,(k//2)*height))
    sheet.save(path.with_name('hulls_video_review.jpg'))
    np.savez_compressed(motion_path,set_angle_camera_direction=np.asarray(motion),
                        fixed_head_world_m=model.head)
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames,r_frame_rate',
                          '-show_entries','format=duration','-of','json',str(path)],check=True,capture_output=True,text=True)
    disk=json.loads(probe.stdout)
    if int(disk['streams'][0]['nb_frames'])!=len(motion):raise ValueError('Incomplete hull-orbit movie')
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    final_geometry=all(g.get('display_final_support',False) for g in groups)
    info=dict(file=str(path.resolve()),sha256=sha256(path),width=width,height=height,fps=fps,
              frames=len(motion),duration_s=len(motion)/fps,task_count=len(model.ids),
              right_tasks_per_set=4,groups=metadata(families),all_original_pressure_samples_drawn=True,
              actual_saved_contact_membrane_geometry=True,actual_final_geometry=final_geometry,
              full_design_certification_claimed=final_geometry,full_turn_degrees_per_set=360,
              ground_definition='tangent_plane_at_lowest_point_of_contact_membrane_in_each_installation',
              all_membranes_directly_touch_their_ground=True,pressure_recomputed_at_each_ground_height=True,
              left_panel='same saved blue membrane and all pressure clouds/hulls on its tangent ground planes',
              right_panel='four task views; each selected membrane and hull share the displayed z=0 ground',
              no_text=True,no_axes=True,no_unfolded_panels=True,
              complete_video_decode_passed=True,motion_file=str(motion_path.resolve()),motion_sha256=sha256(motion_path),
              poster_file=str(poster_path.resolve()),poster_sha256=sha256(poster_path),
              source_sha256=sha256(__file__),geometry_and_task_poses_changed=False)
    write_json(path.with_name('hulls_presentation.json'),info)
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--output-folder',type=Path)
    args=parser.parse_args()
    folder=args.folder or Path(json.loads((ROOT/'subdivide_task/data/B/step5_active.json').read_text())['workdir'])
    report=json.loads((folder/'report.json').read_text())
    if report.get('method')=='direct_contact_membrane_feet':
        from .grounded_feet import result_model
        model=result_model(report)
    else:
        model,_,_=load_model(ROOT/'subdivide_task/data'/report['object']/'step3_membranes')
        model=model_for_report(model,report)
    out=args.output_folder or folder
    out.mkdir(parents=True,exist_ok=True)
    render_movie(model,report['groups'],out/'result.mp4',out/'video_motion.npz')


if __name__=='__main__':main()
