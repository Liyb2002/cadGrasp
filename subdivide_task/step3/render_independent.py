"""Task work poses and their unchanged Step1 sectors on the common plate."""
import math
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
from subdivide_task.step1.needle_render import Renderer as BaseRenderer,sector_lines,rod
from subdivide_task.step5.video import raster as raster
from subdivide_task.step1.common import ROOT,sha256

BLUE=(.23,.55,.76)
VIEW=np.array([0.,1.,1.55]);VIEW/=np.linalg.norm(VIEW)


class Renderer(BaseRenderer):
    def draw(self,items,points,direction,lines=None):
        flat=[]
        for shape,color,mask in items:
            if mask is None:
                vertices=shape.triangles.reshape(-1,3)
                shape=trimesh.Trimesh(vertices,np.arange(len(vertices)).reshape(-1,3),process=False)
            flat.append((shape,color,mask))
        return super().draw(flat,points,direction,lines)


def head_context(workspace):
    origin=np.asarray(workspace['reference_head_front_world_m'])
    axis=np.asarray(workspace['nominal_axis_world'])
    flange=origin-workspace['head_front_to_flange_m']*axis
    head=trimesh.creation.icosphere(subdivisions=3,radius=workspace['head_proxy_radius_m'])
    head.apply_translation(flange)
    stem=rod(flange-.015*axis,flange-.035*axis,.008)
    # The root and gold sector show the tool reference frame. No arbitrary
    # shortened or penetrating contact needle is invented for the figure.
    return [(head,(.67,.71,.75),None),(stem,(.48,.53,.58),None)]


def working_poses(source,records,plate_top,folder,report):
    width,height=800,720;columns=4
    canvas=Image.new('RGB',(columns*width,math.ceil(len(records)/columns)*height),'white')
    renderer=Renderer(width,height)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
    station_xy=np.asarray(report['source_workspace']['reference_head_front_world_m'])[:2]
    metadata=[]
    try:
        for index,record in enumerate(records):
            # Keep the approved close-ups as local pose references. The
            # overview below displays their common-height installed layout.
            workspace=dict(record['workspace_local']);root=np.asarray(workspace['reference_head_front_world_m'])
            local_shift=np.r_[station_xy-root[:2],0.]
            workspace['reference_head_front_world_m']=(root+local_shift).tolist()
            T=np.asarray(record['local_world_from_object']).copy();T[:3,3]+=local_shift
            shift=T[:3,3]-np.asarray(record['world_from_object'])[:3,3]
            body=source.copy();body.apply_transform(T)
            mask=np.zeros(len(source.faces),bool);mask[record['working_face_ids']]=True
            extent=body.extents[:2]+.06
            patch=trimesh.creation.box([*extent,plate_top])
            patch.apply_translation([*body.bounds.mean(0)[:2],plate_top/2])
            context=head_context(workspace);lines=sector_lines(workspace)
            bounds=np.vstack([body.vertices,lines,*[m.vertices for m,_,_ in context]])
            picture=renderer.draw([(patch,BLUE,None),(body,raster.GRAY,mask),*context],
                bounds,[1.1,1.4,.9],lines)
            ImageDraw.Draw(picture).text((20,height-40),record['id'],font=font,fill='#344d60')
            canvas.paste(picture,((index%columns)*width,(index//columns)*height))
            metadata.append(dict(id=record['id'],display_translation_m=shift.tolist(),
                world_from_object=T.tolist(),head_front_world_m=workspace['reference_head_front_world_m'],
                nominal_axis_world=workspace['nominal_axis_world'],half_angle_deg=workspace['half_angle_deg'],
                radius_m=workspace['needle_length_m']))
        canvas.save(folder/'working_poses.png')
    finally:renderer.close()
    return dict(size=list(canvas.size),scenes=metadata,same_nominal_world_direction=True,
        same_head_xy=True,head_height_follows_resting_object=True,rotation_unchanged_from_saved_pose=True,
        local_pose_reference_before_riser_alignment=True,
        reference_plate_patch=True,actual_support_not_drawn_in_pose_only_views=True,
        contact_trajectory_claimed=False)


def render(source,support,bedding,records,plate_bounds,folder,report):
    lo,hi=plate_bounds
    objects=[];centers=[];context=[];lines=[]
    for record in records:
        body=source.copy();body.apply_transform(np.asarray(record['world_from_object']))
        mask=np.zeros(len(source.faces),bool);mask[record['working_face_ids']]=True
        objects.append((body,raster.GRAY,mask));centers.append(body.bounds.mean(0))
        context.extend(head_context(record['workspace']));lines.append(sector_lines(record['workspace']))
    lines=np.vstack(lines)
    bounds=np.vstack([support.vertices,*[m.vertices for m,_,_ in objects],lines,
        *[m.vertices for m,_,_ in context]])
    floor=trimesh.creation.box([hi[0]-lo[0]+.7,hi[1]-lo[1]+.5,.005])
    floor.apply_translation([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,-.0025])
    ground=(floor,(.96,.97,.98),None)
    scenes=[[(bedding,BLUE,None),*objects],[(support,BLUE,None),*objects],[(support,BLUE,None)]]
    width,height=5600,680;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
    renderer=Renderer(width,height);canvas=Image.new('RGB',(width,3*height),'white')
    right=np.cross([0,0,1],VIEW);right/=np.linalg.norm(right);up=np.cross(VIEW,right)
    projected=bounds@np.array([right,up,VIEW]).T;low,high=projected.min(0),projected.max(0)
    scale=max(high[0]-low[0],(high[1]-low[1])*width/height)*1.16
    try:
        for row,items in enumerate(scenes):
            picture=renderer.draw([ground,*items,*context] if row<2 else [ground,*items],
                bounds,VIEW,lines if row<2 else None)
            draw=ImageDraw.Draw(picture)
            for record,center in zip(records,centers):
                x=width/2+(center@right-(low[0]+high[0])/2)/scale*width
                draw.text((x-26,height-36),record['id'],font=font,fill='#344d60')
            canvas.paste(picture,(0,row*height))
        canvas.save(folder/'overview.png')
    finally:renderer.close()
    report['visualization']=dict(engine='OpenGL rasterization',overview_size=list(canvas.size),
        row_order=['task_objects_on_height_aligned_risers','occupied_negative_molds','empty_negative_molds'],
        same_camera_and_same_task_poses_in_all_rows=True,actual_saved_support_geometry=True,
        support_color='blue',working_area_color='pink',task_order=[r['id'] for r in records],
        gold_sectors_from_saved_task_workspaces=True,head_context_in_first_two_rows=True,
        same_installed_head_height_m=report['common_tool_head_height_m'],
        first_row_bedding_is_a_subset_of_saved_support=True,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),Path(raster.__file__)]})
    report['visualization']['working_poses']=working_poses(source,records,hi[2],folder,report)
