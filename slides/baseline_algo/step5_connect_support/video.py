"""Render MP4/GIF only after replaying a verified full-solid insertion path."""
from step1.needs import COORD
import json
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from step5_connect_support import rigid_path as P,piecewise_path as PP,solids as S,belt_geometry as B,visual_details as V


def render(domain,data,path,out):
    if not path.get('passed') or not path.get('continuous_sweep_verified'):
        raise ValueError('A successful continuous trajectory is required before rendering insertion video')
    parts=S.unpack_parts(data);scene=B.Scene(domain.mesh);origin=np.asarray(path['origin_m'])
    piecewise=path.get('path_kind')=='piecewise_rigid'
    (PP.replay if piecewise else P.replay)(scene,parts,path)
    if piecewise:
        poses=np.asarray(path['poses'])[::-1]
        lengths=np.array([np.linalg.norm(b[:3,3]-a[:3,3])/scene.scale+
            np.linalg.norm(P.Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).as_rotvec())*.2
            for a,b in zip(poses[:-1],poses[1:])])
        cumulative=np.r_[0,np.cumsum(lengths)];samples=np.linspace(0,cumulative[-1],72)
        transforms=[]
        for value in samples:
            i=min(len(poses)-2,int(np.searchsorted(cumulative,value,side='right')-1))
            f=(value-cumulative[i])/max(lengths[i],1e-20)
            transforms.append(PP.interpolate(poses[i],poses[i+1],f))
        move=lambda points,k:PP.transform(points,origin,transforms[k])
    else:
        amounts=np.linspace(path['final_withdrawal_amount'],0,72)
        move=lambda points,k:P.transform(points,origin,scene.scale,np.asarray(path['motion']),amounts[k])
    points=np.vstack([domain.mesh.vertices]+[move(data['union_vertices_m'],i) for i in range(72)])
    travel=move(data['union_vertices_m'],0).mean(axis=0)-move(data['union_vertices_m'],71).mean(axis=0)
    horizontal=np.linalg.norm(COORD.floor(travel))
    sight=np.array([travel[2],.55*horizontal,-travel[0]]) if horizontal>1e-8 else np.array([1.,.5,-1.])
    basis=V.R.axes(sight)
    view=V.camera(domain,[],basis=basis,points=points)
    fixed_floor=V.floor_triangles(points,.025*scene.scale);frames=[]
    for k in range(72):
        moved=dict(data)
        for key in ('union_vertices_m','part_vertices_m'):moved[key]=move(data[key],k)
        modules=[(dict(candidate_id='assembly'),moved)]
        frame,_=V.render(V.arrays(domain,[],modules,ground=fixed_floor),view,640)
        frames.append(frame)
    frames=[frames[0]]*12+frames+[frames[-1]]*24
    with imageio.get_writer(out/'insertion.mp4',fps=12,codec='libx264',quality=None,ffmpeg_params=['-crf','0','-vf','scale=in_range=full:out_range=full','-color_range','pc'],macro_block_size=16) as writer:
        for frame in frames:writer.append_data(np.asarray(frame))
    frames[0].save(out/'insertion.gif',save_all=True,append_images=frames[1:],duration=83,loop=0)
    if (out/'video_review.png').exists():
        width,height=frames[0].size
        review=Image.new('RGB',(3*width,height),V.R.PAPER)
        for column,index in enumerate((12,54,83)):
            review.paste(frames[index],(column*width,0))
        review.save(out/'video_review.png')
    (out/'video_metadata.json').write_text(json.dumps(dict(fps=12,frame_count=len(frames),
        contact_labels_drawn=False,frame_numbers_drawn=False,
        text_drawn=False,panel_count=1,view_kind='full_path',frame_size_px=list(frames[0].size),
        complete_trajectory_replayed_before_render=True,object_fixed=True,
        robot_kinematics_verified=False,bearing_verified=bool(path.get("bearing_verified")),contact_refinement_proposal=bool(path.get("contact_refinement_proposal")),animation_frames_used_for_collision_acceptance=False),indent=2)+'\n')
