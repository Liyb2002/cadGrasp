"""One stationary translucent object with all poses' original floor demands."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step0_pose_selection import floor_points as F
from step4_connect_support.baseline_current import clean_render as V,publish_compact as P
from step4_connect_support.baseline_current.run_reseated import groups

COLORS=['#3286be','#d78a3d','#8c76b6','#64a38b','#bc708c']


def camera_for(points):
    # Fixed presentation orientation: bunny upright as in the user's reference.
    # This changes only the camera, never object/sample coordinates.
    # Original Stanford bunny mesh is Y-up, unlike task worlds which are Z-up.
    view=np.array([1.,.5,1.3]);view/=np.linalg.norm(view)
    right=np.cross([0,1,0],view);right/=np.linalg.norm(right)
    basis=np.array([right,np.cross(view,right),view])
    projected=points@basis.T;lo,hi=projected.min(0),projected.max(0)
    return ((lo+hi)/2)@basis,basis,float(max(hi[:2]-lo[:2])*1.12)


def cloud_piece(points,color,camera,pixels,size):
    # Screen-facing square markers preserve all sample centers, including
    # duplicates. Their size is illustrative and has no physical meaning.
    _,basis,span=camera
    radius=pixels*span/size
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])@basis[:2]*radius
    triangles=(points[:,None,:]+corners)[...,[[0,1,2],[0,2,3]],:].reshape(-1,3,3)
    return dict(triangles=triangles,normals=np.broadcast_to(basis[2],triangles.shape),
        colors=np.broadcast_to(P.CPU.rgb(color),triangles.shape),bias=.000001,unlit=True)


def polygon_piece(vertices,color):
    center=vertices.mean(0)
    triangles=np.array([[center,a,b] for a,b in zip(vertices,np.roll(vertices,-1,axis=0))])
    mesh=trimesh.Trimesh(triangles.reshape(-1,3),np.arange(triangles.size//3).reshape(-1,3),process=False)
    part=P.piece(mesh,color);part.update(opacity=.18,unlit=True)
    return part


def boundary_piece(vertices,color):
    segments=[]
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        length=np.linalg.norm(b-a)
        for start in np.arange(0,length,.0032):
            end=min(start+.0019,length)
            segments.append(trimesh.creation.cylinder(radius=.00012,sections=6,
                segment=np.array([a+(b-a)*start/length,a+(b-a)*end/length])))
    part=P.piece(trimesh.util.concatenate(segments),color);part['unlit']=True
    return part


def render(group):
    output=group/'step4';data=output/'data'
    poses=['pose_'+p for p in group.name.removeprefix('pose').split('+')]
    tasks=[read_task(group.parent.name,pose,
        folder=group/'step3_scheculer/independent_poses_floor2mm'/pose/'step_1_needs') for pose in poses]
    frames=F.validate_frames([t.domain.data['frame']['T_world_mesh'] for t in tasks])
    # Inverting each original task placement brings every object vertex to
    # the same mesh coordinates. Saved independent fixture seating is unused.
    vertices=F.map_points(tasks[0].domain.mesh.vertices,frames[0],np.eye(4))
    faces=tasks[0].domain.mesh.faces
    clouds=[];hulls=[];planes=[];records=[];arrays={}
    for k,(task,frame) in enumerate(zip(tasks,frames)):
        mapped_object=F.map_points(task.domain.mesh.vertices,frame,np.eye(4))
        object_residual=float(np.max(np.abs(mapped_object-vertices)))
        assert object_residual<1e-10
        np.testing.assert_array_equal(task.domain.mesh.faces,faces)
        xy=F.pressure_centers(task.targets/task.scale,task.domain.com)[0]
        local=F.ground_points(xy)
        common=F.map_points(local,frame,np.eye(4))
        roundtrip=common@frame[:3,:3].T+frame[:3,3]
        residual=float(np.max(np.abs(roundtrip-local)))
        assert residual<1e-12
        hull=ConvexHull(xy)
        assert np.max(xy@hull.equations[:,:2].T+hull.equations[:,2])<1e-10
        color=COLORS[k%len(COLORS)]
        clouds.append(common);hulls.append(common[hull.vertices])
        lo,hi=xy.min(0)-.007,xy.max(0)+.007
        rectangle=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],
            [hi[0],hi[1],0],[lo[0],hi[1],0]])
        planes.append(F.map_points(rectangle,frame,np.eye(4)))
        records.append(dict(pose=task.pose,color=color,original_point_count=len(xy),
            rendered_point_count=len(common),subsampled=False,T_world_mesh=frame.tolist(),
            T_mesh_world=np.linalg.inv(frame).tolist(),floor_plane_object=frame[2].tolist(),
            maximum_transform_roundtrip_error_m=residual,
            maximum_object_alignment_error_m=object_residual,convex_hull_indices=hull.vertices.tolist()))
        arrays[task.pose+'_task_xy_m']=xy
        arrays[task.pose+'_object_points_m']=common
    arrays.update(T_world_mesh=frames,object_vertices_m=vertices,object_faces=faces)
    np.savez_compressed(data/'floor_demands_common_frame.npz',**arrays)
    size=1600
    camera=camera_for(np.vstack([vertices]+clouds+planes))
    parts=[]
    for k,task in enumerate(tasks):
        color=records[k]['color']
        plane=polygon_piece(planes[k],'#c4cbd0');plane['opacity']=.17
        area=polygon_piece(hulls[k],color);area['opacity']=.055
        parts.extend(P.CPU.placed(p) for p in [plane,area,
            boundary_piece(hulls[k],color),cloud_piece(clouds[k],color,camera,.52,size)])
    ghost=P.CPU.piece(dict(v=vertices.ravel(),f=faces.ravel()),'#88949b',smooth=False)
    ghost['opacity']=.22
    parts.append(P.CPU.placed(ghost))
    renderer=V.Renderer()
    canvas=renderer.render(parts,camera,size)
    filename='floor_demands_common_frame.png';canvas.save(data/filename)
    reference=I.OUTPUTS/'B/pose5+7/step4/data/refer.png'
    record=dict(complete=True,poses=poses,coordinate_frame='one fixed original object mesh frame',
        transform_row_vector='p_object = (p_task - T_world_mesh[:3,3]) @ T_world_mesh[:3,:3]',
        placement_source='Original Step1 task T_world_mesh; independent fixture seating is not used',
        points_projected_onto_object=False,planes_merged=False,fixture_solid_drawn=False,
        image_description='One fixed translucent object; every pose floor cloud and its own planar convex hull',
        object_instance_count=1,object_opacity=.22,per_pose=records,
        ground_rectangles_are_display_only=True,final_fixture_acceptance_claim=False,
        camera=dict(focus_object_m=camera[0].tolist(),basis_object=camera[1].tolist(),span_m=float(camera[2])),
        marker_half_size_pixels=.52,marker_size_is_display_only=True,
        transparent_rendering='Nearest translucent surface per pixel; sample centers are never relocated',
        image_size=list(canvas.size),text_in_images=False,arrows_in_images=False,axes_in_images=False,
        html_generated=False,videos_generated=False,
        provenance=dict(inputs=I.hashes([reference]+[p for t in tasks for p in t.inputs]),
            code=I.hashes([Path(__file__),Path(F.__file__),Path(V.__file__),Path(P.__file__),Path(P.CPU.__file__),
                Path(V.__file__).with_name('translucent_raster.cpp')])),
        artifacts={name:I.sha256(data/name) for name in (filename,'floor_demands_common_frame.npz')})
    I.save(data/'floor_demands_common_frame.json',record)
    print('COMMON FRAME FIGURE',data/filename,'points',sum(r['rendered_point_count'] for r in records),flush=True)


def batch(name,jobs):
    def worker(group):
        with (group/'step4/data/floor_demands_render.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),group.name,'--object',name],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print('OBJECT FRAME COMPLETE',group.name,flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:list(pool.map(worker,groups(name)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group',nargs='?',default='pose5+7')
    p.add_argument('--object',default='B');p.add_argument('--all',action='store_true');p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args()
    if args.all:batch(args.object,args.jobs)
    else:render(I.OUTPUTS/args.object/args.group)
