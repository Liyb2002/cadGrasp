"""Export pose-owned Oracle cones and text-free isometric orbit videos."""
from pathlib import Path
import argparse, hashlib, html, json, subprocess, sys, time
import numpy as np
import trimesh
from PIL import ImageDraw

COOPT=Path(__file__).resolve().parents[1];ROOT=COOPT.parents[1]
sys.path[:0]=[str(COOPT/'helper_func'),str(COOPT/'vis_func')]
import _bootstrap
import artifact_store as A
from co_common import S, save
from base_oracle import BaseOracle
from pose_oracle import SCHEMA, transform_points
from continuous_support.stable_gradient import StableGradientModel
from whole_pipeline import load_layout
from fixture_state import installed_states
from whole_search.common import transform_solid, unpack_solid, transform_mesh
from floor_oracle_vis import floor_boundaries, demo_scenes
import step51_render as V


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sphere(center,radius):
    mesh=trimesh.creation.icosphere(subdivisions=4,radius=radius)
    mesh.apply_translation(center)
    return mesh


def scene(model,constraints,task):
    U=task.T_world_object
    body=transform_mesh(model.mesh,U)
    center=transform_points(model.mesh.bounds.mean(0),U)
    radius=model.extent*.90
    crop=sphere(center,radius);stock=S.solid(crop)
    pink,floor_rows=floor_boundaries(constraints,task,center,radius)
    work=task.work_access
    length=work.required_length(transform_points(crop.vertices,np.linalg.inv(U)),minimum=.001)
    yellow=unpack_solid(transform_solid(work.solid(length),U)^stock)
    faces=model.mesh.faces[task.work_face_ids]
    area=trimesh.Trimesh(model.mesh.vertices.copy(),faces.copy(),process=False)
    area=transform_mesh(area,U)
    # Offset only the displayed work surface to avoid coincident raster depth;
    # the saved cone equations and working triangles remain exact.
    triangles=area.triangles+.00005*area.face_normals[:,None,:]
    highlight=trimesh.Trimesh(triangles.reshape(-1,3),np.arange(triangles.size//3).reshape(-1,3),process=False)
    return dict(body=body,pink=pink,yellow=yellow,area=area,highlight=highlight,
        floor_boundaries=floor_rows,
        center=center,radius=radius,span=radius*2.65,cap_length=length)


def packed(families):
    counts=np.array([len(p) for p in families],int)
    planes=np.zeros((len(families),max(counts,default=0),4))
    for k,p in enumerate(families):planes[k,:len(p)]=p
    return planes,counts


def export_pose_constraints(model,layout,folder,oracle=None):
    """Final diagnostic assets; original saved layouts/masks/models untouched.

    OBJ is a finite spherical excerpt for inspection. NPZ retains original
    unbounded inequalities in object, world and fixture coordinates.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    work_folder=folder/'work_cones';work_folder.mkdir(exist_ok=True)
    oracle=oracle or BaseOracle(model);constraints=oracle.pose_constraints(layout)
    definition=constraints.definition();rows=[];scenes=[]
    for task in constraints.poses:
        s=scene(model,constraints,task);scenes.append(s)
        cone=work_folder/(task.pose+'.obj');s['yellow'].export(cone,digits=16)
        arrays={}
        for frame,families in [('object',task.work_local),('fixture',task.work_common),('world',task.work_world)]:
            arrays['work_planes_'+frame],arrays['work_plane_counts_'+frame]=packed(families)
        np.savez_compressed(work_folder/(task.pose+'.npz'),**arrays,
            work_face_ids=task.work_face_ids,
            work_cell_source_face_ids=task.work_access.cell_face_ids,
            half_angle_deg=task.half_angle_deg,
            T_world_fixture=task.T_world_fixture,Q_fixture_object=task.Q_fixture_object,T_world_object=task.T_world_object,
            all_floor_planes_fixture=constraints.floor_planes_common,
            all_floor_planes_object=constraints.floor_planes_object,
            all_floor_planes_current_world=np.array([(p.T_world_fixture@np.linalg.inv(task.T_world_fixture))[2] for p in constraints.poses]),
            display_floor_vertices_world_m=s['pink'].vertices,display_floor_faces=s['pink'].faces,
            work_area_vertices_world_m=s['area'].vertices,work_area_faces=s['area'].faces,
            display_crop_center_world_m=s['center'],display_crop_radius_m=s['radius'])
        rows.append(dict(pose=task.pose,work_cone_obj=str(cone.relative_to(folder)),
            unbounded_constraints_npz=str((work_folder/(task.pose+'.npz')).relative_to(folder)),
            work_face_count=len(task.work_face_ids),half_angle_deg=task.half_angle_deg,
            object_lowest_world_z_m=float(s['body'].vertices[:,2].min()),
            spherical_display_radius_m=s['radius'],finite_obj_is_display_only=True,
            pink='zero-thickness floor boundary sheets, pulled into this task world; negative side excludes material',
            floor_boundaries=s['floor_boundaries'],floor_thickness_m=0.,
            yellow='only this task outward work half-rays; certified whole-triangle self-occlusion clipped, uncertain rays conservatively retained',
            highlighted_work_area='only this task original work triangles',
            no_other_task_work_cone_shown=True,exit_sweeps_shown=False))
    definition.update(per_pose_files=rows,spherical_crop_display_only=True,unbounded_npz_is_authoritative=True,
        floor_display='zero-thickness rectangular boundary sheets; no excluded-volume solid',
        other_floor_sheets_are_pulled_back_constraints_not_extra_physical_world_floors=True)
    save(folder/'constraints.json',definition)
    return definition,scenes


def frame(renderer,s,angle,size):
    direction=np.array([np.cos(angle),np.sin(angle),1/np.sqrt(2)])
    direction/=np.linalg.norm(direction)
    right=np.array([-np.sin(angle),np.cos(angle),0.])
    camera=(s['center'],np.array([right,np.cross(direction,right),direction]),s['span'])
    parts=[]
    for mesh,color,opacity in [(s['pink'],'#F1B9CC',.18),(s['yellow'],'#F5DF8A',.24),
                                (s.get('fixture',trimesh.Trimesh()),V.BLUE,.70),
                                (s['body'],V.GRAY,1.),(s['highlight'],'#DAB652',1.)]:
        if len(mesh.faces):
            piece=V.piece(mesh,color,opacity)
            if mesh is s['highlight']:piece['unlit']=True;piece['bias']=1e-5
            if mesh is s['pink']:piece['unlit']=True;piece['bias']=-1e-5
            parts.append(V.CPU.placed(piece))
    if 'probe' in s:
        piece=V.piece(s['probe'],'#D04D83',1.);piece['unlit']=True
        parts.append(V.CPU.placed(piece))
    result=renderer.render(parts,camera,size)
    if 'height_guide_world_m' in s:
        points=(s['height_guide_world_m']-camera[0])@camera[1].T
        pixels=size/2+points[:,:2]*np.array([1,-1])*size/camera[2]
        start,end=pixels;length=np.linalg.norm(end-start);ink=ImageDraw.Draw(result)
        for offset in np.arange(7,length,7):
            a=start+(end-start)*offset/length
            b=start+(end-start)*min(offset+3,length)/length
            ink.line([tuple(a),tuple(b)],fill='#C85D8B',width=2)
        x,y=end;ink.ellipse([x-3,y-3,x+3,y+3],outline='#C85D8B',width=2)
    return result


def orbit(renderer,s,path,seconds=8.,fps=24,size=800):
    count=round(seconds*fps)
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{size}x{size}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:
        for k in range(count):
            angle=-np.pi/4+2*np.pi*k/count
            # A two-state sequence cuts directly between installed states.
            # The physical ground stays z=0; no transfer-motion test is added.
            current=s[min(len(s)-1,k*len(s)//count)] if isinstance(s,list) else s
            image=frame(renderer,current,angle,size)
            encoder.stdin.write(image.convert('RGB').tobytes())
            if k%48==0:print('ORBIT',path.stem,k,'/',count,flush=True)
    finally:encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError(f'Video encoding failed: {path}')
    return dict(frames=count,fps=fps,seconds=count/fps,isometric_elevation_deg=35.264389682754654,
                camera_orbit_degrees=360,text=False,fixture_motion=isinstance(s,list),
                transfer_motion_shown=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set',default='pose2+20');parser.add_argument('--object',default='B')
    parser.add_argument('--experiment',default='base_oracle_v3');parser.add_argument('--seconds',type=float,default=8.)
    parser.add_argument('--fps',type=int,default=24);parser.add_argument('--size',type=int,default=800)
    parser.add_argument('--floor-demo',action='store_true',help='Show a real multi-state fixture with one hypothetical floor-rejected material point')
    parser.add_argument('--replace',action='store_true');args=parser.parse_args()
    if args.seconds<=0 or args.fps<1 or args.size<100 or args.size%2:raise ValueError('Invalid video dimensions/duration')
    root=A.result_root(args.object)/args.set;source=root/'step4/step4.2'/args.experiment
    paths=[source/'layout.npz',source/'report.json',ROOT/'objects'/args.object/'mesh.stl']
    fixture_path=root/'step5/step5.2/connected_base_v2/support_with_base.obj'
    if args.floor_demo:paths.append(fixture_path)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=json.loads((source/'report.json').read_text());poses=report['poses']
    model=StableGradientModel(poses,args.object,demand_nodes=32);layout=load_layout(source/'layout.npz')
    if not np.allclose(installed_states(model.native,layout),np.load(source/'layout.npz')['native_world'],atol=1e-12,rtol=0):
        raise ValueError('Saved fixture states do not match original tasks')
    folder=A.VISUALS/args.object/args.set/'step4/step4.2/oracle'
    if folder.exists() and not args.replace:raise FileExistsError('Oracle diagnostics exist; use --replace')
    prior_note=None
    if (folder/'report.json').exists():
        previous=json.loads((folder/'report.json').read_text())
        prior_note=previous.get('prior_independent_regeneration_note')
        if not previous.get('query_origin') and not previous['query']['passed']:
            prior_note=dict(status=previous['query']['status'],
                enclosure_passed=previous['query']['enclosure_passed'],
                connected_passed=previous['query']['connected_passed'],
                meaning='an optional independent finite-window regeneration did not find a component; saved accepted geometry and original witness are retained',
                used_to_modify_saved_acceptance=False)
    began=time.monotonic();oracle=BaseOracle(model)
    # Visualization reads the already accepted final result. Rebuilding the
    # connection witness is neither needed for a plane picture nor acceptance.
    answer=report['base_oracle']
    definition,scenes=export_pose_constraints(model,layout,folder,oracle)
    renderer=V.Renderer();videos=[]
    for task,s in zip(oracle.pose_constraints(layout).poses,scenes):
        path=folder/(task.pose+'.mp4')
        result=orbit(renderer,s,path,args.seconds,args.fps,args.size)
        videos.append(dict(pose=task.pose,video=path.name,**result))
    proof=None
    if args.floor_demo:
        fixture=trimesh.load(fixture_path,process=False)
        sequence,proof=demo_scenes(oracle.pose_constraints(layout),fixture,scenes)
        path=folder/'floor_states.mp4';result=orbit(renderer,sequence,path,args.seconds,args.fps,args.size)
        proof.update(fixture_source=str(fixture_path.relative_to(ROOT)),video=path.name,
            display_only=True,final_fixture_modified=False,states_cut_without_intermediate_motion=True)
        save(folder/'floor_example.json',proof)
        videos.append(dict(pose='Floor states',video=path.name,**result,
            final_state_count=2,actual_floor_always_world_z_zero=True,
            pink_cube_is_hypothetical_material=True))
    for p,h in hashes.items():
        if sha(ROOT/p)!=h:raise RuntimeError(f'Original input changed: {p}')
    save(folder/'report.json',dict(schema=SCHEMA,pose_set=args.set,source_experiment=args.experiment,
        query=answer,query_origin='saved_final_experiment',base_oracle_query_run=False,
        prior_independent_regeneration_note=prior_note,
        definition='constraints.json',videos=videos,floor_example=proof,seconds=time.monotonic()-began,
        provenance=dict(inputs=hashes,code={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'slides/Co-optimize/step4.2/pose_oracle.py':sha(COOPT/'step4.2/pose_oracle.py'),
            'slides/Co-optimize/step4.2/base_oracle.py':sha(COOPT/'step4.2/base_oracle.py'),
            'slides/Co-optimize/step4.2/floor_oracle_vis.py':sha(COOPT/'step4.2/floor_oracle_vis.py'),
            'slides/Co-optimize/helper_func/work_access.py':sha(COOPT/'helper_func/work_access.py'),
            'slides/Co-optimize/helper_func/work_access_shadow.py':sha(COOPT/'helper_func/work_access_shadow.py')}),
        original_layout_and_force_results_changed=False,force_solver_run=False))
    tiles=''.join(f'<section><h2>{html.escape(row["pose"])}</h2><video controls loop muted playsinline preload="metadata" src="{row["video"]}"></video>'
        +(f'<p><a href="work_cones/{row["pose"]}.obj">Work cone excerpt OBJ</a> · <a href="work_cones/{row["pose"]}.npz">Unbounded constraints NPZ</a></p>' if row['pose']!='Floor states' else '<p>Blue: saved accepted fixture. Dark pink cube: hypothetical material above the first floor and below the second; it is not part of the accepted fixture. The dashed guide projects this point vertically onto the floor. The physical floor always stays horizontal at world Z=0. The cut skips transfer motion.</p>')+'</section>' for row in videos)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1500px;margin:24px auto}main{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}video{width:100%}</style>'
        f'<h1>{html.escape(args.set)} · Pose-specific Oracle</h1><p>Grey: current object. Pink sheets: zero-thickness floor boundaries. Pale yellow: current pose work cone. Gold: its working area. Camera orbits a stationary object; no transfer-motion constraint.</p>'
        '<p>A tilted sheet in a single-pose orbit is another final floor pulled back to the current material placement, not another physical sloped floor. The floor-states example uses only the actual horizontal world Z=0 plane.</p>'
        '<p>Cones are outward half-rays from the gold working faces, never bidirectional lines. Certified self-occluded angular sectors are clipped; uncertain sectors remain reserved conservatively.</p>'
        '<p>OBJ volumes have a spherical display crop; NPZ inequalities are unbounded. Other tasks\' cones are omitted from each video; the shared rigid fixture must still pass every task in its own state. All final floor planes are saved in object and fixture coordinates; object-only Z translation changes the former.</p><main>'+tiles+'</main>\n')
    print('ORACLE VIDEOS COMPLETE',len(videos),'passed',answer['passed'],'output',folder,flush=True)


if __name__=='__main__':main()
