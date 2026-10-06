"""Show shared support regrowth while one independent exit direction changes."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from exit_clearance import ExitClearance
import argparse, time, subprocess


def direction_schedule(transforms, steps=32):
    directions=np.array([T[:3,:3].T@np.array([0.,0.,1.]) for T in transforms])
    yield -1,0.,directions.copy()
    for active,T in enumerate(transforms):
        for u in np.linspace(0,1,steps+1)[1:]:
            smooth=u*u*(3-2*u)
            tilt=np.pi/2*smooth
            azimuth=np.pi/4
            world=np.array([np.sin(tilt)*np.cos(azimuth),np.sin(tilt)*np.sin(azimuth),np.cos(tilt)])
            directions[active]=T[:3,:3].T@world
            yield active,float(u),directions.copy()


def remaining_support(seed,sweeps):
    # Always start from S0. Changing a sweep restores material automatically.
    return seed-union(sweeps)


def pending_red(queue,current,frame,delay_frames):
    active=[part for born,part in queue if frame-born<delay_frames]
    return union(active)-current if active else F.md.Manifold()


def pending_red_mesh(queue,current,frame,delay_frames):
    # Display fragments separately: avoids coplanar unions of thin red slivers.
    # Every fragment excludes regrown blue material and obeys its own timer.
    meshes=[S.unpack(part-current) for born,part in queue if frame-born<delay_frames]
    meshes=[mesh for mesh in meshes if len(mesh.faces)]
    return trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()


class Renderer:
    def __init__(self,obj,seedmesh,transforms,poses,length):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.patches import Rectangle, FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d
        self.arrow_patch=FancyArrowPatch;self.project=proj3d.proj_transform
        self.plt=plt;self.collection=Poly3DCollection;self.transforms=transforms;self.obj=obj;self.length=length;self.sweep_artists=[None]*len(poses)
        n=len(poses);cols=2 if n==4 else min(3,n);rows=(n+cols-1)//cols
        self.fig=plt.figure(figsize=(cols*10,rows*7),dpi=100,facecolor='white');self.axes=[];self.artists=[];self.arrows=[];self.borders=[]
        elevation=float(np.degrees(np.arctan(1/np.sqrt(2))));azimuth=-45.
        self.light=np.array([1.,-1.,1.])/np.sqrt(3)+np.array([-.15,-.2,.65]);self.light/=np.linalg.norm(self.light)
        for i,(T,pose) in enumerate(zip(transforms,poses)):
            slot_x=(i%cols)/cols;slot_y=1-(i//cols+1)/rows
            ax=self.fig.add_axes([slot_x,slot_y,1/cols,1/rows],projection='3d');self.axes.append(ax)
            points=transform_points(np.vstack([obj.vertices,seedmesh.vertices]),T)*1000
            # Object-centered closeup; full sweeps remain the actual cutting geometry.
            center=(points.min(0)+points.max(0))/2;radius=float(np.ptp(points,axis=0).max())*.56
            for axis,value in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
            ax.set_box_aspect((1,1,1),zoom=1.34);ax.set_proj_type('ortho');ax.view_init(elev=elevation,azim=azimuth,roll=0);ax.set_axis_off()
            ax.text2D(.04,.05,pose.replace('_',' '),transform=ax.transAxes,color='#303840',fontsize=15)
            lo=points[:,:2].min(0)-8;hi=points[:,:2].max(0)+8
            boundary=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
            ax.plot(*boundary.T,color='#d4d4d4',linewidth=.6)
            border=Rectangle((slot_x+.01/cols,slot_y+.01/rows),.98/cols,.98/rows,fill=False,edgecolor='#ffffff',linewidth=2,transform=self.fig.transFigure,clip_on=False)
            self.fig.add_artist(border);self.borders.append(border)
            self.artists.append(None);self.arrows.append(None)
        self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)

    def frame(self,support,directions,active,sweeps,red=None):
        for i,(ax,T) in enumerate(zip(self.axes,self.transforms)):
            triangles=[];colors=[]
            materials=[(self.obj,'#a4a8ac'),(support,'#319cd7')]
            if red is not None:materials.append((red,'#ef4938'))
            for source,color in materials:
                if not len(source.faces):continue
                world=source.copy();world.apply_transform(T)
                rgb=np.array(self.plt.matplotlib.colors.to_rgb(color));brightness=.58+.42*np.maximum(world.face_normals@self.light,0)
                triangles.append(world.triangles*1000);colors.append(np.c_[brightness[:,None]*rgb,np.ones(len(world.faces))])
            if self.artists[i] is not None:self.artists[i].remove()
            self.artists[i]=self.collection(np.concatenate(triangles),facecolors=np.concatenate(colors),edgecolors='none',zsort='average');ax.add_collection3d(self.artists[i])
            if self.sweep_artists[i] is not None:self.sweep_artists[i].remove()
            sweep=S.unpack(sweeps[i]);sweep.apply_transform(T)
            self.sweep_artists[i]=self.collection(sweep.triangles*1000,facecolor='#6ebad8',edgecolors='none',alpha=.084,zsort='average');ax.add_collection3d(self.sweep_artists[i]);self.sweep_artists[i].set_clip_on(True);self.sweep_artists[i].set_clip_box(ax.bbox)
            if self.arrows[i] is not None:self.arrows[i].remove()
            native=transform_points(self.obj.vertices,T)*1000;origin=native.mean(0);d=T[:3,:3]@directions[i]
            lo=np.array([ax.get_xlim()[0],ax.get_ylim()[0],ax.get_zlim()[0]])
            hi=np.array([ax.get_xlim()[1],ax.get_ylim()[1],ax.get_zlim()[1]])
            near=0.;far=self.length*1000
            for k in range(3):
                if abs(d[k])>1e-12:
                    limits=sorted([(lo[k]-origin[k])/d[k],(hi[k]-origin[k])/d[k]])
                    near=max(near,limits[0]);far=min(far,limits[1])
            center=origin+(max(near,far)+near)/2*d
            def screen(point):
                x,y,_=self.project(*point,ax.get_proj())
                return ax.transAxes.inverted().transform(ax.transData.transform([x,y]))
            middle=np.clip(screen(center),.17,.83);vector=screen(center+10*d)-screen(center-10*d);vector/=np.linalg.norm(vector)
            self.arrows[i]=self.arrow_patch(middle-.14*vector,middle+.14*vector,arrowstyle='simple,head_length=0.7,head_width=0.8,tail_width=0.22',mutation_scale=55,facecolor='#ffe000',edgecolor='#8f7100',linewidth=1.6,transform=ax.transAxes,zorder=100,clip_on=True)
            self.arrows[i].set_clip_box(ax.bbox);self.fig.add_artist(self.arrows[i])
            self.borders[i].set_edgecolor('#e78b28' if active==i else '#ffffff')
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:,:,:3].copy()


def run(group,steps,fps):
    began=time.monotonic();base=HERE/'output/B'/group['id'];out=base/'step4/step4.2';(out/'data').mkdir(parents=True,exist_ok=True);(out/'process').mkdir(exist_ok=True)
    seed_path=base/'step3/step3.3/support_with_rings.obj';object_path=base/'step3/step3.1/registered_object.obj'
    seedmesh=trimesh.load(seed_path,force='mesh',process=False);seed=S.solid(seedmesh);obj=trimesh.load(object_path,force='mesh',process=False)
    states=[state('B',p) for p in group['poses']];transforms=[T for task,T,m in states]
    length=max(.5,float(np.linalg.norm(seedmesh.vertices,axis=1).max()+np.linalg.norm(obj.vertices,axis=1).max())+.02)
    policy=ExitClearance(obj)
    with np.load(base/'step3/step3.2/data/contacts.npz') as z:allowed=z['allowed_faces']
    def construct(parts,dirs,fan=8):return policy.construct(seed,parts,[policy.sweep(length*d,fan) for d in dirs],allowed[np.max(obj.face_normals[allowed]@np.asarray(dirs).T,axis=1)<=1e-9],check_contacts=False)
    schedule=list(direction_schedule(transforms,steps));directions=schedule[0][2]
    sweeps=[S.solid(S.swept_solid(obj,length*d)) for d in directions]
    renderer=Renderer(obj,seedmesh,transforms,group['poses'],length);height,width=renderer.frame(S.unpack(construct(sweeps,directions)['remaining']),directions,-1,sweeps).shape[:2]
    output_fps=2*fps;red_delay_frames=int(round(.5*output_fps));red_queue=[];red_events=[]
    video=out/'process/exit_direction_changes.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(output_fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE);rows=[];previous=None;previous_directions=None;video_frames=0
    try:
        for index,(active,u,directions) in enumerate(schedule):
            changed=[] if previous_directions is None else np.flatnonzero(np.linalg.norm(directions-previous_directions,axis=1)>1e-10).tolist()
            assert len(changed)<=1 and (not changed or changed==[active])
            if index:sweeps[active]=S.solid(S.swept_solid(obj,length*directions[active]))
            construction=construct(sweeps,directions);fan=8
            for trial_fan in [2,16,64]:
                if max(construction['diagnostics'][k] for k in ['nominal_sweep_overlap_m3','padded_sweep_overlap_outside_contact_cores_m3','partition_error_m3'])<1e-10:break
                trial_sweeps=[policy.sweep(length*d,trial_fan,padded=False) for d in directions]
                trial=construct(trial_sweeps,directions,trial_fan)
                metric=lambda c:max(c['diagnostics'][k] for k in ['nominal_sweep_overlap_m3','padded_sweep_overlap_outside_contact_cores_m3','partition_error_m3'])
                if metric(trial)<metric(construction):construction=trial;sweeps=trial_sweeps;fan=trial_fan
            current=construction['remaining']
            score=max(construction['diagnostics'][k] for k in ['nominal_sweep_overlap_m3','padded_sweep_overlap_outside_contact_cores_m3','partition_error_m3'])
            overlap=construction['diagnostics']['nominal_sweep_overlap_m3'];partition=construction['diagnostics']['partition_error_m3']
            support=S.unpack(current)
            grown=0. if previous is None else material_volume(current-previous)*1e6
            lost=0. if previous is None else material_volume(previous-current)*1e6
            overlap=max(material_volume(current^sweep) for sweep in sweeps)
            removed=seed-current;partition=abs(material_volume(seed)-material_volume(current)-material_volume(removed))
            world=[T[:3,:3]@d for T,d in zip(transforms,directions)]
            assert all(d[2]>=-1e-12 for d in world)
            for d in directions:assert ((obj.vertices+length*d)@d).min()>(seedmesh.vertices@d).max()+1e-9
            if previous is not None and lost>1e-8:
                red_queue.append((video_frames,previous-current))
                red_events.append(dict(birth_video_frame=video_frames,expires_at_video_frame=video_frames+red_delay_frames,lost_volume_cm3=lost))
            repeat=fps if index==0 else (fps if u==1 else 1)
            if index==len(schedule)-1:repeat+=fps
            frame=None;last_pending_key=None;first_red_volume=0.
            for offset in range(repeat):
                now=video_frames+offset
                red_queue=[(born,part) for born,part in red_queue if now-born<red_delay_frames]
                key=tuple(born for born,part in red_queue)
                if frame is None or key!=last_pending_key:
                    red=pending_red_mesh(red_queue,current,now,red_delay_frames)
                    frame=renderer.frame(support,directions,active,sweeps,red)
                    red_volume=abs(float(np.einsum('ij,ij->i',red.triangles[:,0],np.cross(red.triangles[:,1],red.triangles[:,2])).sum()/6))*1e6
                    if offset==0:first_red_volume=red_volume
                    last_pending_key=key
                encoder.stdin.write(frame.tobytes())
            if index==0:
                from PIL import Image
                Image.fromarray(frame).save(out/'process/exit_direction_changes_poster.png')
            rows.append(dict(state_index=index,active_pose=None if active<0 else group['poses'][active],progress=u,changed_pose_indices=changed,directions_fixture=directions.tolist(),directions_world=[d.tolist() for d in world],remaining_volume_cm3=material_volume(current)*1e6,grown_since_previous_cm3=grown,lost_since_previous_cm3=lost,maximum_remaining_sweep_overlap_m3=overlap,partition_error_m3=partition,exit_margin_overlap_m3=construction['diagnostics']['padded_sweep_overlap_outside_contact_cores_m3'],boolean_retry_fan_in=fan,support_shared_across_all_panels=True,first_video_frame=video_frames,frame_repetitions=repeat,pending_red_volume_at_first_frame_cm3=first_red_volume,pending_red_volume_at_last_frame_cm3=red_volume))
            video_frames+=repeat;previous=current;previous_directions=directions.copy()
            if index%8==0:print('ANIMATION',index,'/',len(schedule)-1,'active',active,'volume',rows[-1]['remaining_volume_cm3'],flush=True)
        encoder.stdin.close();code=encoder.wait()
        if code:raise RuntimeError(f'ffmpeg failed with status {code}')
    except BaseException:
        encoder.kill();encoder.wait();raise
    finally:renderer.plt.close(renderer.fig)
    inputs=[seed_path,object_path]+[p for task,T,m in states for p in task.inputs]
    report=dict(exit_clearance=policy.metadata,complete=True,pose_set=group['id'],poses=group['poses'],fps=output_fps,base_fps=fps,playback_speed_multiplier=2,red_removal_delay_seconds=.5,red_removal_delay_frames=red_delay_frames,red_is_shared_across_all_panels=True,red_render_policy='Timed cut fragments individually exclude current blue material; surfaces concatenated for display, no unstable union of coplanar red slivers',red_events=red_events,frame_count=video_frames,duration_seconds=video_frames/output_fps,geometry_state_count=len(rows),full_exit_length_m=length,frame_size_px=[width,height],support_color='blue',object_color='gray',active_arrow_color='yellow',other_arrow_color='yellow',arrow_position='center of visible sweep centerline',arrow_style='large filled yellow arrow with dark outline; always drawn above sweep and object',view_kind='orthographic isometric',support_policy=policy.metadata['policy'],only_one_exit_direction_changes_at_a_time=True,all_pose_panels_use_one_shared_support=True,paths_stay_above_native_floor=True,endpoint_separation_checked=True,motion_kind='Fixed azimuth 45 degrees; monotonic meridian tilt from native +Z to equator; object remains stationary',full_sweeps_displayed=False,full_sweep_geometry_used=True,sweep_overlay_on_object=True,sweep_view_cropped_to_object_neighborhood=True,support_closeup_per_pose=True,single_view_per_pose=True,object_screen_scale_vs_previous=700/400*1.34/1.14,sweep_display_length_m=length,sweep_opacity=.084,fixed_world_azimuth_deg=45.,final_world_elevation_deg=0.,demo_only=True,contact_preservation_acceptance_required=False,optimizer_run=False,force_acceptance_run=False,full_fixture_accepted=False,validation_policy='Visual demo only; numerical Boolean troubleshooting preserves the exit margin, without contact-area or mechanical acceptance of animation states',states=rows,states_with_regrowth=sum(r['grown_since_previous_cm3']>1e-5 for r in rows),states_with_loss=sum(r['lost_since_previous_cm3']>1e-5 for r in rows),seconds=time.monotonic()-began,provenance=provenance(inputs,[Path(__file__),HERE/'helper_func/exit_clearance.py',HERE/'helper_func/co_common.py',Path(S.__file__).with_name('translation_sweep.py')]),artifacts={'../process/exit_direction_changes.mp4':I.sha256(video),'../process/exit_direction_changes_poster.png':I.sha256(out/'process/exit_direction_changes_poster.png')})
    save(out/'data/exit_direction_animation.json',report);I.check_report(out/'data/exit_direction_animation.json')
    (out/'data/exit_direction_animation.md').write_text('# Exit direction animation\n\nLarge yellow arrows sit at the center of the visible sweeps; the orange border identifies the pose whose direction currently changes. Playback is 2x faster. Newly cut material remains red in all pose panels for exactly 0.5 seconds of output playback, then disappears. If it regrows during that interval, blue takes precedence. Each active exit follows a fixed azimuth monotonically from native +Z to the equator (parallel to its floor), then stays there. The full-length continuous exit sweep is overlaid transparently on the object-centered closeup. Its distant end may lie outside the zoomed viewport; its complete geometry is retained for cutting. Each pose has one enlarged view, with object screen size about 2.06 times the previous closeup. Other exits are held fixed. All blue supports are the same common-coordinate solid, transformed into each native pose, and change together. Each frame starts from the exact current Step3.3 support and applies all current full exit sweeps with the shared 1% clearance and bearing-contact exceptions. Old shadows can regrow; new shadows cut material away.\n\nThis is a geometry demonstration, not an optimizer trajectory or force/connectivity/ground acceptance. Saved historical Step4.2 results remain separate. Animation states are not subjected to contact-area or mechanical acceptance. Numerical Boolean troubleshooting values are recorded only to maintain the intended exit margin.\n')
    print('VIDEO COMPLETE',video,report['duration_seconds'],'seconds; regrowth',report['states_with_regrowth'],'loss',report['states_with_loss'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--set',default='pose2+3+4+7');parser.add_argument('--steps',type=int,default=32);parser.add_argument('--fps',type=int,default=10);args=parser.parse_args()
    if args.steps<4 or args.fps<1:parser.error('steps >= 4 and fps >= 1 required')
    groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    group=next((g for g in groups if g['id']==args.set),None)
    if group is None:parser.error('Unknown pose set')
    run(group,args.steps,args.fps)

if __name__=='__main__':main()
