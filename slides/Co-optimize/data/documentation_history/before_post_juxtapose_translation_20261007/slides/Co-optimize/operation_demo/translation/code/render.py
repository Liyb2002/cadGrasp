"""Same B shape and four-panel style as the direction demonstration."""
import sys,json,time,subprocess
from pathlib import Path
import numpy as np
CO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(CO/'helper_func'))
import _bootstrap
from co_common import *
from PIL import Image
class Renderer:
    def __init__(self,obj,seedmesh,transforms,poses,length,columns=None,labels=True,final_offset=None):
        self.final_offset=np.zeros(3) if final_offset is None else np.asarray(final_offset)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.patches import Rectangle, FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d
        self.arrow_patch=FancyArrowPatch;self.project=proj3d.proj_transform
        self.plt=plt;self.collection=Poly3DCollection;self.transforms=transforms;self.obj=obj;self.length=length;self.sweep_artists=[None]*len(poses)
        n=len(poses);cols=columns if columns is not None else (2 if n==4 else min(3,n));rows=(n+cols-1)//cols
        self.fig=plt.figure(figsize=(cols*(6 if columns is not None else 10),rows*(6 if columns is not None else 7)),dpi=100,facecolor='white');self.axes=[];self.artists=[];self.arrows=[];self.borders=[]
        elevation=float(np.degrees(np.arctan(1/np.sqrt(2))));azimuth=-45.
        self.light=np.array([1.,-1.,1.])/np.sqrt(3)+np.array([-.15,-.2,.65]);self.light/=np.linalg.norm(self.light)
        for i,(T,pose) in enumerate(zip(transforms,poses)):
            slot_x=(i%cols)/cols;slot_y=1-(i//cols+1)/rows
            ax=self.fig.add_axes([slot_x,slot_y,1/cols,1/rows],projection='3d');self.axes.append(ax)
            layout=np.vstack([obj.vertices,seedmesh.vertices,obj.vertices+self.final_offset,seedmesh.vertices+self.final_offset])
            points=transform_points(layout,T)*1000
            # Object-centered closeup; full sweeps remain the actual cutting geometry.
            center=(points.min(0)+points.max(0))/2;radius=float(np.ptp(points,axis=0).max())*.60
            for axis,value in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
            ax.set_box_aspect((1,1,1),zoom=1.34);ax.set_proj_type('ortho');ax.view_init(elev=elevation,azim=azimuth,roll=0);ax.set_axis_off()
            if labels:ax.text2D(.04,.05,pose.replace('_',' '),transform=ax.transAxes,color='#303840',fontsize=15)
            lo=points[:,:2].min(0)-8;hi=points[:,:2].max(0)+8
            boundary=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
            ax.plot(*boundary.T,color='#d4d4d4',linewidth=.6)
            border=Rectangle((slot_x+.01/cols,slot_y+.01/rows),.98/cols,.98/rows,fill=False,edgecolor='#ffffff',linewidth=2,transform=self.fig.transFigure,clip_on=False)
            self.fig.add_artist(border);self.borders.append(border)
            self.artists.append(None);self.arrows.append(None)
        self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)

    def frame(self,support,directions,active,sweeps,red=None,green=None):
        for i,(ax,T) in enumerate(zip(self.axes,self.transforms)):
            triangles=[];colors=[]
            materials=[(self.obj,'#a4a8ac'),(support,'#319cd7')]
            # Sliding-channel view keeps the cavity visible; omit removed fragments.
            if green is not None:materials.append((green,'#21c875'))
            for source,color in materials:
                if not len(source.faces):continue
                world=source.copy()
                if source is self.obj:world.apply_translation(self.offsets[i])
                world.apply_transform(T)
                rgb=np.array(self.plt.matplotlib.colors.to_rgb(color));brightness=.58+.42*np.maximum(world.face_normals@self.light,0)
                triangles.append(world.triangles*1000);colors.append(np.c_[brightness[:,None]*rgb,np.full(len(world.faces),1.0)])
            if self.artists[i] is not None:self.artists[i].remove()
            self.artists[i]=self.collection(np.concatenate(triangles),facecolors=np.concatenate(colors),edgecolors='none',zsort='average');ax.add_collection3d(self.artists[i])
            self.borders[i].set_edgecolor('#e78b28' if active==i else '#ffffff')
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:,:,:3].copy()


def main():
    out=Path(os.environ.get('CADGRASP_TRANSLATION_OUT',str(Path(__file__).resolve().parents[1]/'vis')));out.mkdir(parents=True,exist_ok=True)
    base=CO/'output/B/pose1+2+4+6'
    object_path=base/'step3/step3.1/registered_object.obj'
    wrap_path=base/'step3/step3.2/wrapped_support.obj'
    init_path=base/'step4/step4.1/data/report.json'
    from layout_geometry import Layout
    layout=Layout()
    obj=layout.obj;wrap=layout.wrap;poses=layout.poses
    states=layout.states;transforms=layout.transforms;directions=layout.directions
    length=.5;fps=12;steps=16;distance=layout.distance/3
    offsets=np.zeros((4,3));moves=np.zeros((4,3));moves[0]=layout.final_offset/3
    initial,_=layout.construct(offsets)
    renderer=Renderer(obj,wrap,transforms,poses,length,columns=2,labels=True,final_offset=moves[0])
    renderer.fig.text(.5,.98,f'Placement adjustment | only pose 1 moves {distance*1000:.1f} mm',ha='center',fontsize=15,color='#303840')
    renderer.fig.text(.5,.02,'Gray: object | blue: retained support | green: new support | no sliding channel',ha='center',fontsize=12,color='#303840')
    renderer.offsets=offsets.copy();width,height=renderer.fig.canvas.get_width_height()
    video=out/'pose1+2+4+6_translation.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE);rows=[];previous=None;frames=0
    schedule=[(-1,0.,offsets.copy())]
    for active in [0]:
        for u in np.linspace(0,1,steps+1)[1:]:
            offsets[active]=moves[active]*u
            schedule.append((active,float(u),offsets.copy()))
    try:
        for index,(active,u,offsets) in enumerate(schedule):
            current,sweeps=layout.construct(offsets)
            overlap=max(material_volume(current^sweep) for sweep in sweeps)
            red=None if previous is None else S.unpack(previous-current)
            renderer.offsets=offsets
            grown=F.md.Manifold() if index==0 else current-initial
            retained=current if index==0 else current^initial
            frame=renderer.frame(S.unpack(retained.simplify(1e-9/S.SCALE)),directions,active,sweeps,red,S.unpack(grown.simplify(1e-9/S.SCALE)))
            repeat=fps if index==0 or u==1 else 2
            if index==len(schedule)-1:repeat+=2*fps
            for _ in range(repeat):encoder.stdin.write(frame.tobytes())
            if index==0:Image.fromarray(frame).save(out/'poster.png')
            if index==len(schedule)-1:Image.fromarray(frame).save(out/'final.png')
            rows.append(dict(state=index,active_pose=None if active<0 else poses[active],offsets_fixture_m=offsets.tolist(),offsets_native_mm=[(T[:3,:3]@o*1000).tolist() for T,o in zip(transforms,offsets)],remaining_volume_cm3=material_volume(current)*1e6,nominal_exit_overlap_m3=overlap,new_support_vs_initial_cm3=material_volume(grown)*1e6,restored_inside_original_wrap_cm3=material_volume(grown^layout.wrap_solid)*1e6,expanded_outside_original_wrap_cm3=material_volume(grown-layout.wrap_solid)*1e6,gained_material_cm3=0 if previous is None else material_volume(current-previous)*1e6,lost_material_cm3=0 if previous is None else material_volume(previous-current)*1e6,first_frame=frames,repetitions=repeat))
            frames+=repeat;previous=current
            if index%1==0:print('TRANSLATION',index,'/',len(schedule)-1,flush=True)
        renderer.offsets=offsets
        # End by merging the highlighted new regions into ordinary blue support.
        blue_frame=renderer.frame(S.unpack(current.simplify(1e-9/S.SCALE)),directions,0,sweeps)
        for _ in range(2*fps):encoder.stdin.write(blue_frame.tobytes())
        frames+=2*fps
        encoder.stdin.close()
        if encoder.wait():raise RuntimeError('Video encoding failed')
    except BaseException:
        encoder.kill();encoder.wait();raise
    finally:renderer.plt.close(renderer.fig)
    record=dict(complete=True,demo_only=True,optimizer_run=False,force_acceptance_run=False,poses=poses,shape='Same registered B and Step3.2 wrap as direction video; no base',directions_fixture_fixed=directions.tolist(),motion='Only pose1 translates one third of its native x body width along its native floor +x; poses2/4/6 stay fixed; every direction unchanged',seed_policy='Union of wraps only at current final placements, minus current complete saved-direction exits and repositioned saved per-pose cuts; no relocation sweep or convex hull',colors='gray object, blue common remaining material, green all currently available new material relative to the initial support (including restored and expanded regions), orange active-panel border; support opaque, removed fragments omitted',panels='2x2: pose1/pose2, pose4/pose6',translation_distance_m=distance,exit_arrows_displayed=False,sweep_displayed=False,nominal_boolean_diagnostics_only=True,display_mesh_simplification_tolerance_m=1e-9,relocation_path_carved=False,highlight_policy='New regions green during translation and final pause, then merge into blue for 2 seconds',fps=fps,frame_count=frames,duration_seconds=frames/fps,full_exit_length_m=length,states=rows,provenance=provenance([object_path,wrap_path,init_path,layout.baseline_path]+layout.cut_paths+[p for task,_,_ in states for p in task.inputs],[Path(__file__),Path(__file__).with_name("layout_geometry.py")]),video_sha256=I.sha256(video))
    (out/'animation.json').write_text(json.dumps(record,indent=2)+'\n')
    print('COMPLETE',video,flush=True)

if __name__=='__main__':main()
