"""An isolated, mechanically evaluated illustration of exit / material coupling.

Read saved task poses and Step4.1 exits. Only exit vectors change. Every frame
cuts the same saved Step3.1 wrap, clips all original installation floors, keeps
one material component, and tests one fixed original wrench for each task with
the existing seven-dimensional, unilateral, frictionless equilibrium solver.
This is a controlled illustration, not a replay of a search or a full fixture
certificate. All original numerical results and representation figures are read
only. Generated data stay in representation/assets/data/exit_wrench/.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
HERE = Path(__file__).resolve().parents[1]
CO = HERE.parents[1]
ROOT = CO.parents[1]
DATA = HERE / 'assets/data/exit_wrench'
sys.path[:0] = [str(CO / p) for p in ['helper_func', 'step5.2', 'step5.3', 'vis_func']]
import _bootstrap
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
import co_common as C
import cpu_mesh as R
from current_task import current_task
from whole_search.common import unpack_solid
from whole_search.classify import classify
from prune_support import material_components

INK = '#22313D'
MUTED = '#71808D'
BLUE = '#359AC1'
EXIT = '#DBA52F'
FORCE = '#DC6454'
TORQUE = '#8B6BB4'
GREEN = '#59A175'
RED = '#C96354'
PALETTE = ['#D86456', '#C39231', '#398CA6', '#8A6CB4']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2) + '\n')


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def slerp(a, b, u):
    angle = np.arccos(np.clip(a @ b, -1., 1.))
    if angle < 1e-10:
        return b.copy()
    return unit(np.sin((1-u)*angle)*a + np.sin(u*angle)*b)


class Experiment:
    def __init__(self, label):
        self.label = label
        self.base = CO / 'data/results/B' / label
        initial = self.base / 'step4/step4.1'
        self.sources = [initial/'layout.npz', initial/'report.json',
                        self.base/'step3/step3.1/registered_object.obj',
                        self.base/'step3/step3.1/wrapped_support.obj',
                        self.base/'step3/step3.1/data/report.json']
        report = json.loads((initial/'report.json').read_text())
        self.poses = report['poses']
        with np.load(initial/'layout.npz') as z:
            self.native = z['native_world'].copy()
            self.target = z['directions'].copy()
            np.testing.assert_allclose(z['placements'], np.tile(np.eye(4), (len(self.poses),1,1)))
            np.testing.assert_array_equal(z['hosts'], np.arange(len(self.poses)))
        self.tasks = [current_task('B', pose) for pose in self.poses]
        for task in self.tasks:
            self.sources += task.inputs
        self.mesh = trimesh.load(self.sources[2], process=False)
        self.allowed = [np.setdiff1d(np.arange(len(self.mesh.faces)), t.domain.work_ids)
                        for t in self.tasks]
        self.all_allowed = np.unique(np.concatenate(self.allowed))
        self.seed = C.S.solid(trimesh.load(self.sources[3], process=False))
        for T, task in zip(self.native, self.tasks):
            np.testing.assert_allclose(T, task.domain.data['frame']['T_world_mesh'], atol=1e-12)
            self.seed = self.seed.trim_by_plane(tuple(T[2,:3]), float(-T[2,3]/C.S.SCALE))
        self.start = np.array([T[:3,:3].T @ [0., 0., 1.] for T in self.native])
        self.length = max(.5, float(np.linalg.norm(self.mesh.extents))*4)
        self.rows = []
        print('INPUTS', label, self.poses, 'saved floor-clipped seed ready', flush=True)

    def geometry(self, directions):
        for T, d in zip(self.native, directions):
            assert (T[:3,:3] @ d)[2] >= -1e-12
        sweeps = [C.S.solid(C.S.swept_solid(self.mesh, self.length*d)) for d in directions]
        remaining = self.seed - C.union(sweeps)
        parts = material_components(remaining)
        if not parts:
            return trimesh.Trimesh(), dict(component_count_before_selection=0, volume_cm3=0.)
        solid = parts[0]
        mesh = unpack_solid(solid)
        return mesh, dict(component_count_before_selection=len(parts),
                          retained_component_count=1,
                          volume_cm3=C.material_volume(solid)*1e6,
                          watertight=bool(mesh.is_watertight))

    def supplies(self, support, directions):
        # All registered seats coincide, so intersect the actual boundary once.
        if len(support.faces):
            triangles, sources = C.contact_boundary(self.mesh, support, self.all_allowed)
        else:
            triangles, sources = np.empty((0,3,3)), np.empty(0, int)
        full, areas = [], []
        for k, task in enumerate(self.tasks):
            valid = np.isin(sources, self.allowed[k])
            valid &= self.mesh.face_normals[sources] @ directions[k] <= 1e-9
            tri, ids = triangles[valid], sources[valid]
            full.append(C.supply(task, self.native[k], tri, ids))
            areas.append(float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]),axis=1).sum()/2))
        return full, areas

    def select_wrenches(self):
        DATA.mkdir(parents=True, exist_ok=True)
        begin, _ = self.geometry(self.start)
        end, _ = self.geometry(self.target)
        print('ENDPOINT GEOMETRY', len(begin.faces), len(end.faces), flush=True)
        start_supply, _ = self.supplies(begin, self.start)
        end_supply, _ = self.supplies(end, self.target)
        self.selected, self.wrenches, self.selection = [], [], []
        for k, task in enumerate(self.tasks):
            raw = task.targets / task.scale
            moments = np.linalg.norm(raw[:,3:], axis=1)
            indices = np.unique(np.r_[np.linspace(0,len(raw)-1,64,dtype=int),
                                      np.argsort(moments)[-64:]])
            end_mask, _ = classify(end_supply[k], task.targets[indices])
            start_mask, _ = classify(start_supply[k], task.targets[indices])
            eligible = np.flatnonzero(end_mask & ~start_mask)
            changed = bool(len(eligible))
            if not changed:
                eligible = np.flatnonzero(end_mask)
            if not len(eligible):
                raise ValueError('No supported endpoint candidate for ' + self.poses[k])
            chosen = eligible[np.argmax(moments[indices[eligible]])]
            index = int(indices[chosen])
            self.selected.append(index)
            self.wrenches.append(-raw[index])  # actual external force and COM moment
            self.selection.append(dict(pose=self.poses[k], original_sample_index=index,
                candidate_count=len(indices), initial_supported=bool(start_mask[chosen]),
                final_supported=bool(end_mask[chosen]), changed_from_failure_to_success=changed,
                applied_external_wrench_world=(-raw[index]).tolist(),
                required_reaction_wrench_world=raw[index].tolist(),
                moment_origin_world_m=task.domain.com.tolist()))
            print('WRENCH', self.selection[-1], flush=True)
        self.wrenches = np.asarray(self.wrenches)
        save_json(DATA/'wrenches.json', dict(pose_set=self.label, selection=self.selection))

    def evaluate(self, steps):
        self.select_wrenches()
        directions = self.start.copy()
        schedule = [(-1, 0., directions.copy())]
        for active in range(len(self.poses)):
            for progress in np.linspace(0.,1.,steps+1)[1:]:
                directions[active] = slerp(self.start[active],self.target[active],progress)
                schedule.append((active,float(progress),directions.copy()))
        for index, (active, progress, directions) in enumerate(schedule):
            began=time.monotonic()
            mesh, record = self.geometry(directions)
            supplies, areas = self.supplies(mesh,directions)
            flags, classifiers = [], []
            for k, task in enumerate(self.tasks):
                mask, info = classify(supplies[k], task.targets[[self.selected[k]]])
                flags.append(bool(mask[0]))
                classifiers.append(info)
            filename = f'state_{index:03d}.npz'
            np.savez_compressed(DATA/filename, vertices=mesh.vertices, faces=mesh.faces,
                                directions=directions, supported=np.asarray(flags),
                                native_world=self.native, applied_wrenches_world=self.wrenches)
            self.rows.append(dict(index=index, active_pose_index=active, progress=progress,
                directions_material=directions.tolist(),
                directions_world=np.einsum('kij,kj->ki',self.native[:,:3,:3],directions).tolist(),
                supported=flags, contact_areas_m2=areas, classifiers=classifiers,
                geometry=filename, geometry_sha256=sha(DATA/filename),
                evaluation_seconds=time.monotonic()-began, **record))
            save_json(DATA/'states.json', self.rows)
            print('STATE', index, '/', len(schedule)-1, flags,
                  round(record['volume_cm3'],2), 'cm3', round(time.monotonic()-began,2),'s',flush=True)
        assert all(self.rows[-1]['supported'])
        assert not all(self.rows[0]['supported'])
        np.testing.assert_allclose(self.rows[-1]['directions_material'],self.target,atol=1e-12)
        record = dict(complete=True, object='B', pose_set=self.label, poses=self.poses,
            fixed_original_native_world=self.native.tolist(), selection=self.selection,
            fixed_original_poses=True, fixed_original_translations=True,
            fixed_applied_wrenches=True, only_exit_directions_change=True,
            original_saved_step41_target_exits=self.target.tolist(),
            initial_exit_policy='native world +Z in every task',
            illustration_policy='controlled spherical direction interpolation; not optimizer replay',
            shared_material_policy='one saved Step3.1 wrap, all nominal full exits, all original floors; largest actual material component',
            support_geometry_rebuilt_from_same_seed=True,
            contacts='actual retained material boundary intersected with original allowed workpiece faces',
            force_model='unchanged original seven-dimensional frictionless nonnegative normal reactions and shared no-uplift slack',
            original_workpiece_floor_reactions=True,
            physical_scope='balance of exactly one fixed original sampled external wrench per pose; no full task-set or ground-base certificate',
            original_full_fixture_results_modified=False, search_rerun=False,
            final_all_displayed_wrenches_supported=all(self.rows[-1]['supported']),
            state_count=len(self.rows),
            input_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(self.sources))},
            numerical_sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
                Path(C.__file__),CO/'helper_func/whole_search/classify.py',
                CO/'helper_func/legacy_kernel/step3_scheculer/passive_support.py',
                CO/'helper_func/legacy_kernel/step4_connect_support/build_coupled_saddle.py',
                CO/'step5.2/prune_support.py']})
        save_json(DATA/'evidence.json',record)


def project(point, camera, width, height):
    focus,basis,span=camera
    p=(np.asarray(point)-focus) @ basis.T
    return np.array([width/2+p[0]*height/span,height/2-p[1]*height/span])


def arrow(draw, points, color, width=5, head=14):
    points=np.asarray(points,float)
    points=[tuple(v) for v in points]
    draw.line(points,fill='white',width=width+4,joint='curve')
    draw.line(points,fill=color,width=width,joint='curve')
    a,b=np.asarray(points[-2]),np.asarray(points[-1]);d=unit(b-a)
    normal=np.array([-d[1],d[0]])
    triangle=[tuple(b),tuple(b-head*d+head*.46*normal),tuple(b-head*d-head*.46*normal)]
    draw.polygon(triangle,fill=color)


def text(draw, xy, string, size=22, fill=INK, bold=False, anchor=None):
    draw.text(tuple(xy),string,font=font(size,bold),fill=fill,anchor=anchor)


class Movie:
    def __init__(self, experiment, size=(1920,1080)):
        self.e=experiment
        self.w,self.h=size
        self.raster=R.Renderer()
        self.panel_size=(910,367)
        self.cameras=[]
        self.fixed=[]
        extent=float(self.e.mesh.extents.max())
        self.length=extent*.83
        view=np.array([1.,-1.,.8])
        for k,(task,T) in enumerate(zip(self.e.tasks,self.e.native)):
            world=self.e.mesh.copy();world.apply_transform(T)
            points=np.vstack([world.vertices,task.domain.com+self.length*np.eye(3),
                              task.domain.com-self.length*np.eye(3)])
            camera=R.fit(points,view,self.panel_size[0]/self.panel_size[1],padding=1.12)
            self.cameras.append(camera)
            colors=np.broadcast_to(R.rgb('#AAB3BA'),(len(world.faces),3,3)).copy()
            colors[task.domain.work_ids]=R.rgb(GREEN)
            body=R.piece(dict(v=world.vertices,f=world.faces),colors)
            low=world.vertices[:,:2].min(0)-extent*.15
            high=world.vertices[:,:2].max(0)+extent*.15
            floor=np.array([[low[0],low[1],0.],[high[0],low[1],0.],
                            [high[0],high[1],0.],[low[0],high[1],0.]])
            plane=R.piece(dict(v=floor,f=np.array([[0,1,2],[0,2,3]])),'#EEF2F5')
            plane['unlit']=True
            self.fixed.append((body,plane,floor))

    def panel(self,k,mesh,direction):
        T=self.e.native[k];task=self.e.tasks[k]
        body,floor,boundary=self.fixed[k]
        parts=[R.placed(floor),R.placed(body)]
        if len(mesh.faces):
            support=R.piece(dict(v=mesh.vertices,f=mesh.faces),BLUE)
            parts.append(R.placed(support,T[:3,:3],T[:3,3]))
        width,height=self.panel_size
        picture=self.raster.render(parts,self.cameras[k],width,height,scale=1.25)
        draw=ImageDraw.Draw(picture)
        P=lambda q:project(q,self.cameras[k],width,height)
        center=task.domain.com
        origin=P(center)
        d=T[:3,:3]@direction
        end=P(center+self.length*d)
        arrow(draw,[origin,end],EXIT,9,19)
        text(draw,end+np.array([10,-17]),'d',23,EXIT,True)
        # Resultant external force and its paired COM moment never change.
        force=self.e.wrenches[k,:3];torque=self.e.wrenches[k,3:]
        tip=P(center+self.length*.52*unit(force))
        arrow(draw,[origin,tip],FORCE,6,17)
        text(draw,tip+np.array([9,0]),'F',23,FORCE,True)
        camera=self.cameras[k];axis=unit(torque)
        tangent=np.cross(axis,camera[1][2])
        if np.linalg.norm(tangent)<1e-5:tangent=np.cross(axis,[1.,0.,0.])
        u=unit(tangent);v=np.cross(axis,u)
        # A torque is a free couple: offset for legibility, same world axis/sign.
        anchor=center+self.length*(.75*camera[1][0]+.35*camera[1][1])
        theta=np.linspace(.15,1.8*np.pi,50)
        circle=anchor+self.length*.23*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)
        curve=[P(point) for point in circle]
        draw.line([tuple(origin),tuple(P(anchor))],fill='#C7CCD3',width=2)
        arrow(draw,curve,TORQUE,5,14)
        text(draw,P(anchor)+np.array([30,-13]),'τ',24,TORQUE,True)
        draw.ellipse((origin[0]-4,origin[1]-4,origin[0]+4,origin[1]+4),fill=INK,outline='white',width=1)
        return picture

    def frame(self,index,phase):
        row=self.e.rows[index]
        with np.load(DATA/row['geometry']) as z:
            mesh=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
            directions=z['directions'].copy()
        image=Image.new('RGB',(self.w,self.h),'#FAFBFC')
        draw=ImageDraw.Draw(image)
        text(draw,(52,32),'Exit directions shape what can be supported',39,INK,True)
        text(draw,(54,90),'Fixed task poses. Fixed force–torque pairs. One shared support.',23,MUTED)
        count=sum(row['supported'])
        for k in range(4):
            x=42+(k%2)*939;y=151+(k//2)*421
            active=row['active_pose_index']==k and phase=='adjust'
            draw.rounded_rectangle((x,y,x+914,y+399),radius=17,fill='white',
                                   outline=EXIT if active else '#DFE6EC',width=4 if active else 2)
            picture=self.panel(k,mesh,directions[k])
            image.paste(picture,(x+2,y+29))
            text(draw,(x+22,y+13),self.e.poses[k].replace('_',' ').capitalize(),24,PALETTE[k],True)
            status=row['supported'][k];color=GREEN if status else RED
            draw.rounded_rectangle((x+674,y+12,x+890,y+47),radius=10,fill='#EDF6F0' if status else '#FAEFEC')
            text(draw,(x+782,y+29),'SUPPORTED' if status else 'UNSUPPORTED',16,color,True,'mm')
            # White strip keeps labels independent from the projected geometry.
            draw.rectangle((x+10,y+345,x+900,y+390),fill='white')
            force=np.linalg.norm(self.e.wrenches[k,:3]);moment=np.linalg.norm(self.e.wrenches[k,3:])*1000
            text(draw,(x+22,y+358),f'Fixed (F, τ)   |F| = {force:.2f} mg   |τ| = {moment:.1f} mg·mm',17,MUTED)
            if active:
                text(draw,(x+890,y+358),'ADJUST EXIT',17,EXIT,True,'ra')
        draw=ImageDraw.Draw(image)
        if phase=='intro':caption='Start with independent upward exits'
        elif phase=='hold':caption='The same four force–torque pairs can now be balanced'
        else:caption=f'Adjust {self.e.poses[row["active_pose_index"]].replace("_"," ")} exit → update shared material → recheck all four loads'
        text(draw,(54,1000),caption,23,INK,True)
        text(draw,(1867,1002),f'{count}/4 shown wrenches supported',22,GREEN if count==4 else RED,True,'ra')
        text(draw,(54,1045),'Blue: shared support    Green: working surface    Gold: exit    Coral: force    Purple: torque',17,MUTED)
        text(draw,(1867,1045),'One fixed original sampled wrench per pose; arrows scaled for visibility.',16,MUTED,False,'ra')
        return image

    def render(self,fps=24):
        video=HERE/'exit_wrench_coupling.mp4'
        cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{self.w}x{self.h}','-r',str(fps),'-i','-','-an','-c:v','libx264',
             '-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        timings=[];frame_count=0
        steps=(len(self.e.rows)-1)//4
        # Three seconds of initial explanation; four four-second adjustments;
        # five seconds on the unchanged successful endpoint.
        repeats=[fps*3]+[round(fps*4/steps)]*(len(self.e.rows)-1)
        samples={0,len(self.e.rows)//3,2*len(self.e.rows)//3,len(self.e.rows)-1}
        posters=[]
        try:
            for index in range(len(self.e.rows)):
                phase='intro' if index==0 else 'adjust'
                picture=self.frame(index,phase)
                first=frame_count
                for _ in range(repeats[index]):
                    encoder.stdin.write(picture.tobytes());frame_count+=1
                if index in samples:
                    filename=DATA/f'preview_{index:03d}.png';picture.save(filename);posters.append((index,picture.copy()))
                timings.append(dict(state_index=index,first_video_frame=first,frame_count=repeats[index]))
                if index%8==0:print('RENDER',index,'/',len(self.e.rows)-1,flush=True)
            picture=self.frame(len(self.e.rows)-1,'hold')
            picture.save(HERE/'exit_wrench_coupling.png')
            for _ in range(fps*5):encoder.stdin.write(picture.tobytes());frame_count+=1
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Video encoding failed')
        except BaseException:
            encoder.kill();encoder.wait();raise
        storyboard=Image.new('RGB',(1920,1080),'white')
        for panel,(index,picture) in enumerate(posters):
            tile=picture.resize((960,540),Image.Resampling.LANCZOS)
            storyboard.paste(tile,((panel%2)*960,(panel//2)*540))
        storyboard.save(HERE/'exit_wrench_storyboard.png')
        save_json(DATA/'video.json',dict(complete=True,file=video.name,sha256=sha(video),
            width=self.w,height=self.h,fps=fps,frame_count=frame_count,duration_seconds=frame_count/fps,
            state_video_frames=timings,final_hold_frames=fps*5,
            final_poster='exit_wrench_coupling.png',storyboard='exit_wrench_storyboard.png',
            source_sha256=sha(__file__),all_labels_language='English'))
        print('COMPLETE',video,frame_count/fps,'seconds',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set',default='pose1+2+3+27')
    parser.add_argument('--steps',type=int,default=24)
    parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    experiment=Experiment(args.set)
    if args.render_only:
        record=json.loads((DATA/'evidence.json').read_text())
        assert record['pose_set']==args.set
        experiment.rows=json.loads((DATA/'states.json').read_text())
        experiment.selection=record['selection']
        experiment.wrenches=np.array([r['applied_external_wrench_world'] for r in experiment.selection])
    else:
        experiment.evaluate(args.steps)
    Movie(experiment).render(args.fps)


if __name__=='__main__':main()
