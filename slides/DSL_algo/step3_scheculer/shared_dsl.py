"""One physical, object-attached head program for ALL poses.

Heads are compiled once in the first task frame, then rigidly viewed in each
other task. Installed views must coincide, including their finite-depth solids.
This restricted placement family is explicit: it does not search arbitrary
regrasp registrations. Global deletes/merges count physical heads, not head uses.
"""
from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from step3_scheculer import contact_dsl as DSL, contacts as I, exit_options as EXIT
from step3_scheculer.pair_geometry import transform_contact
from step3_scheculer.floor_margin import transforms_from_owner
from step2_local_support import geometry as G, circles as P

STAGE = 'dsl_shared'
BODY_STAGE = 'dsl_shared_support'
SCHEMA = 'joint_shared_head_descent_v4'


def placement(tasks):
    """Row-vector world -> fixture; all object meshes coincide in the fixture."""
    transforms = transforms_from_owner(tasks[0], tasks)
    bases = np.asarray([t[:3, :3] for t in transforms])
    offsets = np.asarray([-t[:3, 3]@b for t, b in zip(transforms, bases)])
    for task, b, o in zip(tasks, bases, offsets):
        np.testing.assert_allclose(task.domain.mesh.vertices@b+o, tasks[0].domain.mesh.vertices,
                                   atol=2e-12, rtol=0)
    return dict(bases=bases.tolist(), offsets=offsets.tolist(),
                family='one object-attached fixture, rigidly reoriented with each saved task pose',
                optimized=False, arbitrary_regrasp_registration_searched=False)


class ViewCompiler:
    def __init__(self, joint, index):
        self.joint, self.index = joint, index
        self.task = joint.tasks[index]
        self.exit_catalogue = EXIT.catalogue(self.task.domain.mesh)
        self.backend = joint.canonical.backend
        self.scale = joint.scale
        self.min_radius, self.max_radius = joint.min_radius, joint.max_radius
    def compile(self, program):
        canonical = self.joint.canonical.compile(program)
        if canonical is None: return None
        return [transform_contact(dict(c, candidate_id=f'SH{p.ident:04d}'), self.joint.transforms[self.index])
                for p, c in zip(program.patches, canonical)]


class JointCompiler:
    def __init__(self, tasks, config):
        self.tasks = tasks
        self.transforms = transforms_from_owner(tasks[0], tasks)
        excluded = np.unique(np.concatenate([t.domain.work_ids for t in tasks]))
        self.canonical = DSL.Compiler(tasks[0], tasks, device=config['device'],
            gpu_iterations=config['gpu_iterations'], excluded_work_ids=excluded)
        self.scale = self.canonical.scale
        self.min_radius, self.max_radius = self.canonical.min_radius, self.canonical.max_radius
        self.backend = self.canonical.backend
        self.views = [ViewCompiler(self, k) for k in range(len(tasks))]
        self.vectors = [np.asarray(v.exit_catalogue['vectors']) for v in self.views]
        self.fixture_vectors = [v@t[:3,:3] for v,t in zip(self.vectors,self.transforms)]
        self.force_cache = {}

    def compile(self, program): return self.canonical.compile(program)
    def seeds(self, count): return self.canonical.seeds(count)

    def normal_objective(self, contacts):
        """Soft preference for similar exits; no hard common direction or cutoff.

        Opening penalties dominate directional preference. Exact continuous
        paths and actual sweep unions are checked later, not claimed here.
        """
        blocks=[]
        for task, transform, vectors in zip(self.tasks,self.transforms,self.vectors):
            faces=np.unique(np.concatenate([c['source_faces'] for c in contacts]))
            normals=task.domain.mesh.face_normals[faces]
            blocks.append(np.maximum(-(normals@vectors.T).min(axis=0),0.)**2)
        ids=[int(np.argmin(v)) for v in blocks]
        for _ in range(2):
            for k in range(len(ids)):
                others=[self.fixture_vectors[j][ids[j]] for j in range(len(ids)) if j!=k]
                angle=np.mean([1-np.clip(self.fixture_vectors[k]@v,-1,1) for v in others],axis=0) if others else 0.
                ids[k]=int(np.argmin(blocks[k]+.02*angle))
        opening=float(np.mean([b[i] for b,i in zip(blocks,ids)]))
        closeness=DSL.alignment([v[i] for v,i in zip(self.fixture_vectors,ids)])
        return opening,closeness

    def loss(self, program, sample_ids, reference=None, exits=False):
        return self.loss_batch([program],sample_ids,reference,exits)[0]

    def loss_batch(self, programs, sample_ids, reference=None, exits=False):
        # One GPU batch for all perturbations of a pose. Each pose has its own
        # immutable loads and floor rays; EVERY pose contributes to each gradient.
        compiled=[self.canonical.compile(p) for p in programs]
        losses=np.full((len(programs),len(self.tasks)),100.)
        for k,(task,view) in enumerate(zip(self.tasks,self.views)):
            pending={}
            keys=[]
            for i,(p,c) in enumerate(zip(programs,compiled)):
                key=(k,tuple(sample_ids),tuple((x.center,x.radius) for x in p.patches))
                keys.append(key)
                if c is not None and key not in self.force_cache and key not in pending:
                    viewed=[transform_contact(row,self.transforms[k]) for row in c]
                    pending[key]=DSL.reduced_rays(task.supply(viewed))
            if pending:
                values=self.backend.solve(list(pending.values()),task.targets[sample_ids])
                for key,row in zip(pending,values):
                    self.force_cache[key]=float(row.mean()+.5*row.max(initial=0.))
            for i,key in enumerate(keys):losses[i,k]=self.force_cache.get(key,100.)
        rows=[]
        for p,c,force in zip(programs,compiled,losses):
            opening,closeness=self.normal_objective(c) if exits and c else (0.,0.)
            # Count is a discrete objective; deletes/merges change it. Descent
            # repairs the shared geometry after such a rewrite. No soft heads.
            count=.002*len(p.patches)
            force_loss=float(force.mean()+force.max())
            total=force_loss+count+(.5*opening+.01*closeness if exits else 0.)
            rows.append((total,dict(force=force_loss,normal=opening,alignment=closeness,
                                   head_count=count,per_pose_force=force.tolist())))
        if len(self.force_cache)>16000:self.force_cache.clear()
        return rows


class JointSearch:
    def __init__(self, tasks, group, config, Search, seed_contacts):
        self.tasks,self.group,self.config=tasks,group,config
        self.out=group/'step3_scheculer'/STAGE
        self.out.mkdir(parents=True,exist_ok=True)
        self.compiler=JointCompiler(tasks,config)
        self.input_paths=[p for t in tasks for p in t.inputs]
        self.legacy_seed_count=0
        patches=[]
        for k,task in enumerate(tasks):
            contacts,source=seed_contacts(group,task.pose)
            if source:self.input_paths.append(source)
            self.legacy_seed_count+=len(contacts)
            inverse=np.linalg.inv(self.compiler.transforms[k])
            for c in contacts:
                center=np.asarray(c['center_m'])@inverse[:3,:3].T+inverse[:3,3]
                if not any(np.linalg.norm(center-np.asarray(p.center))<self.compiler.scale*1e-6 for p in patches):
                    patches.append(DSL.Patch(len(patches)+1,tuple(center),float(c['radius_m'])))
        self.seeds=self.compiler.seeds(config['seeds'])
        if not patches:patches=[replace(p,ident=i+1) for i,p in enumerate(self.seeds[:3])]
        # Resolve invalid wraps by shrinking actual radii; do not invent normals.
        fixed=[]
        for p in patches:
            for factor in (1.,.7,.45,.25,.1):
                trial=replace(p,radius=max(self.compiler.min_radius,p.radius*factor))
                if self.compiler.canonical.compile_patch(trial) is not None:
                    contact=self.compiler.canonical.compile_patch(trial)
                    if not any(np.linalg.norm(contact['center_m']-self.compiler.canonical.compile_patch(q)['center_m'])<self.compiler.scale*1e-6 for q in fixed):
                        fixed.append(trial)
                    break
        self.initial=DSL.Program(tuple(fixed))
        self.base_sample_ids=np.unique(np.linspace(0,32767,config['samples'],dtype=int))
        self.sample_ids=self.base_sample_ids.copy()
        self.sample_cap=max(config['samples'],config.get('proposal_sample_cap',128))
        self.next_ident=2000
        self.events=[]
        self.views=[]
        self.depth=P.DEPTH_FRACTION*self.compiler.scale
        from step3_scheculer.pair_geometry import FreePaths
        from step3_scheculer.floor_margin import transforms_from_owner
        for k,task in enumerate(tasks):
            # Reuse authoritative LP/path checks, but never a pose-local search.
            s=Search.__new__(Search)
            s.task,s.out,s.config=task,self.out/task.pose,config
            s.compiler=self.compiler.views[k]
            s.floor_tasks=tasks;s.floor_transforms=transforms_from_owner(task,tasks)
            s.catalogue=s.compiler.exit_catalogue;s.depth=self.depth
            s.analyzer=EXIT.PathAnalyzer(task.domain.mesh,self.depth)
            s.paths=FreePaths(task.domain.mesh,np.array([[0.,0.,1.,0.]]))
            s.path_cache={};s.lp_cache={};s.direction_cache={};s.events=[]
            s.initial=self.initial;s.sample_ids=self.sample_ids.copy()
            self.views.append(s)

    def classify(self,program,refine=False):
        rows=[s.classify(program,refine=refine) for s in self.views]
        if refine:
            combined=np.unique(np.concatenate([s.sample_ids for s in self.views]))
            hard=np.setdiff1d(combined,self.base_sample_ids)
            budget=max(0,self.sample_cap-len(self.base_sample_ids))
            if len(hard)>budget:
                hard=hard[np.linspace(0,len(hard)-1,budget,dtype=int)] if budget else np.array([],int)
            self.sample_ids=np.unique(np.r_[self.base_sample_ids,hard])
            # Prevent the per-pose proposal sets from accumulating without bound.
            # The authoritative LP masks still contain ALL 32768 original loads.
            for s in self.views:s.sample_ids=self.sample_ids.copy()
        return dict(passed=all(r['passed'] for r in rows),covered=sum(r['covered'] for r in rows),
                    minimum_covered=min(r['covered'] for r in rows),rows=rows)

    def exits(self,program):return [s.exits(program) for s in self.views]
    def feasible(self,program):
        force=self.classify(program)
        return force['passed'] and all(ids for ids,_ in self.exits(program))

    def emit(self,operation,program,**data):
        row=dict(operation=operation,physical_heads=len(program.patches),**data)
        self.events.append(row)
        I.save(self.out/'joint_progress.json',dict(complete=False,events=self.events))
        print('JOINT',self.group.name,operation,'physical heads',len(program.patches),data.get('covered',''),flush=True)

    def improve(self,program,exits,operation):
        p,history=DSL.descend(self.compiler,program,self.sample_ids,exits=exits,steps=self.config['steps'])
        row=self.classify(p,refine=True)
        self.emit(operation,p,covered=row['covered'],proposal_sample_count=len(self.sample_ids),gradient_steps=history)
        return p,row

    def mutations(self,program,kind,exits):
        proposals=[]
        for seed in self.seeds:
            patch=replace(seed,ident=self.next_ident);self.next_ident+=1
            proposals.extend(program.substitute(i,patch) for i in range(len(program.patches))) if kind=='replace' else proposals.append(program.add(patch))
        values=self.compiler.loss_batch(proposals,self.sample_ids,exits=exits)
        return [p for _,p in sorted(zip([v[0] for v in values],proposals),key=lambda x:x[0])][:self.config['rewrite_trials']]

    def force_stage(self):
        p=self.initial;score=self.classify(p,refine=True)
        self.emit('joint_saved_seed_force',p,covered=score['covered'])
        for s,r in zip(self.views,score['rows']):s.events=[dict(operation='joint_saved_seed_force',covered=r['covered'])]
        candidate,row=self.improve(p,False,'joint_force_descent')
        if (row['minimum_covered'],row['covered']) >= (score['minimum_covered'],score['covered']):p,score=candidate,row
        for attempt in range(self.config['force_rewrites']):
            if score['passed']:break
            kind='replace' if attempt%2==0 or len(p.patches)>=self.config['max_heads'] else 'add'
            for candidate in self.mutations(p,kind,False):
                candidate,row=self.improve(candidate,False,'joint_force_'+kind)
                if (row['minimum_covered'],row['covered'])>(score['minimum_covered'],score['covered']):p,score=candidate,row
                if score['passed']:break
        for s,row in zip(self.views,score['rows']):
            s.out.mkdir(parents=True,exist_ok=True)
            I.save_contacts(s.out/'force_only_contacts.npz',row['contacts'])
            np.savez_compressed(s.out/'force_only_coverage.npz',passed=row['mask'])
            I.save(s.out/'force_only_report.json',dict(complete=True,pose=s.task.pose,force_passed=row['passed'],
                covered_count=row['covered'],sample_count=32768,head_count=len(p.patches),program=asdict(p),
                acceleration=self.compiler.backend.info,exit_required=False,
                provenance=dict(inputs=I.hashes(self.input_paths),code=I.hashes(sources())),
                artifacts={n:I.sha256(s.out/n) for n in ('force_only_contacts.npz','force_only_coverage.npz')}))
        return p

    def prune(self,p,require_exits=False):
        changed=True
        while changed and len(p.patches)>1:
            changed=False
            proposals=[p.delete(i) for i in range(len(p.patches))]
            # Explicit merge rewrites: one adjustable head replaces two heads.
            # A merge is physical geometry + global LP/path checks, never relabeling.
            for i in range(len(p.patches)):
                for j in range(i+1,len(p.patches)):
                    a,b=p.patches[i],p.patches[j]
                    distance=np.linalg.norm(np.asarray(a.center)-b.center)
                    if distance>2*(a.radius+b.radius):continue
                    patch=DSL.Patch(a.ident,tuple((np.asarray(a.center)+b.center)/2),
                                    min(self.compiler.max_radius,max(a.radius,b.radius)+distance*.5))
                    proposals.append(p.delete(j).delete(i).add(patch))
            values=self.compiler.loss_batch(proposals,self.sample_ids,exits=require_exits)
            ranked=sorted(zip([v[0] for v in values],proposals),key=lambda x:x[0])
            for n,(_,trial) in enumerate(ranked):
                score=self.classify(trial,refine=True)
                valid=score['passed'] and (not require_exits or all(ids for ids,_ in self.exits(trial)))
                if not valid and n<self.config['rewrite_trials']:
                    trial,score=self.improve(trial,require_exits,'joint_delete_or_merge_repair')
                    valid=score['passed'] and (not require_exits or all(ids for ids,_ in self.exits(trial)))
                if valid:
                    self.emit('accept_global_delete_or_merge',trial,covered=score['covered'])
                    p=trial;changed=True;break
        return p

    def repair(self,p):
        if self.feasible(p):
            # Even feasible shared programs participate in the closeness objective.
            trial,row=self.improve(p,True,'joint_exit_closeness_descent')
            return trial if row['passed'] and self.feasible(trial) else p
        trial,row=self.improve(p,True,'joint_exit_opening_descent')
        best=trial if row['passed'] else p
        if self.feasible(best):return best
        for attempt in range(self.config['exit_rewrites']):
            for trial in self.mutations(best,'replace',True):
                trial,row=self.improve(trial,True,'joint_exit_replace')
                if row['passed'] and self.feasible(trial):return trial
                if row['passed'] and self.compiler.loss(trial,self.sample_ids,exits=True)[0]<self.compiler.loss(best,self.sample_ids,exits=True)[0]:best=trial
        return best

    def export(self,p):
        reports=[]
        for s in self.views:
            s.events=[dict(operation='joint_saved_seed_force',covered=self.classify(self.initial)['rows'][self.views.index(s)]['covered'])]+self.events
            report=s.export(p,None,self.input_paths)
            report.update(schema=SCHEMA,shared_physical_head_ids=[f'SH{x.ident:04d}' for x in p.patches],
                shared_program=True,independent_pose_optimization=False,inter_pose_angle_penalty=True,
                search='one joint program; global head deletes/merges plus joint all-pose numerical descent')
            report['provenance']['code'].update(I.hashes(sources()))
            I.save(s.out/'report.json',report)
            reports.append(report)
        canonical=self.compiler.canonical.compile(p) or []
        canonical=[dict(c,candidate_id=f'SH{x.ident:04d}') for x,c in zip(p.patches,canonical)]
        I.save_contacts(self.out/'shared_heads.npz',canonical)
        place=placement(self.tasks)
        rows=[]
        for c in canonical:
            errors=[]
            for s,b,o in zip(self.views,np.asarray(place['bases']),np.asarray(place['offsets'])):
                viewed=next(v for v in s.compiler.compile(p) if v['candidate_id']==c['candidate_id'])
                errors.append(float(np.max(np.abs(viewed['triangles_m']@b+o-c['triangles_m']))))
            if max(errors)>2e-12:raise RuntimeError('Head views do not share one physical contact geometry')
            rows.append(dict(id=c['candidate_id'],active_poses=[t.pose for t in self.tasks],
                             fixture_frame_contact_max_error_m=max(errors)))
        sharing=dict(complete=True,physical_head_count=len(canonical),head_pose_use_count=len(canonical)*len(self.tasks),
            shared_head_count=len(canonical) if len(self.tasks)>1 else 0,
            saved_independent_seed_head_count=self.legacy_seed_count,
            definition='same installed contact triangles AND same finite solid, not equal IDs or shared rods',
            placement=place,heads=rows,program=asdict(p),
            provenance=dict(inputs=I.hashes(self.input_paths),code=I.hashes(sources())),
            artifacts={'shared_heads.npz':I.sha256(self.out/'shared_heads.npz')})
        I.save(self.out/'sharing.json',sharing)
        draw_sharing(self.tasks,canonical,self.out/'shared_heads.png')
        return dict(complete=True,schema=SCHEMA,group=self.group.name,poses=[t.pose for t in self.tasks],passed=False,
            local_contact_passed=all(r['local_contact_passed'] for r in reports),force_passed=all(r['force_passed'] for r in reports),
            exit_passed=all(r['exit_passed'] for r in reports),total_heads=len(canonical),physical_head_count=len(canonical),
            seed_total_heads=self.legacy_seed_count,seed_unique_heads=len(self.initial.patches),
            head_counts=[r['head_count'] for r in reports],total_head_pose_uses=len(canonical)*len(self.tasks),
            shared_head_count=sharing['shared_head_count'],exit_option_counts=[len(r['exit_options']) for r in reports],covered_counts=[r['covered_count'] for r in reports],
            placement=place,shared_head_depth_m=self.depth,inter_pose_angle_penalty=True,
            objective=['minimize distinct physical heads via global deletes/merges','soft exit direction closeness','minimize actual exit sweep union via joint path selection'],
            minimum_head_count_proved=False,placement_family_is_restricted=True,
            arbitrary_regrasp_registration_searched=False,complete_fixture_verified=False,config=self.config,
            per_pose_reports=[str((s.out/'report.json').relative_to(I.ROOT)) for s in self.views],
            provenance=dict(inputs=I.hashes(self.input_paths+[self.out/'sharing.json']),code=I.hashes(sources())))


def draw_sharing(tasks,contacts,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(11,5))
    ax=fig.add_subplot(121,projection='3d')
    mesh=tasks[0].domain.mesh
    ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolor='#cccccc',alpha=.18,edgecolor='none'))
    for i,c in enumerate(contacts):
        color=plt.get_cmap('tab10')(i%10)
        ax.add_collection3d(Poly3DCollection(c['triangles_m'],facecolor=color,edgecolor='none'))
        ax.text(*c['center_m'],c['candidate_id'],fontsize=9,color=color)
    lo,hi=mesh.bounds;center=(lo+hi)/2;span=mesh.extents.max()*.6
    ax.set(xlim=(center[0]-span,center[0]+span),ylim=(center[1]-span,center[1]+span),zlim=(center[2]-span,center[2]+span))
    ax.set_box_aspect((1,1,1));ax.set_axis_off()
    ax.set_title(f'{len(contacts)} physical heads, compiled once')
    matrix=fig.add_subplot(122)
    if contacts:
        matrix.imshow(np.ones((len(tasks),len(contacts))),vmin=0,vmax=1,cmap='Greens',aspect='auto')
        matrix.set_xticks(range(len(contacts)),[c['candidate_id'] for c in contacts],rotation=45)
        matrix.set_yticks(range(len(tasks)),[t.pose for t in tasks])
        for k in range(len(tasks)):
            for j in range(len(contacts)):matrix.text(j,k,'same head',ha='center',va='center',color='white',fontsize=8)
    matrix.set_title('Available physical heads in each pose')
    fig.suptitle('One shared fixture; head identities verified by installed geometry')
    fig.tight_layout();fig.savefig(output,dpi=150);plt.close(fig)


def sources():return [Path(__file__)]+DSL.sources()
