"""Feasible-incumbent DSL: pose-specific heads/paths, optional real head sharing.

The seed geometry is not recompiled or forced into one object-attached layout.
Only fully feasible, objective-improving proposals replace the incumbent.
"""
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,time,traceback
import numpy as np
import trimesh
from step3_scheculer import contacts as I,contact_dsl as D,exit_options as E
from step3_scheculer.run_dsl import saved_task,seed_contacts
from step3_scheculer.pair_scoring import J
from step4_connect_support import boxed_support as F,build_coupled_saddle as S,deterministic_space as SPACE
from step4_connect_support import process_access as ACCESS,select_exit_paths as PATHS,space_budget as BOX
from step4_connect_support.baseline_current.envelope_growth import EnvelopeGrow as CoverageGrow
from step4_connect_support.fixture_view import cells_for
from step0_pose_selection.floor_points import pressure_centers
from step2_local_support import geometry as G,circles as P,surface as SURF

STAGE='dsl_operations';BODY_STAGE='dsl_operations_support';SCHEMA='feasible_operations_dsl_v6'
ROOT_DEPTH=.0008

@dataclass(frozen=True)
class State:
    groups: tuple
    bases: np.ndarray
    offsets: np.ndarray
    paths: tuple

    @property
    def physical_count(self):return len({c['candidate_id'] for row in self.groups for c in row})
    @property
    def shared_count(self):
        ids=[{c['candidate_id'] for c in row} for row in self.groups]
        return sum(sum(ident in row for row in ids)>1 for ident in set.union(*ids))


def ray(direction,ident=-1):
    d=np.asarray(direction,float);d=d/np.linalg.norm(d)
    return dict(id=int(ident),kind='ray',support_withdrawal_world=d.tolist(),
        object_translation_waypoints_world_m=[[0.,0.,0.],(-.5*d).tolist()],
        initial_object_exit_world=(-d).tolist(),terminal_object_exit_world=(-d).tolist(),
        path_length_m=.5,rotation_allowed=False)


def angle_loss(tasks,state):
    vectors=[np.asarray(p['initial_object_exit_world'])@np.asarray(t.domain.data['frame']['T_world_mesh'])[:3,:3]
             for p,t in zip(state.paths,tasks)]
    return D.alignment(vectors)


def objective(tasks,state):return (state.physical_count,angle_loss(tasks,state))


def improves(before,after,tol=1e-8):
    return after[0]<=before[0] and after[1]<=before[1]+tol and (after[0]<before[0] or after[1]<before[1]-tol)


def with_group(state,k,row):
    groups=list(state.groups);groups[k]=tuple(row)
    return replace(state,groups=tuple(groups))


def roots(tasks,state):
    owned={};seeds=[];errors=[]
    for k,(task,row) in enumerate(zip(tasks,state.groups)):
        unit_offsets,valid=G.vertex_offsets(task.domain.mesh,1.)
        out=[]
        for c in row:
            ident=c['candidate_id'];b,o=state.bases[k],state.offsets[k]
            if ident not in owned:
                patch=c["triangles_m"].reshape(-1,3)@b+o
                clearances=[float(((patch-oj)@bj.T)[:,2].min()) for bj,oj in zip(state.bases,state.offsets)]
                vertices=np.unique(task.domain.mesh.faces[np.unique(c["source_faces"])])
                if min(clearances)<=1e-10:raise ValueError("Contact has no all-floor clearance")
                ratio=float(np.linalg.norm(unit_offsets[vertices],axis=1).max())
                depth=min(ROOT_DEPTH,.5*min(clearances)/ratio)
                cells=cells_for(c,task.domain,depth*unit_offsets)
                owned[ident]=(k,[v@b+o for v in cells],c['triangles_m']@b+o)
            owner,installed,patch=owned[ident]
            actual=c['triangles_m']@b+o
            from step4_connect_support.head_registration import cloud_error
            error=cloud_error(patch.reshape(-1,3),actual.reshape(-1,3))
            if error>1e-8:raise ValueError('Shared ID has different installed contact geometry')
            errors.append(error)
            # Reuse the very same physical owner solid; no per-pose re-extrusion.
            out.append([(v-o)@b.T for v in installed])
        seeds.append(out)
    return seeds,owned,max(errors,default=0.)


class Grow(CoverageGrow):
    def __init__(self,group,tasks,state,out,reference_path,reference_report):
        self.group,self.out=group,Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.bases,self.offsets=state.bases,state.offsets
        self.directions=np.asarray([p['support_withdrawal_world'] for p in state.paths])
        seeds,owned,error=roots(tasks,state);self.sharing_error=error
        self.case=SimpleNamespace(name='B',pair=group,output=self.out,poses=[t.pose for t in tasks],tasks=tasks,
            groups=state.groups,heads=[[list(c['triangles_m']) for c in row] for row in state.groups],
            support_seeds=seeds,demands=[pressure_centers(t.targets/t.scale,t.domain.com)[0] for t in tasks],
            schedule=dict(passed=True,covered_counts=[32768]*len(tasks)),paths=[p for t in tasks for p in t.inputs],
            exit_paths=list(state.paths),preview_directions=self.directions)
        self.case.source_ground_centers={r["pose"]:r.get("centers_xy_m",[]) for r in reference_report.get("construction",{}).get("feet",[])}
        self.case.root_solids=[F.union([S.solid(G.hull_mesh(v)) for cells in row for v in cells]) for row in seeds]
        self.reference_report=dict(reference_report,placement=dict(bases=self.bases.tolist(),offsets=self.offsets.tolist(),directions=self.directions.tolist()))
        self.foot_reference=trimesh.load(reference_path,force='mesh',process=False)
        self.reference_report['space_budget']=SPACE.measure(self.case,self.foot_reference,self.bases,self.offsets)
        self.source=self.out/'comparison_input';self.source.mkdir(exist_ok=True)
        SPACE.export_exact_obj(self.foot_reference,self.source/'shape.obj');I.save(self.source/'report.json',self.reference_report)
        self.guard=ACCESS.Guard(self.case,self.reference_report['placement'])
        sphere=trimesh.creation.icosphere(subdivisions=2,radius=.0026);self.bead=sphere.vertices
        self.guaranteed_radius=float(np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]))))
        self.terminals=[];self.feet=[];self.paths=[];self.original_roots=[];self.original_soles=[]
        self.thickness=[];self.core_solids=[];self.seed_positions=[];self.segments=[]
        self.grid_ready=False;self.journal=[];self.partial_steps=0
        installed_roots=[F.transform(root,b,o) for root,b,o in zip(self.case.root_solids,self.bases,self.offsets)]
        cloud=[t.domain.mesh.vertices for t in tasks]+[np.c_[v,np.zeros(len(v))] for v in self.case.demands]
        material=np.vstack([S.unpack(v).vertices for v in installed_roots])
        cloud += [(material-o)@b.T for b,o in zip(self.bases,self.offsets)]
        cloud += [np.c_[np.asarray(points),np.zeros(len(points))] for points in self.case.source_ground_centers.values() if len(points)]
        self.navigation_window=SPACE.enlarged(BOX.box(cloud),.05)
        options=[dict(exit_options=[dict(plan=p)]) for p in state.paths]
        states,detail=PATHS.select(tasks,options,installed_roots,self.bases,self.offsets,self.navigation_window,beam_width=1)
        I.save(self.out/'exit_selection.json',detail)
        if not states:raise ValueError(detail['error'])
        self.selected_state=states[0]
        selected=states[0]['selected'];self.sweep_meshes=[v['raw'] for v in selected]
        self.search=SimpleNamespace(sweep_meshes=self.sweep_meshes,sweep_conditioning=[],precompute=lambda:None)
        self.forbidden=F.union([v['padded'] for v in selected])
        self.mask=(F.bounded_space(self.navigation_window,self.bases,self.offsets)-self.forbidden)+F.union(installed_roots)
        self.reference=S.unpack(self.mask);self.occupied_lo=None;self.occupied_hi=None
        self.contact_vertices=np.unique(np.vstack([c["triangles_m"].reshape(-1,3)@b+o for row,b,o in zip(state.groups,state.bases,state.offsets) for c in row]),axis=0)
        self.buried_contact_rejections=0;self.timings={}

    def roots(self):
        started=time.monotonic()
        seen=set()
        for k,(row,contacts,b,o) in enumerate(zip(self.case.support_seeds,self.case.groups,self.bases,self.offsets)):
            for cells,c in zip(row,contacts):
                if c['candidate_id'] in seen:continue
                seen.add(c['candidate_id']);points=[v@b+o for v in cells]
                self.attach(F.union([S.solid(G.hull_mesh(v)) for v in points]),np.vstack(points),self.case.poses[k]+':'+c['candidate_id'])
        self.timings['contact_starts']=time.monotonic()-started
        self.save_stage('roots',F.union(self.original_roots))

    def verify(self,mesh):
        # Baseline's constructor is retained. DSL accepts object motion paths;
        # its existing path-aware geometry acceptance replaces the historical
        # support-moving straight-ray floor assumption, once per construction.
        from step4_connect_support.growing_support import Grow as PathAwareGrow
        previous=self.reference_report
        self.reference_report=dict(previous,space_budget=SPACE.measure(self.case,mesh,self.bases,self.offsets))
        started=time.monotonic()
        try:return PathAwareGrow.verify(self,mesh)
        finally:
            self.timings.setdefault('validation_passes',[]).append(time.monotonic()-started)
            self.reference_report=previous

    def connect(self):
        full,construction=super().connect()
        cores=self.core_solids+[(f'beam_{j}',p) for j,p in enumerate(self.beams)]+[(f'sole_{j}',p) for j,p in enumerate(self.original_soles)]
        checks=[dict(name=n,missing_volume_m3=abs(float((p-full).volume()))*S.SCALE**3) for n,p in cores]
        if not all(c['missing_volume_m3']<=8e-14 for c in checks):raise RuntimeError('Complete 5 mm core missing')
        construction['minimum_branch_thickness']=dict(minimum_required_diameter_mm=5.,guaranteed_inscribed_beam_diameter_mm=self.guaranteed_radius*2000,all_complete_cores_preserved=True,constructed_unclipped_core_checks=checks,local_branches=self.thickness)
        return full,construction


class Checker:
    def __init__(self,tasks):
        self.tasks=tasks;self.force_cache={};self.path_cache={}
        self.analyzers=[E.PathAnalyzer(t.domain.mesh,ROOT_DEPTH) for t in tasks]

    def check(self,state):
        rows=[]
        for k,(task,row,path,a) in enumerate(zip(self.tasks,state.groups,state.paths,self.analyzers)):
            key=(k,hashlib.sha256(b''.join(c['triangles_m'].tobytes()+c['source_faces'].tobytes() for c in row)).hexdigest())
            if key not in self.force_cache:
                mask,info=J.classify(task.supply(row),task.targets);self.force_cache[key]=(mask,info)
            mask,info=self.force_cache[key]
            pkey=(key,json.dumps(path,sort_keys=True))
            if pkey not in self.path_cache:self.path_cache[pkey]=a.test(a.heads(row),path)
            pathcheck=self.path_cache[pkey]
            work=not any(np.intersect1d(c['source_faces'],task.domain.work_ids).size for c in row)
            floor=all(c['triangles_m'][:,:,2].min()>=.002-1e-10 for c in row)
            rows.append(dict(pose=task.pose,covered=int(mask.sum()),force_passed=bool(mask.all()),
                             exit_passed=pathcheck['clear'],work_passed=work,floor_passed=floor))
        return dict(passed=all(r['force_passed'] and r['exit_passed'] and r['work_passed'] and r['floor_passed'] for r in rows),per_pose=rows)


def sources():
    from step4_connect_support import feasible_coverage_growth,direct_head_growth,growing_support,run_thick_batch,fixture_view
    from step3_scheculer import feasible_seating,shared_dsl,run_dsl,pair_scoring,passive_support,floor_support
    return list(dict.fromkeys([Path(__file__)]+list((Path(__file__).resolve().parent.parent/'step4_connect_support/baseline_current').rglob('*.py'))+D.sources()+PATHS.sources()+ACCESS.sources()+[Path(m.__file__) for m in (feasible_seating,shared_dsl,run_dsl,pair_scoring,passive_support,floor_support,J,feasible_coverage_growth,direct_head_growth,growing_support,run_thick_batch,fixture_view,F,S,SPACE,BOX,E,I)]))


def restrict_compiler(compiler,direction):
    """Connected real patches on faces that preserve a chosen escape opening."""
    d=np.asarray(direction)
    compiler.polygons={f:p for f,p in compiler.polygons.items() if compiler.mesh.face_normals[f]@d>=-1e-10}
    if not compiler.polygons:raise ValueError('No admissible faces for this exit')
    compiler.surface=P.SurfaceCircles(compiler.mesh,compiler.polygons)
    compiler.triangles=np.concatenate([SURF.fan(p) for p in compiler.polygons.values()])
    compiler.triangle_faces=np.concatenate([np.full(len(p),f,int) for f,p in compiler.polygons.items()])
    from scipy.spatial import cKDTree
    compiler.tree=cKDTree(compiler.triangles.mean(axis=1));compiler.cache.clear()
    return compiler


def repair_initial_pose(task,device='cuda',count=96):
    """Find a force-complete pose WITH its own path before starting optimization."""
    base=D.Compiler(task,[task],device=device);seeds=base.seeds(count)
    analyzer=E.PathAnalyzer(task.domain.mesh,ROOT_DEPTH)
    trials=[]
    directions=[np.array([0,0,-1.])]
    for elevation in (45,60,30,15):
        for azimuth in range(0,360,45):
            a=np.deg2rad(azimuth);e=np.deg2rad(elevation)
            directions.append(-np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)]))
    for number,d in enumerate(directions):
        plan=ray(d,100000+number)
        try:c=restrict_compiler(D.Compiler(task,[task],device=device),d)
        except ValueError:continue
        pool=[]
        for seed in seeds:
            for factor in (1.,.5,.25):
                head=c.compile_patch(replace(seed,radius=seed.radius*factor))
                if head is not None and analyzer.test(analyzer.heads([head]),plan)['clear']:
                    if not any(np.linalg.norm(head['center_m']-v['center_m'])<c.scale*1e-6 for v in pool):pool.append(head)
                    break
        selected=[];ids=np.linspace(0,32767,48,dtype=int);covered=0
        for _ in range(min(16,len(pool))):
            residuals=c.backend.solve([D.reduced_rays(task.supply(selected+[v])) for v in pool],task.targets[ids])
            j=int(np.argmin(residuals.mean(axis=1)+residuals.max(axis=1)));selected.append(pool.pop(j))
            mask,_=J.classify(task.supply(selected),task.targets);covered=int(mask.sum())
            print('INIT REPAIR',task.pose,'exit',number,'heads',len(selected),'covered',covered,flush=True)
            if mask.all() and analyzer.test(analyzer.heads(selected),plan)['clear']:
                return selected,plan,dict(repaired=True,trials=trials,chosen_exit=number,covered=covered)
            failed=np.flatnonzero(~mask)
            ids=np.unique(np.r_[np.linspace(0,32767,24,dtype=int),failed[np.linspace(0,len(failed)-1,min(72,len(failed)),dtype=int)]])
        trials.append(dict(exit=number,covered=covered,heads=len(selected)))
    raise RuntimeError('Could not initialize a force-complete pose with a certified exit: '+task.pose)


def save_state(folder,tasks,state,check):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ids={}
    for k,(task,row) in enumerate(zip(tasks,state.groups)):
        I.save_contacts(folder/f'contacts_{task.pose}.npz',row)
        for c in row:ids.setdefault(c['candidate_id'],[]).append(task.pose)
    data=dict(complete=True,schema=SCHEMA,poses=[t.pose for t in tasks],physical_head_count=state.physical_count,
        shared_head_count=state.shared_count,active_head_ids=[[c['candidate_id'] for c in row] for row in state.groups],
        physical_heads=[dict(id=ident,active_poses=active) for ident,active in ids.items()],
        placement=dict(bases=state.bases.tolist(),offsets=state.offsets.tolist()),exit_paths=list(state.paths),
        force_and_local_path_checks=check,objective=list(objective(tasks,state)),
        all_paths_equal_required=False,head_sharing_forced=False,
        artifacts={f'contacts_{t.pose}.npz':I.sha256(folder/f'contacts_{t.pose}.npz') for t in tasks})
    I.save(folder/'state.json',data)
    return data


def baseline(group,tasks):
    source=group/'step4/data/growing_support'
    if not (source/'report.json').exists():source=group/'step4/data'
    report=json.loads((source/'report.json').read_text())
    shape=source/'shape.obj'
    if not shape.exists():shape=group/'step4/shape.obj'
    b=np.asarray(report['placement']['bases']);o=np.asarray(report['placement']['offsets'])
    groups=[];inputs=[]
    for task in tasks:
        contacts,p=seed_contacts(group,task.pose)
        if not contacts:raise ValueError('No exact saved seed contacts: '+task.pose)
        inputs.append(p)
        groups.append(tuple(dict(c,candidate_id=task.pose+'_'+c['candidate_id']) for c in contacts))
    state=State(tuple(groups),b,o,tuple(ray(d,-1000-k) for k,d in enumerate(report['placement']['directions'])))
    return state,source,shape,report,inputs


class BaseOptimizer:
    def __init__(self,group,tasks,config):
        self.group,self.tasks,self.config=group,tasks,config
        self.out=group/'step3_scheculer'/STAGE;self.out.mkdir(parents=True,exist_ok=True)
        self.checker=Checker(tasks);self.events=[];self.trial=0
        self.seed,self.source,self.reference,self.reference_report,self.seed_inputs=baseline(group,tasks)
        self.incumbent=None;self.witness=None;self.check=None
        self.compilers=[D.Compiler(t,[t],device=config['device']) for t in tasks]

    def event(self,operation,**fields):
        self.events.append(dict(operation=operation,**fields))
        I.save(self.out/'progress.json',dict(complete=False,events=self.events))
        print('FEASIBLE',self.group.name,operation,fields.get('physical_heads',''),flush=True)

    def construct(self,state,check,kind,allow_reference=False):
        folder=self.out/'trials'/f'{self.trial:04d}_{kind}';self.trial+=1
        save_state(folder,self.tasks,state,check)
        try:
            grow=Grow(self.group,self.tasks,state,folder,self.reference,self.reference_report)
            grow.case.paths += [folder/'state.json']+[folder/f'contacts_{t.pose}.npz' for t in self.tasks]
            if allow_reference:
                mesh=trimesh.load(self.reference,force='mesh',process=False)
                geometry,cert=grow.verify(mesh)
                SPACE.export_exact_obj(mesh,folder/'shape.obj');np.savez_compressed(folder/'geometry_certificate.npz',**cert)
                report=dict(complete=True,passed=True,constructed=True,seed_fixture_reused=True,
                    validation_policy='one in-memory seed geometry acceptance; no exported-model replay',result=geometry,
                    volume_cm3=float(mesh.volume*1e6),space_budget=SPACE.measure(grow.case,mesh,state.bases,state.offsets),
                    source_fixture=str(self.reference.relative_to(I.ROOT)))
            else:
                report=grow.run(pitch=.008)
            report.update(schema=SCHEMA,physical_head_count=state.physical_count,shared_head_count=state.shared_count,
                installed_shared_geometry_error_m=grow.sharing_error,selected_exit_paths=list(state.paths),
                placement=dict(bases=state.bases.tolist(),offsets=state.offsets.tolist()))
            report['provenance']=dict(inputs=I.hashes([folder/'state.json',self.reference,self.source/'report.json']+grow.case.paths),code={**report.get('provenance',{}).get('code',{}),**I.hashes(sources())})
            report['artifacts']={p:I.sha256(folder/p) for p in ('shape.obj','geometry_certificate.npz')}
            I.save(folder/'report.json',report)
            return folder,report,grow
        except Exception as error:
            I.save(folder/'report.json',dict(complete=True,passed=False,error=str(error),checks=getattr(error,'checks',None),traceback=traceback.format_exc()))
            self.event('reject_construction',reason=str(error),trial=str(folder.relative_to(I.ROOT)))
            return None

    def initialize(self):
        state=self.seed;check=self.checker.check(state)
        save_state(self.out/'original_seed',self.tasks,state,check)
        repairs=[]
        for k,row in enumerate(check['per_pose']):
            if row['force_passed'] and row['exit_passed']:continue
            heads,path,info=repair_initial_pose(self.tasks[k],self.config['device'],self.config['init_seeds'])
            heads=tuple(dict(c,candidate_id=self.tasks[k].pose+'_INIT'+str(i)) for i,c in enumerate(heads))
            state=with_group(state,k,heads);paths=list(state.paths);paths[k]=path;state=replace(state,paths=tuple(paths))
            repairs.append(dict(pose=self.tasks[k].pose,before=row,repair=info))
        check=self.checker.check(state)
        if not check['passed']:raise RuntimeError('Initialization is not fully feasible')
        witness=self.construct(state,check,'initial',allow_reference=not repairs)
        if witness is None and not repairs:witness=self.construct(state,check,'initial_regrowth')
        if witness is None:
            from step3_scheculer.feasible_seating import restore,saved_ground_parts
            placement=self.reference_report['placement']
            ground=saved_ground_parts(self.reference,np.asarray(placement['bases']),np.asarray(placement['offsets']))
            for attempt in range(8):
                repaired=restore(self.tasks,state,roots,attempt=attempt,ground_parts=ground)
                if repaired is None:continue
                seated,cuts=repaired
                witness=self.construct(seated,check,'initial_collision_seating')
                self.event('initial_seating_proposal',attempt=attempt,separation_cuts=cuts,passed=witness is not None)
                if witness is not None:state=seated;break
        if witness is None:
            # Local inputs remain unchanged. Only search the seating translations
            # needed to keep inactive heads and ground regions mutually legal.
            for extra in (.01,.03,.06,.12):
                seated=seat(self.tasks,state,extra)
                if seated is None:continue
                witness=self.construct(seated,check,'initial_seating')
                if witness is not None:state=seated;break
        if witness is None:raise RuntimeError('No complete feasible initialization yet')
        self.incumbent,self.witness,self.check=state,witness,check
        self.initial_state=state
        I.save(self.out/'initialization.json',dict(complete=True,passed=True,repairs=repairs,
            original_seed_force_complete=all(r['force_passed'] for r in self.checker.check(self.seed)['per_pose']),
            initialized_physical_heads=state.physical_count,initialized_objective=list(objective(self.tasks,state))))
        save_state(self.out/'initial',self.tasks,state,check)
        self.event('initialize_feasible',physical_heads=state.physical_count,objective=list(objective(self.tasks,state)),witness=str((witness[0]/'report.json').relative_to(I.ROOT)))

    def accept(self,state,kind,external=None):
        before=objective(self.tasks,self.incumbent);after=objective(self.tasks,state)
        if not improves(before,after):return False
        check=self.checker.check(state)
        if not check['passed']:
            self.event('reject_local_keep_incumbent',proposal=kind,checks=check)
            return False
        witness=self.construct(state,check,kind) if external is None else external
        if witness is None:return False
        self.incumbent,self.witness,self.check=state,witness,check
        self.event('accept_feasible_improvement',proposal=kind,before=list(before),after=list(after),physical_heads=state.physical_count,checks=check,witness=str((witness[0]/"report.json").relative_to(I.ROOT)))
        return True

    def delete(self):
        changed=True
        while changed:
            changed=False
            for k,row in enumerate(self.incumbent.groups):
                if len(row)<2:continue
                for j in range(len(row)):
                    state=with_group(self.incumbent,k,row[:j]+row[j+1:])
                    if self.accept(state,'delete_head'):
                        changed=True;break
                if changed:break

    def align_paths(self):
        for k,(task,path) in enumerate(zip(self.tasks,self.incumbent.paths)):
            rotation=np.asarray(task.domain.data['frame']['T_world_mesh'])[:3,:3]
            others=[np.asarray(p['initial_object_exit_world'])@np.asarray(t.domain.data['frame']['T_world_mesh'])[:3,:3]
                    for j,(p,t) in enumerate(zip(self.incumbent.paths,self.tasks)) if j!=k]
            if not others:continue
            u=np.asarray(path['initial_object_exit_world'])
            desired=np.mean(others,axis=0)@rotation.T
            if np.linalg.norm(desired)<1e-10:continue
            desired/=np.linalg.norm(desired)
            for alpha in (.25,.1,.04):
                candidate=(1-alpha)*u+alpha*desired
                if candidate[2]<-1e-10:continue
                paths=list(self.incumbent.paths);paths[k]=ray(-candidate,200000+k)
                if self.accept(replace(self.incumbent,paths=tuple(paths)),'align_exit'):
                    break

    def parameter_descent(self):
        # GPU gradients propose changes. A line search commits only fully
        # feasible steps, including the complete support witness.
        for k,(task,c) in enumerate(zip(self.tasks,self.compilers)):
            row=self.incumbent.groups[k]
            program=D.Program(tuple(D.Patch(i+1,tuple(v['center_m']),float(v['radius_m'])) for i,v in enumerate(row)))
            # Preserve exact saved primitives at the initial parameter values.
            for p,v in zip(program.patches,row):c.cache[(p.ident,*p.center,p.radius)]=v
            sample_ids=np.linspace(0,32767,self.config['samples'],dtype=int)
            trial,history=D.descend(c,program,sample_ids,exits=False,steps=self.config['steps'])
            heads=c.compile(trial)
            if heads is None:continue
            heads=[dict(v,candidate_id=old['candidate_id']) for v,old in zip(heads,row)]
            state=with_group(self.incumbent,k,heads)
            # This geometry step is accepted only when it enables a verified
            # delete or improves a feasible exit. It never replaces a feasible
            # state solely because a surrogate force loss got smaller.
            for j in range(len(heads)):
                candidate=with_group(state,k,heads[:j]+heads[j+1:])
                if len(heads)>1 and self.accept(candidate,'descent_then_delete'):break
            self.event('gradient_proposal_checked',pose=task.pose,gradient_steps=history)

    def optimize(self):
        self.initialize()
        for operation in (self.delete,self.parameter_descent,self.delete,self.align_paths,self.shared_proposals):
            try:operation()
            except (ValueError,RuntimeError,AssertionError,KeyError) as error:
                self.event("reject_search_error_keep_incumbent",proposal=operation.__name__,reason=str(error),traceback=traceback.format_exc())
        self.publish()

    def shared_proposals(self):
        # Reuse a previously certified shared candidate as a warm proposal.
        path=self.group/'step3_scheculer/dsl_shared/report.json'
        if path.exists():
            try:old=I.check_report(path)
            except (AssertionError,RuntimeError):old={}
            if old.get('passed'):
                place=old['placement']
                groups=tuple(tuple(I.read_contacts(self.group/'step3_scheculer/dsl_shared'/t.pose/f'final_contacts_{t.pose}.npz')) for t in self.tasks)
                state=State(groups,np.asarray(place['bases']),np.asarray(place['offsets']),tuple(old['selected_exit_paths']))
                self.accept(state,'warm_shared_registration')
        # Search PARTIAL sharing, two poses at a time. All other poses keep
        # their own heads, paths and seating. Failed registrations are discarded.
        from itertools import combinations,product
        from step3_scheculer.shared_dsl import JointSearch
        from step3_scheculer.run_dsl import Search
        from step4_connect_support.head_registration import fixed_placements,floor_compatibility
        for i,j in combinations(range(len(self.tasks)),2):
            pair=[self.tasks[i],self.tasks[j]]
            b,o=fixed_placements(pair)
            demands=[pressure_centers(t.targets/t.scale,t.domain.com)[0] for t in pair]
            try:floor=floor_compatibility(pair,demands,b,o)
            except ValueError:continue
            if not floor['passed']:
                self.event('reject_shared_registration_floor_bound',pair=[t.pose for t in pair]);continue
            scratch=self.out/'proposals'/f'pair_{i}_{j}'
            cfg=dict(device=self.config['device'],gpu_iterations=600,steps=self.config['steps'],samples=self.config['samples'],
                     seeds=24,max_heads=16,rewrite_trials=1,force_rewrites=1,exit_rewrites=1,exit_witnesses=6)
            originals={self.tasks[k].pose:self.incumbent.groups[k] for k in (i,j)}
            search=JointSearch(pair,scratch,cfg,Search,lambda _,pose:(originals[pose],None))
            for view in search.views:
                view.depth=ROOT_DEPTH;view.analyzer=E.PathAnalyzer(view.task.domain.mesh,ROOT_DEPTH)
            p=search.force_stage()
            if not search.classify(p)['passed']:continue
            p=search.prune(p,require_exits=False);p=search.repair(p)
            if not search.feasible(p):
                self.event('reject_shared_registration_keep_incumbent',pair=[t.pose for t in pair]);continue
            p=search.prune(p,require_exits=True)
            groups=list(self.incumbent.groups)
            prefix=f'PAIR{i}_{j}_'
            for k,view in zip((i,j),search.views):
                groups[k]=tuple(dict(c,candidate_id=prefix+c['candidate_id']) for c in view.compiler.compile(p))
            bases=self.incumbent.bases.copy();offsets=self.incumbent.offsets.copy()
            # Cluster gauge is the first member's current seating, not the
            # first pose of the whole group. Other seating is untouched.
            transform=search.compiler.transforms[1]
            bases[j]=transform[:3,:3]@bases[i]
            offsets[j]=offsets[i]-transform[:3,3]@bases[j]
            choices=[]
            for view in search.views:
                ids,_=view.exits(p)
                choices.append([view.catalogue['options'][ident] for ident in ids])
            proposals=[]
            for selected in product(*choices):
                paths=list(self.incumbent.paths)
                for k,plan in zip((i,j),selected):paths[k]=plan
                proposal=State(tuple(groups),bases,offsets,tuple(paths))
                if improves(objective(self.tasks,self.incumbent),objective(self.tasks,proposal)):
                    proposals.append(proposal)
            proposals.sort(key=lambda state:objective(self.tasks,state))
            for proposal in proposals[:2]:
                if self.accept(proposal,'partial_shared_registration'):break

    def publish(self):
        import shutil
        state=self.incumbent;folder,body,grow=self.witness
        final=self.out/'final';save_state(final,self.tasks,state,self.check)
        for name in ('shape.obj','geometry_certificate.npz'):shutil.copy2(folder/name,final/name)
        output=self.group/'step4/data'/BODY_STAGE;output.mkdir(parents=True,exist_ok=True)
        for name in ('shape.obj','geometry_certificate.npz'):shutil.copy2(final/name,output/name)
        assert I.sha256(final/'shape.obj')==I.sha256(output/'shape.obj')
        F.draw(output/'overview.png',grow.case,trimesh.load(output/'shape.obj',force='mesh',process=False),state.bases,state.offsets)
        report=dict(complete=True,schema=SCHEMA,group=self.group.name,passed=True,constructed=True,
            original_seed_physical_heads=self.seed.physical_count,initial_physical_heads=self.initial_state.physical_count,physical_head_count=state.physical_count,shared_head_count=state.shared_count,
            head_area_fractions=[[float(c['triangle_areas_m2'].sum()/t.domain.mesh.area) for c in row] for t,row in zip(self.tasks,state.groups)],maximum_head_area_fraction=.02,
            original_seed_coverage=[r['covered'] for r in self.checker.check(self.seed)['per_pose']],
            covered_counts=[r['covered'] for r in self.check['per_pose']],
            initial_angle_loss=angle_loss(self.tasks,self.initial_state),final_angle_loss=angle_loss(self.tasks,state),
            feasible_incumbent_retained=True,all_accepted_states_feasible=True,selected_exit_paths=list(state.paths),
            placement=dict(bases=state.bases.tolist(),offsets=state.offsets.tolist()),
            witness=str((folder/'report.json').relative_to(I.ROOT)),events=self.events,
            volume_cm3=body['volume_cm3'],space_budget=body['space_budget'],
            provenance=dict(inputs=I.hashes([folder/'report.json',final/'state.json']+self.seed_inputs),code=I.hashes(sources())),
            artifacts={p:I.sha256(final/p) for p in ('shape.obj','geometry_certificate.npz')})
        # Artifacts are relative to their report directory.
        I.save(final/'report.json',report)
        top=dict(report,artifacts={},provenance=dict(inputs=I.hashes([final/'report.json']),code=I.hashes(sources())))
        I.save(self.out/'report.json',top)
        published=dict(report,published_model_identical_to_step3_witness=True,new_construction_search=True,latest_baseline_envelope_growth=True,
                       provenance=dict(inputs=I.hashes([final/'report.json']),code=I.hashes(sources())))
        I.save(output/'report.json',published)
        I.save(self.out/'progress.json',dict(complete=True,events=self.events))


def seat(tasks,state,margin):
    """Translation-only feasibility restoration; force/local paths are invariant."""
    from scipy.optimize import linprog
    n=len(tasks);matrix=[];rhs=[]
    clouds=[]
    for t,row,b in zip(tasks,state.groups,state.bases):
        demand=pressure_centers(t.targets/t.scale,t.domain.com)[0]
        clouds.append(np.vstack([np.c_[demand,np.zeros(len(demand))],
                                 np.vstack([c['triangles_m'].reshape(-1,3) for c in row])])@b)
    for k,points in enumerate(clouds):
        for j,b in enumerate(state.bases):
            if j==k:continue
            normal=b[2];row=np.zeros(6*n);row[3*k:3*k+3]=-normal;row[3*j:3*j+3]=normal
            clearance=0. if np.dot(state.bases[k][2],normal)>1-1e-10 else margin
            matrix.append(row);rhs.append(float(np.min(points@normal)-clearance))
    for i,x in enumerate(state.offsets.ravel()):
        for sign in (-1,1):
            row=np.zeros(6*n);row[i]=sign;row[3*n+i]=-1
            matrix.append(row);rhs.append(sign*x)
    result=linprog(np.r_[np.zeros(3*n),np.ones(3*n)],A_ub=np.asarray(matrix),b_ub=rhs,
                   bounds=[(None,None)]*(3*n)+[(0,None)]*(3*n),method='highs')
    if not result.success:return None
    return replace(state,offsets=result.x[:3*n].reshape(n,3))

class Optimizer(BaseOptimizer):
    """Atomic area/exchange/merge edits: exact feasible incumbent throughout."""
    def __init__(self,group,tasks,config):
        self.group,self.tasks,self.config=group,tasks,config
        self.out=group/'step3_scheculer'/STAGE;self.out.mkdir(parents=True,exist_ok=True)
        self.checker=Checker(tasks);self.events=[];self.trial=0;self.serial=0
        previous=group/'step3_scheculer/dsl_feasible/final'
        data=I.check_report(previous/'report.json');record=json.loads((previous/'state.json').read_text())
        placement=record['placement']
        self.seed=State(tuple(tuple(I.read_contacts(previous/f'contacts_{t.pose}.npz')) for t in tasks),np.asarray(placement['bases']),np.asarray(placement['offsets']),tuple(record['exit_paths']))
        self.source=previous;self.reference=previous/'shape.obj';self.reference_report=data
        self.seed_inputs=[previous/'report.json',previous/'state.json']+[previous/f'contacts_{t.pose}.npz' for t in tasks]
        self.incumbent=None;self.witness=None;self.check=None
        self.compilers=[D.Compiler(t,[t],device=config['device']) for t in tasks]

    def initialize(self):
        check=self.checker.check(self.seed)
        if not check['passed']:raise RuntimeError('Previous qualified state failed its original loads/paths')
        witness=self.construct(self.seed,check,'initial_latest_baseline_step4')
        if witness is None:raise RuntimeError('Latest baseline Step4 construction failed; previous public fixture retained')
        self.incumbent=self.initial_state=self.seed;self.witness=witness;self.check=check
        save_state(self.out/'initial',self.tasks,self.seed,check)
        I.save(self.out/'initialization.json',dict(complete=True,passed=True,initialized_physical_heads=self.seed.physical_count,source='qualified V5 final',latest_baseline_step4_rerun=True))
        self.event('initialize_feasible',physical_heads=self.seed.physical_count,objective=list(objective(self.tasks,self.seed)),witness=str((witness[0]/'report.json').relative_to(I.ROOT)))

    def accept(self,state,kind,external=None):
        for task,row in zip(self.tasks,state.groups):
            if any(float(c['triangle_areas_m2'].sum())>task.domain.mesh.area*.020001 for c in row):return False
        try:return super().accept(state,kind,external)
        except (ValueError,RuntimeError,AssertionError,KeyError) as error:
            self.event('reject_proposal_keep_incumbent',proposal=kind,reason=str(error));return False

    def fitted(self,k,center,fraction):
        compiler=self.compilers[k];center,face=compiler.project(center)
        polygons,fit=compiler.surface.fit(center,face,float(fraction)*compiler.mesh.area)
        if not polygons or not fit['wrap_limit_satisfied']:return None
        triangles=np.concatenate([SURF.fan(p) for p in polygons.values()]);areas=SURF.areas(triangles)
        if areas.sum()>compiler.mesh.area*.020001:return None
        self.serial+=1
        return dict(candidate_index=900000+self.serial,candidate_id=f'{self.tasks[k].pose}_OP{self.serial}',center_m=center,center_face=face,radius_m=fit['radius_m'],triangles_m=triangles,source_faces=np.concatenate([np.full(len(p),f,int) for f,p in polygons.items()]),triangle_areas_m2=areas)

    def ranked(self,k,rows):
        """CUDA residual ranking only; exact all-load LP decides acceptance."""
        ids=np.linspace(0,32767,self.config['samples'],dtype=int);task=self.tasks[k]
        losses=self.compilers[k].backend.solve([D.reduced_rays(task.supply(row)) for row in rows],task.targets[ids])
        return sorted(range(len(rows)),key=lambda i:float(losses[i].mean()+losses[i].max()))

    def try_rows(self,k,candidates,kind,limit=12):
        if not candidates:return False
        # Unique physical geometry, bounded expensive exact checks.
        rows=[];seen=set()
        for row in candidates:
            signature=hashlib.sha256(b''.join(c['triangles_m'].tobytes() for c in row)).digest()
            if signature not in seen:seen.add(signature);rows.append(tuple(row))
        order=self.ranked(k,rows)
        for i in order[:limit]:
            if self.accept(with_group(self.incumbent,k,rows[i]),kind):return True
        self.event('operator_checked',pose=self.tasks[k].pose,proposal=kind,generated=len(rows),exact_checked=min(limit,len(rows)),backend=self.config['device'])
        return False

    def operations(self):
        from collections import Counter
        for sweep in range(2):
            changed=False
            for k,task in enumerate(self.tasks):
                row=self.incumbent.groups[k]
                active=Counter(c['candidate_id'] for group in self.incumbent.groups for c in group)
                editable=[i for i,c in enumerate(row) if active[c['candidate_id']]==1]
                if len(row)<2 or not editable:continue
                # Area edits are atomic with deletion: never install a force-only plateau.
                resized=[]
                for fraction in (.0125,.015,.02):
                    enlarged=[self.fitted(k,c['center_m'],fraction) if i in editable else c for i,c in enumerate(row)]
                    if any(c is None for c in enlarged):continue
                    for j in editable:resized.append(enlarged[:j]+enlarged[j+1:])
                if self.try_rows(k,resized,'resize_then_delete'):
                    changed=True;continue
                # Two old heads -> one real connected patch, <=2% true surface area.
                merges=[]
                for ai,i in enumerate(editable):
                    for j in editable[ai+1:]:
                        a,b=row[i],row[j]
                        for center in (a['center_m'],b['center_m'],(np.asarray(a['center_m'])+np.asarray(b['center_m']))*.5):
                            c=self.fitted(k,center,.02)
                            if c is not None:merges.append([v for q,v in enumerate(row) if q not in (i,j)]+[c])
                if self.try_rows(k,merges,'merge_two_heads'):
                    changed=True;continue
                # Exchange one head for a new center/area; simultaneously seek a deletion.
                seeds=self.compilers[k].seeds(24);exchanges=[]
                for patch in seeds:
                    c=self.fitted(k,patch.center,.02)
                    if c is None:continue
                    for ai,i in enumerate(editable):
                        for j in editable[ai+1:]:exchanges.append([v for q,v in enumerate(row) if q not in (i,j)]+[c])
                if self.try_rows(k,exchanges,'exchange_then_delete',limit=16):changed=True
            if not changed:break

    def exit_area_operations(self):
        for k,(task,path) in enumerate(zip(self.tasks,self.incumbent.paths)):
            counts={c['candidate_id']:sum(c['candidate_id']==v['candidate_id'] for group in self.incumbent.groups for v in group) for c in self.incumbent.groups[k]}
            if any(v>1 for v in counts.values()):continue
            rotation=np.asarray(task.domain.data['frame']['T_world_mesh'])[:3,:3]
            others=[np.asarray(p['initial_object_exit_world'])@np.asarray(t.domain.data['frame']['T_world_mesh'])[:3,:3] for j,(p,t) in enumerate(zip(self.incumbent.paths,self.tasks)) if j!=k]
            desired=np.mean(others,axis=0)@rotation.T
            if np.linalg.norm(desired)<1e-10:continue
            desired/=np.linalg.norm(desired);u=np.asarray(path['initial_object_exit_world'])
            old=self.compilers[k]
            for alpha in (.5,.25,.1):
                direction=(1-alpha)*u+alpha*desired
                if direction[2]<-1e-10:continue
                try:
                    self.compilers[k]=restrict_compiler(D.Compiler(task,[task],device=self.config['device']),-direction)
                    heads=[self.fitted(k,c['center_m'],.02) for c in self.incumbent.groups[k]]
                    if any(c is None for c in heads):continue
                    paths=list(self.incumbent.paths);paths[k]=ray(-direction,400000+k)
                    state=replace(with_group(self.incumbent,k,heads),paths=tuple(paths))
                    if self.accept(state,'resize_exchange_align_exit'):break
                finally:self.compilers[k]=old

    def optimize(self):
        self.initialize()
        for operation in (self.operations,self.parameter_descent,self.delete,self.align_paths,self.exit_area_operations):
            try:operation()
            except (ValueError,RuntimeError,AssertionError,KeyError) as error:self.event('reject_search_error_keep_incumbent',proposal=operation.__name__,reason=str(error),traceback=traceback.format_exc())
        self.publish()
