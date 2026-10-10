"""Sample direction changes and floor-tangent pose translations together.

Cheap contact-lock screening never constructs solids. Exact finalists rebuild
translated seeds and complete exits, with current contact-core clearance policy.
"""
import time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from trimesh.ray.ray_pyembree import RayMeshIntersector
from co_common import (HERE,ROOT,state,save,union,material_volume,contact_boundary,
    supply,transform_points,wrap_offsets,provenance,code_sources,np,trimesh,G,S,D,F,U,J)
from exit_clearance import CONTACT_DEPTH_M,_PRISM_FACES
from physics_guided_padded_sweep import ConservativeExitClearance
from physics_guided_geometry import tangent_frames,retract
from physics_guided_cone_iterative import cone_projection
from contact_recovery import preserves_loads
from worst_wrench_descent import farthest_load


def shifted(solid,offset):return solid.translate(np.asarray(offset)/S.SCALE)

def fixture_transform(T,offset):
    result=T.copy();result[:3,3]-=T[:3,:3]@offset
    return result


def colocated_allowed(normals,allowed,owner,directions,offsets):
    coincident=np.linalg.norm(offsets-offsets[owner],axis=1)<1e-10
    blocked=np.max(normals[allowed]@directions[coincident].T,axis=1)>1e-9
    return allowed[~blocked]


def separation_layout(normals,spacing):
    frames=tangent_frames(normals);offsets=np.zeros_like(normals)
    for k in range(1,len(normals)):
        for attempt in range(1000):
            radius=spacing*(1+attempt//24)
            angle=2*np.pi*((attempt%24)/24+k*.38196601125)
            candidate=frames[k]@(radius*np.array([np.cos(angle),np.sin(angle)]))
            if np.min(np.linalg.norm(offsets[:k]-candidate,axis=1))>=spacing:
                offsets[k]=candidate;break
        else:raise RuntimeError('separation layout exhausted')
    return offsets


class PlacementSearch:
    def __init__(self,name,group,out,seed=42):
        import json
        self.name=name;self.group=group;self.out=Path(out)/'data';self.out.mkdir(parents=True,exist_ok=True)
        self.base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']
        self.states=[state(name,p)[:2] for p in group['poses']]
        self.mesh=state(name,group['poses'][0])[2];self.n=len(self.states)
        self.normals=np.array([T[:3,:3].T@np.array([0.,0.,1.]) for task,T in self.states])
        self.frames=tangent_frames(self.normals);self.rng=np.random.default_rng(seed)
        self.initial_path=self.base/'step4/step4.1/data/report.json'
        self.initial=json.loads(self.initial_path.read_text());save(self.out/'initialization_input.json',self.initial)
        self.initial_directions=np.array([r['direction_fixture'] for r in self.initial['state_results']])
        self.length=self.initial['initialization']['full_length_m'];self.extent=float(self.mesh.extents.max())
        self.seed_path=self.base/'step3/step3.3/support_with_rings.obj'
        shell_path=self.base/'step3/step3.2/wrapped_support.obj'
        shell=S.solid(trimesh.load(shell_path,force='mesh',process=False))
        self.seed_parts=[];self.inputs=[self.initial_path,shell_path]
        for pose in group['poses']:
            ring_path=self.base/'step3/step3.3/data'/f'{pose}_perimeter.obj'
            self.seed_parts.append(union([shell,S.solid(trimesh.load(ring_path,force='mesh',process=False))]))
            self.inputs.append(ring_path)
        self.contact_path=self.base/'step3/step3.2/data/contacts.npz';self.inputs.append(self.contact_path)
        with np.load(self.contact_path) as z:
            self.tri=z['triangles_mesh_m'];self.src=z['source_faces'];self.allowed=z['allowed_faces']
        self.clearance=ConservativeExitClearance(self.mesh)
        work=np.unique(np.concatenate([task.domain.work_ids for task,T in self.states]))
        offsets=wrap_offsets(self.mesh,.005)
        self.work_obstacle=union([S.solid(G.hull_mesh(G.head_cell(self.mesh,self.mesh.triangles[i],i,offsets))) for i in work])
        bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3],[1/3,1/3,1/3]])
        self.points=np.einsum('av,tvc->tac',bary,self.tri).reshape(-1,3)
        self.sources=np.repeat(self.src,4);self.outward=self.mesh.face_normals[self.sources]
        areas=.5*np.linalg.norm(np.cross(self.tri[:,1]-self.tri[:,0],self.tri[:,2]-self.tri[:,0]),axis=1)
        self.areas=np.repeat(areas/4,4);self.intersector=RayMeshIntersector(self.mesh)
        self.epsilon=self.extent*1e-7
        self.floors=[];self.rays=[]
        for task,T in self.states:
            from co_common import FLOOR
            points=transform_points(self.points,T);normal=-task.domain.mesh.face_normals[self.sources]
            self.rays.append(U.heads(np.c_[normal,np.cross(points-task.domain.com,normal)],task.scale))
            self.floors.append(U.floor(FLOOR.columns(task.floor,task.domain.com),task.scale))
        self.projection_cache={};self.exact_calls=0;self.proxy_calls=0;self.lock_cache={}
        seed_radius=max(np.linalg.norm(S.unpack(s).vertices,axis=1).max() for s in self.seed_parts)
        object_radius=np.linalg.norm(self.mesh.vertices,axis=1).max()
        self.baseline_spacing=float(seed_radius+object_radius+self.length+2*self.clearance.metadata['maximum_kernel_radius_m']+.01)
        self.inputs.extend(p for task,T in self.states for p in task.inputs)
        self.targets=[];self.saved_masks=[]
        for k,pose in enumerate(group['poses']):
            with np.load(self.base/'step4/step4.1/data'/f'{pose}.npz') as z:
                mask=z['force_mask'];sup=z['supply_7d']
                self.saved_masks.append(mask.copy())
            bad=np.flatnonzero(~mask)
            if len(bad):
                # Exact worst scan with safe bounds; no original load resampling.
                temporary=dict(serial=-k-1,masks=[mask],supplies=[sup])
                from types import SimpleNamespace
                adapter=SimpleNamespace(states=[self.states[k]],projection=lambda r,kk,i:cone_projection(sup,U.target(self.states[k][0].targets[i])))
                worst,scan=farthest_load(adapter,temporary);indices=[worst['load_index']]
            else:indices=[0]
            for i in indices:self.targets.append((k,int(i),U.target(self.states[k][0].targets[i])))
            print('PREPARED POSE',pose,'failed',len(bad),flush=True)

    def projection(self,result,k,index):
        key=(result['serial'],k,index)
        if key not in self.projection_cache:
            value=cone_projection(result['supplies'][k],U.target(self.states[k][0].targets[index]))
            if value['kkt_max_violation']>1e-7:raise RuntimeError('projection KKT unresolved')
            self.projection_cache[key]=value
        return self.projection_cache[key]

    def locks(self,owner,blocker,directions,offsets):
        relative=offsets[owner]-offsets[blocker]
        key=(owner,blocker,tuple(relative),tuple(directions[blocker]))
        if key in self.lock_cache:return self.lock_cache[key]
        origins=self.points+relative+self.epsilon*self.outward
        if np.linalg.norm(relative)<1e-12:
            blocked=self.outward@directions[blocker]>1e-9
        else:blocked=self.intersector.contains_points(origins)
        ids=np.flatnonzero(~blocked)
        if len(ids):
            locations,rays,faces=self.intersector.intersects_location(origins[ids],np.tile(-directions[blocker],(len(ids),1)),multiple_hits=False)
            distance=np.linalg.norm(locations-origins[ids[rays]],axis=1)
            blocked[ids[rays[distance<=self.length]]]=True
        if len(self.lock_cache)>2048:self.lock_cache.clear()
        self.lock_cache[key]=blocked
        return blocked

    def proxy(self,directions,offsets):
        began=time.perf_counter();self.proxy_calls+=1
        active=[];areas=[]
        for owner in range(self.n):
            locks=np.array([self.locks(owner,j,directions,offsets) for j in range(self.n)])
            available=~locks.any(axis=0);active.append(available);areas.append(float(self.areas[available].sum()))
        losses=[]
        for k,i,target in self.targets:
            value=cone_projection(np.vstack([self.floors[k],self.rays[k][active[k]]]),target)
            if value['kkt_max_violation']>1e-7:raise RuntimeError('proxy cone KKT unresolved')
            losses.append(value['loss'])
        # Feasibility first; compactness breaks equal-deficit ties.
        span=float(np.linalg.norm(np.ptp(offsets,axis=0)))
        return dict(losses=losses,score=(float(max(losses)),float(sum(losses)),span),
                    available_areas_m2=areas,active=active,seconds=time.perf_counter()-began)

    def exact(self,directions,offsets):
        if np.min(np.sum(directions*self.normals,axis=1))<-1e-12:raise ValueError('illegal exit direction')
        if np.max(np.abs(np.sum(offsets*self.normals,axis=1)))>1e-10:raise ValueError('translation leaves native floor plane')
        self.exact_calls+=1;began=time.monotonic()
        seed=union([shifted(s,o) for s,o in zip(self.seed_parts,offsets)])
        work=union([shifted(self.work_obstacle,o) for o in offsets])
        seed=seed-work
        nominal_sweeps=[shifted(self.clearance.sweep(self.length*d,padded=False),o) for d,o in zip(directions,offsets)]
        padded_sweeps=[shifted(self.clearance.sweep(self.length*d,padded=True),o) for d,o in zip(directions,offsets)]
        nominal_cut=union(nominal_sweeps);padded_cut=union(padded_sweeps);nominal=seed-nominal_cut
        nominal_mesh=S.unpack(nominal);meshes=[];allowed=[];core_parts=[]
        for k,o in enumerate(offsets):
            mesh=self.mesh.copy();mesh.apply_translation(o);meshes.append(mesh)
            ids=colocated_allowed(self.mesh.face_normals,self.allowed,k,directions,offsets);allowed.append(ids)
            tris,src=contact_boundary(mesh,nominal_mesh,ids)
            for tri,source in zip(tris,src):
                prism=trimesh.Trimesh(np.vstack([tri,tri+CONTACT_DEPTH_M*mesh.face_normals[source]]),_PRISM_FACES,process=False)
                if prism.volume<0:prism.invert()
                part=S.solid(prism)
                if material_volume(part^nominal)>1e-16:core_parts.append(part)
        core=union(core_parts)^nominal
        remaining=((seed-padded_cut)+core)-nominal_cut
        remaining=remaining-(padded_cut-core)
        # Remove any Boolean residual that actually violates the core exception.
        # This changes material conservatively rather than relaxing tolerances.
        for _ in range(1):
            residual=(remaining-core)^padded_cut
            if material_volume(residual)<1e-10:break
            remaining=remaining-residual
        diagnostics=dict(nominal_overlap_m3=max(material_volume(remaining^s) for s in nominal_sweeps),
            padded_overlap_outside_contact_cores_m3=material_volume((remaining-core)^padded_cut),
            working_region_overlap_m3=material_volume(remaining^work),
            partition_error_m3=abs(material_volume(seed)-material_volume(remaining)-material_volume(seed-remaining)))
        if max(diagnostics.values())>=1e-10:raise RuntimeError('exit/clearance/work/partition unresolved: '+str(diagnostics))
        parts=[S.unpack(p) for p in remaining.decompose() if material_volume(p)>=1e-12]
        if not parts:raise RuntimeError('no positive material')
        masks=[];supplies=[];patches=[];transforms=[]
        for k,((task,T),o) in enumerate(zip(self.states,offsets)):
            triangles=[];sources=[]
            for part in parts:
                tri,src=contact_boundary(meshes[k],part,allowed[k]);triangles.append(tri);sources.append(src)
            tri=np.concatenate(triangles);src=np.concatenate(sources);Tf=fixture_transform(T,o);transforms.append(Tf)
            full=supply(task,Tf,tri,src);mask,info=J.classify(full,task.targets)
            masks.append(mask);supplies.append(full);patches.append(dict(pose=self.group['poses'][k],triangles=len(tri),
                area_m2=float(.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum())))
        endpoint=max(material_volume(shifted(S.solid(self.mesh),o+self.length*d)^seed) for o,d in zip(offsets,directions))
        if endpoint>=1e-10:raise RuntimeError('complete exit endpoint unresolved')
        result=dict(serial=self.exact_calls,directions=directions.copy(),offsets=offsets.copy(),masks=masks,supplies=supplies,
            counts=[int(m.sum()) for m in masks],remaining=remaining,diagnostics=diagnostics,endpoint_overlap_m3=endpoint,
            contact_patches=patches,transforms=transforms,seconds=time.monotonic()-began)
        print('PLACEMENT EXACT',self.exact_calls,result['counts'],'offset max',float(np.linalg.norm(offsets,axis=1).max()),flush=True)
        return result

    def candidates(self,directions,offsets,round_index,count):
        result=[];frames=tangent_frames(directions)
        scale=[.001,.003,.006,.012,.025,.05,.1,.2][min(round_index,7)]
        failed=[k for k,i,t in self.targets];angles=[2.,5.,10.]
        for i in range(count):
            d=directions.copy();o=offsets.copy();kind=['translation','direction','joint','coherent-direction'][i%4]
            k=int(self.rng.choice(failed if i%3 else np.arange(self.n)))
            if kind in ['translation','joint']:
                if k==0:k=1+int(self.rng.integers(self.n-1))
                v=self.rng.normal(size=2);v/=np.linalg.norm(v)
                o[k]+=self.frames[k]@(v*scale*self.rng.choice([.5,1.,2.]))
            if kind in ['direction','joint','coherent-direction']:
                z=np.zeros((self.n,2));radius=np.deg2rad(angles[min(round_index//2,2)])
                if kind=='coherent-direction':
                    axis=self.rng.normal(size=3)
                    delta=np.cross(np.tile(axis,(self.n,1)),d)
                    z=np.einsum('nki,nk->ni',frames,delta);z*=radius/max(np.linalg.norm(z),1e-12)
                else:
                    v=self.rng.normal(size=2);z[k]=radius*v/np.linalg.norm(v)
                d=retract(d,frames,z)
                for j in range(self.n):
                    dot=d[j]@self.normals[j]
                    if dot<0:d[j]-=dot*self.normals[j];d[j]/=np.linalg.norm(d[j])
            result.append((kind,d,o))
        # Deterministic separation proposals supplement local random moves.
        for k in range(1,self.n):
            for axis in range(2):
                o=offsets.copy();o[k]+=scale*4*self.frames[k,:,axis]
                result.append(('translation-expansion',directions.copy(),o))
        result.append(('native-up-directions',self.normals.copy(),offsets.copy()))
        return result

    def run(self,iterations=8,candidates=24,finalists=2):
        began=time.monotonic();events=[];errors=[]
        directions=self.initial_directions.copy();offsets=np.zeros_like(directions)
        baseline_offsets=separation_layout(self.normals,self.baseline_spacing)
        baseline=None
        try:
            baseline=self.exact(self.normals.copy(),baseline_offsets)
            save(self.out/'separated_baseline.json',dict(counts=baseline['counts'],force_exit_passed=all(m.all() for m in baseline['masks']),
                offsets_m=baseline_offsets.tolist(),directions=self.normals.tolist(),spacing_m=self.baseline_spacing,
                diagnostics=baseline['diagnostics'],full_fixture_accepted=False))
        except (RuntimeError,ValueError) as error:
            errors.append(dict(stage='separated_baseline',error=str(error)));save(self.out/'separated_baseline.json',errors[-1])
        current=None
        try:current=self.exact(directions,offsets)
        except (RuntimeError,ValueError) as error:errors.append(dict(stage='initial',error=str(error)))
        initial_counts=[int(m.sum()) for m in self.saved_masks]
        exact_initial_counts=current['counts'] if current else None
        stop='iteration_limit';accepted_steps=0
        for iteration in range(iterations):
            if current is not None and all(m.all() for m in current['masks']):stop='sampled_force_exit_feasible';break
            base_proxy=self.proxy(directions,offsets)
            candidates_ranked=[];proxy_started=time.monotonic()
            for kind,d,o in self.candidates(directions,offsets,iteration,candidates):
                try:
                    proxy=self.proxy(d,o)
                    candidates_ranked.append((proxy['score'],kind,d,o,proxy))
                except (RuntimeError,ValueError) as error:errors.append(dict(stage='proxy',error=str(error)))
            candidates_ranked.sort(key=lambda row:row[0]);row=dict(iteration=iteration+1,base_proxy_score=base_proxy['score'],
                proposals=len(candidates_ranked),proxy_seconds=time.monotonic()-proxy_started,trials=[])
            accepted=False
            for score,kind,d,o,proxy in candidates_ranked[:finalists]:
                record=dict(kind=kind,proxy_score=score,offsets_m=o.tolist(),directions=d.tolist(),
                    released_area_m2=[float(self.areas[~a&b].sum()) for a,b in zip(base_proxy['active'],proxy['active'])],
                    newly_locked_area_m2=[float(self.areas[a&~b].sum()) for a,b in zip(base_proxy['active'],proxy['active'])])
                try:
                    trial=self.exact(d,o);record['counts']=trial['counts']
                    protected=preserves_loads(current if current is not None else dict(masks=self.saved_masks),trial)
                    def real_score(r):
                        fractions=np.array(r['counts'])/np.array([len(t.targets) for t,T in self.states])
                        return (float(fractions.min()),float(fractions.sum()))
                    accepted=protected and (current is None or real_score(trial)>real_score(current))
                    if protected and not accepted and score[:2]<base_proxy['score'][:2]:
                        worst,_=farthest_load(self,current);new_worst,_=farthest_load(self,trial)
                        accepted=(0. if new_worst is None else new_worst['loss'])<worst['loss']-1e-8
                    if accepted:current=trial;directions=d;offsets=o;accepted_steps+=1
                    record['accepted']=accepted
                except (RuntimeError,ValueError) as error:record.update(error=str(error),accepted=False)
                row['trials'].append(record)
                if accepted:break
            row.update(accepted=accepted,counts=current['counts'] if current else None);events.append(row)
            save(self.out/'sampling_trace.json',events)
            print('SAMPLING ROUND',iteration+1,'accepted',accepted,row['counts'],'proxy seconds',row['proxy_seconds'],flush=True)
        # Sample contractions of the validated separation layout; retain the
        # smallest actually validated layout, never infer exact success from rays.
        baseline_contractions=[]
        if baseline is not None and all(m.all() for m in baseline['masks']):
            feasible=[]
            for factor in [.5,.25,.125,.0625,.03125]:
                o=baseline_offsets*factor
                proxy=self.proxy(self.normals,o)
                baseline_contractions.append(dict(factor=factor,proxy_score=proxy['score']))
                if proxy['score'][0]<1e-8:feasible.append((factor,o))
            for factor,o in reversed(feasible):
                try:
                    trial=self.exact(self.normals,o)
                    baseline_contractions.append(dict(factor=factor,counts=trial['counts']))
                    if all(m.all() for m in trial['masks']):
                        baseline=trial;break
                except (RuntimeError,ValueError) as error:
                    baseline_contractions.append(dict(factor=factor,error=str(error)))
        used_baseline=False
        if current is None or not all(m.all() for m in current['masks']):
            if baseline is not None and all(m.all() for m in baseline['masks']):
                current=baseline;directions=baseline['directions'];offsets=baseline['offsets'];used_baseline=True;stop='separated_baseline_force_exit_feasible'
        passed=current is not None and all(m.all() for m in current['masks'])
        if current is not None:D.export_exact_obj(S.unpack(current['remaining']),self.out/'remaining_support.obj')
        np.savez_compressed(self.out/'layout.npz',directions=directions,offsets_m=offsets,
            T_fixture_to_world=np.asarray(current['transforms']) if current else np.empty((0,4,4)))
        report=dict(complete=True,algorithm='multi-candidate sampling of exit directions and floor-tangent pose translations',
            pose_set=self.group['id'],poses=self.group['poses'],initial_counts=initial_counts,exact_initial_counts=exact_initial_counts,
            baseline_contractions=baseline_contractions,
            final_counts=current['counts'] if current else None,force_exit_passed=bool(passed),force_passed=bool(passed),
            geometry_constructed=current is not None,clearance_certified=current is not None,full_fixture_accepted=False,
            baseline_used=used_baseline,baseline_passed=baseline is not None and all(m.all() for m in baseline['masks']),
            stop_reason=stop,accepted_sampling_steps=accepted_steps,iterations=events,errors=errors,
            offsets_m=offsets.tolist(),directions=directions.tolist(),fixed_reference_pose=self.group['poses'][0],
            maximum_translation_m=float(np.linalg.norm(offsets,axis=1).max()),layout_span_m=float(np.linalg.norm(np.ptp(offsets,axis=0))),
            exact_evaluations=self.exact_calls,proxy_evaluations=self.proxy_calls,seconds=time.monotonic()-began,
            original_loads_reused=True,load_count_per_pose=[len(t.targets) for t,T in self.states],
            no_uplift=U.description(),connectivity_required=False,component_count=len(current['remaining'].decompose()) if current else None,
            diagnostics=current['diagnostics'] if current else None,contact_patches=current['contact_patches'] if current else None,
            seed_policy='union of translated common non-work wrap plus each owning pose original ring; translated original work exclusions',
            proxy_policy='four equal-area samples per triangle; all pose continuous reverse-ray lock union; no force capacities',
            acceptance='complete continuous nominal exits; 1% clearance outside available contact cores; original all-load equilibrium/no-uplift',
            deferred='connectivity, installed whole-fixture ground legality/support, and strength; illegal group is not a full fixture success',
            provenance=provenance(self.inputs,[Path(__file__),Path(__file__).with_name('physics_guided_padded_sweep.py')]+code_sources()))
        save(self.out/'report.json',report)
        print('PLACEMENT FINAL',self.group['id'],report['final_counts'],stop,'seconds',report['seconds'],flush=True)
        return report
