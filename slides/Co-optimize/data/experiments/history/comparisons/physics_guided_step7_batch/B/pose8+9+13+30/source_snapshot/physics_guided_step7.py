"""Joint continuous exit optimization with physics-envelope direction feedback.

The optimization surrogate is not physical acceptance. Continuous trajectory
distance-field sensitivities are numerical; normal compatibility and LP envelopes are
analytic. Original unbounded mechanics and exact construction remain intact.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from exit_clearance import ExitClearance
from physics_guided_cone import cone_projection
from physics_guided_geometry import (acquisition, tangent_frames, retract,
    retraction_jacobian, SweepDistanceModel, distance_cost)
from physics_guided_objective import acquisition_equilibrium, aggregate
from physics_guided_paths import MaterialPathModel
from physics_guided_progress import progress_decision
from scipy.optimize import minimize
import argparse, time


class PhysicsSearch:
    def __init__(self, name, group, out=None, native=False, geometry='sweep', directions=None, connectivity=False):
        self.name, self.group = name, group
        self.base = HERE/'output'/name/group['id']
        self.out = out or HERE/'output/physics_guided_v2'/name/group['id']
        self.out.mkdir(parents=True, exist_ok=True)
        self.states = [state(name, p)[:2] for p in group['poses']]
        self.mesh = state(name, group['poses'][0])[2]
        self.seed_path = self.base/'step3/step3.3/support_with_rings.obj'
        self.seed_mesh = trimesh.load(self.seed_path, force='mesh', process=False)
        self.seed = S.solid(self.seed_mesh)
        self.contact_path = self.base/'step3/step3.2/data/contacts.npz'
        with np.load(self.contact_path) as z:
            self.tri, self.src, self.allowed = z['triangles_mesh_m'], z['source_faces'], z['allowed_faces']
        self.initial_path = self.base/'step4/step4.1/data/report.json'
        self.initial = json.loads(self.initial_path.read_text())
        self.initial_snapshot = self.out/'initialization_input.json'
        save(self.initial_snapshot,self.initial)
        self.length = self.initial['initialization']['full_length_m']
        self.normals = np.array([T[:3,:3].T @ np.array([0.,0.,1.]) for task,T in self.states])
        self.clearance = ExitClearance(self.mesh)
        self.ray_normals = np.repeat(self.mesh.face_normals[self.src],3,axis=0)
        self.rays, self.floors = [], []
        for task,T in self.states:
            points = transform_points(self.tri.reshape(-1,3),T)
            normals = np.repeat(-task.domain.mesh.face_normals[self.src],3,axis=0)
            raw = np.c_[normals,np.cross(points-task.domain.com,normals)]
            self.rays.append(U.heads(raw,task.scale))
            self.floors.append(U.floor(FLOOR.columns(task.floor,task.domain.com),task.scale))
        self.geometry = geometry
        self.distance_model = SweepDistanceModel(self.clearance,self.tri.reshape(-1,3),
            self.ray_normals,self.length,float(self.mesh.extents.max())) if geometry=='sweep' else None
        self.connectivity = connectivity
        self.path_model = MaterialPathModel(self.seed_mesh,self.clearance,self.tri.reshape(-1,3),
            self.ray_normals,self.length,float(self.mesh.extents.max())) if connectivity else None
        self.native, self.start_directions = native, directions
        self.warm_path = None
        self.exact_calls, self.trace = 0, []
        self.load_bank = set()
        self.projection_cache = {}

    def warm(self):
        if self.start_directions is not None:
            self.warm_path = Path(self.start_directions)
            return np.load(self.warm_path)['directions'].copy()
        if self.native:
            return self.normals.copy()
        folder = self.base/'step4/step4.2/data/connected_run/data'
        for name in ['balanced_best_paths.npz','best_connected_paths.npz','best_paths.npz']:
            path = folder/name
            if path.exists():
                self.warm_path = path
                return np.load(path)['directions'].copy()
        path = self.base/'step4/step4.2/data/report.json'
        self.warm_path = path if path.exists() else self.initial_path
        report = json.loads(self.warm_path.read_text())
        return np.asarray([r['direction_fixture'] for r in report['state_results']])

    def classify(self, triangles, sources):
        masks, infos, supplies = [], [], []
        for task,T in self.states:
            full = supply(task,T,triangles,sources)
            mask,info = J.classify(full,task.targets)
            masks.append(mask); infos.append(info); supplies.append(full)
        return masks,infos,supplies

    def exact(self,d):
        d = np.asarray(d,float)
        if d.shape != self.normals.shape or not np.isfinite(d).all():
            raise ValueError('invalid direction array')
        if not np.allclose(np.linalg.norm(d,axis=1),1.,atol=1e-10):
            raise ValueError('directions must be unit vectors')
        if np.min(np.sum(d*self.normals,axis=1)) < -1e-12:
            raise ValueError('below native floor hemisphere')
        self.exact_calls += 1
        physical = self.allowed[np.max(self.mesh.face_normals[self.allowed]@d.T,axis=1)<=1e-9]
        errors=[]
        for fan in [8,2,16]:
            try:
                sweeps=[self.clearance.sweep(self.length*x,fan,padded=False) for x in d]
                padded=[self.clearance.sweep(self.length*x,fan) for x in d]
                c=self.clearance.construct(self.seed,sweeps,padded,physical,boundary=contact_boundary)
                overlap=max(material_volume(c['remaining']^s) for s in sweeps)
                partition=abs(material_volume(self.seed)-material_volume(c['remaining'])-material_volume(c['removed']))
                diag=c['diagnostics']
                if not diag['contact_area_preserved'] or max(overlap,partition,diag['padded_sweep_overlap_outside_contact_cores_m3'])>=1e-10:
                    raise RuntimeError('clearance/contact/partition unresolved')
                endpoint_overlap=[]
                for x in d:
                    moved=self.mesh.copy();moved.apply_translation(self.length*x)
                    endpoint_overlap.append(material_volume(S.solid(moved)^self.seed))
                if max(endpoint_overlap)>=1e-10:
                    raise RuntimeError('exit endpoint not fully detached')
                components=c['remaining'].decompose()
                # Zero-volume coplanar remnants do not provide physical
                # contact. Extract contacts on positive material components
                # separately before computing even the aggregate force loss.
                patches=[contact_boundary(self.mesh,S.unpack(part),physical)
                         for part in components if material_volume(part)>=1e-12]
                if not patches:raise RuntimeError('no positive material contacts')
                tri=np.concatenate([p[0] for p in patches])
                src=np.concatenate([p[1] for p in patches])
                masks,infos,supplies=self.classify(tri,src)
                result=dict(serial=self.exact_calls,directions=d.copy(),construction=c,masks=masks,
                    infos=infos,supplies=supplies,counts=[int(m.sum()) for m in masks],triangles=tri,
                    sources=src,components=components,overlap=overlap,
                    partition=partition,endpoint_overlap=endpoint_overlap,fan=fan)
                if self.connectivity:
                    parts=[part for part in result['components'] if material_volume(part)>=1e-12]
                    if not parts:raise RuntimeError('no positive bearing component')
                    # Fixed local anchor policy: grow the largest actual
                    # component. This does not claim globally best anchoring.
                    anchor=max(parts,key=material_volume)
                    am,ai,asupply=self.classify(*contact_boundary(self.mesh,S.unpack(anchor),physical))
                    result.update(bearing_masks=am,bearing_supplies=asupply,anchor=anchor,
                        bearing_counts=[int(m.sum()) for m in am])
                print('EXACT',self.exact_calls,result['counts'],'components',len(result['components']),
                      'anchor',result.get('bearing_counts'),flush=True)
                return result
            except (RuntimeError,ValueError) as error:
                errors.append(str(error))
        raise RuntimeError('exact construction unresolved: '+'; '.join(errors))

    def projection(self,result,k,index):
        key=(result['serial'],k,index)
        if key not in self.projection_cache:
            target=U.target(self.states[k][0].targets[index])
            value=cone_projection(result.get('bearing_supplies',result['supplies'])[k],target)
            if value['kkt_max_violation']>1e-7*max(1.,np.linalg.norm(target)):
                raise RuntimeError('cone projection KKT unresolved')
            self.projection_cache[key]=value
        return self.projection_cache[key]

    def choose_loads(self,result):
        # Select from unchanged saved loads; full classification identifies all
        # new failures. Rank a deterministic spread by continuous cone loss.
        for k,mask in enumerate(result.get('bearing_masks',result['masks'])):
            bad=np.flatnonzero(~mask)
            candidates=bad[np.linspace(0,len(bad)-1,min(16,len(bad))).astype(int)] if len(bad) else np.array([0])
            ranked=sorted((self.projection(result,k,int(j))['loss'],int(j)) for j in candidates)
            self.load_bank.update((k,index) for loss,index in ranked[-3:])
        return sorted(self.load_bank)

    def actual_loss(self,result,loads):
        values=[self.projection(result,k,index)['loss'] for k,index in loads]
        return max(values),values

    def local_costs(self,origin,frames,coordinates,sweep_model=None,width=.02):
        d=retract(origin,frames,coordinates)
        costs,cartesian=acquisition(d,self.ray_normals,width)
        mapping=retraction_jacobian(origin,frames,coordinates)
        gradients=np.einsum('jnk,nki->jni',cartesian,mapping)
        if sweep_model is not None:
            if sweep_model.get('contact') is not None:
                distance_values,distance_jacobian=sweep_model['contact']
                c,g=distance_cost(distance_values,distance_jacobian,coordinates)
                costs+=c;gradients+=g
            if sweep_model.get('path') is not None:
                values,jac=sweep_model['path']
                c,g=self.path_model.cost(values,jac,coordinates)
                costs+=c;gradients+=g
        return costs,gradients

    def objective(self,origin,frames,coordinates,loads,sweep_model=None,width=.02):
        costs,gradients=self.local_costs(origin,frames,coordinates,sweep_model,width)
        values,derivatives=[],[]
        for k,index in loads:
            target=U.target(self.states[k][0].targets[index])
            physical=acquisition_equilibrium(self.floors[k],self.rays[k],target,costs)
            values.append(physical['value'])
            derivatives.append(np.einsum('j,jni->ni',physical['cost_gradient'],gradients))
        return aggregate(values,derivatives)

    def nonlinear_costs(self,d):
        """Re-query geometry at candidate directions; never use a predicted decrease."""
        costs=acquisition(d,self.ray_normals)[0]
        zero=np.zeros((len(d),2))
        for model,is_path in [(self.distance_model,False),
                              (self.path_model.sweep if self.path_model else None,True)]:
            if model is None:continue
            values=np.column_stack([model.distances(x) for x in d])
            jac=np.zeros(values.shape+(2,))
            addition=(self.path_model.cost(values,jac,zero) if is_path else distance_cost(values,jac,zero))[0]
            costs+=addition
        return costs

    def contact_values(self,costs,loads):
        """Freeze useful-contact sensitivities for the entire local acceptance search."""
        values=[];weights=[]
        for k,index in loads:
            physical=acquisition_equilibrium(self.floors[k],self.rays[k],
                U.target(self.states[k][0].targets[index]),costs)
            values.append(physical['value']);weights.append(physical['cost_gradient'])
        values=np.asarray(values)
        factors=np.exp((values-values.max())/.005);factors/=factors.sum()
        weights=np.einsum('l,lj->j',factors,np.asarray(weights))
        total=weights.sum()
        return weights/total if total>1e-15 else np.zeros_like(weights)

    def optimize(self,iterations=8):
        began=time.monotonic()
        d=self.warm();d/=np.linalg.norm(d,axis=1)[:,None]
        # Freeze the exact starting directions before another batch can update
        # a checkpoint. This local file is the actual input for this run.
        np.savez_compressed(self.out/'initial_directions.npz',directions=d)
        result=self.exact(d);best_result=result;initial_counts=result['counts'];initial_anchor_counts=result.get('bearing_counts');radius=np.deg2rad(8.)
        stop='iteration_limit'
        for iteration in range(iterations):
            if all(m.all() for m in result.get('bearing_masks',result['masks'])):
                stop='single_component_force_feasible' if self.connectivity else 'aggregate_force_feasible';break
            loads=self.choose_loads(result)
            frames=tangent_frames(d);zero=np.zeros((len(d),2))
            print('LINEARIZE',iteration,self.geometry,'connectivity',self.connectivity,flush=True)
            geometric=dict(contact=self.distance_model.linearize(d,frames) if self.distance_model else None,
                path=self.path_model.linearize(d,frames,S.unpack(result['anchor'])) if self.path_model else None)
            actual_costs=self.nonlinear_costs(d)
            contact_values=self.contact_values(actual_costs,loads)
            geometry_before=float(contact_values@actual_costs)
            def geometry_fun(x):
                c,g=self.local_costs(d,frames,x.reshape(zero.shape),geometric)
                return float(contact_values@c),np.einsum('j,jni->ni',contact_values,g)
            fun=lambda x:self.objective(d,frames,x.reshape(zero.shape),loads,geometric)
            value,gradient=fun(zero)
            magnitude=np.linalg.norm(gradient)
            if magnitude<1e-12:
                stop='zero_surrogate_gradient';break
            descent=-gradient/magnitude
            analytic=float(np.sum(gradient*descent))
            audits=[]
            for h in [1e-3,3e-4,1e-4,3e-5]:
                plus=fun(h*descent)[0];minus=fun(-h*descent)[0]
                fd=(plus-minus)/(2*h)
                forward=(plus-value)/h;backward=(value-minus)/h
                relative=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12)
                envelope=forward-1e-3*magnitude <= analytic <= backward+1e-3*magnitude
                audits.append(dict(step=h,finite_difference=fd,forward=forward,backward=backward,
                    relative_error=relative,envelope_consistent=bool(envelope)))
            best_audit=min(audits,key=lambda row:row['relative_error'])
            fd=best_audit['finite_difference'];derivative_error=best_audit['relative_error']
            record=dict(iteration=iteration,surrogate=value,gradient_norm=float(magnitude),
                analytic_directional_derivative=analytic,finite_difference=fd,
                derivative_relative_error=derivative_error,loads=[dict(pose=self.group['poses'][k],index=j) for k,j in loads],trials=[])
            # LP value is piecewise smooth. A finite difference at a kink need
            # not match a selected LP subgradient; shrink/check, do not claim
            # a verified ordinary derivative when it does not.
            record['gradient_audits']=audits
            record['gradient_check_passed']=bool(derivative_error<1e-3)
            record['nonsmooth_envelope_consistent']=bool(any(row['envelope_consistent'] for row in audits))
            if not record['gradient_check_passed'] and not record['nonsmooth_envelope_consistent']:
                self.trace.append(record);stop='physics_sensitivity_unresolved';break
            accepted=False
            for trust_attempt in range(3):
                def wrapped(x):
                    v,g=geometry_fun(x);return v,g.flatten()
                def hemisphere(x):
                    return np.sum(retract(d,frames,x.reshape(zero.shape))*self.normals,axis=1)
                opt=minimize(wrapped,zero.flatten(),jac=True,method='SLSQP',
                    constraints=[dict(type='ineq',fun=hemisphere),
                        dict(type='ineq',fun=lambda x:radius**2-float(x@x))],
                    options={'maxiter':30,'ftol':1e-9})
                step=opt.x.reshape(zero.shape)
                record['optimizer_status']=dict(success=bool(opt.success),message=str(opt.message))
                if not np.isfinite(step).all() or np.linalg.norm(step)>radius*(1+1e-5):
                    radius*=.5;continue
                for fraction in [1.,.5,.25]:
                    coordinates=fraction*step
                    candidate=retract(d,frames,coordinates)
                    if np.min(np.sum(candidate*self.normals,axis=1)) < -1e-12:continue
                    try:
                        trial=self.exact(candidate)
                        # Include newly failed loads BEFORE deciding a step;
                        # evaluate old and new geometry on the same enlarged
                        # working set, so improvement cannot hide regression.
                        expanded=self.choose_loads(trial)
                        before,before_values=self.actual_loss(result,expanded)
                        after,after_values=self.actual_loss(trial,expanded)
                        regression=max(np.asarray(after_values)-np.asarray(before_values))
                        no_new_failure=all(np.all(new[old]) for old,new in zip(result['masks'],trial['masks']))
                        best_loss,_=self.actual_loss(best_result,expanded)
                        geometry_after=float(contact_values@self.nonlinear_costs(candidate))
                        decision=progress_decision(before,after,best_loss,geometry_before,geometry_after)
                        accept=decision['accepted']
                        record['trials'].append(dict(fraction=fraction,radius_rad=radius,anchor_counts=trial.get('bearing_counts'),
                            old_loss=before,new_loss=after,largest_working_set_regression=float(regression),
                            preserves_previously_feasible_demands=bool(no_new_failure),
                            best_constructed_loss=best_loss,geometry_before=geometry_before,geometry_after=geometry_after,
                            contact_value_sum=float(contact_values.sum()),counts=trial['counts'],**decision))
                        if accept:
                            if after<=best_loss+1e-12:best_result=trial
                            d=candidate;result=trial;accepted=True;radius=min(np.deg2rad(12.),radius*1.4);break
                    except (RuntimeError,ValueError) as error:
                        record['trials'].append(dict(fraction=fraction,error=str(error),accepted=False))
                if accepted:break
                radius*=.5
            record['accepted']=accepted;self.trace.append(record)
            save(self.out/'optimization_trace.json',self.trace)
            np.savez_compressed(self.out/'directions.npz',directions=d)
            print('ITERATION',iteration,'accepted',accepted,'derivative error',derivative_error,flush=True)
            if not accepted:
                stop='exact_line_search_stalled';break
        # Intermediate geometry steps may trade force coverage temporarily.
        # Return the best actual constructed state on the final common load bank.
        continuation_counts=result['counts'].copy()
        np.savez_compressed(self.out/'continuation_directions.npz',directions=d)
        returned_best=False
        if self.load_bank and self.actual_loss(best_result,sorted(self.load_bank))[0] < self.actual_loss(result,sorted(self.load_bank))[0]-1e-12:
            result=best_result;d=result['directions'].copy();returned_best=True
        component_rows=[];winner=None
        physical=self.allowed[np.max(self.mesh.face_normals[self.allowed]@d.T,axis=1)<=1e-9]
        for part in sorted(result['components'],key=lambda x:-material_volume(x)):
            if material_volume(part)<1e-12:continue
            tri,src=contact_boundary(self.mesh,S.unpack(part),physical)
            masks,infos,supplies=self.classify(tri,src)
            component_rows.append(dict(volume_cm3=material_volume(part)*1e6,counts=[int(m.sum()) for m in masks]))
            if all(m.all() for m in masks):winner=part;break
        retained=winner if winner is not None else result['construction']['remaining']
        single=winner is not None and len(retained.decompose())==1
        final_overlap=max(material_volume(retained^self.clearance.sweep(self.length*x,padded=False)) for x in d)
        passed=bool(single and all(m.all() for m in result['masks']) and final_overlap<1e-10)
        D.export_exact_obj(S.unpack(retained),self.out/'remaining_support.obj')
        np.savez_compressed(self.out/'directions.npz',directions=d)
        code=[Path(__file__),HERE/'helper_func/physics_guided_cone.py',HERE/'helper_func/physics_guided_geometry.py',
            HERE/'helper_func/physics_guided_objective.py',HERE/'helper_func/physics_guided_field.py',HERE/'helper_func/exit_clearance.py',HERE/'helper_func/co_common.py',
            HERE/'helper_func/physics_guided_paths.py',HERE/'helper_func/physics_guided_progress.py',Path(J.__file__),Path(C.W.__file__),Path(U.__file__),Path(S.__file__).with_name('translation_sweep.py')]
        field_inputs=[self.distance_model.field.path] if self.distance_model else []
        if self.path_model:field_inputs.append(self.path_model.sweep.field.path)
        inputs=field_inputs+[self.seed_path,self.contact_path,self.initial_snapshot,self.out/'initial_directions.npz']+[p for task,T in self.states for p in task.inputs]
        report=dict(complete=True,algorithm='joint continuous physics-envelope exit optimization',geometry_mode=self.geometry,
            pose_set=self.group['id'],initial_counts=initial_counts,final_counts=result['counts'],
            load_count_per_pose=[len(task.targets) for task,T in self.states],force_passed=all(m.all() for m in result['masks']),
            single_component_passed=single,passed=passed,component_count=len(retained.decompose()),component_results=component_rows,
            stop_reason=stop,exact_evaluations=self.exact_calls,iterations=self.trace,seconds=time.monotonic()-began,
            intermediate_progress_policy='true force decrease OR frozen physics-weighted nonlinear geometry decrease; track best constructed force state; final feasibility unchanged',
            returned_best_constructed_state=returned_best,
            continuation_counts=continuation_counts,
            exit_clearance=self.clearance.metadata,clearance_diagnostics=result['construction']['diagnostics'],
            nominal_sweep_overlap_m3=final_overlap,partition_error_m3=result['partition'],endpoint_overlap_m3=result['endpoint_overlap'],
            geometry_gradient='analytic normal compatibility + central differences of a fixed trilinear object distance field with smooth full-exit time quadrature' if self.distance_model else 'analytic initial normal compatibility only',
            sweep_distance_evaluations=self.distance_model.evaluations if self.distance_model else 0,
            guidance_discretization=dict(distance_grid_max_extent_divisor=96,exit_time_points=129),
            physics_gradient='analytic LP envelope; normalized unbounded rays pay acquisition cost for guidance only',
            connectivity_optimized=self.connectivity,initial_anchor_counts=initial_anchor_counts,
            path_geometry_gradient='central differences of the fixed trajectory distance field on local material paths' if self.path_model else None,
            final_anchor_counts=result.get('bearing_counts'),material_graph_nodes=len(self.path_model.points) if self.path_model else 0,full_fixture_accepted=False,original_loads_reused=True,
            load_subsampling_for_final_acceptance=False,directions=d.tolist(),
            validation_policy='exact construction/full original demands; connected-component pruning if independently sufficient; no exported replay',
            provenance=provenance(inputs,code))
        save(self.out/'optimization_trace.json',self.trace)
        report['artifacts']={name:I.sha256(self.out/name) for name in ['remaining_support.obj','directions.npz','continuation_directions.npz','optimization_trace.json']}
        save(self.out/'report.json',report)
        I.check_report(self.out/'report.json')
        print('FINAL',json.dumps({k:report[k] for k in ['pose_set','initial_counts','final_counts','passed','seconds']}),flush=True)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set',default='pose1+4+7+12+21+27')
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--geometry',choices=['sweep','normal'],default='sweep')
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--connectivity',action='store_true',help='Optimize material paths to the largest current bearing component')
    parser.add_argument('--directions',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    group=next(g for g in groups if g['id']==args.set)
    search=PhysicsSearch('B',group,out=args.out,native=args.native,geometry=args.geometry,directions=args.directions,connectivity=args.connectivity)
    try:
        search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,
            pose_set=group['id'],error=str(error),exact_evaluations=search.exact_calls,
            policy='A numerical construction/optimization failure is not proof of physical infeasibility'))
        raise


if __name__=='__main__':main()
