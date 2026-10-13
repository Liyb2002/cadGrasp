"""Existing whole-gradient operations with a layout-dependent base gate.

Force gradients and their demand measure are unchanged. Explicit multi-pose
Juxtapose samples cross floor-incompatible state assignments; oracle queries
then rank common XYZ offsets, before original-load checks. +Z is not assumed
monotonic: every actual sampled height is queried before selection.
"""
import numpy as np
from continuous_support.stable_gradient import StableGradientModel, StableGradientSearch
from whole_search.common import legal_direction
from whole_search.search import failed_count
from whole_search.reuse_first import registered
from base_oracle import BaseOracle
from co_common import save
from ground_support import CONTACT_CLEARANCE_M, allowed_contacts, contact_event_step, floor_states
from translation.proposals import translation_proposals, coherent_translation_proposals
from translation.installation import installation_proposals
from fixture_state import installed_state

def candidate_rank(loss, envelope_cm3, spacing, height):
    """Numerically zero projection losses tie; prefer a compact envelope.

    This guides proposal order only. Every finalist still needs the original
    full-demand and connected-base gates; material descent remains unchanged.
    """
    return (0. if loss<=1e-28 else loss, envelope_cm3, spacing, height)


class OracleModel(StableGradientModel):
    def __init__(self,*args,oracle_work_mode='normal',**kwargs):
        self.support_floor_clearance_m=CONTACT_CLEARANCE_M
        super().__init__(*args,**kwargs)
        self.base_oracle=BaseOracle(self,work_mode=oracle_work_mode)
        self.oracle_gate_active=False

    def contact_allowed(self,layout,owner):
        return super().contact_allowed(layout,owner) & allowed_contacts(self,layout,owner)

    def translation_contact_step(self,layout,owners,axis,sign,minimum):
        return contact_event_step(self,layout,owners,axis,sign,minimum)

    def material_floor_allowed(self,layout,points):
        planes=floor_states(self,layout)
        return np.all(np.asarray(points)@planes[:,:3].T+planes[:,3]>=-1e-10,axis=1)

    def restore_registered(self,layout,owner):
        """Return to the original seat without dropping the task's XYZ pose."""
        trial=layout.copy()
        world_direction=self.native[int(layout.hosts[owner]),:3,:3]@layout.directions[owner]
        world=installed_state(self.native,layout,layout.hosts[owner])@layout.placements[owner]
        new=self.native[owner].copy();new[:3,3]=world[:3,3]
        # If this host already serves other tasks, keep their world placements.
        for k in layout.active:
            if int(layout.hosts[k])==owner:
                old=installed_state(self.native,layout,owner)@layout.placements[k]
                trial.placements[k]=np.linalg.inv(new)@old
        trial.state_offsets[owner]=new[:3,3]-self.native[owner,:3,3]
        trial.hosts[owner]=owner;trial.placements[owner]=np.eye(4)
        trial.directions[owner]=legal_direction(self.native[owner,:3,:3].T@world_direction,
                                                self.floor_normal(trial,owner))
        return trial

    def evaluate(self,layout):
        if getattr(self,'require_connected_core',False):
            from connected_core import prepare
            from material_force import certify_material_contacts
            core,core_record=prepare(self,layout)
            # Direction/translation proxies retain the existing cheap guidance;
            # full masks always belong to the retained, serialized actual core.
            result=super().evaluate(layout)
            previous=failed_count(result)
            proof=certify_material_contacts(self,layout,core)
            result.update(proof,remaining_mesh=core,connected_contact_core=core_record,
                volume_cm3=abs(float(core.volume))*1e6,
                grid_failed_demands=previous,material_contact_certification_run=True,
                force_material_geometry_certified=all(v.all()for v in proof['masks'].values()),
                force_acceptance_geometry='support.obj')
            result['base_oracle']=self.base_oracle.query(layout)
            return result
        oracle=self.base_oracle.query(layout)
        if self.oracle_gate_active and not oracle['enclosure_passed']:
            raise ValueError('Base oracle rejected candidate: '+oracle['status'])
        result=super().evaluate(layout)
        result['base_oracle']=oracle
        return result


class OracleSearch(StableGradientSearch):
    def __init__(self,*args,frontier_width=4,search_policy='frontier',**kwargs):
        if frontier_width<1:
            raise ValueError('Frontier width must be positive')
        super().__init__(*args,**kwargs)
        self.frontier_width=int(frontier_width)
        if search_policy not in ('frontier','material','population32','population_soft'):
            raise ValueError('Unknown search policy')
        self.search_policy=search_policy

    def layout_envelope_cm3(self,layout):
        """Cheap static layout/pressure-hull envelope, without a base Boolean."""
        m=self.model
        points=[m.mesh.vertices@layout.placements[k,:3,:3].T+layout.placements[k,:3,3]
                for k in layout.active]
        for row in m.base_oracle.rows(layout):
            inverse=np.linalg.inv(row['T'])
            floor=np.c_[row['hull'],np.zeros(len(row['hull']))]
            points.append(floor@inverse[:3,:3].T+inverse[:3,3])
        points=np.vstack(points)
        worlds=[points@installed_state(m.native,layout,host)[:3,:3].T+installed_state(m.native,layout,host)[:3,3]
                for host in sorted(set(int(layout.hosts[k]) for k in layout.active))]
        extent=np.ptp(np.vstack(worlds),axis=0)+2*m.thickness
        return float(np.prod(extent)*1e6)

    def local_proposals(self,current,targets,worst,translation_guests=()):
        rows,base=super().local_proposals(current,targets,worst,translation_guests)
        layout=current['layout'];m=self.model
        # Native rotating-reuse poses may lift/translate without changing host.
        # The same XYZ helper and all-pose loss judge these continuous moves.
        native=[k for k in layout.active if registered(layout,k)]
        native.sort(key=lambda k:-base['residual_loss'][list(layout.active).index(k)])
        chosen=native[:3]
        fine=getattr(self,'fine_resolution',False)
        rows.extend(translation_proposals(m,layout,chosen,fine=fine))
        rows.extend(coherent_translation_proposals(m,layout,chosen,fine=fine))
        rows.extend(installation_proposals(m,layout,fine=fine))
        if chosen:
            self.last_coordinate_selection['translation_poses'] += [m.poses[k] for k in chosen]
        return rows,base

    def refine_fixture_states(self,current,rounds=2):
        """Repair floor/base feasibility through installed-state XYZ motion."""
        m=self.model
        for iteration in range(rounds):
            layout=current['layout'];answer=m.base_oracle.query(layout)
            if answer['enclosure_passed']:break
            objective=lambda q:m.base_oracle.query(q)['gap_m']
            with m.hold_measure(current):
                proposals=installation_proposals(m,layout,objective=objective)
                ranked=[]
                before=(not answer['enclosure_passed'],answer['gap_m'])
                for kind,trial,detail in proposals:
                    oracle=m.base_oracle.query(trial)
                    score=(not oracle['enclosure_passed'],oracle['gap_m'])
                    if score<before:
                        ranked.append((score,kind,trial,detail,oracle))
                ranked.sort(key=lambda r:r[0])
                # Only a bounded shortlist needs all-demand force guidance.
                finalists=[]
                for score,kind,trial,detail,oracle in ranked[:12]:
                    try:
                        proxy=m.proxy(trial)
                        finalists.append((score,proxy['loss'],self.layout_envelope_cm3(trial),
                                          kind,trial,detail,oracle))
                    except (RuntimeError,ValueError):continue
                finalists.sort(key=lambda r:r[:3])
                best=None;trials=[]
                for score,loss,extent,kind,trial,detail,oracle in finalists[:3]:
                    try:
                        branch=m.evaluate(trial)
                        rank=(score,failed_count(branch),branch['volume_cm3'])
                        trials.append(dict(operation=kind,detail=detail,oracle=oracle['status'],
                                           gap_m=oracle['gap_m'],counts=branch['counts']))
                        if best is None or rank<best[0]:best=(rank,branch)
                    except (RuntimeError,ValueError,AssertionError) as error:
                        trials.append(dict(operation=kind,error=str(error)))
            event=dict(phase='fixture_installation_translation',iteration=iteration+1,
                before_oracle=answer['status'],candidate_count=len(proposals),trials=trials,
                accepted=best is not None,world_axes='xyz',zero_floor_margin=True)
            self.record(event)
            if best is None:break
            current=best[1];m.commit(current)
            self.process_snapshot(current,event['phase'],event)
        return current

    def shortlist(self,proposals,targets):
        selected,count=super().shortlist(proposals,targets)
        kept=[]
        for row in selected:
            answer=self.model.base_oracle.query(row[4])
            row[5]['base_oracle_status']=answer['status']
            row[5]['base_enclosure_passed']=answer['enclosure_passed']
            if not self.model.oracle_gate_active or answer['enclosure_passed']:kept.append(row)
        return kept,count

    def collapse(self,layout,spacing,height,pattern=0,direction='up'):
        """Explicit multi-pose Juxtapose; all native task rotations preserved."""
        m=self.model;trial=layout.copy();host=layout.active[0];T=installed_state(m.native,trial,host)
        columns=int(np.ceil(np.sqrt(len(layout.active))))
        for order,k in enumerate(layout.active):
            trial.hosts[k]=host
            trial.placements[k]=np.linalg.inv(T)@m.native[k]
            if pattern==0:xy=np.array([order%columns,order//columns],float)
            elif pattern==1:xy=np.array([order,0.],float)
            else:
                angle=2*np.pi*order/len(layout.active);xy=np.array([np.cos(angle),np.sin(angle)])*max(1.,len(layout.active)/4)
            delta=np.r_[xy*m.extent*spacing,height]
            trial.placements[k,:3,3]+=T[:3,:3].T@delta
            world=(m.native[layout.hosts[k],:3,:3]@layout.directions[k])
            if direction=='up':world=np.array([0.,0.,1.])
            if direction=='blend':world=world+np.array([0.,0.,1.])
            trial.directions[k]=legal_direction(T[:3,:3].T@world,m.floor_normal(trial,k))
        return trial

    def oracle_round(self,current,revision=0):
        m=self.model;old=current['layout'];candidates=[];seen=set()
        # Pose placements are structural samples; continuous force refinement
        # uses the original direction/XYZ helper functions afterward.
        scales=([.33,.67,1.,1.33] if revision==0 else [.25,.5,.85,1.15,1.5,2.])
        heights=np.array([1.,2.,4.,6.,8.,12.])*m.extent
        if revision:heights=np.r_[heights,16*m.extent,24*m.extent]
        patterns=[0] if revision==0 else [0,1,2]
        for pattern in patterns:
            for spacing in scales:
                for height in heights:
                    trial=self.collapse(old,spacing,float(height),pattern)
                    answer=m.base_oracle.query(trial)
                    self.oracle_trace.append(dict(operation='juxtapose-cohort+xyz-sample',pattern=pattern,
                        spacing_body_fraction=spacing,height_m=float(height),status=answer['status'],
                        enclosure_passed=answer['enclosure_passed'],gap_m=answer['gap_m']))
                    if answer['enclosure_passed']:
                        # Find smaller feasible heights by actual bisection
                        # between this witness and its preceding sampled height.
                        index=int(np.flatnonzero(heights==height)[0]);lower=float(heights[index-1]) if index else 0.
                        upper=float(height)
                        for _ in range(3):
                            mid=(lower+upper)/2;probe=self.collapse(old,spacing,mid,pattern)
                            if m.base_oracle.query(probe)['enclosure_passed']:upper=mid;trial=probe
                            else:lower=mid
                        for direction in ['up','native','blend']:
                            candidate=self.collapse(old,spacing,upper,pattern,direction)
                            if candidate.key() in seen:continue
                            seen.add(candidate.key());value=m.proxy(candidate)
                            if not m.base_oracle.query(candidate)['enclosure_passed']:continue
                            # Projection roundoff must not select a larger seat
                            # arrangement when both demand losses are zero.
                            candidates.append((value['loss'],spacing,float(upper),pattern,direction,candidate))
                        break
        candidates.sort(key=lambda r:candidate_rank(r[0],self.layout_envelope_cm3(r[5]),r[1],r[2]))
        m.oracle_gate_active=True
        best=current;accepted=None
        for _,spacing,height,pattern,direction,layout in candidates[:(6 if revision==0 else 15)]:
            try:
                branch=m.evaluate(layout);m.commit(branch)
                self.process_snapshot(branch,'oracle_juxtapose_cohort',dict(spacing=spacing,height_m=height,
                    pattern=pattern,direction=direction,oracle=branch['base_oracle']['status']))
                if failed_count(branch):
                    movable=[k for k in layout.active if not registered(layout,k)]
                    branch=self.refine(branch,4,translation_guests=movable,phase='oracle_gated_whole_gradient')
                    if failed_count(branch)<=512 and failed_count(branch):branch=self.fine_refine(branch,3)
                if failed_count(branch)<failed_count(best) or not best['base_oracle']['enclosure_passed']:best=branch
                if not failed_count(branch):
                    oracle=m.base_oracle.query(branch['layout'],connected=True)
                    branch['base_oracle']=oracle
                    if oracle['passed']:
                        accepted=branch;break
            except (RuntimeError,ValueError,AssertionError) as error:
                self.oracle_trace.append(dict(operation='oracle-finalist',spacing=spacing,height_m=height,error=str(error)))
        m.oracle_gate_active=False
        return accepted or best

    def oracle_layout_proposals(self,current,revision=0):
        """Original cohort patterns as candidates, without committing a winner."""
        rows=[];scales=([.33,.67,1.,1.33] if not revision else [.25,.5,.85,1.15,1.5,2.])
        heights=([1.,2.,4.,6.,8.,12.] if not revision else [1.,2.,4.,6.,8.,12.,16.,24.])
        for pattern in ([0] if not revision else [0,1,2]):
            for spacing in scales:
                for height in heights:
                    for direction in ['up','native','blend']:
                        trial=self.collapse(current['layout'],spacing,height*self.model.extent,pattern,direction)
                        rows.append(('oracle-cohort-sample',trial,dict(pattern=pattern,
                            spacing_body_fraction=spacing,height_m=height*self.model.extent,
                            direction=direction)))
        return rows

    def solve_frontier(self,current,anchor=None,insertion_guest=None):
        if getattr(self,'search_policy','frontier')=='population_soft':
            from population_soft import PopulationSoftSearch
            return PopulationSoftSearch(self).solve(current)
        if getattr(self,'search_policy','frontier')=='population32':
            from population32 import Population32Search
            return Population32Search(self).solve(current)
        if getattr(self,'search_policy','frontier')=='material':
            from material_search import MaterialSearch
            return MaterialSearch(self).solve(current)
        if self.frontier_width==1:
            return self.solve_single_frontier(current,anchor,insertion_guest)
        from layout_frontier import LayoutFrontier
        self.oracle_trace=[]
        return LayoutFrontier(self).solve(current,anchor)

    def solve_single_frontier(self,current,anchor=None,insertion_guest=None):
        self.oracle_trace=[]
        # Keep the old local operations first. The oracle records every evaluated
        # candidate; failed intermediate states remain search states, never PASS.
        if not failed_count(current):
            answer=self.model.base_oracle.query(current['layout'],connected=True)
            current['base_oracle']=answer
            if answer['passed']:return current
        current=self.refine_fixture_states(current)
        oracle=current['base_oracle']
        # Direction changes neither installed floors nor working exclusions.
        # A floor certificate / work-only rejection makes force-only local
        # optimization a dead end for base feasibility in this search model.
        if oracle.get('work_only_enclosure_passed',False):
            current=super().solve_frontier(current,anchor,insertion_guest)
        else:
            self.record(dict(phase='oracle_route_to_layout_sampling',reason=oracle['status'],
                working_exclusions_or_floors_require_position_or_state_change=True))
        if not failed_count(current):
            answer=self.model.base_oracle.query(current['layout'],connected=True)
            current['base_oracle']=answer
            if answer['passed']:return current
        for revision in range(2):
            current=self.oracle_round(current,revision)
            save(self.out/'oracle_sampling.json',self.oracle_trace)
            if not failed_count(current) and current['base_oracle']['passed']:break
        return current

    def restore_reuse(self,current):
        self.model.oracle_gate_active=bool(current['base_oracle']['passed'])
        try:
            trial=super().restore_reuse(current)
            if current['base_oracle']['passed'] and not self.model.base_oracle.query(trial['layout'],connected=True)['passed']:
                self.model.commit(current);return current
            return trial
        finally:self.model.oracle_gate_active=False

    def polish_volume(self,current,rounds=2):
        self.model.oracle_gate_active=bool(current['base_oracle']['passed'])
        try:
            trial=super().polish_volume(current,rounds)
            if current['base_oracle']['passed'] and not self.model.base_oracle.query(trial['layout'],connected=True)['passed']:
                self.model.commit(current);return current
            return trial
        finally:self.model.oracle_gate_active=False

    def save(self,current,mode,began,initialization,**extra):
        selected=getattr(self,'material_selected',None)
        if selected is not None and selected.layout.key()==current['layout'].key():
            # Preserve the exact accepted evidence after LRU eviction. Saving
            # must not launch another geometry decision or lose its witness.
            answer=selected.oracle
        else:
            answer=self.model.base_oracle.query(current['layout'],connected=not failed_count(current))
        oracle_passed=bool(answer.get('connected_passed',False) and answer['passed'])
        current['base_oracle']=answer
        witness=(selected.witness if selected is not None else None)
        if witness is None:
            witness=getattr(self,'frontier_witnesses',{}).get(current['layout'].key())
        if oracle_passed and witness is not None:
            self.model.base_oracle.witness=current['layout'].key(),witness
        if selected is not None and selected.nominal is not None:
            self.model.material_nominal_export=(selected.layout.key(),selected.nominal)
        report=super().save(current,mode,began,initialization,**extra)
        if getattr(self.model,'require_connected_core',False):
            report.update(connected_contact_core=current['connected_contact_core'],
                force_acceptance_geometry='support.obj',
                step4_contact_core_connected=True,connection_material_added_to_core=False)
        report.update(passed=report['force_passed'] and oracle_passed,base_oracle_passed=oracle_passed,
            base_oracle=answer,acceptance='original_search_force_demands_AND_base_enclosure_AND_common_connection',
            base_oracle_seconds=self.model.base_oracle.seconds,base_oracle_queries=self.model.base_oracle.queries,
            base_oracle_component_queries=self.model.base_oracle.component_queries,
            rotation_transfer_constrained=False,base_witness_is_optimized_base=False,
            support_contact_floor_clearance_m=self.model.support_floor_clearance_m,
            support_contact_floor_scope='every final fixture state',
            native_pose_xyz_translation_allowed=True)
        report.update(fixture_state_xyz_translation_allowed=True,
            fixture_state_offsets_world_m=current['layout'].state_offsets.tolist(),
            oracle_foot_clearance_m=self.model.base_oracle.foot_width)
        report.update(frontier_width=self.frontier_width,
            frontier_search=getattr(self,'frontier_statistics',None),
            oracle_work_access_mode=self.model.base_oracle.work_mode)
        if oracle_passed and getattr(self.model.base_oracle,'witness',(None,None))[0]==current['layout'].key():
            self.model.base_oracle.witness[1].export(self.out/'oracle_connected_witness.obj')
        if selected is not None:
            from material_evaluator import membership,target_gap,mixed_target_applicable
            reuse,jux=membership(current['layout'])
            completed=bool(selected.completed and selected.layout.key()==current['layout'].key())
            applicable=mixed_target_applicable(current['layout'])
            if self.search_policy=='population_soft':
                applicable=False
            report.update(solid_constructed_during_search=completed,
                mixed_target_applicable=applicable,
                mixed_target_satisfied=not target_gap(current['layout']) if applicable else None,
                target_rotating_reuse_range=[2,4] if applicable else None,
                target_juxtaposed_range=[2,4] if applicable else None,
                physical_force_space_passed=report['passed'])
            if self.search_policy=='population_soft':
                report.update(sampling_count_center=len(current['layout'].active)/3.,
                    count_penalty_used=False,juxtaposed_fraction=jux/len(current['layout'].active))
            report['passed']=bool(report['passed'] and completed)
            if completed:
                from co_common import D,I
                D.export_exact_obj(selected.solid,self.out/'support_with_base.obj')
                fields=['layout.npz','support.obj','oracle_connected_witness.obj','support_with_base.obj']
                prebuilt=dict(complete=True,construction=selected.construction,
                    evaluation=selected.evaluation,fixture_weight_ratio=0.,
                    support='support_with_base.obj',layout_changed=False,force_solver_run=False,
                    source_sha256={name:I.sha256(self.out/name) for name in fields})
                save(self.out/'preconstructed_base.json',prebuilt)
                report['preconstructed_base']='preconstructed_base.json'
                report['artifacts'].update({name:I.sha256(self.out/name) for name in
                    ['support_with_base.obj','preconstructed_base.json','oracle_connected_witness.obj']})
            elif selected.solid is not None:
                from co_common import D,I
                D.export_exact_obj(selected.solid,self.out/'diagnostic_material_support.obj')
                report['diagnostic_material_support']='diagnostic_material_support.obj'
                report['diagnostic_construction']=selected.construction
                if current.get('material_contact_certification_run'):
                    report['force_acceptance_geometry']='diagnostic_material_support.obj'
                report['artifacts']['diagnostic_material_support.obj']=I.sha256(self.out/'diagnostic_material_support.obj')
        save(self.out/'base_oracle.json',answer);save(self.out/'report.json',report)
        return report
