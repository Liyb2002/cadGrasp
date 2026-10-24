"""Co-optimize population logic with station-preserving layout proposals.

Same failure weights, 32 persistent slots, positive common demand measure,
.25/.25/.25 PSO and complete-solid stopping rule. Translation proposals couple
Q and T so U=TQ remains the computed robot work pose.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from .adapter import TaskModel
from whole_task.step1.common import write_json
from population_soft import PopulationSoftSearch, physical_rank, weighted_pose, action_probabilities
from material_search import freeze_population_measure, sample_id
from whole_search.search import failed_count
from whole_search.common import legal_direction, tangent_frame
from continuous_support.fast_gradient import turn
from fixture_state import installed_state


def lock_seats(model, layout):
    """Installations are the variables; all object world poses are fixed."""
    q=layout.copy()
    for k in q.active:
        host=int(q.hosts[k]);R=model.native[host,:3,:3]
        # Inverse(T) @ U is analytically identity for a registered seat.
        # Keep it exact: tiny artificial rotations create near-coincident
        # copies of one body and destabilize downstream solid Booleans.
        Q=np.eye(4)
        if not np.array_equal(model.native[k,:3,:3],R):
            Q[:3,:3]=R.T@model.native[k,:3,:3]
        Q[:3,3]=R.T@(model.native[k,:3,3]-model.native[host,:3,3]-q.state_offsets[host])
        q.placements[k]=Q
    model.check_station(q)
    return q


def shift_seat(model, layout, owner, delta_material):
    q=layout.copy();host=int(q.hosts[owner])
    R=model.native[host,:3,:3]
    # Moving the seat by delta in material coordinates requires the opposite
    # installation displacement to leave the actual robot task unchanged.
    q.state_offsets[host]-=R@np.asarray(delta_material)
    return lock_seats(model,q)


def rehost(model, layout, owner, host):
    q=layout.copy();old=int(q.hosts[owner])
    world_direction=model.native[old,:3,:3]@q.directions[owner]
    q.hosts[owner]=int(host)
    q=lock_seats(model,q)
    q.directions[owner]=legal_direction(model.native[host,:3,:3].T@world_direction,model.floor_normal(q,owner))
    return q


class StationSearch(PopulationSoftSearch):
    def initialize_population(self,current):
        start=self.checked(self.preview_cached(current['layout'],result=current))
        self.evaluated[start.layout.key()]=start
        self.slots=[start]*self.population_size
        self.pso_best=list(self.slots)
        self.certify(self.slots)
        write_json(self.search.out/'initial_population.json',dict(
            slots=self.population_size,identical_registered_start=True,
            predefined_juxtaposed_poses=[],warm_start=False))
        return start

    def discrete(self, layout, owner, weights):
        m=self.model;rng=self.search.rng
        probabilities=action_probabilities(layout,owner,weights)
        # Registered versus nonregistered identity is inherited; proposed
        # hosts are still selected dynamically, never assigned in advance.
        action=str(rng.choice(list(probabilities),p=list(probabilities.values())))
        hosts=[k for k in layout.active if k!=owner]
        host=owner if action=='restore' else int(rng.choice(hosts))
        q=rehost(m,layout,owner,host)
        if action=='role-exchange':
            candidates=[k for k in layout.active if k!=owner and q.hosts[k]!=k]
            if candidates:q=rehost(m,q,int(rng.choice(candidates)),owner)
        # Bounded seat displacement, accompanied by the matching installation.
        if action!='restore' and rng.random()<.5:
            vector=rng.normal(size=3);vector/=np.linalg.norm(vector)
            q=shift_seat(m,q,owner,vector*m.extent*float(rng.choice([1/16,1/8,1/4])))
        q.directions[owner]=legal_direction(q.directions[owner],m.floor_normal(q,owner))
        return q,dict(action=action,owner=owner,host=int(q.hosts[owner]),object_world_pose_locked=True)

    def pso(self,layout,personal,social,velocity,owner):
        m=self.model;rng=self.search.rng;c=m.mesh.center_mass
        def center(q):return q.placements[owner,:3,:3]@c+q.placements[owner,:3,3]
        current,pbest,gbest=map(center,(layout,personal,social))
        velocity[owner]=.25*velocity[owner]+.25*rng.random(3)*(pbest-current)+.25*rng.random(3)*(gbest-current)
        q=shift_seat(m,layout,owner,velocity[owner])
        if rng.random()<.35:
            axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
            q=turn(m,q,owner,np.radians(12)*axis)
        return q,dict(owner=owner,velocity_m=velocity[owner].tolist(),coefficients=[.25,.25,.25],object_world_pose_locked=True)

    def continuation_rows(self,parent,round_index):
        """Bounded all-owner direction and coupled seat/install adjustments."""
        m=self.model;q=parent.layout
        weights,_=self.weights(parent)
        owners=sorted(q.active,key=lambda k:-weights[k])[:2]
        rows=[]
        for owner in owners:
            frame=tangent_frame(q.directions[owner])
            for degrees in (2.,8.):
                for axis in range(2):
                    for sign in (-1,1):
                        rows.append(('direction',turn(m,q,owner,sign*np.radians(degrees)*frame[:,axis]),
                                     dict(owner=owner,degrees=sign*degrees,axis=axis)))
            for axis in range(3):
                for sign in (-1,1):
                    delta=sign*m.extent/32*np.eye(3)[axis]
                    rows.append(('coupled-seat-installation',shift_seat(m,q,owner,delta),dict(owner=owner,delta_m=delta.tolist())))
        return rows

    def checked_adjustment(self,raw,iteration,slot,parent=None):
        refined=self.checked(self.refine_preview(raw,iteration,maximum=1))
        return min([raw,refined],key=lambda x:physical_rank(x,slot))

    def solve(self,current):
        """Port of PopulationSoftSearch.solve with the above legal proposals."""
        s,m=self.search,self.model
        start=self.initialize_population(current)
        for iteration in range(s.iterations):
            if self.incumbent is not None:break
            reference=min(self.slots,key=lambda x:failed_count(x.result)).result
            with m.hold_measure(reference):
                freeze_population_measure(m,self.slots)
                for item in self.slots+self.pso_best:item.loss=float(m.proxy(item.layout)['loss'])
                for slot,item in enumerate(self.slots):self.update_pso_best(slot,item)
                personal=[x.layout.copy() for x in self.pso_best]
                social=[min(self.pso_best,key=lambda x:physical_rank(x,p)).layout.copy() for p in (0,1)]
                events=[]
                for slot,parent in enumerate(list(self.slots)):
                    self.active_slot=slot;self.protected=()
                    weights,source=self.weights(parent)
                    owner=weighted_pose(parent.layout.active,weights,s.rng)
                    # A direction-only branch remains available at a registered
                    # seat; it is valuable when station locking removes free
                    # task translations available in the reference problem.
                    branch='hold' if owner is None else str(s.rng.choice(['discrete','direction','pso'],p=[.4,.3,.3]))
                    event=dict(slot=slot,branch=branch,owner=owner,accepted=False,weight_source=source)
                    try:
                        if branch!='hold':
                            if branch=='discrete':q,detail=self.discrete(parent.layout,owner,weights)
                            elif branch=='pso':q,detail=self.pso(parent.layout,personal[slot],social[slot%2],self.pso_velocity[slot],owner)
                            else:
                                axis=s.rng.normal(size=3);axis/=np.linalg.norm(axis)
                                q=turn(m,parent.layout,owner,np.radians(12)*axis);detail=dict(owner=owner,degrees=12)
                            raw=self.checked(self.preview_cached(q,branch,sample_id(parent.layout),detail))
                            trial=self.checked_adjustment(raw,iteration,slot,parent)
                            better=physical_rank(trial,slot)<physical_rank(parent,slot)
                            exploratory=branch=='discrete' and not parent.completed and s.rng.random()<.1
                            if better or exploratory:
                                self.slots[slot]=trial;self.accepted+=1
                                self.exploration_accepted+=int(not better)
                                if branch=='discrete':self.pso_velocity[slot]=0.
                                event['accepted']=True
                            self.update_pso_best(slot,self.slots[slot])
                            event.update(move=detail,counts=self.slots[slot].result['counts'])
                    except (ValueError,RuntimeError,AssertionError) as error:
                        event['error']=str(error);self.errors.append(dict(stage='slot',iteration=iteration+1,slot=slot,error=str(error)))
                    events.append(event)
                    if slot%4==3:
                        print('Step4.2 round',iteration+1,'slot',slot+1,'best counts',self.best_checked(start).result['counts'],flush=True)
                    # Run the inherited connected-space/full-solid gate at
                    # its usual finalist boundary, stopping at the first pass.
                    if self.evaluator.space_candidate(self.slots[slot]):self.certify([self.slots[slot]],connections=1,constructions=1)
                    if self.incumbent is not None:break
                self.certify(self.slots);self.propagate_evidence()
                best=self.best_checked(start)
                row=dict(iteration=iteration+1,events=events,best_counts=best.result['counts'],completed=best.completed,
                         statistics=dict(self.evaluator.stats))
                self.audit.append(row);write_json(s.out/'population_trace.json',self.audit)
                m.commit(best.result)
                s.process_snapshot(best.result,'population',dict(iteration=iteration+1))
                print('Step4.2 ROUND',iteration+1,'counts',best.result['counts'],'complete',best.completed,flush=True)
        selected=self.best_checked(start)
        self.search.material_selected=selected
        write_json(s.out/'search_summary.json',dict(population_size=self.population_size,rounds=len(self.audit),
            configured_rounds=s.iterations,accepted_updates=self.accepted,exploratory_updates=self.exploration_accepted,
            statistics=dict(self.evaluator.stats),errors=self.errors[-50:],fixed_robot_task_poses=True,
            stopped_on_first_complete_solid=self.incumbent is not None))
        return selected


def context(model,out,iterations=12,population=32,seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def snapshot(result,phase,details):
        np.savez_compressed(out/'latest_layout.npz',**result['layout'].arrays(model.native))
        write_json(out/'latest.json',dict(phase=phase,counts=result['counts'],**details))
    return SimpleNamespace(model=model,out=out,iterations=iterations,frontier_width=population,rng=np.random.default_rng(seed),
        allow_oracle_fallback=True,allow_coupled_adjustment=True,allow_group_seating=True,
        allow_joint_reseating=False,allow_placement_guidance=True,allow_height_repair=False,
        stop_when_feasible=True,allow_native_sliding=True,allow_geometry_validation=True,
        population_continuous_policy='pso',record=lambda row:None,process_snapshot=snapshot)
