"""Joint and insertion search with atomic Juxtapose/refinement branches."""
import time
from common import *
from model import Layout
from cohort import candidates as cohort_candidates


def failed_count(result):
    return sum(len(mask)-int(mask.sum()) for mask in result['masks'].values())


def lost_protected(anchor, result):
    if anchor is None:
        return 0
    return sum(int(np.count_nonzero(mask & ~result['masks'][k]))
               for k,mask in anchor['masks'].items() if k in result['masks'])


def rank(result, anchor=None):
    return (lost_protected(anchor,result),failed_count(result),
            result['maximum_projected_footprint_m2'],result['volume_cm3'])


class Search:
    def __init__(self,model,out,iterations=8,finalists=3,branch_rounds=3,seed=42):
        self.model,self.out=model,out
        self.iterations,self.finalists,self.branch_rounds=iterations,finalists,branch_rounds
        self.rng=np.random.default_rng(seed)
        self.events=[]

    def record(self,row):
        self.events.append(row)
        C.save(self.out/'trace.json',self.events)

    def checkpoint(self,current,label):
        layout=current['layout']
        np.savez_compressed(self.out/'accepted_layout.npz',placements=layout.placements,
            directions=layout.directions,hosts=layout.hosts,active=np.array(layout.active))
        C.save(self.out/'accepted_layout.json',dict(label=label,counts=current['counts'],
            exact_serial=current['serial'],failed_load_count=failed_count(current)))

    def baseline(self,current):
        C.D.export_exact_obj(unpack_solid(current['remaining']),self.out/'initial_support.obj')
        C.save(self.out/'initial_metrics.json',dict(counts=current['counts'],
            volume_cm3=current['volume_cm3'],
            maximum_projected_footprint_m2=current['maximum_projected_footprint_m2'],
            already_feasible=failed_count(current)==0))

    def initialize_exact(self,layout,phase):
        trials=[layout]
        for axis in range(2):
            for degrees in [.25,-.25,1.,-1.]:
                trial=layout.copy()
                for k in layout.active:
                    tangent=tangent_frame(layout.directions[k])[:,axis]
                    trial.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(degrees))*tangent,
                                                       self.model.floor_normal(layout,k))
                trials.append(trial)
        for index,trial in enumerate(trials):
            try:
                result=self.model.exact(trial)
                if index:
                    self.record(dict(phase=phase,initial_direction_numerical_recovery=True,
                                     candidate_index=index,counts=result['counts']))
                return result
            except (RuntimeError,ValueError,AssertionError) as error:
                self.record(dict(phase=phase,initialization_unresolved=True,
                                 candidate_index=index,error=str(error),accepted=False))
        raise RuntimeError('All initialization direction candidates unresolved')

    def targets(self,current):
        if current.get('pending_geometry'):
            k=current['pending_pose']
            saved=self.model.tasks[k].targets
            index=int(np.argmax(np.linalg.norm(saved,axis=1)))
            worst=dict(pose_index=k,load_index=index,
                       proposal_seed_only=True,reason='New layout geometry unresolved; not an evaluated hardest demand')
            scan=[]
        else:
            worst,scan=self.model.hardest(current)
        if worst is None:
            return [],None,scan
        targets=[(worst['pose_index'],worst['load_index'])]
        # A small working set guides proposals; exact acceptance still checks
        # every original stored load. Include a failing load from other poses.
        for k,mask in current['masks'].items():
            bad=np.flatnonzero(~mask)
            if len(bad) and (k,int(bad[0])) not in targets:
                targets.append((k,int(bad[0])))
            # Original extreme demands guard other tasks during screening.
            # Their indices are selected from saved loads, never regenerated.
            saved=self.model.tasks[k].targets
            for axis in range(6):
                for i in [int(np.argmax(saved[:,axis])),int(np.argmin(saved[:,axis]))]:
                    if (k,i) not in targets:
                        targets.append((k,i))
        return targets,worst,scan

    def local_proposals(self,current,targets,worst,translation_guests=()):
        layout=current['layout'];m=self.model
        proposals=[]
        base=m.proxy(layout,targets)
        # Differentiate contact availability/reaction-cone guidance only.
        # Choose the target owner and all blockers for joint direction changes.
        gradients={};h=np.radians(.75)
        for k in layout.active:
            frame=tangent_frame(layout.directions[k]);g=np.zeros(2)
            for axis in range(2):
                values=[]
                for sign in [-1.,1.]:
                    trial=layout.copy()
                    trial.directions[k]=legal_direction(layout.directions[k]+sign*h*frame[:,axis],m.floor_normal(layout,k))
                    values.append(m.proxy(trial,targets)['loss'])
                g[axis]=(values[1]-values[0])/(2*h)
            gradients[k]=(frame,g)
        total=np.sqrt(sum(float(g@g) for frame,g in gradients.values()))
        if total>1e-12:
            for degrees in [.25,.5,1.,2.,4.,8.]:
                trial=layout.copy()
                for k,(frame,g) in gradients.items():
                    trial.directions[k]=legal_direction(layout.directions[k]-np.radians(degrees)*frame@g/total,m.floor_normal(layout,k))
                proposals.append(('direction-gradient-joint',trial,dict(step_degrees=degrees)))
        owners=[worst['pose_index']]+[k for k,mask in current['masks'].items() if not mask.all() and k!=worst['pose_index']]
        for k in owners[:3]:
            frame,g=gradients[k]
            vectors=[np.array([1.,0.]),np.array([-1.,0.]),np.array([0.,1.]),np.array([0.,-1.])]
            if np.linalg.norm(g)>1e-12:
                vectors.insert(0,-g/np.linalg.norm(g))
            for degrees in [.5,2.,5.]:
                for vector in vectors:
                    trial=layout.copy()
                    trial.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(degrees))*frame@vector,m.floor_normal(layout,k))
                    proposals.append(('direction',trial,dict(pose=m.poses[k],step_degrees=degrees)))
        # Coherent changes release contact patches locked by several similar
        # directions; a single-pose derivative can be zero in this situation.
        for axis in np.eye(3):
            for degrees in [-2.,2.,-5.,5.]:
                trial=layout.copy()
                for k in layout.active:
                    trial.directions[k]=legal_direction(layout.directions[k]+np.radians(degrees)*np.cross(axis,layout.directions[k]),m.floor_normal(layout,k))
                proposals.append(('direction-coherent',trial,dict(axis=axis.tolist(),step_degrees=degrees)))
        for k in translation_guests:
            frame=tangent_frame(m.floor_normal(layout,k));g=np.zeros(2);distance=m.extent/100
            for axis in range(2):
                values=[]
                for sign in [-1.,1.]:
                    trial=layout.copy();trial.placements[k,:3,3]+=sign*distance*frame[:,axis]
                    values.append(m.proxy(trial,targets)['loss'])
                g[axis]=(values[1]-values[0])/(2*distance)
            vectors=[np.array([1.,0.]),np.array([-1.,0.]),np.array([0.,1.]),np.array([0.,-1.])]
            if np.linalg.norm(g)>1e-12:
                vectors.insert(0,-g/np.linalg.norm(g))
            for fraction in [1/128,1/64,1/32,1/16,1/8]:
                for vector in vectors:
                    trial=layout.copy();trial.placements[k,:3,3]+=m.extent*fraction*frame@vector
                    proposals.append(('translation-gradient' if np.linalg.norm(g)>1e-12 and np.allclose(vector,vectors[0]) else 'translation',
                                      trial,dict(pose=m.poses[k],step_m=m.extent*fraction)))
        return proposals,base

    def shortlist(self,proposals,targets):
        scored=[];seen=set()
        for kind,layout,detail in proposals:
            key=layout.key()
            if key in seen or key in getattr(self.model,'unresolved_layouts',{}):
                continue
            seen.add(key)
            try:
                proxy=self.model.proxy(layout,targets)
                scored.append((proxy['loss'],proxy['sum_loss'],proxy['span_m'],kind,layout,detail,proxy))
            except (RuntimeError,ValueError) as error:
                continue
        scored.sort(key=lambda r:r[:3])
        # Reserve distinct operation/step candidates: one proxy favorite must
        # not consume the whole actual-geometry budget with duplicate scales.
        selected=[]
        def diversity(row):
            if row[3]=='juxtapose':
                return ('juxtapose',row[5].get('guest_index'),row[5].get('host'))
            if row[3]=='juxtapose-cohort':
                return ('juxtapose-cohort',row[5].get('host'),row[5].get('radius_fraction'))
            return (row[3].split('-')[0],row[5].get('step_m',row[5].get('step_degrees',0.)))
        # Reserve one actual evaluation per distinct operation before giving
        # extra slots to different step sizes of the proxy favorite.
        tools={row[3].split('-')[0] for row in scored}
        for tool in sorted(tools,key=lambda tool:min(r[0] for r in scored if r[3].split('-')[0]==tool)):
            row=next(r for r in scored if r[3].split('-')[0]==tool)
            selected.append(row)
            if len(selected)>=self.finalists:
                return selected,len(scored)
        selected_keys={row[4].key() for row in selected}
        used={diversity(row) for row in selected}
        for row in scored:
            if row[4].key() in selected_keys:
                continue
            group=diversity(row)
            if group in used:
                continue
            selected.append(row);used.add(group)
            if len(selected)>=self.finalists:
                break
        return selected,len(scored)

    def refine(self,current,rounds,anchor=None,translation_guests=(),phase='direction'):
        for iteration in range(rounds):
            if failed_count(current)==0:
                break
            targets,worst,scan=self.targets(current)
            proposals,base=self.local_proposals(current,targets,worst,translation_guests)
            selected,screened=self.shortlist(proposals,targets)
            row=dict(phase=phase,iteration=iteration+1,before_counts=current['counts'],
                     hardest=worst,hardest_scan=scan,proxy_before=base,screened_candidates=screened,trials=[])
            best=current
            for _,_,_,kind,layout,detail,proxy in selected:
                record=dict(operation=kind,detail=detail,proxy=proxy,accepted=False)
                try:
                    trial=self.model.exact(layout)
                    record.update(counts=trial['counts'],lost_protected_loads=lost_protected(anchor,trial),rank=rank(trial,anchor))
                    if rank(trial,anchor)<rank(best,anchor):
                        best=trial
                        record['eligible']=True
                except (RuntimeError,ValueError,AssertionError) as error:
                    record['error']=str(error)
                row['trials'].append(record)
                if failed_count(best)==0 and lost_protected(anchor,best)==0:
                    break
            accepted=best is not current
            if accepted:
                current=best
                self.checkpoint(current,phase)
                for record in row['trials']:
                    if record.get('rank')==rank(best,anchor):
                        record['accepted']=True
            row.update(accepted=accepted,after_counts=current['counts'])
            self.record(row)
            print('REFINE',phase,iteration+1,'accepted',accepted,'failed',failed_count(current),flush=True)
            if not accepted:
                break
        return current

    def juxtapose_proposals(self,current,guests,targets):
        m=self.model;layout=current['layout'];rows=[]
        for guest in guests:
            hosts=[k for k in layout.active if k!=guest]
            hosts.sort(key=lambda k:(not current['masks'][k].all(),-int(current['masks'][k].sum())))
            for host in hosts[:3]:
                base=m.juxtapose(layout,guest,host)
                if np.allclose(base.placements[guest],layout.placements[guest],atol=1e-10):
                    continue
                frame=tangent_frame(m.floor_normal(base,guest))
                # Native alignment and centroid alignment in the same floor
                # plane are direct overlapping placements, not packing.
                center_guest=C.transform_points(m.mesh.center_mass[None],base.placements[guest])[0]
                center_host=C.transform_points(m.mesh.center_mass[None],layout.placements[host])[0]
                alignment=frame @ (frame.T @ (center_host-center_guest))
                shifts=[np.zeros(3),alignment]
                for fraction in [1/8,1/3]:
                    for angle in np.arange(0,2*np.pi,np.pi/2):
                        shifts.append(alignment+m.extent*fraction*frame @ [np.cos(angle),np.sin(angle)])
                for shift in shifts:
                    trial=base.copy();trial.placements[guest,:3,3]+=shift
                    a=transform_mesh(m.mesh,trial.placements[guest]).bounds
                    b=transform_mesh(m.mesh,layout.placements[host]).bounds
                    intersection=np.maximum(0,np.minimum(a[1],b[1])-np.maximum(a[0],b[0]))
                    overlap=float(np.prod(intersection)/min(np.prod(a[1]-a[0]),np.prod(b[1]-b[0])))
                    if overlap<.03:
                        continue
                    alternatives={'host':trial.directions[guest],
                        'world-up':m.floor_normal(trial,guest),
                        'previous-world':m.native[trial.hosts[guest],:3,:3].T @
                            m.native[layout.hosts[guest],:3,:3] @ layout.directions[guest]}
                    for label,direction in alternatives.items():
                        alternative=trial.copy()
                        alternative.directions[guest]=legal_direction(direction,m.floor_normal(alternative,guest))
                        rows.append(('juxtapose',alternative,dict(guest=m.poses[guest],host=m.poses[host],guest_index=guest,
                                    horizontal_offset_m=shift.tolist(),body_bbox_overlap_fraction=overlap,direction_choice=label)))
        return rows

    def rescue(self,current,guests,anchor=None,secondary=True):
        targets,worst,scan=self.targets(current)
        proposals=self.juxtapose_proposals(current,guests,targets)
        selected,screened=self.shortlist(proposals,targets)
        row=dict(phase='juxtapose',guests=[self.model.poses[k] for k in guests],hardest=worst,
                 hardest_scan=scan,screened_candidates=screened,trials=[],before_counts=current['counts'])
        best=current
        for _,_,_,kind,layout,detail,proxy in selected:
            record=dict(operation=kind,detail=detail,proxy=proxy,accepted=False)
            try:
                branch=self.model.exact(layout)
                record['juxtapose_counts']=branch['counts']
                if failed_count(branch):
                    damaged=[] if anchor is None else sorted(
                        [k for k,mask in anchor['masks'].items() if np.any(mask & ~branch['masks'][k])],
                        key=lambda k:-int(np.count_nonzero(anchor['masks'][k] & ~branch['masks'][k])))
                    translate=tuple(dict.fromkeys([detail['guest_index']]+damaged[:2]))
                    branch=self.refine(branch,self.branch_rounds,anchor=anchor,
                                       translation_guests=translate,phase='post_juxtapose')
                    if secondary and lost_protected(anchor,branch):
                        broken=sorted([k for k,mask in anchor['masks'].items()
                                       if np.any(mask & ~branch['masks'][k])],
                                      key=lambda k:-int(np.count_nonzero(anchor['masks'][k] & ~branch['masks'][k])))
                        branch=self.rescue(branch,broken[:1],anchor=anchor,secondary=False)
                record.update(counts=branch['counts'],lost_protected_loads=lost_protected(anchor,branch),rank=rank(branch,anchor))
                if lost_protected(anchor,branch)==0 and rank(branch,anchor)<rank(best,anchor):
                    best=branch;record['eligible']=True
            except (RuntimeError,ValueError,AssertionError) as error:
                record['error']=str(error)
            row['trials'].append(record)
            if failed_count(best)==0 and lost_protected(anchor,best)==0:
                break
        accepted=best is not current
        if accepted:
            self.checkpoint(best,'juxtapose')
            for record in row['trials']:
                if record.get('rank')==rank(best,anchor):
                    record['accepted']=True
        row.update(accepted=accepted,after_counts=best['counts'])
        self.record(row)
        print('JUXTAPOSE accepted',accepted,'failed',failed_count(best),flush=True)
        return best

    def coordinated_rescue(self,current,anchor=None):
        """An overlapping multi-task Juxtapose bundle after single-task stalls."""
        targets,worst,scan=self.targets(current)
        ordered=sorted(current['layout'].active,
                       key=lambda k:(not current['masks'][k].all(),-int(current['masks'][k].sum())))
        hosts=list(dict.fromkeys([current['layout'].active[0]]+ordered))[:2]
        selected,screened=self.shortlist(cohort_candidates(self.model,current['layout'],hosts=hosts),targets)
        row=dict(phase='coordinated_juxtapose',hardest=worst,hardest_scan=scan,
                 before_counts=current['counts'],screened_candidates=screened,trials=[])
        best=current
        for _,_,_,kind,layout,detail,proxy in selected:
            record=dict(operation=kind,detail=detail,proxy=proxy)
            try:
                branch=self.model.exact(layout)
                record['juxtapose_counts']=branch['counts']
                if failed_count(branch):
                    guests=sorted([k for k,mask in branch['masks'].items() if not mask.all()],
                                  key=lambda k:int(branch['masks'][k].sum()))[:3]
                    branch=self.refine(branch,self.branch_rounds,anchor=anchor,
                                       translation_guests=guests,phase='post_coordinated_juxtapose')
                record.update(counts=branch['counts'],rank=rank(branch,anchor),
                              lost_protected_loads=lost_protected(anchor,branch))
                if lost_protected(anchor,branch)==0 and rank(branch,anchor)<rank(best,anchor):
                    best=branch;record['eligible']=True
            except (RuntimeError,ValueError,AssertionError) as error:
                record['error']=str(error)
            row['trials'].append(record)
            if failed_count(best)==0 and lost_protected(anchor,best)==0:
                break
        row.update(accepted=best is not current,after_counts=best['counts'])
        if best is not current:
            self.checkpoint(best,'coordinated_juxtapose')
        self.record(row)
        print('COORDINATED JUXTAPOSE accepted',best is not current,'failed',failed_count(best),flush=True)
        return best

    def joint(self):
        began=time.monotonic()
        layout,initialization=self.model.initial()
        current=self.initialize_exact(layout,'joint_initial');initial_counts=current['counts']
        self.baseline(current)
        self.checkpoint(current,'joint_initial')
        for iteration in range(self.iterations):
            if failed_count(current)==0:
                break
            before=current
            current=self.refine(current,2,anchor=current,phase='joint_direction')
            if failed_count(current)==0:
                break
            worst,_=self.model.hardest(current)
            guests=sorted([k for k,mask in current['masks'].items() if not mask.all()],
                          key=lambda k:(k!=worst['pose_index'],int(current['masks'][k].sum())))
            current=self.rescue(current,guests[:3],anchor=current)
            if failed_count(current):
                current=self.coordinated_rescue(current,anchor=current)
            if current is before:
                break
        return self.model.save(current,self.out,dict(strategy='joint',initialization=initialization,
                    initial_counts=initial_counts,seconds=time.monotonic()-began,events=self.events,
                    failure_scope='Search budget exhausted, not an infeasibility proof',separated_fallback_used=False))

    def incremental(self):
        began=time.monotonic()
        layout,initialization=self.model.initial(active=(0,))
        current=self.initialize_exact(layout,'incremental_initial')
        self.baseline(current)
        current=self.refine(current,self.iterations,phase='first_pose')
        committed=current if failed_count(current)==0 else None
        self.checkpoint(current,'incremental_initial')
        stages=[dict(added_pose=self.model.poses[0],counts=current['counts'],passed=failed_count(current)==0)]
        for guest in range(1,len(self.model.poses)):
            anchor=committed
            layout=current['layout'].copy();layout.active=tuple(range(guest+1))
            # Keep all existing optimized directions. Project their mean onto
            # the new task's legal hemisphere instead of reinitializing them.
            mean=layout.directions[list(current['layout'].active)].sum(axis=0)
            layout.directions[guest]=legal_direction(mean,self.model.floor_normal(layout,guest))
            try:
                trial=self.model.exact(layout)
                trial=self.refine(trial,2,anchor=anchor,phase='insert_direction')
            except (RuntimeError,ValueError,AssertionError) as error:
                # This is an UNEVALUATED frontier used only to propose a real
                # alternative. It can neither become a feasible checkpoint nor
                # be exported as a final evaluated support.
                self.record(dict(phase='insert_registered_unresolved',added_pose=self.model.poses[guest],
                                 error=str(error),accepted=False))
                trial=dict(current,layout=layout,pending_geometry=True,pending_pose=guest,
                    masks={**current['masks'],guest:np.zeros(len(self.model.tasks[guest].targets),bool)},
                    supplies={**current['supplies'],guest:self.model.floors[guest]},
                    counts={**current['counts'],str(guest):None})
            for iteration in range(self.iterations):
                if failed_count(trial)==0 and lost_protected(anchor,trial)==0:
                    break
                before=trial
                trial=self.rescue(trial,[guest],anchor=anchor)
                if failed_count(trial):
                    trial=self.coordinated_rescue(trial,anchor=anchor)
                if trial is before:
                    break
            # Preserve the unsatisfied state as an explicitly failed stage;
            # never claim that an insertion which breaks old loads succeeded.
            current=trial
            stages.append(dict(added_pose=self.model.poses[guest],counts=current['counts'],
                               passed=failed_count(current)==0 and lost_protected(anchor,current)==0,
                               geometry_evaluated=not current.get('pending_geometry',False),
                               lost_existing_passed_loads=lost_protected(anchor,current)))
            if stages[-1]['passed']:
                committed=current
            C.save(self.out/'insertion_stages.json',stages)
            print('INSERT',self.model.poses[guest],'failed',failed_count(current),flush=True)
        return self.model.save(current,self.out,dict(strategy='incremental',initialization=initialization,
                    insertion_stages=stages,seconds=time.monotonic()-began,events=self.events,
                    failure_scope='Search budget exhausted, not an infeasibility proof',separated_fallback_used=False))
