"""Failure-weighted poses and a soft count bias in exactly 32 slots."""
import numpy as np

from population32 import registered_population, centered_seat, POPULATION_SIZE
from material_search import MaterialSearch, freeze_population_measure, sample_id
from whole_search.reuse_first import registered
from whole_search.search import failed_count
from whole_search.common import legal_direction
from continuous_support.fast_gradient import turn
from translation.layout_update import shift
from fixture_state import installed_state


def failure_weights(layout, result):
    if result is None or result['layout'].key() != layout.key():
        raise ValueError('Pose probabilities require full masks for this exact layout')
    return {k: float(np.count_nonzero(~np.asarray(result['masks'][k], dtype=bool))) /
            len(result['masks'][k]) for k in layout.active}


def weighted_pose(indices, weights, rng):
    choices = [int(k) for k in indices if weights.get(k, 0.) > 0.]
    if not choices:
        return None
    probabilities = np.array([weights[k] for k in choices], dtype=float)
    probabilities /= probabilities.sum()
    return int(rng.choice(choices, p=probabilities))


def problem_weights(model, item, allow_oracle_fallback=False):
    weights = failure_weights(item.layout, item.result)
    force_failed = any(weights.values())
    if force_failed and not allow_oracle_fallback:
        return weights, 'original_force_failure_fraction'
    oracle = item.oracle
    if item.completed or not allow_oracle_fallback or oracle.get('passed'):
        return weights, 'original_force_failure_fraction' if force_failed else 'no_failed_pose'
    if oracle.get('enclosure_passed') and oracle.get('status') == 'enclosure_witness':
        return weights, 'original_force_failure_fraction' if force_failed else 'awaiting_connection_certification'
    names = {model.poses[k]: k for k in item.layout.active}
    oracle_weights = dict.fromkeys(item.layout.active, 0.)
    for row in oracle.get('certificates', []):
        for field in ['pose', 'other_pose']:
            if row.get(field) in names:
                oracle_weights[names[row[field]]] += max(0., float(row.get('gap_m', 0.)))
    for row in oracle.get('component_per_pose', []) or oracle.get('per_pose', []):
        if not row.get('passed') and row.get('pose') in names:
            oracle_weights[names[row['pose']]] += max(0., float(row.get('gap_m', 0.)))
    if any(oracle_weights.values()):
        if force_failed:
            scale=max(float(model.extent),1e-12)
            return {k:weights[k]+oracle_weights[k]/scale for k in weights}, 'original_force_plus_oracle_failure_gap'
        return oracle_weights, 'oracle_failure_gap_after_all_force_passed'
    if force_failed:
        return weights, 'original_force_failure_fraction'
    return {k: 1. for k in item.layout.active}, 'unattributed_oracle_failure_after_all_force_passed'


def action_probabilities(layout, owner, weights):
    guests = [k for k in layout.active if not registered(layout, k)]
    reuse = [k for k in layout.active if registered(layout, k)]
    deficit = np.clip(len(layout.active)/3. - len(guests), -3., 3.)
    if owner in reuse:
        values = dict(create=float(np.exp(deficit)))
        if any(weights.get(k, 0.) > 0 for k in guests):
            values['role-exchange'] = 2.
    else:
        values = dict(restore=float(np.exp(-deficit)), reseat=1.)
        if any(weights.get(k, 0.) > 0 for k in reuse):
            values['role-exchange'] = 2.
    total = sum(values.values())
    return {name: value/total for name, value in values.items()}


def replace_guest(model, layout, outgoing, incoming):
    """Try the old seat first, with native rotations; never invent a guest."""
    old_host = int(layout.hosts[outgoing])
    T = installed_state(model.native, layout, old_host)
    old_U = T @ layout.placements[outgoing]
    incoming_T = installed_state(model.native, layout, int(layout.hosts[incoming]))
    incoming_U = incoming_T @ layout.placements[incoming]
    c = np.asarray(model.mesh.center_mass)
    target = incoming_U[:3, :3] @ c + incoming_U[:3, 3]
    target[:2] = (old_U[:3, :3] @ c + old_U[:3, 3])[:2]
    world_direction = incoming_T[:3, :3] @ layout.directions[incoming]
    q = model.restore_registered(layout, outgoing)
    T = installed_state(model.native, q, old_host)
    R = T[:3, :3].T @ model.native[incoming, :3, :3]
    q.hosts[incoming] = old_host
    Q = np.eye(4); Q[:3, :3] = R
    Q[:3, 3] = T[:3, :3].T @ (target-T[:3, 3]) - R @ c
    q.placements[incoming] = Q
    q.directions[incoming] = legal_direction(T[:3, :3].T @ world_direction,
                                             model.floor_normal(q, incoming))
    q = shift(model, q, incoming, np.zeros(3))
    retained_seat = not registered(q, incoming)
    if not retained_seat:
        # Replacing the old anchor in its own registered seat is not a swap.
        q = centered_seat(model, model.restore_registered(layout, outgoing), incoming, outgoing)
    for k in q.active:
        model.validate_placement(q, k)
    return q, retained_seat


def carry_installation(model, before, moved, owner,exclude=()):
    """Move the owner's current cohort so the old installation really disappears."""
    old_host=int(before.hosts[owner]);new_host=int(moved.hosts[owner])
    if old_host==new_host:return moved
    old_T=installed_state(model.native,before,old_host)
    new_T=installed_state(model.native,moved,new_host)
    old_U=old_T@before.placements[owner];new_U=new_T@moved.placements[owner]
    delta=new_U[:3,3]-old_U[:3,3];q=moved.copy()
    for k in before.active:
        if k==owner or k in exclude or int(before.hosts[k])!=old_host:continue
        U=old_T@before.placements[k];U[:3,3]+=delta
        q.hosts[k]=new_host;q.placements[k]=np.linalg.inv(new_T)@U
        q.directions[k]=legal_direction(new_T[:3,:3].T@old_T[:3,:3]@before.directions[k],model.floor_normal(q,k))
        q=shift(model,q,k,np.zeros(3));model.validate_placement(q,k)
    return q


def native_slide(model,layout,owner,anchor,rng):
    """Change the seat while retaining this task's native installation rotation."""
    q=model.restore_registered(layout,owner);c=np.asarray(model.mesh.center_mass)
    T=installed_state(model.native,q,owner);U=T@q.placements[owner]
    A=installed_state(model.native,layout,int(layout.hosts[anchor]))@layout.placements[anchor]
    delta=np.zeros(3);delta[:2]=(A[:3,:3]@c+A[:3,3]-U[:3,:3]@c-U[:3,3])[:2]
    q=shift(model,q,owner,T[:3,:3].T@delta)
    if registered(q,owner):
        angle=float(rng.uniform(0,2*np.pi));delta=model.extent/16*np.array([np.cos(angle),np.sin(angle),0.])
        q=shift(model,q,owner,T[:3,:3].T@delta)
    model.validate_placement(q,owner)
    return q


def joint_discrete_move(model,layout,weights,rng,owner,native_family=False):
    """Choose a complete role change during optimization, using physical floor certificates."""
    active=list(layout.active);q=layout.copy()
    for k in active:
        world=installed_state(model.native,layout,int(layout.hosts[k]))[:3,:3]@layout.directions[k]
        q.hosts[k]=k;q.placements[k]=np.eye(4);q.state_offsets[k]=0.
        q.directions[k]=legal_direction(model.native[k,:3,:3].T@world,model.floor_normal(q,k))
    number=min(len(active)-1,max(1 if all(registered(layout,k)for k in active)else 0,int(rng.binomial(len(active),1/3))))
    guests=[owner]if number else[]
    available=[k for k in active if k!=owner]
    maximum=max(weights.values(),default=0.)
    for _ in range(max(0,number-1)):
        extra=weighted_pose(available,{k:weights[k]+.1*max(maximum,1e-12)for k in available},rng)
        guests.append(extra);available.remove(extra)
    reuse=[k for k in active if k not in guests];prior=getattr(model,'registered_floor_prior',None)
    assessment={};removed=[]
    if prior is not None:
        while len(reuse)>1:
            assessment=prior.assess(reuse)
            if assessment.get('status')!='certified_infeasible':break
            rows=assessment['certificate']['rows']
            conflict=list(set(r['owner']for r in rows)|set(r['floor']for r in rows))
            choices=[k for k in reuse if k in conflict]or reuse
            selected=weighted_pose(choices,{k:weights[k]+.1*max(maximum,1e-12)for k in choices},rng)
            reuse.remove(selected);guests.append(selected);removed.append(selected)
        assessment=prior.assess(reuse)
        if assessment.get('status')=='necessary_feasible':q.state_offsets[reuse,2]=assessment['heights_m']
    for k in guests:
        anchor=int(rng.choice(reuse))
        q=native_slide(model,q,k,anchor,rng)if native_family else centered_seat(model,q,k,anchor)
    for k in active:model.validate_placement(q,k)
    return q,dict(action='joint-reseat',selected_pose=owner,poses=guests,registered_indices=reuse,
        floor_certificate_removed=removed,initial_sampled_juxtaposed_count=number,native_state_sliding=native_family,
        initialization='world_xy_center_of_mass_alignment',contact_matching_run=False)


def align_guest_centers(model,layout):
    q=layout.copy();c=np.asarray(model.mesh.center_mass)
    for k in q.active:
        if registered(q,k):continue
        host=int(q.hosts[k]);members=[j for j in q.active if int(q.hosts[j])==host and j!=k]
        if not members:continue
        anchor=next((j for j in members if registered(q,j)),members[0])
        target=q.placements[anchor,:3,:3]@c+q.placements[anchor,:3,3]
        q.placements[k,:3,3]=target-q.placements[k,:3,:3]@c
        q=shift(model,q,k,np.zeros(3));model.validate_placement(q,k)
    return q


def discrete_move(model, layout, weights, rng, owner=None, carry_group=False,allow_native_slide=False):
    owner = weighted_pose(layout.active, weights, rng) if owner is None else owner
    if owner is None:
        return None, dict(action='hold', reason='no_failed_pose')
    probabilities = action_probabilities(layout, owner, weights)
    if carry_group and any(int(layout.hosts[k])!=int(layout.hosts[owner])for k in layout.active):
        probabilities['merge-installations']=1.
    if allow_native_slide:probabilities['native-slide']=1.
    total=sum(probabilities.values());probabilities={k:v/total for k,v in probabilities.items()}
    names = list(probabilities)
    action = str(rng.choice(names, p=[probabilities[name] for name in names]))
    detail = dict(action=action, selected_pose=owner, action_probabilities=probabilities)
    if action=='native-slide':
        anchor=int(rng.choice([k for k in layout.active if k!=owner]))
        return native_slide(model,layout,owner,anchor,rng),dict(detail,pose=owner,anchor=anchor,native_state_preserved=True)
    if action == 'restore':
        return model.restore_registered(layout, owner), dict(detail, pose=owner)
    if action == 'role-exchange':
        if registered(layout, owner):
            incoming = owner
            outgoing = weighted_pose([k for k in layout.active if not registered(layout, k)], weights, rng)
        else:
            outgoing = owner
            incoming = weighted_pose([k for k in layout.active if registered(layout, k)], weights, rng)
        q, kept = replace_guest(model, layout, outgoing, incoming)
        if carry_group:q=carry_installation(model,layout,q,incoming,exclude=(outgoing,))
        return q, dict(detail, pose=incoming, restored=outgoing, retained_previous_seat=kept)
    anchors=[k for k in layout.active if k!=owner and (action!='merge-installations' or int(layout.hosts[k])!=int(layout.hosts[owner]))]
    anchor = int(rng.choice(anchors))
    q=centered_seat(model, layout, owner, anchor)
    if carry_group and (action=='merge-installations' or registered(layout,owner)):
        q=carry_installation(model,layout,q,owner)
    return q, dict(
        detail, pose=owner, anchor=anchor, initialization='world_xy_center_of_mass_alignment',
        contact_matching_run=False)


def differential_move(model, layout, population, slot, weights, rng, iteration, owner=None):
    owner = (weighted_pose([k for k in layout.active if not registered(layout, k)], weights, rng)
             if owner is None else owner)
    if owner is None:
        return None, dict(action='hold', reason='no_failed_guest')
    donors = [i for i in range(len(population)) if i != slot]
    a, b = map(int, rng.choice(donors, size=2, replace=False))
    first, second = population[a], population[b]
    c = np.asarray(model.mesh.center_mass)
    ca = first.placements[owner, :3, :3] @ c + first.placements[owner, :3, 3]
    cb = second.placements[owner, :3, :3] @ c + second.placements[owner, :3, 3]
    noise = model.extent*(1/16 if iteration < 8 else 1/64)*rng.normal(size=3)
    q = shift(model, layout, owner, (.6 if iteration < 8 else .3)*(ca-cb)+noise)
    if rng.random() < .35:
        axis = rng.normal(size=3); axis /= max(float(np.linalg.norm(axis)), 1e-12)
        q.directions[owner] = turn(model, q, owner, np.radians(12.)*axis).directions[owner]
    for k in q.active:
        model.validate_placement(q, k)
    return q, dict(selected_pose=owner, poses=[owner], donor_slots=[a, b],
                   coordinate_frame='common_material', better_donors_required=False)


def preserves_poses(before, after, protected):
    for k in protected:
        if int(before.hosts[k]) != int(after.hosts[k]):
            return False
        for old, new in [(before.placements[k], after.placements[k]),
                         (before.directions[k], after.directions[k]),
                         (before.state_offsets[int(before.hosts[k])], after.state_offsets[int(after.hosts[k])])]:
            if not np.allclose(old, new, atol=1e-12, rtol=0.):
                return False
    return True


def physical_rank(item, slot=0):
    if item.completed:
        return (0, 0., 0., 0., 0., *item.cost)
    failures = failed_count(item.result) if item.result is not None else float('inf')
    space = (int(not item.oracle.get('enclosure_passed')), float(item.oracle.get('gap_m', 0.)))
    if slot % 2:
        return (1, *space, int(failures!=0), float(item.loss), failures, item.lower_litres)
    if failures==0:
        return (1,0,*space,float(item.loss),item.lower_litres)
    return (1,1,float(item.loss),failures,*space,item.lower_litres)


class PopulationSoftSearch(MaterialSearch):
    def __init__(self, search):
        if search.frontier_width < 3:
            raise ValueError('population_soft requires at least 3 persistent slots')
        self.population_size=search.frontier_width
        if (getattr(search, 'initialization_seeds', None) or
                getattr(search, 'material_warm_layouts', []) or
                getattr(search, 'warm_checkpoint_experiment', None)):
            raise ValueError('population_soft is a registered cold initialization')
        super().__init__(search)
        self.slots = []; self.audit = []; self.protected = ()
        self.allow_oracle_fallback = bool(getattr(search, 'allow_oracle_fallback', False))
        self.allow_coupled_adjustment = bool(getattr(search, 'allow_coupled_adjustment', False))
        self.allow_group_seating = bool(getattr(search, 'allow_group_seating', False))
        self.allow_joint_reseating = bool(getattr(search, 'allow_joint_reseating', False))
        self.allow_placement_guidance = bool(getattr(search, 'allow_placement_guidance', False))
        self.allow_height_repair = bool(getattr(search, 'allow_height_repair', False))
        self.stop_when_feasible = bool(getattr(search, 'stop_when_feasible', False))
        self.allow_native_sliding = bool(getattr(search, 'allow_native_sliding', False))
        self.allow_geometry_validation = bool(getattr(search, 'allow_geometry_validation', False))
        if self.allow_geometry_validation:self.evaluator.space_candidate=self.geometry_candidate
        self.branch_counts = dict(discrete=0, differential=0, hold=0)
        self.accepted = 0; self.exploration_accepted = 0

    def goal_gap(self, layout):
        return 0

    def choose_pool(self, items, width):
        selected = []; seen = set()
        for slot in range(width):
            available = [x for x in items if x.layout.key() not in seen]
            if not available:
                break
            item = min(available, key=lambda x: physical_rank(x, slot))
            selected.append(item); seen.add(item.layout.key())
        return selected

    def weights(self, item):
        if (self.allow_placement_guidance and not item.completed and item.construction is not None
                and not item.construction.get('passed') and not failed_count(item.result)):
            return dict.fromkeys(item.layout.active,1.),'construction_failure_structural_exploration'
        return problem_weights(self.model, item, self.allow_oracle_fallback)

    def geometry_candidate(self,item):
        if getattr(self.model,'require_connected_core',False):
            core=item.result.get('connected_contact_core',{})if item.result is not None else{}
            return (item.result is not None and failed_count(item.result)==0
                and core.get('connected',False)and core.get('downward_resultant_passed',False)
                and item.result.get('force_material_geometry_certified',False))
        # Eligibility only; construct() still verifies all original demands on
        # actual contact polygon vertices before it can mark a model complete.
        return (item.result is not None and item.result['layout'].key()==item.layout.key()
            and len(item.layout.active)==len(self.model.poses)
            and all(len(item.result['masks'][k])==32768 for k in item.layout.active))

    def certification_rank(self,item):
        if not self.allow_geometry_validation:return super().certification_rank(item)
        failures=failed_count(item.result)
        return (int(failures!=0),item.loss,failures,item.lower_litres,item.result['volume_cm3'])

    def adjustable_poses(self, parent):
        if self.allow_coupled_adjustment and not parent.completed and (self.allow_group_seating or not parent.oracle.get('passed')):
            return list(parent.layout.active)
        weights, _ = self.weights(parent)
        return [k for k in parent.layout.active if weights[k] > 0 and k not in self.protected]

    def adjustable_hosts(self, parent):
        active = set(self.adjustable_poses(parent)); q = parent.layout
        return [h for h in sorted(set(int(q.hosts[k]) for k in q.active))
                if all(k in active for k in q.active if int(q.hosts[k]) == h)]

    def continuation_rows(self, parent, round_index):
        # Keep the original multiple directions, signs and step sizes.
        rows=list(super().continuation_rows(parent, round_index))
        if getattr(self.model,'require_connected_core',False)and 0<failed_count(parent.result)<=512:
            from whole_search.common import tangent_frame
            from translation.layout_update import translation_frame
            owners=sorted(parent.layout.active,key=lambda k:-np.count_nonzero(~parent.result['masks'][k]))[:2]
            fine=[]
            for k in owners:
                if k in self.protected:continue
                angular=tangent_frame(parent.layout.directions[k])
                for degrees in [.03125,.125]:
                    for axis in range(2):
                        for sign in [-1.,1.]:
                            fine.append(('near-feasible-direction',turn(self.model,parent.layout,k,
                                sign*np.radians(degrees)*angular[:,axis]),dict(pose=k,degrees=sign*degrees)))
                if not registered(parent.layout,k):
                    frame=translation_frame(self.model,parent.layout,k)
                    for fraction in [1/2048,1/512]:
                        for axis in range(3):
                            for sign in [-1.,1.]:
                                fine.append(('near-feasible-translation',shift(self.model,parent.layout,k,
                                    sign*self.model.extent*fraction*frame[:,axis]),dict(pose=k,axis=axis,fraction=fraction,sign=sign)))
            rows=fine+rows
        if self.allow_group_seating and not parent.completed:
            from reuse_proposals import set_exit_directions
            from material_floor import project_floor_layout
            from translation.layout_update import translation_frame
            q=parent.layout;m=self.model
            closed=set_exit_directions(m,q,'close')
            rows=[('direction-global-close',closed,dict(all_pose_objective=True)),
                  ('direction-global-up',set_exit_directions(m,q,'up'),dict(all_pose_objective=True))]+rows
            projected=project_floor_layout(m,closed)
            if projected is not None:rows.append(('global-close-floor-projection',projected,dict(original_demands=True)))
            guests=[k for k in q.active if not registered(q,k)]
            if guests:
                owner=max(guests,key=lambda k:np.count_nonzero(~parent.result['masks'][k]))
                angle=float(self.search.rng.uniform(0,2*np.pi))
                for base in [q,closed]:
                    for fraction in [1/16,1/3,2/3]:
                        vector=m.extent*fraction*translation_frame(m,base,owner)[:,:2]@np.array([np.cos(angle),np.sin(angle)])
                        rows.append(('seat-separation',shift(m,base,owner,vector),dict(pose=owner,fraction=fraction,global_close=base is closed)))
                if self.allow_joint_reseating:
                    for family in range(3):
                        trial=closed.copy()if family%2==0 else set_exit_directions(m,q,'up')
                        for k in guests:
                            angle=float(self.search.rng.uniform(0,2*np.pi));fraction=float(self.search.rng.choice([1/16,1/3,2/3]))
                            vector=m.extent*fraction*translation_frame(m,trial,k)[:,:2]@np.array([np.cos(angle),np.sin(angle)])
                            trial=shift(m,trial,k,vector)
                        rows.append(('joint-seat-separation',trial,dict(poses=guests,family=family,all_pose_objective=True)))
                if self.allow_placement_guidance:
                    from reuse_proposals import assign_seat
                    aligned=align_guest_centers(m,q)
                    rows.append(('guest-centers-3d',aligned,dict(poses=guests,legal_floor_projection=True)))
                    rows.append(('guest-centers-3d-close',set_exit_directions(m,aligned,'close'),dict(poses=guests)))
                    for family in range(4):
                        trial=q.copy()
                        for order,k in enumerate(guests):
                            members=[j for j in q.active if int(q.hosts[j])==int(q.hosts[k])and j!=k]
                            if not members and self.allow_native_sliding:members=[j for j in q.active if j!=k]
                            if not members:continue
                            anchor=next((j for j in members if registered(q,j)),members[0])
                            trial=assign_seat(m,trial,k,anchor,preserve_state=True,peak=family%2,
                                offset_fraction=[1/3,1/16,2/3,1/3][family],angle=(order+self.active_slot%4)*np.pi/2)
                        trial=set_exit_directions(m,trial,'close'if family%2==0 else'up')
                        rows.append(('current-seat-contact-translation',trial,dict(poses=guests,family=family,
                            role_identity_changed=False,host_selection_changed=False)))
        return [row for row in rows
                if preserves_poses(parent.layout, row[1], self.protected)]

    def refine_preview(self,parent,round_index,maximum=1):
        if not self.allow_group_seating:return super().refine_preview(parent,round_index,maximum)
        current=parent
        for step in range(maximum):
            options=[current]
            for kind,q,detail in self.continuation_rows(current,round_index+step):
                if self.allow_height_repair and kind.startswith(('current-seat-','guest-centers-','seat-separation','joint-seat-separation')):
                    q=self.repair_height(q)
                options.append(self.preview_cached(q,kind,sample_id(current.layout),detail))
            def rank(x):
                space=(not x.oracle.get('enclosure_passed'),float(x.oracle.get('gap_m',0.)))
                return (*space,x.loss,x.lower_litres)if self.active_slot%2 else(x.loss,*space,x.lower_litres)
            current=min(options,key=rank)
        return current

    def preview_cached(self, layout, operation='parent', parent='', detail=None, result=None):
        saved = self.evaluated.get(layout.key())
        item = self.preview(layout, operation, parent, detail,
                            result=saved.result if saved else result)
        if saved:
            for name in ['witness', 'nominal', 'solid', 'construction', 'evaluation']:
                setattr(item, name, getattr(saved, name))
            if saved.witness is not None:
                item.oracle = saved.oracle
        return item

    def checked(self, item):
        rows = self.classify([item], 1)
        if not rows:
            raise RuntimeError('Original full-demand classification failed')
        return rows[0]

    def checked_adjustment(self,raw,iteration,slot):
        refined=self.checked(self.refine_preview(raw,iteration,maximum=1))
        return min([raw,refined],key=lambda item:physical_rank(item,slot))

    def certify(self, items, connections=3, constructions=2):
        if self.allow_geometry_validation and getattr(self.model,'require_connected_core',False):
            seen=set();candidates=[]
            for item in sorted(items,key=lambda x:(not x.oracle.get('enclosure_passed'),failed_count(x.result),x.loss)):
                if item.layout.key()in seen:continue
                seen.add(item.layout.key())
                if not item.result.get('material_contact_certification_run'):candidates.append(item)
            for item in candidates[:connections]:
                try:
                    self.evaluator.certify_core(item)
                    self.evaluated[item.layout.key()]=item
                except (ValueError,RuntimeError,AssertionError)as error:
                    self.errors.append(dict(stage='retained-core-polygons',error=str(error)))
        super().certify(items, connections, constructions)
        for item in items:
            saved = self.evaluated.get(item.layout.key())
            if saved is None or (item.completed, item.witness is not None) > (saved.completed, saved.witness is not None):
                self.evaluated[item.layout.key()] = item

    def propagate_evidence(self):
        for i, item in enumerate(self.slots):
            self.slots[i] = self.preview_cached(item.layout, item.operation, item.parent, item.detail, item.result)

    def describe(self, item):
        row = super().describe(item)
        row.update(sampling_count_center=len(item.layout.active)/3., count_penalty_used=False)
        if item.result is not None:
            row['per_pose_failed_demands'] = {str(k): int(np.count_nonzero(~item.result['masks'][k]))
                                            for k in item.layout.active}
        return row

    def best_checked(self, start):
        if self.incumbent is not None:
            return self.incumbent
        return min(self.slots+[start], key=lambda x: (
            failed_count(x.result) != 0, not x.oracle.get('enclosure_passed'),
            float(x.oracle.get('gap_m', 0.)), failed_count(x.result), x.loss, x.lower_litres))

    def solve(self, current):
        from co_common import save
        s, m = self.search, self.model
        start = self.preview_cached(current['layout'], result=current)
        self.evaluated[start.layout.key()] = start
        identical=bool(getattr(m,'require_connected_core',False))
        layouts, common = registered_population(m, current['layout'], s.rng,self.population_size,identical=identical)
        initialization = s.out.parent.parent/'step4.1'/s.out.name
        initialization.mkdir(parents=True, exist_ok=False)
        for slot, q in enumerate(layouts):
            np.savez_compressed(initialization/f'slot_{slot:02d}.npz', **q.arrays(m.native))
        save(initialization/'population.json', dict(population_size=self.population_size, common_exit_directions=common,
            identical_registered_initial_directions=identical,
            source='saved original registered Step4.1', warm_start=False,
            predefined_juxtaposed_pose_ids=[], numerical_inputs_changed=False))
        with m.hold_measure(current):
            self.slots=[]
            for i,q in enumerate(layouts):
                try:
                    item=self.checked(self.preview_cached(q,'registered-initialize',detail=dict(slot=i)))
                except (ValueError,RuntimeError)as error:
                    if not getattr(m,'require_connected_core',False):raise
                    # A tiny optional direction perturbation can give an
                    # unresolved LP. Keep the already checked registered seed;
                    # never label the unresolved candidate as force-passed.
                    item=self.preview_cached(current['layout'],'registered-initialize',
                        detail=dict(slot=i,numerical_direction_fallback=True,error=str(error)),result=current)
                    np.savez_compressed(initialization/f'slot_{i:02d}.npz',**item.layout.arrays(m.native))
                    self.errors.append(dict(stage='registered-direction-fallback',slot=i,error=str(error)))
                self.slots.append(item)
            self.certify(self.slots); self.propagate_evidence()
        save(s.out/'initial_population_summary.json',[dict(slot=i,**self.describe(item))for i,item in enumerate(self.slots)])
        save(s.out/'population32_trace.json',self.audit)
        for iteration in range(s.iterations):
            if self.stop_when_feasible and self.incumbent is not None:break
            reference = min(self.slots, key=lambda x: failed_count(x.result)).result
            with m.hold_measure(reference):
                freeze_population_measure(m, self.slots)
                for item in self.slots:
                    item.loss = float(m.proxy(item.layout)['loss'])
                snapshot = [item.layout.copy() for item in self.slots]
                events = []
                for slot, parent in enumerate(list(self.slots)):
                    self.active_slot=slot
                    weights, source = self.weights(parent)
                    coupled = self.allow_coupled_adjustment and not parent.completed and (self.allow_group_seating or not parent.oracle.get('passed'))
                    self.protected = () if coupled else tuple(k for k in parent.layout.active if weights[k] == 0)
                    total = sum(weights.values())
                    probabilities = {str(k): weights[k]/total if total else 0. for k in weights}
                    owner = weighted_pose(parent.layout.active, weights, s.rng)
                    branch = ('hold' if owner is None else 'discrete' if registered(parent.layout, owner) or s.rng.random() < .5
                              else 'differential')
                    self.branch_counts[branch] += 1
                    event = dict(slot=slot, branch=branch, accepted=False, adjustment_passes=0,
                        pose_probabilities=probabilities, weight_source=source,
                        selected_pose=owner,
                        coupled_adjustment_enabled=coupled,
                        protected_pose_indices=list(self.protected), before=self.describe(parent))
                    try:
                        if branch == 'hold':
                            event['move'] = dict(action='hold', reason=source)
                        else:
                            if branch == 'discrete':
                                if self.allow_joint_reseating and s.rng.random()<.25:
                                    native_family=self.allow_native_sliding and s.rng.random()<.5
                                    q,detail=joint_discrete_move(m,parent.layout,weights,s.rng,owner,native_family=native_family)
                                else:q, detail = discrete_move(m, parent.layout, weights, s.rng, owner=owner,carry_group=self.allow_group_seating,allow_native_slide=self.allow_native_sliding)
                            else:
                                q, detail = differential_move(m, parent.layout, snapshot, slot, weights, s.rng, iteration, owner=owner)
                            if self.allow_height_repair:q=self.repair_height(q)
                            if q is None or not preserves_poses(parent.layout, q, self.protected):
                                raise ValueError('Mutation changed a protected pose or has no eligible owner')
                            raw = self.checked(self.preview_cached(q, branch, sample_id(parent.layout), detail))
                            event['adjustment_passes'] = 1
                            trial = self.checked_adjustment(raw,iteration,slot)
                            if not preserves_poses(parent.layout, trial.layout, self.protected):
                                raise ValueError('Adjustment changed a protected pose')
                            trial = self.checked(trial)
                            better = physical_rank(trial, slot) < physical_rank(parent, slot)
                            exploratory = (branch == 'discrete' and not parent.completed and
                                trial.layout.key() != parent.layout.key() and s.rng.random() < .1)
                            if better or exploratory:
                                self.slots[slot] = trial; self.accepted += 1
                                self.exploration_accepted += int(not better)
                                event.update(accepted=True, reason='physical-rank' if better else 'structural-exploration')
                            event.update(move=detail,raw=self.describe(raw), trial=self.describe(trial),protected_pose_parameters_unchanged=True,
                                successful_pose_parameters_unchanged=preserves_poses(parent.layout,trial.layout,
                                    [k for k in parent.layout.active if parent.result['masks'][k].all()]))
                        event['after'] = self.describe(self.slots[slot])
                    except (ValueError, RuntimeError, AssertionError) as error:
                        event['error'] = str(error)
                        self.errors.append(dict(stage='population-soft-slot', slot=slot, error=str(error)))
                    events.append(event)
                    if slot % 8 == 7:
                        print('SOFT SLOTS', s.out, iteration+1, slot+1, '/', self.population_size, flush=True)
                self.protected = ()
                self.certify(self.slots); self.propagate_evidence()
                selected = self.best_checked(start)
                row = dict(phase='population_soft', iteration=iteration+1, population_size=len(self.slots),
                    generated_trials=self.population_size, maximum_trials_per_slot=1, count_penalty_used=False,
                    extra_role_seeds=0, events=events, branches=dict(self.branch_counts),
                    statistics=dict(self.evaluator.stats), retained=[dict(slot=i, **self.describe(x))
                    for i, x in enumerate(self.slots)], incumbent=self.describe(self.incumbent) if self.incumbent else None)
                self.audit.append(row); save(s.out/'population32_trace.json', self.audit); s.record(row)
                m.commit(selected.result)
                signature=(selected.layout.key(),failed_count(selected.result),bool(selected.completed))
                if not self.allow_placement_guidance or signature!=getattr(self,'snapshot_signature',None):
                    s.process_snapshot(selected.result, 'population_soft', dict(iteration=iteration+1))
                    self.snapshot_signature=signature
                print('SOFT32 ROUND', s.out, iteration+1, self.describe(selected), flush=True)
        self.certify(self.slots); selected = self.best_checked(start)
        s.frontier_witnesses = self.witnesses; s.frontier_states = [x.result for x in self.slots]
        s.frontier_statistics = dict(self.evaluator.stats, width=self.population_size, rounds=len(self.audit),configured_rounds=s.iterations,
            policy='failure_weighted_soft_count', maximum_retained=self.population_size, population_size=self.population_size,
            generated_trials=len(self.audit)*self.population_size, maximum_trials_per_slot=1, initialization_role_seeds=0,
            warm_start=False, count_penalty_used=False, sampling_count_center=len(current['layout'].active)/3.,
            oracle_fallback_when_all_force_passed=self.allow_oracle_fallback,
            coupled_adjustment_for_oracle_failure=self.allow_coupled_adjustment,
            group_seating_and_global_direction_adjustment=self.allow_group_seating,
            joint_reseating_during_optimization=self.allow_joint_reseating,
            placement_guidance_after_dynamic_role_selection=self.allow_placement_guidance,
            height_repair_after_proposals=self.allow_height_repair,stop_when_feasible=self.stop_when_feasible,
            native_state_sliding_during_optimization=self.allow_native_sliding,
            actual_geometry_validation_beyond_point_screen=self.allow_geometry_validation,
            branch_counts=self.branch_counts, accepted_updates=self.accepted,
            structural_exploration_updates=self.exploration_accepted, errors=self.errors[-30:])
        s.material_selected = selected
        if selected.witness is not None:
            m.base_oracle.witness = selected.layout.key(), selected.witness
        selected.result['base_oracle'] = selected.oracle; m.commit(selected.result)
        return selected.result
