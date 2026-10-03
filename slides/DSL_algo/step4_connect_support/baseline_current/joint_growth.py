"""Bounded deterministic search over joint connectivity and ground growth.

All contact starts exist initially as separate components. Ground contacts are
optional outputs. Partial states may merge components or improve any pose's
coverage; there is no head-first or pose-by-pose completion phase.
"""
from pathlib import Path
import time
import numpy as np
from scipy.spatial import ConvexHull
from step4_connect_support.baseline_current.coverage_growth import CoverageGrow
from step4_connect_support.baseline_current.direct_head_growth import DirectGrow
from step4_connect_support.baseline_current import boxed_support as F,build_coupled_saddle as S
from step4_connect_support.baseline_current import deterministic_space as D,growing_support as L
from step3_scheculer import contacts as I

METHOD='Joint component and ground growth with a four-state deterministic beam'

class JointGrow(CoverageGrow):
    beam_width=4

    def coverage(self,points,k):
        """True distance to a convex set is monotone as ground material grows."""
        points=np.unique(points,axis=0);required=self.required_vertices[k]
        if not len(points):return False,float('inf'),self.required_hulls[k].area
        if len(points)==1:
            distances=np.linalg.norm(required-points[0],axis=1)
        else:
            if len(points)>2 and np.linalg.matrix_rank(points-points[0],tol=1e-12)==2:
                hull=ConvexHull(points);polygon=points[hull.vertices]
                inside=(required@hull.equations[:,:2].T+hull.equations[:,2]).max(axis=1)<=1e-10
            else:
                axis=int(np.argmax(np.ptp(points,axis=0)));polygon=points[[np.argmin(points[:,axis]),np.argmax(points[:,axis])]]
                inside=np.zeros(len(required),bool)
            a=polygon;delta=np.roll(polygon,-1,axis=0)-a
            squared=np.einsum('ij,ij->i',delta,delta);good=squared>1e-24
            fraction=np.clip(np.einsum('nji,ji->nj',required[:,None,:]-a[None,good,:],delta[good])/squared[good],0.,1.)
            nearest=a[None,good,:]+fraction[:,:,None]*delta[None,good,:]
            distances=np.linalg.norm(required[:,None,:]-nearest,axis=2).min(axis=1)
            distances[inside]=0.
        done=bool(np.max(distances)<=1e-10)
        return done,float(distances.sum()),0. if done else float('nan')

    def activate(self,state):
        self.occupied_lo=state['lo'];self.occupied_hi=state['hi']
        self.seed_positions=list(state['seeds'])

    def progress(self,state):
        connectivity=(len(set(state['labels']))-1)/max(1,len(self.terminals)-1)
        missing=[];covered=[]
        for k,points in enumerate(state['floors']):
            done,loss,_=self.coverage(points,k);covered.append(done)
            missing.append(1. if not np.isfinite(loss) else min(1.,loss/self.loss_scales[k]))
        return connectivity,float(np.mean(missing)),all(covered)

    def state_key(self,state):
        segments=tuple(sorted(tuple(sorted((tuple(a),tuple(b)))) for a,b,_ in state['segments']))
        return state['labels'],tuple(sorted(state['feet_used'])),segments

    def rank(self,state):
        extent=state['hi']-state['lo'];conn,missing,_=self.progress(state)
        score=np.prod(extent)/self.initial_volume+np.prod(extent[:2])/self.initial_area+conn+2.*missing+.01*state['length']/self.length_scale
        return float(score),float(np.prod(extent)),float(np.prod(extent[:2])),state['length'],self.state_key(state)

    def complete_key(self,state):
        extent=state['hi']-state['lo']
        return float(np.prod(extent)),float(np.prod(extent[:2])),state['length'],self.state_key(state)

    def ports(self,state,point,exclude=None):
        rows=[(np.asarray(p),owner) for p,owner in zip(state['seeds'],state['seed_owners'])
            if exclude is None or state['labels'][owner]!=exclude]
        for a,b,owner in state['segments']:
            if exclude is not None and state['labels'][owner]==exclude:continue
            delta=b-a;norm=float(delta@delta)
            if norm<=1e-24:continue
            fraction=float(np.clip((point-a)@delta/norm,0.,1.));rows.append((a+fraction*delta,owner))
        unique={tuple(p):(p,owner) for p,owner in rows}
        return sorted(unique.values(),key=lambda row:(float(np.linalg.norm(point-row[0])),state['labels'][row[1]],tuple(row[0])))

    def checked_rod(self,a,b):
        key=tuple(sorted((tuple(a),tuple(b))))
        if key not in self.rod_cache:
            body=self.rod(a,b);self.rod_cache[key]=(body,self.legal(body))
        return self.rod_cache[key]

    def routed(self,state,start,exclude=None):
        self.activate(state)
        ports=self.ports(state,start,exclude)
        if not ports:raise RuntimeError('No component attachment')
        # Include actual projections as attachment sites, with fixed geometry ties.
        chosen=ports[:12]
        points=[p for p,_ in chosen]
        self.seed_positions=points+[start]
        route=self.fallback(len(points),set(range(len(points))))
        endpoint=route[-1]
        owner=min(chosen,key=lambda row:(float(np.linalg.norm(row[0]-endpoint)),row[1]))[1]
        return route,owner

    def action(self,state,kind,route,source_owner,target_owner,k=None,index=None,row=None):
        parts=[]
        for a,b in zip(route,route[1:]):
            body,legal=self.checked_rod(a,b)
            if not legal:return None
            parts.append(body)
        vertices=np.vstack([p+self.bead for p in route]+([] if row is None else [row['vertices']]))
        world=np.vstack([(vertices-o)@b.T for b,o in zip(self.bases,self.offsets)])
        floor_parts=tuple(self.material_floor_points(vertices,j) for j in range(len(self.case.poses)))
        return dict(kind=kind,route=tuple(np.asarray(p) for p in route),parts=parts,source_owner=source_owner,target_owner=target_owner,
            pose_index=k,index=index,row=row,lo=world.min(axis=0),hi=world.max(axis=0),floor_points=floor_parts,
            length=sum(float(np.linalg.norm(b-a)) for a,b in zip(route,route[1:])))

    def merge_actions(self,state):
        if len(set(state['labels']))==1:return []
        proposals=[];seen=set()
        for source,owner in zip(state['seeds'],state['seed_owners']):
            label=state['labels'][owner]
            for endpoint,target_owner in self.ports(state,source,label)[:3]:
                key=tuple(sorted((tuple(source),tuple(endpoint))))
                if key in seen:continue
                seen.add(key);proposals.append((float(np.linalg.norm(source-endpoint)),tuple(source),tuple(endpoint),owner,target_owner))
        result=[]
        for _,a,b,owner,target_owner in sorted(proposals)[:48]:
            route=[np.asarray(a),np.asarray(b)]
            action=self.action(state,'merge',route,owner,target_owner)
            if action is not None:result.append(action)
            if len(result)>=2:break
        if not result:
            for _,a,_,owner,_ in sorted(proposals)[:2]:
                try:route,target_owner=self.routed(state,np.asarray(a),state['labels'][owner])
                except RuntimeError:continue
                action=self.action(state,'merge',route,owner,target_owner)
                if action is not None:result.append(action);break
        return result

    def gains(self,state,k):
        points=state['floors'][k];key=(k,points.tobytes())
        if key not in self.gain_cache:
            _,loss,_=self.coverage(points,k)
            before=self.loss_scales[k] if not np.isfinite(loss) else loss
            gain=[]
            for row in self.ground_candidates[k]:
                _,after,_=self.coverage(np.vstack([points,row['footprint']]),k)
                gain.append(max(0.,before-after)/self.loss_scales[k])
            self.gain_cache[key]=np.asarray(gain)
        return self.gain_cache[key]

    def ground_actions(self,state):
        proposals=[];extent=state['hi']-state['lo'];volume=float(np.prod(extent));area=float(np.prod(extent[:2]))
        self.activate(state)
        for k,pool in self.ground_candidates.items():
            if self.coverage(state['floors'][k],k)[0]:continue
            if sum(pose==k for pose,_ in state['feet_used'])>=32:continue
            gains=self.gains(state,k)
            starts=np.asarray([row['start'] for row in pool]);seeds=np.asarray(state['seeds'])
            lengths=np.linalg.norm(starts[:,None,:]-seeds[None,:,:],axis=2).min(axis=1)
            for a,b,_ in state['segments']:
                delta=b-a;denom=float(delta@delta)
                if denom<=1e-24:continue
                fraction=np.clip((starts-a)@delta/denom,0.,1.)
                lengths=np.minimum(lengths,np.linalg.norm(starts-a-fraction[:,None]*delta,axis=1))
            lo=np.minimum(state['lo'],np.array([row['lo'] for row in pool]));hi=np.maximum(state['hi'],np.array([row['hi'] for row in pool]))
            extents=hi-lo
            cost=(np.prod(extents,axis=1)-volume)/self.initial_volume+(np.prod(extents[:,:2],axis=1)-area)/self.initial_area+.02*lengths/self.length_scale
            for index,row in enumerate(pool):
                if gains[index]<=1e-12 or (k,index) in state['feet_used'] or row.get('legal') is False:continue
                proposals.append((float(cost[index]/gains[index]),float(cost[index]),-float(gains[index]),k,index))
        result=[];used_rows=set()
        orders=[sorted(proposals),sorted(proposals,key=lambda p:(p[2],p[1],p[0],p[3:])),
            sorted(proposals,key=lambda p:(p[1],p[2],p[0],p[3:]))]
        for order in orders:
            trials=0;selected=None
            for _,_,_,k,index in order:
                if (k,index) in used_rows:continue
                row=self.ground_candidates[k][index]
                if not self.legal_foot(row):continue
                for end,owner in self.ports(state,row['start'])[:8]:
                    trials+=1
                    action=self.action(state,'ground',[row['start'],end],owner,owner,k,index,row)
                    if action is not None:selected=action;break
                if selected is not None or trials>=48:break
            if selected is None:
                # A blocked high-progress family must not be starved by cheap
                # direct micro-gains from a different proposal family.
                for _,_,_,k,index in order[:2]:
                    if (k,index) in used_rows:continue
                    row=self.ground_candidates[k][index]
                    if not self.legal_foot(row):continue
                    try:route,owner=self.routed(state,row['start'])
                    except RuntimeError:continue
                    selected=self.action(state,'ground',route,owner,owner,k,index,row)
                    if selected is not None:break
            if selected is not None:
                result.append(selected);used_rows.add((selected['pose_index'],selected['index']))
        if not result:
            for _,_,_,k,index in sorted(proposals)[:8]:
                row=self.ground_candidates[k][index]
                if not self.legal_foot(row):continue
                try:route,owner=self.routed(state,row['start'])
                except RuntimeError:continue
                action=self.action(state,'ground',route,owner,owner,k,index,row)
                if action is not None:result.append(action);break
        return result

    def advance(self,state,action):
        labels=list(state['labels']);source=labels[action['source_owner']];target=labels[action['target_owner']]
        if action['kind']=='merge':
            keep,drop=min(source,target),max(source,target)
            labels=[keep if label==drop else label for label in labels]
        owner=action['target_owner'];route=action['route']
        seeds=state['seeds'];seed_owners=state['seed_owners'];used=state['feet_used']
        if action['kind']=='ground':
            seeds=seeds+(route[0],);seed_owners=seed_owners+(owner,)
            used=used|{(action['pose_index'],action['index'])}
        segments=state['segments']+tuple((a,b,owner) for a,b in zip(route,route[1:]))
        floors=tuple(self.reduce_floor(np.vstack([p,q])) if len(q) else p for p,q in zip(state['floors'],action['floor_points']))
        return dict(labels=tuple(labels),seeds=seeds,seed_owners=seed_owners,segments=segments,floors=floors,
            lo=np.minimum(state['lo'],action['lo']),hi=np.maximum(state['hi'],action['hi']),
            actions=state['actions']+(action,),feet_used=used,length=state['length']+action['length'])

    def select(self,children):
        unique={self.state_key(state):state for state in children};values=list(unique.values())
        if not values:return []
        selected=[];used=set()
        orders=[sorted(values,key=self.rank),
            sorted(values,key=lambda state:(self.progress(state)[1],self.rank(state))),
            sorted(values,key=lambda state:(len(set(state['labels'])),self.rank(state))),
            sorted(values,key=self.complete_key)]
        for order in orders:
            for state in order:
                key=self.state_key(state)
                if key not in used:selected.append(state);used.add(key);break
            if len(selected)>=self.beam_width:break
        return selected

    def assemble(self,actions):
        return F.union([t['solid'] for t in self.terminals]+[part for action in actions for part in
            action['parts']+([] if action['row'] is None else [action['row']['sole'],action['row']['ball']])])

    def prune(self,state):
        active=list(state['actions']);removed=[]
        for action in reversed(state['actions']):
            if action['kind']!='ground':continue
            route=action['route'];dependent=False
            for other in active:
                if other is action:continue
                for endpoint in (other['route'][0],other['route'][-1]):
                    for a,b in zip(route,route[1:]):
                        delta=b-a;denom=float(delta@delta)
                        fraction=np.clip((endpoint-a)@delta/denom,0.,1.) if denom>1e-24 else 0.
                        if np.linalg.norm(endpoint-a-fraction*delta)<=.0061:dependent=True;break
                    if dependent:break
                if dependent:break
            if dependent:continue
            trial=[a for a in active if a is not action]
            if all(self.coverage(np.vstack([self.initial_floors[k]]+[a['floor_points'][k] for a in trial]),k)[0] for k in range(len(self.case.poses))):
                active=trial;removed.append((action['pose_index'],action['index']))
        full=self.assemble(active)
        if len(L.components(full))!=1:active=list(state['actions']);removed=[];full=self.assemble(active)
        return active,full,removed

    def connect(self):
        started=time.monotonic();self.initialize_occupancy()
        self.rod_cache={};self.gain_cache={};self.loss_scales=[]
        vertices=np.vstack([S.unpack(t['solid']).vertices for t in self.terminals])
        self.initial_floors=tuple(self.reduce_floor(self.material_floor_points(vertices,k)) for k in range(len(self.case.poses)))
        initial=dict(labels=tuple(range(len(self.terminals))),seeds=tuple(self.seed_positions),seed_owners=tuple(range(len(self.terminals))),
            segments=(),floors=self.initial_floors,lo=self.occupied_lo.copy(),hi=self.occupied_hi.copy(),actions=(),feet_used=frozenset(),length=0.)
        extent=initial['hi']-initial['lo'];self.initial_volume=float(np.prod(extent));self.initial_area=float(np.prod(extent[:2]));self.length_scale=float(np.linalg.norm(extent))
        for k in range(len(self.case.poses)):
            self.foot_pool(k);corners=np.asarray(self.center_region(k).exterior.coords)[:-1]
            distances=np.linalg.norm(self.required_vertices[k][:,None,:]-corners[None,:,:],axis=2).max(axis=1)
            self.loss_scales.append(max(1e-9,float(distances.sum())))
        frontier=[initial];complete=[];first_complete=None;expanded=0
        for depth in range(120):
            children=[]
            for state in frontier:
                conn,missing,covered=self.progress(state)
                if conn==0 and covered:
                    complete.append(state)
                    if first_complete is None:first_complete=depth
                    continue
                self.activate(state)
                actions=self.merge_actions(state)+self.ground_actions(state)
                expanded+=1
                children.extend(self.advance(state,action) for action in actions)
            frontier=self.select(children)
            print('JOINT SEARCH',self.group.name,'depth',depth,'states',len(frontier),'complete',len(complete),
                'best',None if not frontier else [len(set(frontier[0]['labels'])),round(self.progress(frontier[0])[1],6)],flush=True)
            if not frontier or (first_complete is not None and depth>=first_complete+32):break
        complete.extend(state for state in frontier if self.progress(state)[0]==0 and self.progress(state)[2])
        if not complete:raise RuntimeError('Joint beam exhausted without a complete support')
        candidates=[]
        for state in sorted(complete,key=self.complete_key)[:4]:
            actions,solid,removed=self.prune(state);mesh=S.unpack(solid)
            budget=D.measure(self.case,mesh,self.bases,self.offsets)
            candidates.append((budget['box_volume_cm3'],budget['xy_area_cm2'],float(mesh.volume),state,actions,solid,removed))
        _,_,_,state,actions,full,removed=min(candidates,key=lambda row:row[:3])
        self.beams=[part for action in actions for part in action['parts']];self.original_soles=[]
        self.feet=[dict(pose=pose,centers_xy_m=[],creation_steps=[]) for pose in self.case.poses]
        self.generated_ground=[];self.paths=[];self.journal=[];self.segments=[]
        for step,action in enumerate(actions,1):
            route=action['route'];name=f'component_join_{step}'
            if action['kind']=='ground':
                k=action['pose_index'];row=action['row'];name=f'{self.case.poses[k]}:grown_foot{step}'
                self.original_soles.append(row['sole']);self.core_solids.extend([(name+':sole',row['sole']),(name+':start_ball',row['ball'])])
                self.generated_ground.append((k,row['xy']));self.feet[k]['centers_xy_m'].append(row['xy'].tolist());self.feet[k]['creation_steps'].append(step)
            self.paths.append([p.tolist() for p in route]);self.segments.extend((a,b) for a,b in zip(route,route[1:]))
            self.journal.append(dict(step=step,target=name,target_kind='generated_ground_contact' if action['kind']=='ground' else 'component_merge',
                start_network_point_m=route[-1].tolist(),path_m=[p.tolist() for p in route[::-1]],core_segments=len(action['parts'])))
        self.partial_steps=max(1,len(actions)//2);self.save_stage('partial_growth',self.assemble(actions[:self.partial_steps]));self.save_stage('shared_tree',full)
        if len(L.components(full))!=1:raise RuntimeError('Joint-grown support disconnected')
        self.final_solid=full;self.timings=getattr(self,'timings',{});self.timings['joint_growth']=time.monotonic()-started
        kinds=[a['kind'] for a in actions];first_ground=next((i for i,k in enumerate(kinds) if k=='ground'),None)
        merges_before_ground=sum(k=='merge' for k in kinds[:first_ground]) if first_ground is not None else len(self.terminals)-1
        return full,dict(terminal_count=len(self.terminals),grown_segment_count=len(self.beams),
            direct_connection_count=sum(len(a['parts'])==1 for a in actions),fallback_route_count=sum(len(a['parts'])>1 for a in actions),
            lazy_grid_constructed=self.grid_ready,preset_skeleton=False,preset_grid_anchor=False,initial_local_loft_to_grid=False,
            initial_head_name='All original contact starts',growth_journal=self.journal,partial_stage_step_count=self.partial_steps,
            fixed_ground_targets=False,ground_volume_ranking_resolution_mm=None,ground_minimum_relative_gain=None,
            generated_ground_contact_count=len(self.generated_ground),redundant_ground_branches_removed=removed,
            ground_candidate_source='original demand neighborhoods and initial-envelope floor sections; no previous support footprint',
            routing='all components and all pose coverage grow jointly; bounded diverse beam; no head-first phase',
            beam_width=self.beam_width,expanded_states=expanded,complete_candidates=len(complete),
            final_candidates_measured=len(candidates),merges_before_first_ground=merges_before_ground,
            ground_started_before_head_connectivity=merges_before_ground<len(self.terminals)-1,
            ground_proposal_count=sum(map(len,self.ground_candidates.values())),
            ground_exact_solid_checks=sum('legal' in r for rows in self.ground_candidates.values() for r in rows),
            active_grid_pitch_m=getattr(self,'active_grid_pitch',None))

    def run(self,pitch=.008):
        report=DirectGrow.run(self,pitch)
        report['timing_seconds']=dict(self.timings,run_with_internal_validation=report['seconds'],
            construction=self.timings.get('contact_starts',0.)+self.timings['joint_growth'])
        report['timing_seconds']['setup_and_run']=self.timings.get('setup',0.)+report['seconds']
        report['objective']='Joint connectivity and coverage; select completed supports by occupied XYZ volume, XY area, then material'
        report['construction'].update(method=METHOD,reference_used_as='saved placement only',head_first_phase=False,pose_sequential_phase=False,
            ground_candidate_order=['compact_box_cost_per_coverage_gain','rod_length','fixed_geometric_ties'])
        report['previous_material_volume_cm3']=float(self.foot_reference.volume*1e6)
        report['material_reduction_percent']=100*(1-report['volume_cm3']/report['previous_material_volume_cm3'])
        report['comparison_reference']='boxed_support/shape.obj; compare latest metrics with before_joint_growth/report.json'
        report['provenance']['code'].update(I.hashes([Path(__file__),Path(__file__).with_name('coverage_growth.py')]))
        I.save(self.out/'report.json',report)
        return report
