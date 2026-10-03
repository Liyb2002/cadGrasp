"""Single-state greedy growth guided by a fixed usage envelope."""
from pathlib import Path
import time
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint,Point
from shapely.ops import nearest_points
from step4_connect_support.joint_growth import JointGrow
from step4_connect_support.direct_head_growth import DirectGrow
from step4_connect_support import build_coupled_saddle as S,growing_support as L
from step3_scheculer import contacts as I

METHOD='Greedy support growth close to a fixed usage envelope'

class EnvelopeGrow(JointGrow):
    beam_width=1
    def distances(self,points,required):
        points=np.unique(points,axis=0)
        if not len(points):return np.full(len(required),np.inf)
        if len(points)==1:return np.linalg.norm(required-points[0],axis=1)
        inside=np.zeros(len(required),bool)
        if len(points)>2 and np.linalg.matrix_rank(points-points[0],tol=1e-12)==2:
            hull=ConvexHull(points);polygon=points[hull.vertices]
            inside=(required@hull.equations[:,:2].T+hull.equations[:,2]).max(axis=1)<=1e-10
        else:
            axis=int(np.argmax(np.ptp(points,axis=0)))
            polygon=points[[np.argmin(points[:,axis]),np.argmax(points[:,axis])]]
        delta=np.roll(polygon,-1,axis=0)-polygon
        squared=np.einsum('ij,ij->i',delta,delta);good=squared>1e-24
        a=polygon[good];d=delta[good]
        t=np.clip(np.einsum('nji,ji->nj',required[:,None,:]-a[None,:,:],d)/squared[good],0.,1.)
        result=np.linalg.norm(required[:,None,:]-a[None,:,:]-t[:,:,None]*d[None,:,:],axis=2).min(axis=1)
        result[inside]=0.
        return result

    def overflow(self,lo,hi):
        return float(max(0.,np.max(self.envelope_lo-lo),np.max(hi-self.envelope_hi)))

    def space_cost(self,lo,hi):
        extent=np.maximum(self.envelope_hi,hi)-np.minimum(self.envelope_lo,lo)
        return float(np.prod(extent)-np.prod(self.envelope_hi-self.envelope_lo))

    def key(self,action):
        return max(0.,round(self.space_cost(action['lo'],action['hi']),14)),0 if action['kind']=='merge' else 1,action['length'],tuple(action['route'][0]),tuple(action['route'][-1])

    def weight_navigation(self):
        # Cached guidance stays fixed even after an outward branch is accepted.
        if getattr(self,'fixed_navigation_weighted',False):return
        lo=np.full((len(self.nodes),3),np.inf);hi=np.full((len(self.nodes),3),-np.inf)
        for b,o in zip(self.bases,self.offsets):
            points=(self.nodes-o)@b.T;bead=self.bead@b.T
            lo=np.minimum(lo,points+bead.min(axis=0));hi=np.maximum(hi,points+bead.max(axis=0))
        outside=np.maximum(0.,np.maximum(self.envelope_lo-lo,hi-self.envelope_hi).max(axis=1))
        penalty=1.+outside/.005
        rows=np.repeat(np.arange(len(self.nodes)),np.diff(self.graph_data.indptr))
        active=np.isfinite(self.graph_data.data)
        self.graph_data.data[active]=(self.original_edge_weights*.5*(penalty[rows]+penalty[self.graph_data.indices]))[active]
        self.fixed_navigation_weighted=True

    def graph(self,pitch):
        self.fixed_navigation_weighted=False
        super().graph(pitch)

    def compact_ports(self,state,point,row):
        ports=self.ports(state,point)
        ends=np.asarray([p for p,_ in ports]);lo=np.full((len(ends),3),np.inf);hi=np.full((len(ends),3),-np.inf)
        for b,o in zip(self.bases,self.offsets):
            transformed=(ends-o)@b.T;bead=self.bead@b.T
            start=(point-o)@b.T
            lo=np.minimum(lo,np.minimum(transformed+bead.min(axis=0),start+bead.min(axis=0)))
            hi=np.maximum(hi,np.maximum(transformed+bead.max(axis=0),start+bead.max(axis=0)))
        lo=np.minimum(lo,row['lo']);hi=np.maximum(hi,row['hi'])
        volumes=np.prod(np.maximum(self.envelope_hi,hi)-np.minimum(self.envelope_lo,lo),axis=1)
        order=sorted(range(len(ports)),key=lambda j:(round(float(volumes[j]),14),float(np.linalg.norm(ends[j]-point)),tuple(ends[j])))
        return [ports[j] for j in order]

    def merge_actions(self,state):
        if len(set(state['labels']))==1:return []
        proposals={}
        for source,owner in zip(state['seeds'],state['seed_owners']):
            for end,target in self.ports(state,source,state['labels'][owner])[:3]:
                vertices=np.vstack([source+self.bead,end+self.bead])
                world=np.vstack([(vertices-o)@b.T for b,o in zip(self.bases,self.offsets)])
                outside=self.space_cost(world.min(axis=0),world.max(axis=0))
                key=(outside,float(np.linalg.norm(source-end)),tuple(source),tuple(end),owner,target)
                proposals[tuple(sorted((tuple(source),tuple(end))))]=key
        ordered=sorted(proposals.values())
        for _,_,a,b,owner,target in ordered[:48]:
            action=self.action(state,'merge',[np.asarray(a),np.asarray(b)],owner,target)
            if action is not None:return [action]
        if self.direct_only:return []
        for _,_,a,_,owner,_ in ordered[:2]:
            try:route,target=self.routed(state,np.asarray(a),state['labels'][owner])
            except RuntimeError:continue
            action=self.action(state,'merge',route,owner,target)
            if action is not None:return [action]
        return []

    def ground_actions(self,state):
        gaps=[]
        for k,points in enumerate(state['floors']):
            distances=self.distances(points,self.required_vertices[k])
            if np.max(distances)<=1e-10:continue
            j=int(np.argmax(distances)) if np.isfinite(distances).all() else int(np.argmax(np.linalg.norm(self.required_vertices[k]-self.required_vertices[k].mean(axis=0),axis=1)))
            gaps.append((-float(distances[j]),k,j))
        if not gaps:return []
        # Repair the most exposed demand vertex, not aggregate micro-gains.
        for _,k,j in sorted(gaps):
            point=self.required_vertices[k][j]
            if len(state['floors'][k]):
                nearest=np.asarray(nearest_points(MultiPoint(state['floors'][k]).convex_hull,Point(point))[0].coords[0])
            else:nearest=self.required_vertices[k].mean(axis=0)
            direction=point-nearest;norm=float(np.linalg.norm(direction))
            if norm<1e-12:continue
            direction/=norm;current=float(nearest@direction);target=float(point@direction)
            proposals=[]
            for index,row in enumerate(self.ground_candidates[k]):
                if (k,index) in state['feet_used'] or row.get('legal') is False:continue
                capacity=float((row['footprint']@direction).max())
                if capacity<=current+1e-10:continue
                end,_=self.ports(state,row['start'])[0]
                cost=self.space_cost(row['lo'],row['hi'])
                proposals.append((capacity,index,cost,float(np.linalg.norm(row['start']-end))))
            valid=[p for p in proposals if self.legal_foot(self.ground_candidates[k][p[1]])]
            if not valid:continue
            # Cross the exposed supporting line with one sole-radius of slack.
            # If that slack is unavailable, reach the demand line or advance as
            # far as the legal candidates permit in the deficient direction.
            useful=[p for p in valid if p[0]>=target+.003-1e-10]
            if not useful:useful=[p for p in valid if p[0]>=target-1e-10]
            if not useful:
                capacity=max(p[0] for p in valid)
                useful=[p for p in valid if p[0]>=capacity-1e-9]
            compact=lambda p:(p[2],p[3],p[1])
            tiers=[sorted(useful,key=compact)]
            remaining=[p for p in valid if p not in useful]
            if remaining:tiers.append(sorted(remaining,key=lambda p:(-p[0],compact(p))))
            for ordered in tiers:
                choices=[]
                for _,index,_,_ in ordered:
                    row=self.ground_candidates[k][index]
                    if not self.legal_foot(row):continue
                    for end,owner in self.compact_ports(state,row['start'],row)[:4]:
                        action=self.action(state,'ground',[row['start'],end],owner,owner,k,index,row)
                        if action is not None:choices.append(action);break
                    if len(choices)>=4:break
                if choices:return [min(choices,key=self.key)]
                if self.direct_only:continue
                for _,index,_,_ in ordered[:4]:
                    row=self.ground_candidates[k][index]
                    if not self.legal_foot(row):continue
                    try:route,owner=self.routed(state,row['start'])
                    except RuntimeError:continue
                    action=self.action(state,'ground',route,owner,owner,k,index,row)
                    if action is not None:return [action]
        return []

    def prune(self,state):
        active=list(state['actions']);removed=[]
        for action in reversed(state['actions']):
            if action['kind']!='ground':continue
            route=action['route'];dependent=False
            for other in active:
                if other is action:continue
                for endpoint in (other['route'][0],other['route'][-1]):
                    # The attachment to an existing component survives removal;
                    # another branch sharing that junction is not a dependent.
                    if np.linalg.norm(endpoint-route[-1])<=1e-9:continue
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
        started=time.monotonic();self.initialize_occupancy();self.rod_cache={}
        demands=np.vstack([np.c_[row,np.zeros(len(row))] for row in self.case.demands])
        self.envelope_lo=np.minimum(self.occupied_lo,demands.min(axis=0))-.003
        self.envelope_hi=np.maximum(self.occupied_hi,demands.max(axis=0))+.003
        # Generate candidates once from the fixed reference, not grown bounds.
        self.occupied_lo=self.envelope_lo.copy();self.occupied_hi=self.envelope_hi.copy()
        vertices=np.vstack([S.unpack(t['solid']).vertices for t in self.terminals])
        self.initial_floors=tuple(self.reduce_floor(self.material_floor_points(vertices,k)) for k in range(len(self.case.poses)))
        state=dict(labels=tuple(range(len(self.terminals))),seeds=tuple(self.seed_positions),seed_owners=tuple(range(len(self.terminals))),
            segments=(),floors=self.initial_floors,lo=self.occupied_lo.copy(),hi=self.occupied_hi.copy(),actions=(),feet_used=frozenset(),length=0.)
        for k in range(len(self.case.poses)):self.foot_pool(k)
        for step in range(160):
            if len(set(state['labels']))==1 and all(self.coverage(p,k)[0] for k,p in enumerate(state['floors'])):break
            self.activate(state)
            self.direct_only=True
            joins=self.merge_actions(state)
            candidates=joins if joins and self.key(joins[0])[0]==0. else joins+self.ground_actions(state)
            if not candidates:
                self.direct_only=False
                candidates=self.merge_actions(state)+self.ground_actions(state)
            if not candidates:raise RuntimeError('No legal greedy growth action')
            action=min(candidates,key=self.key);state=self.advance(state,action)
            print('ENVELOPE GROW',self.group.name,step,action['kind'],'outside_mm',round(self.overflow(action['lo'],action['hi'])*1000,3),flush=True)
        else:raise RuntimeError('Greedy growth action limit exceeded')
        actions,full,removed=self.prune(state)
        self.beams=[part for action in actions for part in action['parts']];self.original_soles=[]
        self.feet=[dict(pose=p,centers_xy_m=[],creation_steps=[]) for p in self.case.poses]
        self.generated_ground=[];self.paths=[];self.journal=[];self.segments=[]
        for step,action in enumerate(actions,1):
            route=action['route'];name=f'component_join_{step}'
            if action['kind']=='ground':
                k=action['pose_index'];row=action['row'];name=f'{self.case.poses[k]}:grown_foot{step}'
                self.original_soles.append(row['sole']);self.core_solids.extend([(name+':sole',row['sole']),(name+':start_ball',row['ball'])])
                self.generated_ground.append((k,row['xy']));self.feet[k]['centers_xy_m'].append(row['xy'].tolist());self.feet[k]['creation_steps'].append(step)
            self.paths.append([p.tolist() for p in route]);self.segments.extend((a,b) for a,b in zip(route,route[1:]))
            self.journal.append(dict(step=step,target=name,target_kind='generated_ground_contact' if action['kind']=='ground' else 'component_merge',
                start_network_point_m=route[-1].tolist(),path_m=[p.tolist() for p in route[::-1]],core_segments=len(action['parts']),fixed_envelope_overflow_mm=self.overflow(action['lo'],action['hi'])*1000))
        self.partial_steps=max(1,len(actions)//2);self.save_stage('partial_growth',self.assemble(actions[:self.partial_steps]));self.save_stage('shared_tree',full)
        if len(L.components(full))!=1:raise RuntimeError('Greedy support disconnected')
        self.final_solid=full;self.timings['envelope_growth']=time.monotonic()-started
        first=next((i for i,a in enumerate(actions) if a['kind']=='ground'),len(actions));merges=sum(a['kind']=='merge' for a in actions[:first])
        return full,dict(terminal_count=len(self.terminals),grown_segment_count=len(self.beams),direct_connection_count=sum(len(a['parts'])==1 for a in actions),
            fallback_route_count=sum(len(a['parts'])>1 for a in actions),lazy_grid_constructed=self.grid_ready,preset_skeleton=False,preset_grid_anchor=False,initial_local_loft_to_grid=False,
            initial_head_name='All original contact starts',growth_journal=self.journal,partial_stage_step_count=self.partial_steps,fixed_ground_targets=False,
            beam_width=1,head_first_phase=False,pose_sequential_phase=False,ground_minimum_relative_gain=None,generated_ground_contact_count=len(self.generated_ground),
            redundant_ground_branches_removed=removed,reference_envelope=dict(min_m=self.envelope_lo.tolist(),max_m=self.envelope_hi.tolist()),
            envelope_fixed_during_growth=True,reference_allowance_mm=3.,coverage_line_slack_mm=3.,fixed_envelope_volume_ranking=True,ground_started_before_head_connectivity=merges<len(self.terminals)-1,
            routing='Single-state greedy; fixed-envelope enclosing volume then length; repair exposed coverage vertices',
            ground_candidate_source='demand neighborhoods and fixed-envelope floor sections; no previous support footprint')

    def run(self,pitch=.008):
        r=DirectGrow.run(self,pitch)
        r['timing_seconds']=dict(self.timings,run_with_internal_validation=r['seconds'],construction=self.timings['contact_starts']+self.timings['envelope_growth'])
        r['timing_seconds']['setup_and_run']=self.timings.get('setup',0.)+r['seconds']
        r['objective']='Keep branches close to fixed usage envelope; enclosing volume first, then length'
        r['construction'].update(method=METHOD,reference_used_as='saved placement only',candidate_order=['fixed_envelope_enclosing_volume','rod_length','fixed_geometric_ties'])
        r['provenance']['code'].update(I.hashes([Path(__file__),Path(__file__).with_name('joint_growth.py'),Path(__file__).with_name('coverage_growth.py')]))
        I.save(self.out/'report.json',r);return r
