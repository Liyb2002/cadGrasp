"""Whole-set material minimization, with a small feasible/repair layout beam.

No solid Boolean is used in proposal generation or candidate screening. The
incumbent is a previously checked REAL support, rather than a sampled estimate.
All original loads/geometry are checked again for any newly published layout.
"""
import _bootstrap
import time
from dataclasses import dataclass
from co_common import *
from whole_pipeline import WholeSearch, load_layout
from whole_search.model import Model
from whole_search.reuse_first import registered, juxtaposed_count
from whole_search.search import failed_count
from whole_search.common import tangent_frame, legal_direction, result_mesh


def load_incumbent(model, source):
    report = I.check_report(source/'data/report.json')
    if report.get('schema') != 'whole_step4_v1' or report['poses'] != model.poses:
        raise ValueError('Compaction must continue this whole group, in its saved order')
    if not report.get('force_exit_work_passed') or not report.get('passed'):
        raise ValueError('The retained incumbent must have passed actual force/exit/work checks')
    layout = load_layout(source/'layout.npz')
    assert layout.active == tuple(range(len(model.poses)))
    mesh = trimesh.load(source/'support.obj', force='mesh', process=False)
    if abs(abs(mesh.volume)*1e6-report['volume_cm3']) > 1e-4:
        raise ValueError('Incumbent exported volume does not match its actual certificate')
    contacts, masks, supplies = {}, {}, {}
    for k in layout.active:
        with np.load(source/f'{model.poses[k]}_force.npz') as z:
            masks[k] = z['mask'].copy(); supplies[k] = z['supply_7d'].copy()
            triangles = z['triangles_fixture_m'].copy()
            contacts[k] = dict(triangles=triangles, sources=z['source_faces'].copy(),
                area_m2=float(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                    triangles[:,2]-triangles[:,0]),axis=1).sum()/2))
        if len(masks[k]) != 32768 or not masks[k].all():
            raise ValueError('An incumbent must retain every original load')
    result = dict(layout=layout, serial=0, remaining=None, remaining_mesh=mesh,
        contacts=contacts, masks=masks, supplies=supplies,
        counts={str(k):32768 for k in layout.active},
        classifiers={k:report['classifiers'][model.poses[k]] for k in layout.active},
        **{key:report[key] for key in ['diagnostics','volume_cm3',
            'maximum_projected_footprint_m2','actual_work_surface_checks',
            'component_volumes_cm3','lossless_worker_geometry_transport']}, seconds=0.)
    return result, report


@dataclass
class Node:
    index: int
    parent: int | None
    result: dict
    operation: str
    detail: dict
    generation: int
    gap: tuple = (0., 0.)


def distinct_frontier(nodes, width, repair_width):
    """Keep feasible material minima and separately bounded repair states.

    A temporary failure cannot displace a feasible incumbent. Prefer different
    discrete host patterns before filling spare slots with continuous variants.
    """
    good = sorted((n for n in nodes if failed_count(n.result)==0),
        key=lambda n:(n.result['volume_cm3'],n.result['maximum_projected_footprint_m2'],n.index))
    bad = sorted((n for n in nodes if failed_count(n.result)),
        key=lambda n:(failed_count(n.result),*n.gap,n.result['volume_cm3'],n.index))
    chosen, patterns = [], set()
    for node in good:
        pattern = tuple(node.result['layout'].hosts)
        if pattern not in patterns:
            chosen.append(node);patterns.add(pattern)
            if len(chosen)==width: break
    for node in good:
        if len(chosen)>=width: break
        if all(node.index!=other.index for other in chosen): chosen.append(node)
    return chosen+bad[:repair_width]


class CompactSearch(WholeSearch):
    def __init__(self, model, out, *, mode='beam', rounds=6, width=2,
                 repair_width=1, screen_budget=96, full_budget=48, seed=42):
        super().__init__(model,out,iterations=rounds,finalists=8,branch_rounds=2,seed=seed)
        self.mode,self.rounds = mode,rounds
        self.width = width if mode=='beam' else 1
        self.repair_width = repair_width if mode=='beam' else 0
        self.structural = mode!='greedy'
        self.screen_budget,self.full_budget = screen_budget,full_budget
        self.nodes,self.visited,self.events = [],set(),[]
        self.capture_process = True
        self.out.mkdir(parents=True,exist_ok=True)
        self.full_checks = 0

    def proposals(self,node):
        m,layout=self.model,node.result['layout'];rows=[]
        if failed_count(node.result):
            targets,worst,_=self.targets(node.result)
            movable=[k for k in layout.active if not registered(layout,k)]
            self.fine_resolution=False
            rows,_=self.local_proposals(node.result,targets,worst,movable)
        else:
            for k in layout.active:
                frame=tangent_frame(layout.directions[k])
                for degrees in [.125,.5,2.,5.]:
                    for axis in range(2):
                        for sign in [-1,1]:
                            trial=layout.copy()
                            trial.directions[k]=legal_direction(layout.directions[k]+sign*
                                np.tan(np.radians(degrees))*frame[:,axis],m.floor_normal(layout,k))
                            rows.append(('direction',trial,dict(pose=m.poses[k],step_degrees=sign*degrees,axis=axis)))
            # Several coincident exits can lock the same patch: perturb their
            # directions together, rather than assuming a nonzero derivative.
            for axis in np.eye(3):
                for degrees in [-5.,-2.,-.25,.25,2.,5.]:
                    trial=layout.copy()
                    for k in layout.active:
                        trial.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(degrees))*
                            np.cross(axis,layout.directions[k]),m.floor_normal(layout,k))
                    rows.append(('direction-coherent',trial,dict(step_degrees=degrees,axis=axis.tolist())))
            for k in layout.active:
                if registered(layout,k):continue
                frame=tangent_frame(m.floor_normal(layout,k))
                for fraction in [1/512,1/128,1/32,1/8]:
                    for axis in range(2):
                        for sign in [-1,1]:
                            trial=layout.copy();trial.placements[k,:3,3]+=sign*m.extent*fraction*frame[:,axis]
                            rows.append(('translation',trial,dict(pose=m.poses[k],step_m=sign*m.extent*fraction,axis=axis)))
        if self.structural:
            targets=self.guard_targets(layout)
            # Re-hosting is useful after feasibility too. It can merge material
            # domains or restore contact lost by a previous rescue choice.
            rows.extend(self.juxtapose_proposals(node.result,list(layout.active),targets))
            initial,_=m.initial()
            for k in layout.active:
                if registered(layout,k):continue
                base=layout.copy();base.hosts[k]=k;base.placements[k]=np.eye(4)
                body_direction=layout.placements[k,:3,:3].T@layout.directions[k]
                for direction in [initial.directions[k],body_direction]:
                    for degrees in [0.,-.5,.5,-2.,2.]:
                        trial=base.copy()
                        trial.directions[k]=legal_direction(direction+np.tan(np.radians(degrees))*
                            tangent_frame(direction)[:,0],m.floor_normal(trial,k))
                        rows.append(('restore-reuse',trial,dict(pose=m.poses[k],guest_index=k,step_degrees=degrees)))
        seen=set();unique=[]
        for row in rows:
            key=row[1].key()
            if key not in seen and key not in self.visited:
                seen.add(key);unique.append(row)
        return unique

    def sample_rows(self,rows,budget):
        if len(rows)<=budget:return rows
        groups={}
        for row in rows:
            detail=row[2]
            group=(row[0],detail.get('guest_index',detail.get('pose')),
                detail.get('host'),detail.get('step_degrees',detail.get('step_m')))
            groups.setdefault(group,[]).append(row)
        # Randomize group order deterministically: the first named pose/tool
        # must not consume every slot of a bounded material/guard screen.
        keys=list(groups);order=self.rng.permutation(len(keys))
        selected=[]
        for tool in ['restore-reuse','juxtapose','translation','direction','direction-coherent']:
            matches=[row for row in rows if row[0]==tool]
            if matches and len(selected)<budget:
                selected.append(matches[int(self.rng.integers(len(matches)))])
        used={row[1].key() for row in selected}
        for j in order:
            group=groups[keys[j]]
            row=group[int(self.rng.integers(len(group)))]
            if row[1].key() not in used:
                selected.append(row);used.add(row[1].key())
            if len(selected)==budget:break
        if len(selected)<budget:
            for j in self.rng.permutation(len(rows)):
                row=rows[j]
                if row[1].key() not in used:
                    selected.append(row);used.add(row[1].key())
                if len(selected)==budget:break
        return selected

    def add_node(self,result,parent,operation,detail,generation):
        gap=(0.,0.)
        if failed_count(result):
            targets,worst,_=self.targets(result)
            guidance=self.model.proxy(result['layout'],targets)
            gap=(guidance['loss'],guidance['sum_loss'])
            detail=dict(detail,hardest_original_demand=worst)
        node=Node(len(self.nodes),parent,result,operation,detail,generation,gap)
        self.nodes.append(node);self.visited.add(result['layout'].key())
        directory=self.out/'tree_states';directory.mkdir(exist_ok=True)
        path=directory/f'{node.index:04d}.npz';layout=result['layout']
        np.savez_compressed(path,placements=layout.placements,directions=layout.directions,
            hosts=layout.hosts,active=np.array(layout.active))
        return node

    def write_tree(self,frontier):
        save(self.out/'tree.json',dict(schema='whole_compact_tree_v1',
            frontier=[n.index for n in frontier],nodes=[dict(index=n.index,parent=n.parent,
                operation=n.operation,detail=n.detail,generation=n.generation,
                sampled_failed_loads=failed_count(n.result),estimated_volume_cm3=n.result['volume_cm3'],
                volume_epoch=n.result.get('volume_epoch'),gap=n.gap,
                hosts=n.result['layout'].hosts.tolist(),layout=f'tree_states/{n.index:04d}.npz') for n in self.nodes],
            screening_is_not_actual_geometry_acceptance=True))

    def run(self,baseline):
        began=time.monotonic();m=self.model
        first=m.evaluate(baseline['layout']);m.commit(first)
        root=self.add_node(first,None,'checked-baseline',{},0);frontier=[root]
        for generation in range(1,self.rounds+1):
            if self.full_checks>=self.full_budget:break
            proposals=[]
            quota=max(1,self.screen_budget//len(frontier))
            for node in frontier:
                m.commit(node.result)
                rows=self.sample_rows(self.proposals(node),quota)
                proposals.extend((node,row) for row in rows)
            if not proposals:break
            m.volume_guidance([n.result['layout'] for n in self.nodes]+[row[1] for _,row in proposals])
            for node in self.nodes:m.refresh_volume(node.result)
            estimates=[]
            for parent,(kind,layout,detail) in proposals:
                m.commit(parent.result)
                try:estimates.append((m.volume_delta.estimate(layout),parent,kind,layout,detail))
                except (RuntimeError,ValueError) as error:
                    self.events.append(dict(generation=generation,operation=kind,screen_error=str(error)))
            estimates.sort(key=lambda r:r[0])
            # Test the low-material tail and representative structural moves.
            pool=estimates[:24]
            for tool in ['juxtapose','restore-reuse','translation','direction','direction-coherent']:
                for row in [r for r in estimates if r[2]==tool][:3]:
                    if not any(row[3].key()==r[3].key() for r in pool):pool.append(row)
            screened=[]
            for volume,parent,kind,layout,detail in pool:
                m.commit(parent.result)
                targets=self.guard_targets(layout)
                if failed_count(parent.result):targets=self.targets(parent.result)[0]
                try:
                    proxy=m.proxy(layout,targets)
                    screened.append((proxy['loss'],volume,parent,kind,layout,detail,proxy))
                except (RuntimeError,ValueError):continue
            screened.sort(key=lambda r:r[:2])
            limit=min(8,self.full_budget-self.full_checks)
            selected=[];patterns=set()
            for row in screened:
                if row[0]>1e-12:continue
                key=(row[3],tuple(row[4].hosts))
                if key not in patterns:
                    selected.append(row);patterns.add(key)
                if len(selected)>=max(1,limit-2*self.repair_width):break
            # A bounded repair frontier may cross a sampled infeasible region.
            # Only full actual all-load feasibility can update the incumbent.
            if self.repair_width:
                repairs=[r for r in screened if r[0]>1e-12 and r[1]<root.result['volume_cm3']
                    and r[3] in ['juxtapose','restore-reuse','translation']]
                for row in repairs[:2]:
                    if len(selected)<limit:selected.append(row)
            for row in screened:
                if len(selected)>=limit:break
                if any(row[4].key()==r[4].key() for r in selected):continue
                selected.append(row)
            event=dict(generation=generation,parents=[n.index for n in frontier],
                material_screens=len(estimates),guard_screens=len(screened),trials=[])
            for proxy_loss,volume,parent,kind,layout,detail,proxy in selected:
                if layout.key() in self.visited:continue
                self.full_checks+=1;m.commit(parent.result)
                trial=dict(parent=parent.index,operation=kind,detail=detail,
                    estimated_volume_cm3=volume,guard=proxy)
                try:
                    result=m.evaluate(layout)
                    if self.mode=='greedy' and (failed_count(result) or result['volume_cm3']>=parent.result['volume_cm3']):
                        self.visited.add(layout.key());trial['greedy_rejected']=True
                    else:
                        node=self.add_node(result,parent.index,kind,detail,generation)
                        trial.update(node=node.index,sampled_failed_loads=failed_count(result))
                except (RuntimeError,ValueError,AssertionError) as error:trial['error']=str(error)
                event['trials'].append(trial)
            frontier=distinct_frontier(self.nodes,self.width,self.repair_width)
            if not frontier:frontier=[root]
            event.update(frontier=[n.index for n in frontier],
                best_sampled_volume_cm3=min((n.result['volume_cm3'] for n in self.nodes
                    if not failed_count(n.result)),default=None),full_checks=self.full_checks)
            self.events.append(event);save(self.out/'trace.json',self.events);self.write_tree(frontier)
            print('COMPACT',self.mode,generation,'full',self.full_checks,
                'best sampled',event['best_sampled_volume_cm3'],'frontier',event['frontier'],flush=True)
        self.search_seconds=time.monotonic()-began
        return sorted([n for n in self.nodes if not failed_count(n.result)],key=lambda n:n.result['volume_cm3'])

    def actual_finalists(self,nodes,budget):
        selected=[];patterns=set()
        for node in nodes:
            if node.index==0:continue
            pattern=tuple(node.result['layout'].hosts)
            if pattern not in patterns:
                selected.append(node);patterns.add(pattern)
                if len(selected)==budget:return selected
        for node in nodes:
            if node.index and all(node.index!=n.index for n in selected):
                selected.append(node)
                if len(selected)==budget:break
        return selected

    def export_path(self,selected,actual):
        path=[]
        if selected is not None:
            node=selected
            while node is not None:
                path.append(node);node=self.nodes[node.parent] if node.parent is not None else None
            path.reverse()
        else:path=[self.nodes[0]]
        for node in path:
            self.process_snapshot(node.result,node.operation,
                dict(tree_node=node.index,parent=node.parent,detail=node.detail,
                    provisional_sampled_failed_loads=failed_count(node.result)))
        save(self.out/'selected_path.json',dict(tree_nodes=[n.index for n in path],
            final_layout=actual['layout'].key()==path[-1].result['layout'].key(),
            temporary_sampled_failures_are_provisional=True))

    def validate_and_save(self,nodes,baseline,baseline_report,source,real_budget=3):
        began=time.monotonic();m=self.model;best=baseline;selected=None;attempts=[]
        for node in self.actual_finalists(nodes,real_budget):
            trial=dict(tree_node=node.index,estimated_volume_cm3=node.result['volume_cm3'],accepted=False)
            try:
                actual=m.exact(node.result['layout'])
                passed=(failed_count(actual)==0 and all(row['passed'] for row in actual['actual_work_surface_checks']))
                trial.update(actual_volume_cm3=actual['volume_cm3'],force_exit_work_passed=passed,
                    counts=actual['counts'])
                if passed and actual['volume_cm3']<best['volume_cm3']-1e-4:
                    best,selected=actual,node;trial['accepted_at_time']=True
            except (RuntimeError,ValueError,AssertionError) as error:trial['error']=str(error)
            attempts.append(trial);save(self.out/'actual_validation.json',attempts)
        for trial in attempts:trial['accepted']=selected is not None and trial['tree_node']==selected.index
        save(self.out/'actual_validation.json',attempts)
        self.export_path(selected,best)
        report=Model.save(m,best,self.out,dict(schema='whole_compact_volume_v1',
            strategy='whole',compact_method=self.mode,passed=True,status='pass',
            initialized_all_poses_together=True,baseline_volume_cm3=baseline['volume_cm3'],
            material_saved_cm3=baseline['volume_cm3']-best['volume_cm3'],
            material_saved_fraction=1-best['volume_cm3']/baseline['volume_cm3'],
            baseline_retained=selected is None,selected_tree_node=None if selected is None else selected.index,
            original_baseline_report_sha256=I.sha256(source/'data/report.json'),
            baseline_diagnostics_not_relabelled_as_new_evaluation=selected is None,
            budget=dict(rounds=self.rounds,feasible_width=self.width,repair_width=self.repair_width,
                material_screens_per_round=self.screen_budget,full_load_evaluations=self.full_budget,
                real_finalists=real_budget,seed=42),
            full_load_evaluations_used=self.full_checks,search_seconds=self.search_seconds,
            validation_seconds=time.monotonic()-began,final_validation_attempts=attempts,
            rotating_reuse_pose_count=sum(registered(best['layout'],k) for k in best['layout'].active),
            juxtaposed_pose_count=juxtaposed_count(best['layout']),
            fixture_placement_count_is_not_cost=True,separated_fallback_used=False,
            all_pose_juxtapose_disabled=True,relocation_sweep_carved=False,
            objective='minimize actual solid volume subject to all original loads, work and full legal exits',
            optimization_is_heuristic=True,timing=m.timing))
        report['provenance']['inputs'].update(I.hashes([source/'data/report.json',source/'layout.npz',source/'support.obj']))
        report['provenance']['code'].update(I.hashes([Path(__file__)]))
        save(self.out/'report.json',report)
        return report
