"""Broader contact-start/routing proposals for the copied baseline constructor.

All complete cores remain uncut. The normal baseline proposals are tried first;
recovery expands proposal locations, never relaxes geometry acceptance.
"""
from pathlib import Path
import time
import numpy as np
from step3_scheculer import operation_dsl as F
from step4_connect_support import growing_support as L


class RecoveryGrow(F.Grow):
    def attach(self,original,points,name):
        try:return super().attach(original,points,name)
        except RuntimeError:
            if ':foot' in name:raise
        pose,ident=name.split(':',1);k=self.case.poses.index(pose)
        c=next(c for c in self.case.groups[k] if c['candidate_id']==ident)
        b,o=self.bases[k],self.offsets[k];mesh=self.case.tasks[k].domain.mesh
        triangles=c['triangles_m'];order=np.argsort(-c['triangle_areas_m2'])[:6]
        vertices=triangles.reshape(-1,3)
        extremes=np.unique(np.r_[np.argmin(vertices,axis=0),np.argmax(vertices,axis=0)])
        centers=np.vstack([c['center_m'],triangles.mean(axis=1)[order],vertices[extremes]])@b+o
        normals=np.unique(np.round(mesh.face_normals[np.unique(c['source_faces'])]@b,10),axis=0)
        average=np.sum(mesh.face_normals[c['source_faces']]*c['triangle_areas_m2'][:,None],axis=0)@b
        average/=np.linalg.norm(average)
        from itertools import product
        directions=np.asarray([v for v in product((-1.,0.,1.),repeat=3) if any(v)])
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        normals=np.vstack([average,normals,directions])
        floor_minima=np.asarray([(self.bead@basis.T)[:,2].min() for basis in self.bases])
        if not hasattr(self,'contact_transition_space'):
            raw=F.F.union([F.F.transform(F.S.solid(mesh),basis,offset) for mesh,basis,offset in zip(self.sweep_meshes,self.bases,self.offsets)])
            self.contact_transition_space=F.F.bounded_space(self.navigation_window,self.bases,self.offsets)-raw
        contact_space=self.contact_transition_space+original
        for center in centers:
            for normal in normals:
                for depth in (.0031,.0051,.0061,.0081,.0101,.0121):
                    start=center+depth*normal
                    if any(((start-offset)@basis.T)[2]+lower < -1e-10 for basis,offset,lower in zip(self.bases,self.offsets,floor_minima)):continue
                    ball=self.sphere(start)
                    if not self.legal(ball):continue
                    transition=F.S.solid(F.G.hull_mesh(np.vstack([points,start+self.bead]))) ^ contact_space
                    seed=original+transition+ball
                    if len(L.components(seed))!=1:continue
                    if abs(float((original-seed).volume()))*F.S.SCALE**3>8e-14:continue
                    self.mask=self.mask+transition
                    index=len(self.seed_positions);self.seed_positions.append(start)
                    self.terminals.append(dict(name=name,node=index,solid=seed,kind='head_start'))
                    self.core_solids.append((name+':start_ball',ball))
                    self.thickness.append(dict(name=name,start_m=start.tolist(),guaranteed_core_diameter_mm=self.guaranteed_radius*2000,
                        original_contact_transition_length_mm=depth*1000,transition_normal='actual patch face/centroid recovery',transition_may_taper=True,preset_grid_anchor=False,contact_padding_exception_m=F.S.RELIEF,actual_continuous_sweep_enforced=True))
                    print('RECOVERY CONTACT START',self.group.name,name,'neck_mm',depth*1000,flush=True)
                    return
        raise RuntimeError('No legal complete-core contact start after patch-position recovery: '+name)

    def merge_actions(self,state):
        proposals=super().merge_actions(state)
        if proposals or self.direct_only or len(set(state['labels']))==1:return proposals
        for point,owner in zip(state['seeds'],state['seed_owners']):
            try:route,target=self.routed(state,np.asarray(point),state['labels'][owner])
            except RuntimeError:continue
            action=self.action(state,'merge',route,owner,target)
            if action is not None:return [action]
        return []


class RecoveryOptimizer(F.Optimizer):
    def initialize(self):
        try:return super().initialize()
        except RuntimeError as error:
            if not str(error).startswith('Latest baseline Step4 construction failed'):raise
        if self.seed.shared_count:raise RuntimeError('Shared incumbent seating cannot be changed independently')
        # Force and object-frame exit objectives are translation invariant.
        from step3_scheculer.feasible_seating import restore
        for margin in (.008,.015,.025,.05):
            state=F.seat(self.tasks,self.seed,margin)
            if state is None:continue
            restored=restore(self.tasks,state,F.roots,attempt=0)
            if restored is None:continue
            state,cuts=restored
            check=self.checker.check(state)
            if not check['passed']:continue
            witness=self.construct(state,check,'initial_baseline_seating_recovery')
            self.event('seating_recovery',margin_m=margin,passed=witness is not None)
            if witness is None:continue
            self.incumbent=self.initial_state=state;self.witness=witness;self.check=check
            F.save_state(self.out/'initial',self.tasks,state,check)
            F.I.save(self.out/'initialization.json',dict(complete=True,passed=True,initialized_physical_heads=state.physical_count,
                source='qualified V5 contacts/paths; translations restored for latest constructor',latest_baseline_step4_rerun=True))
            self.event('initialize_feasible',physical_heads=state.physical_count,objective=list(F.objective(self.tasks,state)),witness=str((witness[0]/'report.json').relative_to(F.I.ROOT)))
            return
        raise RuntimeError('Latest baseline constructor exhausted contact-start/routing/seating recovery')


def activate():
    if F.Grow is RecoveryGrow:return
    original_sources=F.sources
    F.Grow=RecoveryGrow;F.Optimizer=RecoveryOptimizer
    F.sources=lambda:list(dict.fromkeys(original_sources()+[Path(__file__),Path(__file__).with_name('run_operation_dsl.py')]))
