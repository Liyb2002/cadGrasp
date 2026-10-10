"""Small-step two-tool sampling with independent and coherent joint updates."""
import numpy as np
from global_placement_sampling import GlobalPlacementSearch
from physics_guided_geometry import tangent_frames,retract,SweepDistanceModel
from physics_guided_field_fast import FastObjectDistanceField
from contact_recovery import RecoveryTarget
import physics_guided_field


class RecoveryAdapter:
    def __init__(self,search):
        self.search=search;self.ray_normals=search.outward
        physics_guided_field.ObjectDistanceField=FastObjectDistanceField
        self.distance_model=SweepDistanceModel(search.clearance,search.points,search.outward,search.length,search.extent)
    def __getattr__(self,key):return getattr(self.search,key)
    def choose_loads(self,current):return [(k,i) for k,i,t in self.search.refresh_targets(current)]


class SamplingRecoverySearch(GlobalPlacementSearch):
    def refresh_targets(self,current):
        self.latest_current=current
        return super().refresh_targets(current)

    def blocks(self,current,count):
        blocks=super().blocks(current,count)
        blocks.extend([('translation',list(range(1,self.n))),('direction',list(range(self.n))),('recovery-direction',list(range(self.n)))])
        return blocks

    def propose(self,directions,offsets,targets,kind,indices,base):
        if kind!='recovery-direction':return super().propose(directions,offsets,targets,kind,indices,base)
        if np.max(np.linalg.norm(offsets,axis=1))>1e-10 or self.latest_current is None:
            return [],dict(kind=kind,skipped='legacy direction guidance only applicable to coincident poses')
        if not hasattr(self,'recovery_adapter'):self.recovery_adapter=RecoveryAdapter(self)
        if getattr(self,'recovery_guide',None) is not self.guide:
            self.recovery_target=RecoveryTarget(self.recovery_adapter,self.latest_current,directions)
            self.recovery_guide=self.guide
        endpoint,stats=self.recovery_target.step(directions)
        frames=tangent_frames(directions);dots=np.sum(endpoint*directions,axis=1)
        coordinates=np.einsum('nki,nk->ni',frames,endpoint/np.maximum(dots[:,None],1e-12)-directions)
        length=np.linalg.norm(coordinates)
        if length<1e-10:return [],dict(kind=kind,stalled=True)
        if length>np.deg2rad(1.):coordinates*=np.deg2rad(1.)/length
        before=self.recovery_target.nonlinear_cost(directions);options=[]
        for fraction in getattr(self,'step_fractions',(1.,.5,.25)):
            d=retract(directions,frames,fraction*coordinates)
            if np.min(np.sum(d*self.normals,axis=1))<-1e-12:continue
            after=self.recovery_target.nonlinear_cost(d)
            if after>=before-max(1e-10,before*1e-5):continue
            value=self.boundary.evaluate(d,offsets,targets)
            options.append(((after,value['loss'],value['sum_loss']),kind,d,offsets.copy(),value,
                dict(step_fraction=fraction,reaction_guidance=True,indices=indices,step=(fraction*coordinates).tolist(),
                    target_key='coincident-recovery-'+str(self.recovery_target.ids.tolist()),guidance_before=before,guidance_after=after)))
        return options,dict(kind=kind,derivative=stats,contacts=len(self.recovery_target.ids))
