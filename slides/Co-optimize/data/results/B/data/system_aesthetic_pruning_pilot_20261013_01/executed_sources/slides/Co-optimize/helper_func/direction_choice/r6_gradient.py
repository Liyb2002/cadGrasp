"""R6 projection-cell reuse for direction and translation event derivatives.

The all-pose objective, demand nodes, weights and normalization are unchanged.
Analytic projection cells are reused only after primal and all-ray dual KKT
checks. Contact availability remains a hard physical predicate, so operation
derivatives are contact-event secants, not derivatives of the load parameters.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
from numba import njit
from continuous_support.projection import project_demands
from whole_search.common import C


def _objects_module():
    from co_common import ROOT
    folder=str(ROOT/'objects')
    if folder not in sys.path:sys.path.insert(0,folder)
    from r6_quadratic import QuadraticR6
    return QuadraticR6


class ForwardNodes:
    """Evaluate original quadrature coordinates through the compiled formula."""
    def __init__(self,task):
        folder=Path(task.inputs[0]).parent/f'angle_{int(task.domain.data["load"]["cone_half_deg"])}'
        self.expression=_objects_module().read(folder);self.task=task
        with np.load(folder/'samples.npz') as source:
            self.faces=source['work_face_index'].copy();self.parameters=source['parameters'].copy()
        if len(self.faces)!=len(task.targets):raise ValueError('Original quadrature count changed')
        np.testing.assert_array_equal(self.expression.domain.com,task.domain.com)
        np.testing.assert_array_equal(self.expression.domain.gravity,task.domain.gravity)

    def evaluate(self,ids):
        # Original accepted coordinates already carry their visibility witness.
        # Build loads forward; never infer a domain from saved wrench samples.
        value=self.expression.spherical(self.faces[ids],self.parameters[ids],False)['need_wrench']
        scaled=value*self.task.scale
        np.testing.assert_allclose(scaled,self.task.targets[ids],atol=1e-13,rtol=0)
        return C.U.target(scaled)


def forward_strata(model,owner,mask,passed_nodes=32,failed_nodes=64):
    generators=getattr(model,'_r6_forward_nodes',None)
    if generators is None:generators=model._r6_forward_nodes={}
    if owner not in generators:
        generators[owner]=ForwardNodes(model.tasks[owner])
        folder=Path(model.tasks[owner].inputs[0]).parent/f'angle_{int(model.tasks[owner].domain.data["load"]["cone_half_deg"])}'
        for name in ['r6_domain.json','r6_quadratic.json','r6_quadratic.npz','needs.json','samples.npz']:
            if folder/name not in model.inputs:model.inputs.append(folder/name)
        if not getattr(model,'_r6_sources_tracked',False):
            from co_common import ROOT
            model.startup_sources.update(C.I.hashes([Path(__file__),ROOT/'objects/r6_quadratic.py',ROOT/'objects/r6_demand.py']))
            model._r6_sources_tracked=True
    ids=[];weights=[]
    for region,budget in [(np.flatnonzero(mask),passed_nodes),(np.flatnonzero(~mask),failed_nodes)]:
        if not len(region):continue
        count=len(region) if len(region)<=256 and budget==failed_nodes else min(budget,len(region))
        selected=region[np.minimum(len(region)-1,((np.arange(count)+.5)*len(region)/count).astype(int))]
        ids.extend(selected);weights.extend([len(region)/len(mask)/count]*count)
    return generators[owner].evaluate(np.asarray(ids,int)),np.asarray(weights)


@njit(cache=True, boundscheck=True)
def _solve_cells(rays, targets, seeds, tolerance=2e-10):
    count, dimension = targets.shape
    residuals = np.empty_like(targets); errors=np.zeros(count)
    bases=np.full((count,32),-1,np.int64)
    for index in range(count):
        target=targets[index]; ids=np.empty(32,np.int64); coefficients=np.zeros(32); size=0
        # Deleted contacts cannot remain in a projection basis. Refit surviving
        # columns until this warm start is primal feasible; it is not acceptance.
        for j in range(seeds.shape[1]):
            if seeds[index,j]>=0:
                ids[size]=seeds[index,j];size+=1
        while size:
            matrix=np.empty((dimension,size))
            for j in range(size):matrix[:,j]=rays[ids[j]]
            proposed=np.linalg.lstsq(matrix,target,rcond=1e-12)[0]
            if np.min(proposed)>1e-12:
                coefficients[:size]=proposed;break
            retained=0
            for j in range(size):
                if proposed[j]>1e-12: ids[retained]=ids[j];retained+=1
            size=retained
        residual=target.copy()
        for j in range(size):residual-=coefficients[j]*rays[ids[j]]
        for iteration in range(160):
            reduced=rays@residual;entering=int(np.argmax(reduced))
            if reduced[entering]<=tolerance or size>=31:break
            duplicate=False
            for j in range(size):
                if ids[j]==entering:duplicate=True
            if duplicate:break
            ids[size]=entering;coefficients[size]=0.;size+=1
            for inner in range(80):
                matrix=np.empty((dimension,size))
                for j in range(size):matrix[:,j]=rays[ids[j]]
                proposed=np.linalg.lstsq(matrix,target,rcond=1e-12)[0]
                if np.min(proposed)>1e-12:
                    coefficients[:size]=proposed;break
                fraction=1.
                for j in range(size):
                    if proposed[j]<=1e-12 and coefficients[j]-proposed[j]>0:
                        fraction=min(fraction,coefficients[j]/(coefficients[j]-proposed[j]))
                coefficients[:size]+=fraction*(proposed-coefficients[:size])
                retained=0
                for j in range(size):
                    if coefficients[j]>1e-12:
                        ids[retained]=ids[j];coefficients[retained]=coefficients[j];retained+=1
                size=retained
                if size==0:break
            residual=target.copy()
            for j in range(size):residual-=coefficients[j]*rays[ids[j]]
        residuals[index]=-residual;reduced=rays@residual
        violation=max(0.,np.max(reduced))
        for j in range(size):
            violation=max(violation,abs(reduced[ids[j]]));bases[index,j]=ids[j]
        errors[index]=violation
    return residuals,errors,bases


class ProjectionCells:
    """Fixed normalized ray bank, stable column identities, bounded bases."""
    def __init__(self, full_rays):
        raw=np.asarray(full_rays,float);lengths=np.linalg.norm(raw,axis=1);keep=lengths>0
        self.bank,inverse=np.unique(raw[keep]/lengths[keep,None],axis=0,return_inverse=True)
        self.column_map=np.full(len(raw),-1,int);self.column_map[keep]=inverse
        self.cells=OrderedDict();self.reused=0;self.solved=0;self.fallbacks=0

    def evaluate(self, column_ids, targets, signature):
        targets=np.asarray(targets,float)
        global_ids=np.unique(self.column_map[np.asarray(column_ids,int)])
        global_ids=global_ids[global_ids>=0];rays=np.ascontiguousarray(self.bank[global_ids])
        if not len(rays):return -targets.copy()
        normalization=max(1.,float(np.linalg.norm(targets,axis=1).max(initial=0.)))
        b=np.ascontiguousarray(targets/normalization)
        mapping=np.full(len(self.bank),-1,int);mapping[global_ids]=np.arange(len(global_ids))
        target_digest=hashlib.sha256(b.tobytes()).digest()
        previous=self.cells.get(signature)
        if previous is not None and previous['target_digest']!=target_digest:previous=None
        seeds=np.full((len(b),32),-1,np.int64)
        residuals=-b.copy();good=np.zeros(len(b),bool)
        if previous is not None:
            residuals=previous['residuals'].copy();good=previous['certified'].copy()
            for i,row in enumerate(previous['bases']):
                ids=row[row>=0];local=mapping[ids]
                good[i]&=bool(np.all(local>=0))
                local=local[local>=0];seeds[i,:len(local)]=local
            # Retained rays already have a dual certificate for this EXACT
            # residual and target. Only added rays need fresh inequalities.
            added=np.setdiff1d(global_ids,previous['available'],assume_unique=True)
            for start in range(0,len(added),512):
                reduced=-self.bank[added[start:start+512]]@residuals.T
                good&=np.max(reduced,axis=0)<=2e-10
        self.reused+=int(good.sum());failed=np.flatnonzero(~good)
        if len(failed):
            r,error,active=_solve_cells(rays,np.ascontiguousarray(b[failed]),np.ascontiguousarray(seeds[failed]))
            residuals[failed]=r;seeds[failed]=active;self.solved+=len(failed)
            fallback=np.flatnonzero(error>2e-8)
            if len(fallback):
                selected=failed[fallback]
                exact=project_demands(rays,b[selected])
                residuals[selected]=exact['residuals'];seeds[selected]=-1
                self.fallbacks+=len(selected)
        certified=np.ones(len(b),bool)
        if len(failed) and len(fallback):certified[failed[fallback]]=False
        out=np.full_like(seeds,-1)
        positive=seeds>=0;out[positive]=global_ids[seeds[positive]]
        self.cells[signature]=dict(bases=out,residuals=residuals.copy(),available=global_ids.copy(),
                                   target_digest=target_digest,certified=certified)
        self.cells.move_to_end(signature)
        while len(self.cells)>8:self.cells.popitem(last=False)
        return residuals*normalization


class R6GradientLoss:
    """Same fixed/stratified integration as model.proxy, cheap cell updates."""
    def __init__(self,model):
        self.model=model
        self.projectors=[ProjectionCells(np.vstack([model.floors[k],model.point_rays[k]]))
                         for k in range(len(model.tasks))]
        self.seconds=0.;self.calls=0

    def evaluate(self,layout):
        m=self.model;began=time.monotonic();self.calls+=1
        flags=m.contact_delta.state(layout).available;fixed=[];adaptive=[]
        for owner in layout.active:
            signature=hashlib.sha256(flags[owner].tobytes()).digest()
            key=(owner,signature,m.floor_signature(layout,owner))
            projector=self.projectors[owner]
            rays,ids=m.supply_at_points(owner,flags[owner],layout)
            targets,weights,scale=m.demand_screen.demands[owner]
            if key not in m.demand_screen.cache:
                residual=projector.evaluate(ids,targets,('fixed',))
                loss=float(weights@(.5*np.sum(residual**2,axis=1))/scale**2)
                maximum=float(np.max(np.abs(residual),initial=0.))
                m.demand_screen.cache[key]=(loss,maximum)
                while len(m.demand_screen.cache)>256:m.demand_screen.cache.popitem(last=False)
            fixed.append(m.demand_screen.cache[key][0])
            if getattr(m,'adaptive_demands',None):
                needs,w,epoch=m.adaptive_demands[owner]
                ak=(owner,epoch,signature,m.floor_signature(layout,owner))
                if ak not in m.adaptive_cache:
                    residual=projector.evaluate(ids,needs,('adaptive',epoch))
                    m.adaptive_cache[ak]=float(w@(.5*np.sum(residual**2,axis=1))/m.original_scales[owner]**2)
                    while len(m.adaptive_cache)>1024:m.adaptive_cache.popitem(last=False)
                adaptive.append(m.adaptive_cache[ak])
        loss=np.asarray(fixed)
        if adaptive:
            loss=(loss+np.asarray(adaptive))/2;loss[loss<1e-28]=0.
        self.seconds+=time.monotonic()-began
        m.timing.update(r6_gradient_loss_s=self.seconds,r6_gradient_loss_calls=self.calls,
                       r6_projection_cells_reused=sum(p.reused for p in self.projectors),
                       r6_projection_nodes_solved=sum(p.solved for p in self.projectors),
                       r6_projection_fallbacks=sum(p.fallbacks for p in self.projectors))
        return float(loss.mean())


def gradient_loss(model,layout):
    """Internal operation evaluator; no search/orchestration signature changes."""
    if not hasattr(model,'demand_screen'):
        return model.proxy(layout)['loss']
    engine=getattr(model,'_r6_gradient_loss',None)
    if engine is None:engine=model._r6_gradient_loss=R6GradientLoss(model)
    return engine.evaluate(layout)
