"""Whole-demand numerical descent on cached contact/occupancy deltas.

The fixed contact discretization has events: finite differences are resolved
secants, not analytic shape derivatives. Small coherent/coordinate samples
cross flat contact cells. Every selected step is checked on all original loads.
No candidate Boolean, hardest-load derivative or new reaction capacity.
"""
import time
from types import SimpleNamespace
import numpy as np
from whole_search.common import tangent_frame, legal_direction
from whole_search.fast_search import FastModel, FastReuseSearch
from whole_search.reuse_first import registered
from whole_search.search import failed_count
from .complete_grid import CompleteGrid
from .boundary_projection import BoundaryDistance
from juxtapose.reuse_search import JumpScreen
from translation.proposals import height_proposals


class GradientModel(FastModel):
    def __init__(self, *args, demand_nodes=32, **kwargs):
        super().__init__(*args, **kwargs)
        distances = [BoundaryDistance(CompleteGrid.from_task(t, 2)) for t in self.tasks]
        self.demand_screen = JumpScreen(SimpleNamespace(model=self, distances=distances), demand_nodes)
        self.gradient_seconds = 0.

    def proxy(self, layout, targets=()):
        self.proxy_calls += 1
        value = self.demand_screen.evaluate(layout)
        # All owners and positive-weight demand strata are fixed BEFORE search.
        # The mean integrates both improvements and losses across all owners.
        return dict(value, sum_loss=sum(value['residual_loss']), span_m=self.layout_span(layout),
                    objective='all_pose_integrated_squared_original_cone_distance',
                    continuous_domain_certified=False)


def turn(model, layout, owner, vector):
    trial = layout.copy()
    d = layout.directions[owner]
    v = vector - d * (d @ vector)
    angle = float(np.linalg.norm(v))
    if angle:
        trial.directions[owner] = legal_direction(
            np.cos(angle)*d + np.sin(angle)*v/angle, model.floor_normal(layout, owner))
    return trial


class GradientSearch(FastReuseSearch):
    def targets(self, current):
        value = self.model.proxy(current['layout'])
        indices = list(current['layout'].active)
        failing = [k for k in indices if not current['masks'][k].all()]
        k = max(failing or indices, key=lambda j: value['residual_loss'][indices.index(j)])
        # The old orchestration needs a guest index; it never differentiates
        # this one owner/demand. ALL owners remain in every proxy evaluation.
        return [], dict(pose_index=k, selection='largest integrated owner loss'), []

    def local_proposals(self, current, targets, worst, translation_guests=()):
        m, layout = self.model, current['layout']
        began = time.monotonic(); base = m.proxy(layout)
        fine = getattr(self, 'fine_resolution', False)
        h = np.radians(.125 if fine else .75)
        rows = []; gradients = {}; measurements = []
        for k in layout.active:
            frame = tangent_frame(layout.directions[k]); g = np.zeros(2)
            for axis in range(2):
                minus = m.proxy(turn(m, layout, k, -h*frame[:, axis]))['loss']
                plus = m.proxy(turn(m, layout, k, h*frame[:, axis]))['loss']
                g[axis] = (plus-minus)/(2*h)
                measurements.append(dict(pose=m.poses[k], axis=axis,
                                         minus=minus, plus=plus, derivative=float(g[axis])))
            gradients[k] = (frame, g)
        norm = np.sqrt(sum(float(g@g) for _, g in gradients.values()))
        degrees = [.03125, .125, .25] if fine else [.25, .5, 1., 2., 4., 8.]
        if norm > 1e-28:
            for step in degrees:
                trial = layout.copy()
                for k, (frame, g) in gradients.items():
                    trial.directions[k] = turn(m, layout, k, -np.radians(step)*frame@g/norm).directions[k]
                rows.append(('direction-gradient-joint', trial,
                    dict(step_degrees=step, finite_difference_degrees=float(np.degrees(h)),
                         gradient_norm=float(norm), gradient=measurements,
                         derivative_kind='finite_resolution_contact_secant', all_poses_in_objective=True)))
        # Common rotation moves several overlapping blockers simultaneously.
        # Differentiate this shared coordinate instead of summing independently
        # flat contact-cell derivatives.
        coherent = np.zeros(3)
        def rotate(axis, angle):
            trial = layout.copy()
            for k in layout.active:
                trial.directions[k] = turn(m, layout, k, angle*np.cross(axis, layout.directions[k])).directions[k]
            return trial
        for axis in range(3):
            unit = np.eye(3)[axis]
            coherent[axis] = (m.proxy(rotate(unit,h))['loss']-m.proxy(rotate(unit,-h))['loss'])/(2*h)
        length = np.linalg.norm(coherent)
        if length > 1e-28:
            for step in degrees:
                rows.append(('direction-gradient-coherent', rotate(-coherent/length,np.radians(step)),
                    dict(step_degrees=step, gradient=coherent.tolist(), gradient_norm=float(length),
                         finite_difference_degrees=float(np.degrees(h)), all_poses_in_objective=True)))
        owners = sorted(layout.active, key=lambda k: -base['residual_loss'][list(layout.active).index(k)])[:3]
        for k in owners:
            frame, g = gradients[k]; length = np.linalg.norm(g)
            if length > 1e-28:
                for step in degrees:
                    rows.append(('direction-gradient', turn(m,layout,k,-np.radians(step)*frame@g/length),
                                 dict(pose=m.poses[k],step_degrees=step,all_poses_in_objective=True)))
            # Limited sampling, explicitly separate from gradient steps.
            for step in ([.03125,.125] if fine else [.5,2.,5.]):
                for axis in range(2):
                    for sign in [-1,1]:
                        rows.append(('direction-sample', turn(m,layout,k,sign*np.radians(step)*frame[:,axis]),
                                     dict(pose=m.poses[k],step_degrees=sign*step,axis=axis)))
        for axis in np.eye(3):
            for step in ([-.125,.125] if fine else [-2.,2.,-5.,5.]):
                rows.append(('direction-coherent-sample',rotate(axis,np.radians(step)),
                             dict(step_degrees=step,axis=axis.tolist())))
        movable = [k for k in translation_guests if not registered(layout,k)]
        for k in movable:
            frame=tangent_frame(m.floor_normal(layout,k)); g=np.zeros(2)
            delta=m.extent*(1/512 if fine else 1/100)
            for axis in range(2):
                values=[]
                for sign in [-1,1]:
                    trial=layout.copy();trial.placements[k,:3,3]+=sign*delta*frame[:,axis]
                    values.append(m.proxy(trial)['loss'])
                g[axis]=(values[1]-values[0])/(2*delta)
            vectors=[(np.eye(2)[axis]*sign,'sample') for axis in range(2) for sign in [-1,1]]
            if np.linalg.norm(g)>1e-28: vectors.insert(0,(-g/np.linalg.norm(g),'gradient'))
            for fraction in ([1/1024,1/512,1/256] if fine else [1/128,1/64,1/32,1/16,1/8]):
                for v,kind in vectors:
                    trial=layout.copy();trial.placements[k,:3,3]+=m.extent*fraction*frame@v
                    rows.append(('translation-'+kind,trial,dict(pose=m.poses[k],step_m=m.extent*fraction,
                        finite_difference_m=delta,gradient=g.tolist(),all_poses_in_objective=True)))
        if len(movable)>1:
            for axis in np.eye(3)[:2]:
                for sign in [-1,1]:
                    trial=layout.copy()
                    for k in movable:
                        trial.placements[k,:3,3]+=m.extent/256*sign*m.native[layout.hosts[k],:3,:3].T@axis
                    rows.append(('translation-coherent-sample',trial,dict(step_m=m.extent/256)))
        if getattr(m,'allow_z_translation',False):
            rows.extend(height_proposals(m,layout,movable,fine=fine))
        m.gradient_seconds += time.monotonic()-began
        return rows,base

    def shortlist(self, proposals, targets):
        # Keep original guest/host/scale diversity, but reserve an actual
        # all-load evaluation for a genuine descent-gradient proposal.
        selected, count = super().shortlist(proposals, targets)
        gradients=[p for p in proposals if 'gradient' in p[0]]
        if gradients:
            ranked=sorted(((self.model.proxy(p[1])['loss'],p) for p in gradients),key=lambda x:x[0])
            base=self.model.proxy(self.model.contact_delta.base.layout)['loss']
            value,(kind,layout,detail)=ranked[0]
            if value < base-max(1e-30,base*1e-8) and not any(r[4].key()==layout.key() for r in selected):
                proxy=self.model.proxy(layout)
                row=(proxy['loss'],proxy['sum_loss'],proxy['span_m'],kind,layout,detail,proxy)
                selected=selected[:max(0,self.finalists-1)]+[row]
        return selected,count

    def record(self, row):
        # Report actual proxy gains/losses for accepted steps, independent of
        # the discrete all-original-load safety gate/ranking.
        for trial in row.get('trials',[]):
            if trial.get('accepted') and 'gradient' in trial.get('operation',''):
                trial['whole_demand_gradient_step']=True
        super().record(row)

    def polish_volume(self, current, rounds=2):
        # Fast occupancy descent preserves ALL original sampled demands.
        # Parent retains a small coordinate/coherent pool to cross voxel cells.
        current=super().polish_volume(current,rounds)
        self.model.timing.update(whole_demand_proxy_s=self.model.demand_screen.seconds,
                                 numerical_gradient_s=self.model.gradient_seconds)
        return current
