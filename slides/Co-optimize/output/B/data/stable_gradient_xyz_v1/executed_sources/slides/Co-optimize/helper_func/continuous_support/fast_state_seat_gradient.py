"""Avoid provably ineligible full checks; guide explicit new seats by loss.

All-owner translation secants may propose a new seat even for a registered
guest. Such a proposal is an EXPLICIT Juxtapose, with another pose as its
geometric anchor; no registered pose silently translates during refinement.
"""
import hashlib
import numpy as np
from .state_seat_gradient import StableGradientModel, StateSeatGradientSearch
from whole_search.common import C, tangent_frame, transform_mesh


class FastStateSeatGradientSearch(StateSeatGradientSearch):
    def local_proposals(self, current, *args, **kwargs):
        rows, base = super().local_proposals(current, *args, **kwargs)
        self.comparison_loss = base['loss']
        return rows, base

    def proposal_groups(self, rows, fine=False):
        base = self.comparison_loss
        for label, group in super().proposal_groups(rows, fine):
            if label.startswith('gradient'):
                yield label, group
                continue
            eligible = []
            for row in group:
                loss = self.model.proxy(row[1])['loss']
                # This is EXACTLY the parent's necessary acceptance condition.
                # Zero proxy distance retains original-count blind-spot repair.
                if loss < base-max(1e-30,base*1e-8) or loss <= 1e-28:
                    eligible.append(row)
                else:
                    timing = self.model.timing
                    timing['ineligible_full_load_candidates_skipped'] = timing.get('ineligible_full_load_candidates_skipped',0)+1
            if eligible:
                yield label, eligible

    def current_state_seats(self, current, guests):
        rows = super().current_state_seats(current, guests)
        model, layout = self.model, current['layout']
        guided=[]
        for guest in guests:
            frame=tangent_frame(model.floor_normal(layout,guest))
            gradient=np.zeros(2);resolution=None
            for fraction in [1/100,1/8]:
                delta=model.extent*fraction
                for axis in range(2):
                    minus=layout.copy();plus=layout.copy()
                    minus.placements[guest,:3,3]-=delta*frame[:,axis]
                    plus.placements[guest,:3,3]+=delta*frame[:,axis]
                    gradient[axis]=(model.proxy(plus)['loss']-model.proxy(minus)['loss'])/(2*delta)
                resolution=delta
                if np.linalg.norm(gradient)>1e-28:
                    break
            length=float(np.linalg.norm(gradient))
            if length<=1e-28:
                continue
            center=C.transform_points(model.mesh.center_mass[None],layout.placements[guest])[0]
            anchors=[k for k in layout.active if k!=guest]
            anchors.sort(key=lambda k:np.linalg.norm(C.transform_points(
                model.mesh.center_mass[None],layout.placements[k])[0]-center))
            for host in anchors[:2]:
                host_bounds=transform_mesh(model.mesh,layout.placements[host]).bounds
                for fraction in [1/32,1/8,1/3]:
                    shift=-model.extent*fraction*frame@gradient/length
                    trial=layout.copy();trial.placements[guest,:3,3]+=shift
                    bounds=transform_mesh(model.mesh,trial.placements[guest]).bounds
                    overlap=np.maximum(0.,np.minimum(bounds[1],host_bounds[1])-np.maximum(bounds[0],host_bounds[0]))
                    overlap=float(np.prod(overlap)/min(np.prod(np.ptp(bounds,axis=0)),np.prod(np.ptp(host_bounds,axis=0))))
                    if overlap<.03:
                        continue
                    signature=hashlib.sha256(repr((layout.key(),trial.key(),'gradient-seat')).encode()).hexdigest()
                    if signature in getattr(self,'seat_hashes',set()):
                        continue
                    detail=dict(guest=model.poses[guest],guest_index=guest,host=model.poses[host],
                        fixture_state=model.poses[int(layout.hosts[guest])],fixture_state_preserved=True,
                        anchored_to_another_pose=True,horizontal_offset_fixture_m=shift.tolist(),
                        body_bbox_overlap_fraction=overlap,direction_host_weight='preserve_current_state',
                        seat_signature=signature,gradient_step=False,
                        proposed_by_all_pose_translation_gradient=True,translation_gradient=gradient.tolist(),
                        finite_difference_m=resolution,all_poses_in_objective=True)
                    guided.append(('juxtapose-current-state',trial,detail))
        # Put the gradient proposal ahead of the coordinate offsets in each
        # guest/anchor's deterministic quota; total 96/3 budgets are unchanged.
        return guided+rows
