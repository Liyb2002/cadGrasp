"""Choose a Juxtapose seat without forcing its anchor's fixture orientation.

A fixture STATE and an object SEAT are distinct. The existing Layout already
represents both: hosts chooses the world fixture transform, placements the
object's seat. Registered guests can obtain an explicit, overlapping new
seat near another pose while keeping their current fixture state. Their
subsequent Translation remains explicit-Juxtapose-only.
"""
import hashlib
import numpy as np
from .balanced_gradient import StableGradientModel, BalancedGradientSearch
from whole_search.common import C, tangent_frame, transform_mesh


class StateSeatGradientSearch(BalancedGradientSearch):
    def current_state_seats(self, current, guests):
        model, layout = self.model, current['layout']
        rows = []
        for guest in guests:
            center = C.transform_points(model.mesh.center_mass[None], layout.placements[guest])[0]
            frame = tangent_frame(model.floor_normal(layout, guest))
            anchors = [k for k in layout.active if k != guest]
            anchors.sort(key=lambda k: (not current['masks'][k].all(),
                -float(self.world_direction(layout, guest) @ self.world_direction(layout, k))))
            for host in anchors[:4]:
                host_center = C.transform_points(model.mesh.center_mass[None], layout.placements[host])[0]
                alignment = frame @ (frame.T @ (host_center-center))
                shifts = [alignment]
                for fraction in [1/32, 1/8, 1/3]:
                    shifts += [alignment + model.extent*fraction*sign*frame[:,axis]
                               for axis in range(2) for sign in [-1., 1.]]
                host_bounds = transform_mesh(model.mesh, layout.placements[host]).bounds
                for shift in shifts:
                    if np.linalg.norm(shift) < model.extent*1e-12:
                        continue
                    trial = layout.copy()
                    trial.placements[guest,:3,3] += shift
                    bounds = transform_mesh(model.mesh, trial.placements[guest]).bounds
                    intersection = np.maximum(0., np.minimum(bounds[1],host_bounds[1])-np.maximum(bounds[0],host_bounds[0]))
                    overlap = float(np.prod(intersection)/min(np.prod(np.ptp(bounds,axis=0)),
                                                             np.prod(np.ptp(host_bounds,axis=0))))
                    if overlap < .03:
                        continue
                    signature = hashlib.sha256(repr((layout.key(), trial.key(), 'current-state-seat')).encode()).hexdigest()
                    if signature in getattr(self, 'seat_hashes', set()):
                        continue
                    detail = dict(guest=model.poses[guest],guest_index=guest,host=model.poses[host],
                        fixture_state=model.poses[int(layout.hosts[guest])],fixture_state_preserved=True,
                        anchored_to_another_pose=True,horizontal_offset_fixture_m=shift.tolist(),
                        body_bbox_overlap_fraction=overlap,direction_host_weight='preserve_current_state',
                        seat_signature=signature,gradient_step=False)
                    rows.append(('juxtapose-current-state',trial,detail))
        return rows

    def juxtapose_proposals(self, current, guests, targets):
        return super().juxtapose_proposals(current,guests,targets)+self.current_state_seats(current,guests)

    def shortlist(self, proposals, targets):
        seats = [r for r in proposals if r[0]=='juxtapose-current-state']
        if not seats:
            return super().shortlist(proposals,targets)
        ordinary = [r for r in proposals if r[0]!='juxtapose-current-state']
        budget = self.screen_budget
        seat_budget = min(24,len(seats),max(1,budget//3))
        # Evenly cover guests/anchors/scales. Ranking uses the SAME all-pose
        # loss; one of the three existing full branches is reserved for a
        # preserved-state seat, without adding full checks or screen slots.
        groups = {}
        for r in seats:
            groups.setdefault((r[2]['guest_index'],r[2]['host']),[]).append(r)
        selected_seats=[]
        for index in range(seat_budget):
            key=list(groups)[index%len(groups)];pool=groups[key]
            offset=(index*5+index//len(groups)*7)%len(pool)
            selected_seats.append(pool[offset])
        self.screen_budget=budget-seat_budget
        try:
            selected,count=super().shortlist(ordinary,targets)
        finally:
            self.screen_budget=budget
        ranked=[]
        for kind,layout,detail in selected_seats:
            proxy=self.model.proxy(layout)
            ranked.append((proxy['loss'],proxy['sum_loss'],proxy['span_m'],kind,layout,detail,proxy))
        ranked.sort(key=lambda r:r[:3])
        # Retain a paired jump slot if the parent explicitly reserved one.
        pairs=[r for r in selected if r[3]=='juxtapose-pair']
        singles=[r for r in selected if r[3]!='juxtapose-pair']
        slots=max(0,self.finalists-1)
        selected=(singles[:max(0,slots-len(pairs[:1]))]+pairs[:1])[:slots]+ranked[:1]
        selected.sort(key=lambda r:r[:3])
        return selected,count+len(selected_seats)
