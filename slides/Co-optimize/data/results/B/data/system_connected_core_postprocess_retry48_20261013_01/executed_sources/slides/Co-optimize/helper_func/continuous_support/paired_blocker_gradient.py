"""Bounded two-pose Juxtapose samples can release jointly locked contacts."""
import hashlib
import numpy as np
from .blocker_gradient import BlockerGradientSearch
from .releasable_blockers import ReleasableBlockerModel


class PairedBlockerGradientSearch(BlockerGradientSearch):
    def jump_guests(self,current):
        guests,blockers,detail=super().jump_guests(current)
        self.jump_blockers=blockers
        return guests,blockers,detail

    def juxtapose_proposals(self,current,guests,targets):
        rows=super().juxtapose_proposals(current,guests,targets)
        blockers=getattr(self,'jump_blockers',[])
        if len(blockers)<2:return rows
        model=self.model;old=model.contact_delta.state(current['layout'])
        weights=model.jump_ray_weights
        chosen=[]
        for blocker in blockers[:2]:
            candidates=[r for r in rows if r[2]['guest_index']==blocker]
            # Geometry-only shortlist for the paired jump: choose placements
            # that remove useful locks, even if another blocker still hides
            # the resulting improvement in the SINGLE candidate's cone.
            ranked=[]
            for kind,layout,detail in candidates:
                if any(abs(v)>1e-12 for v in detail.get('horizontal_offset_fixture_m',[0.,0.,0.])):
                    continue
                state=model.contact_delta.state(layout)
                release=sum(float(weights[owner][old.locks[owner,blocker]&~state.locks[owner,blocker]].sum())
                            for owner in current['layout'].active)
                ranked.append((release,kind,layout,detail))
            ranked.sort(key=lambda r:-r[0]);unique=[];seen=set()
            for r in ranked:
                signature=(r[2].placements[blocker].tobytes(),r[2].directions[blocker].tobytes(),int(r[2].hosts[blocker]))
                if signature not in seen:
                    unique.append(r);seen.add(signature)
                if len(unique)==2:break
            chosen.append(unique)
        pairs=[]
        first,second=blockers[:2]
        for a in chosen[0]:
            for b in chosen[1]:
                trial=current['layout'].copy()
                for k,r in [(first,a),(second,b)]:
                    trial.placements[k]=r[2].placements[k]
                    trial.directions[k]=r[2].directions[k]
                    trial.hosts[k]=r[2].hosts[k]
                signature=hashlib.sha256(repr((current['layout'].key(),trial.key(),'paired')).encode()).hexdigest()
                if signature in getattr(self,'seat_hashes',set()):continue
                detail=dict(guest=model.poses[first],guest_index=first,host='paired',
                            direction_host_weight='paired',second_guest_index=second,
                            paired_poses=[model.poses[first],model.poses[second]],
                            component_details=[a[3],b[3]],seat_signature=signature,
                            release_score=a[0]+b[0],gradient_step=False)
                pairs.append(('juxtapose-pair',trial,detail))
        # Reserve at most four of the same 96 cheap screens for these pairs.
        # Full loss and original-load checks still evaluate their net effects.
        self.pending_pair_rows=pairs
        return rows

    def shortlist(self,proposals,targets):
        pairs=getattr(self,'pending_pair_rows',[])
        if not pairs:return super().shortlist(proposals,targets)
        self.pending_pair_rows=[]
        previous=self.screen_budget;self.screen_budget=max(1,previous-len(pairs))
        try:selected,count=super().shortlist(proposals,targets)
        finally:self.screen_budget=previous
        ranked=[]
        for kind,layout,detail in pairs:
            proxy=self.model.proxy(layout)
            ranked.append((proxy['loss'],proxy['sum_loss'],proxy['span_m'],kind,layout,detail,proxy))
        ranked.sort(key=lambda r:r[:3])
        if ranked:
            # A pair must get a real branch even when single candidates tie
            # because they cannot yet release a multiply locked point.
            selected=selected[:max(0,self.finalists-1)]+ranked[:1]
            selected.sort(key=lambda r:r[:3])
        return selected,count+len(pairs)
