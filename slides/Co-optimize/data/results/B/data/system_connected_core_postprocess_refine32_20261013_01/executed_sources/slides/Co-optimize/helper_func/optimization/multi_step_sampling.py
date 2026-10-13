"""Bounded two-tool, multi-scale search without a packing fallback.

Derivatives use contact boundaries. Only shortlisted candidates reconstruct a
support. Acceptance protects every previously passing original load and requires
strict improvement of the global worst cone distance.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from co_common import S,provenance,code_sources
from sampling_recovery import SamplingRecoverySearch
from two_tool_sampling import VertexRecoveryAdapter


def shortlist_multiscale(rows,limit):
    """Alternate tools and reserve distinct step fractions before repeat blocks."""
    ordered=sorted(rows,key=lambda row:(row[4]['loss'],row[4]['sum_loss']))
    buckets={tool:[r for r in ordered if tool in r[1]] for tool in ('direction','translation')}
    chosen=[];used={tool:set() for tool in buckets}
    while len(chosen)<limit:
        added=False
        for tool,items in buckets.items():
            fresh=[r for r in items if not any(r is c for c in chosen)]
            diverse=[r for r in fresh if r[5].get('step_fraction',1.) not in used[tool]]
            if not fresh:continue
            r=(diverse or fresh)[0];chosen.append(r)
            used[tool].add(r[5].get('step_fraction',1.));added=True
            if len(chosen)>=limit:break
        if not added:break
    return chosen


def improvement_allowed(before_loss,after_loss,protected,expansion,cap):
    return bool(protected and expansion<=cap and after_loss<before_loss-max(1e-10,before_loss*1e-5))


class MultiStepSamplingSearch(SamplingRecoverySearch):
    strict_multistep=True
    step_fractions=(4.,2.,1.,.5,.25,.125,.0625)
    translation_cap_m=.01
    direction_cap_degrees=15.
    footprint_cap=.20

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.footprint_baseline=None;self.exact_cache={};self.exact_cache_hits=0
        self.footprint_frames=[T[:3,:3].T[:,:2] for task,T in self.states]

    def footprint_areas(self,solid):
        vertices=np.asarray(S.unpack(solid).vertices)
        return [float(ConvexHull(vertices@frame).volume) for frame in self.footprint_frames]

    def exact(self,directions,offsets):
        key=(np.round(directions,12).tobytes(),np.round(offsets,12).tobytes())
        if key in self.exact_cache:
            self.exact_cache_hits+=1;return self.exact_cache[key]
        result=super().exact(directions,offsets)
        if len(self.exact_cache)>=24:self.exact_cache.pop(next(iter(self.exact_cache)))
        self.exact_cache[key]=result
        if self.footprint_baseline is None:self.footprint_baseline=np.asarray(self.footprint_areas(result['remaining']))
        return result

    def footprint(self,solid):
        areas=np.asarray(self.footprint_areas(solid))
        return dict(area_m2=areas.tolist(),initial_area_m2=self.footprint_baseline.tolist(),
            maximum_expansion_fraction=float(np.maximum(areas/self.footprint_baseline-1.,0.).max()),
            definition='convex hull of exact remaining support projected on each native floor; maximum relative expansion')

    def layout_limit(self,d,o):
        if np.linalg.norm(o[0])>1e-12:return 'reference pose moved'
        if np.max(np.abs(np.sum(o*self.normals,axis=1)))>1e-10:return 'translation leaves native floor plane'
        if np.max(np.linalg.norm(o,axis=1))>self.translation_cap_m+1e-12:return 'cumulative translation cap'
        angle=np.rad2deg(np.arccos(np.clip(np.sum(d*self.initial_directions,axis=1),-1,1)))
        if angle.max()>self.direction_cap_degrees+1e-9:return 'cumulative direction cap'
        if np.min(np.sum(d*self.normals,axis=1))<-1e-12:return 'illegal exit hemisphere'
        return None

    def propose(self,*args,**kwargs):
        kind=args[3] if len(args)>3 else kwargs['kind']
        if kind=='recovery-direction' and not hasattr(self,'recovery_adapter'):self.recovery_adapter=VertexRecoveryAdapter(self)
        return super().propose(*args,**kwargs)

    def select_candidates(self,proposed,events,limit,directions,offsets):
        # Contact events provide a second source when the differential model is
        # on a plateau. Scale their calculated displacement; do not carve a
        # relocation path or accept an event just because its proxy improves.
        scaled=[];targets=self.refresh_targets(self.latest_current)
        for tool in ('direction','translation'):
            selected=[row for row in events if tool in row[1]][:2]
            for row in selected:
                for fraction in self.step_fractions:
                    d=(1.-fraction)*directions+fraction*row[2]
                    d/=np.linalg.norm(d,axis=1)[:,None]
                    o=offsets+fraction*(row[3]-offsets)
                    if self.layout_limit(d,o):continue
                    value=self.boundary.evaluate(d,o,targets)
                    info=dict(row[5],step_fraction=fraction)
                    scaled.append(((value['loss'],value['sum_loss']),row[1],d,o,value,info))
        valid=[r for r in proposed+scaled if not self.layout_limit(r[2],r[3])]
        # Identical candidates from different guidance models spend one slot.
        unique={}
        for row in valid:
            key=(np.round(row[2],12).tobytes(),np.round(row[3],12).tobytes())
            unique.setdefault(key,row)
        return shortlist_multiscale(list(unique.values()),limit)

    def accept_trial(self,before,after,protected,footprint):
        return improvement_allowed(before,after,protected,footprint['maximum_expansion_fraction'],self.footprint_cap)

    def trial_rank(self,after,before,footprint,d,o):
        # Near-equivalent gains (<0.1% of current worst loss) share a bucket.
        # Within it use physical footprint, then translation and angle. These
        # are lexicographic decisions, never a weighted sum of unlike units.
        width=max(1e-10,before*.001)
        loss_bucket=int(np.floor(after/width))
        angle=float(np.rad2deg(np.arccos(np.clip(np.sum(d*self.initial_directions,axis=1),-1,1))).max())
        return (loss_bucket,footprint['maximum_expansion_fraction'],float(np.linalg.norm(o,axis=1).max()),angle,after)

    def finish_report(self,report,current):
        report.update(algorithm='multi-step two-tool sampling; strict protected-load/global-worst acceptance; no packing fallback',
            small_step_search_passed=report['force_exit_passed'],computed_packing_used=False,
            step_fractions=list(self.step_fractions),exact_cache_hits=self.exact_cache_hits,
            maximum_translation_step_m=.004,maximum_direction_step_degrees=4.,cumulative_translation_cap_m=self.translation_cap_m,
            cumulative_direction_cap_degrees=self.direction_cap_degrees,footprint_expansion_cap=self.footprint_cap,
            footprint=self.footprint(current['remaining']) if current else None,
            acceptance='every passing original load protected; strict global worst cone-gap decrease; exact exit/clearance/work checks; cumulative layout and footprint caps',
            provenance=provenance(self.inputs,[Path(__file__),Path(__file__).with_name('global_placement_sampling.py'),
                Path(__file__).with_name('sampling_recovery.py'),Path(__file__).with_name('joint_placement_target.py'),
                Path(__file__).with_name('local_placement_sampling.py'),Path(__file__).with_name('current_reaction_target.py'),
                Path(__file__).with_name('contact_boundary_model.py'),Path(__file__).with_name('contact_event_steps.py'),
                Path(__file__).with_name('deduplicated_placement.py'),Path(__file__).with_name('two_tool_sampling.py')]+code_sources()))
