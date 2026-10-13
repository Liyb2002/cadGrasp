"""Small-step two-tool search with an explicitly reported computed layout bound."""
import time
from pathlib import Path
import numpy as np
from co_common import save,S,D,provenance,code_sources
from sampling_recovery import SamplingRecoverySearch,RecoveryAdapter
from co_common import transform_points,U
from physics_guided_geometry import SweepDistanceModel
from physics_guided_field_fast import FastObjectDistanceField
import physics_guided_field
from computed_packing import computed_fallback


class VertexRecoveryAdapter(RecoveryAdapter):
    def __init__(self,search):
        self.search=search;self.points=search.tri.reshape(-1,3)
        self.ray_normals=np.repeat(search.mesh.face_normals[search.src],3,axis=0)
        self.rays=[]
        for task,T in search.states:
            world=transform_points(self.points,T);normal=np.repeat(-task.domain.mesh.face_normals[search.src],3,axis=0)
            self.rays.append(U.heads(np.c_[normal,np.cross(world-task.domain.com,normal)],task.scale))
        physics_guided_field.ObjectDistanceField=FastObjectDistanceField
        self.distance_model=SweepDistanceModel(search.clearance,self.points,self.ray_normals,search.length,search.extent)


class TwoToolSamplingSearch(SamplingRecoverySearch):
    def propose(self,*args,**kwargs):
        kind=args[3] if len(args)>3 else kwargs['kind']
        if kind=='recovery-direction' and not hasattr(self,'recovery_adapter'):self.recovery_adapter=VertexRecoveryAdapter(self)
        return super().propose(*args,**kwargs)

    def run(self,iterations=8,candidates=4,finalists=2):
        began=time.monotonic()
        try:report=super().run(iterations,candidates,finalists)
        except (RuntimeError,ValueError) as error:
            report=dict(complete=True,pose_set=self.group['id'],poses=self.group['poses'],force_exit_passed=False,
                initial_counts=[int(m.sum()) for m in self.saved_masks],local_search_error=str(error),iterations=[])
        local_report=report.copy();save(self.out/'local_search_report.json',local_report)
        report['small_step_search_passed']=bool(report['force_exit_passed'])
        report['computed_packing_used']=False
        if not report['force_exit_passed']:
            try:
                result=computed_fallback(self,contraction_trials=4)
                D.export_exact_obj(S.unpack(result['remaining']),self.out/'remaining_support.obj')
                np.savez_compressed(self.out/'layout.npz',directions=result['directions'],offsets_m=result['offsets'],T_fixture_to_world=np.asarray(result['transforms']))
                report.update(force_exit_passed=True,force_passed=True,geometry_constructed=True,clearance_certified=True,
                    computed_packing_used=True,baseline_used=True,baseline_passed=True,stop_reason='computed_packing_force_exit_feasible',
                    final_counts=result['counts'],offsets_m=result['offsets'].tolist(),directions=result['directions'].tolist(),
                    maximum_translation_m=float(np.linalg.norm(result['offsets'],axis=1).max()),diagnostics=result['diagnostics'],
                    component_count=len(result['remaining'].decompose()))
            except (RuntimeError,ValueError) as error:report['computed_packing_error']=str(error)
        report.update(algorithm='two-tool small-step sampling; calculated contact-boundary updates; computed packing bound and contraction fallback',
            full_fixture_accepted=False,connectivity_required=False,seconds=time.monotonic()-began,
            baseline_used=report.get('baseline_used',False),baseline_passed=report.get('baseline_passed',False),
            force_certificate='local all-load exact construction, or individually classified original-load cones preserved by certified pairwise neighbourhood separation; contraction candidates fully reclassified',
            direction_guidance='full contact vertices; analytic normals and finite-difference exit sensitivities; clipped-polygon/event alternatives',
            translation_guidance='floor-tangent derivatives of joint force-balanced shadow costs and contact-boundary events; <=1mm per pose per local update',
            deferred='connectivity, installed whole-fixture ground legality/support, strength; illegal group is not a full fixture success',
            provenance=provenance(self.inputs,[Path(__file__),Path(__file__).with_name('computed_packing.py'),Path(__file__).with_name('sampling_recovery.py'),
                Path(__file__).with_name('global_placement_sampling.py'),Path(__file__).with_name('deduplicated_placement.py'),
                Path(__file__).with_name('joint_placement_target.py'),Path(__file__).with_name('current_reaction_target.py'),
                Path(__file__).with_name('contact_boundary_model.py'),Path(__file__).with_name('contact_event_steps.py')]+code_sources()))
        save(self.out/'report.json',report)
        print('TWO TOOL FINAL',self.group['id'],report.get('final_counts'),'small',report['small_step_search_passed'],'packing',report['computed_packing_used'],flush=True)
        return report
