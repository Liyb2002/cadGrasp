"""Construct full-solid evidence for a DSL Step3 proposal.

Placements are retained; ground coverage grows adaptively.
A failure is explicitly a fixed-placement construction failure, not infeasibility.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
import shutil
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import trimesh
from shapely.geometry import MultiPoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.run_dsl import saved_task
from step0_pose_selection.floor_points import pressure_centers
from step4_connect_support import direct_head_growth as H
from step4_connect_support.coverage_growth import CoverageGrow
from step4_connect_support import select_exit_paths as PATHS
from step4_connect_support import boxed_support as F, build_coupled_saddle as S
from step4_connect_support import deterministic_space as D, process_access as A
from step4_connect_support import growing_support as L, space_budget as B
from step4_connect_support.fixture_view import cells_for
from step2_local_support import geometry as G
from step2_local_support import circles as P


class DSLGrow(CoverageGrow):
    def __init__(self, group, choice=0, out=None, prepared=None, selected_option_ids=None, proposal=False):
        self.group = group
        self.out = Path(out) if out is not None else group/'step4/data/dsl_shared_support'
        self.out.mkdir(parents=True, exist_ok=True)
        dsl_path = group/'step3_scheculer/dsl_shared/contact_proposal.json'
        schedule = I.check_report(dsl_path)
        if not schedule['local_contact_passed']:
            raise ValueError('Contact proposal has not passed force, exit and root-connectivity checks')
        if not proposal:
            accepted = I.check_report(group/'step3_scheculer/dsl_shared/report.json')
            if not accepted['passed'] or not accepted['complete_fixture_verified']:
                raise ValueError('DSL Step3 has no accepted complete construction witness')
        tasks = [saved_task(group, pose) for pose in schedule['poses']]
        reports = [I.check_report(group/'step3_scheculer/dsl_shared'/pose/'report.json') for pose in schedule['poses']]
        contacts = [I.read_contacts(group/'step3_scheculer/dsl_shared'/pose/f'final_contacts_{pose}.npz') for pose in schedule['poses']]
        self.source = group/'step4/data/growing_support'
        if not (self.source/'report.json').is_file():
            self.source = group/'step4/data'
        self.has_comparison_fixture = (self.source/'report.json').is_file()
        if self.has_comparison_fixture:
            old = json.loads((self.source/'report.json').read_text())
            shape = self.source/'shape.obj' if (self.source/'shape.obj').exists() else group/'step4/shape.obj'
        else:
            if len(tasks) != 1:
                raise ValueError('A multi-pose proposal needs explicit placement inputs')
            # A single-pose fixture has a defined identity placement. Mandatory
            # roots are input geometry, never a fabricated prior accepted body.
            self.source = group/'step3_scheculer/dsl_shared/construction_inputs'
            self.source.mkdir(parents=True, exist_ok=True)
            old = dict(placement=dict(bases=[np.eye(3).tolist()],offsets=[[0.,0.,0.]]),
                       comparison_kind='mandatory roots only; no previous fixture')
            I.save(self.source/'report.json',old)
            task = tasks[0]
            offsets = G.vertex_offsets(task.domain.mesh,P.DEPTH_FRACTION*task.domain.mesh.extents.max())[0]
            cells = [v for c in contacts[0] for v in cells_for(c,task.domain,offsets)]
            shape = self.source/'shape.obj'
            D.export_exact_obj(S.unpack(F.union([S.solid(G.hull_mesh(v)) for v in cells])),shape)
        self.reference_report = dict(old)
        if schedule.get('schema') == 'joint_shared_head_descent_v4':
            # The baseline comparison body does not define the new placement.
            old = dict(old, placement=schedule['placement'])
        self.bases = np.asarray(old['placement']['bases'])
        self.offsets = np.asarray(old['placement']['offsets'])
        self.reference = trimesh.load(shape,force='mesh',process=False)
        self.foot_reference = self.reference.copy()
        self.directions = np.array([r['selected_withdrawal_world'] for r in reports])
        paths = [dsl_path, shape,group/'step3_scheculer/dsl_shared/sharing.json',group/'step3_scheculer/dsl_shared/shared_heads.npz']+[p for t in tasks for p in t.inputs]
        paths.append(self.source/'report.json')
        comparison = self.out/'comparison_input'
        comparison.mkdir(exist_ok=True)
        shutil.copy2(self.source/'report.json',comparison/'report.json')
        shutil.copy2(shape,comparison/'shape.obj')
        self.source = comparison
        paths += [group/'step3_scheculer/dsl_shared'/p/f'final_contacts_{p}.npz' for p in schedule['poses']]
        paths += [group/'step3_scheculer/dsl_shared'/p/'report.json' for p in schedule['poses']]
        seeds = []
        canonical_offsets = G.vertex_offsets(tasks[0].domain.mesh, schedule['shared_head_depth_m'])[0]
        from step3_scheculer.floor_margin import transforms_from_owner
        transforms = transforms_from_owner(tasks[0],tasks)
        canonical_contacts = I.read_contacts(group/'step3_scheculer/dsl_shared/shared_heads.npz')
        canonical_cells = {c['candidate_id']:cells_for(c,tasks[0].domain,canonical_offsets) for c in canonical_contacts}
        for task, row, transform in zip(tasks, contacts, transforms):
            # Match the real constructor's roots. Thin Step3 exit probes are
            # proposal checks; they cannot bridge the constructor's 0.4mm relief.
            seeds.append([[v@transform[:3,:3].T+transform[:3,3] for v in canonical_cells[c['candidate_id']]] for c in row])
        demands = [pressure_centers(t.targets/t.scale, t.domain.com)[0] for t in tasks]
        self.case = SimpleNamespace(name='B', pair=group, output=self.out, poses=schedule['poses'], tasks=tasks,
            groups=contacts, heads=[[list(c['triangles_m']) for c in row] for row in contacts],
            support_seeds=seeds, demands=demands, schedule=dict(passed=True, covered_counts=schedule['covered_counts']),
            paths=paths)
        self.reference_report['space_budget'] = D.measure(self.case,self.foot_reference,self.bases,self.offsets)
        self.case.root_solids = [F.union([S.solid(G.hull_mesh(v)) for cells in row for v in cells]) for row in seeds]
        self.shared_geometry_errors=[]
        for k,row in enumerate(seeds):
            for c,cells in zip(contacts[k],row):
                installed=[v@self.bases[k]+self.offsets[k] for v in cells]
                expected=canonical_cells[c['candidate_id']]
                error=max(float(np.max(np.abs(a-b))) for a,b in zip(installed,expected))
                if error>2e-12:raise ValueError('Shared head solids do not coincide across poses')
                self.shared_geometry_errors.append(error)
        self.case.preview_directions = self.directions
        self.reference_report['placement'] = dict(bases=self.bases.tolist(), offsets=self.offsets.tolist(), directions=self.directions.tolist())
        self.guard = A.Guard(self.case, self.reference_report['placement'])
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=.0026)
        self.bead = sphere.vertices
        self.guaranteed_radius = float(np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]))))
        self.terminals=[]; self.feet=[]; self.paths=[]; self.original_roots=[]; self.original_soles=[]
        self.thickness=[]; self.core_solids=[]; self.seed_positions=[]; self.segments=[]
        self.grid_ready=False; self.journal=[]; self.partial_steps=0
        roots = [F.transform(root,b,o) for root,b,o in zip(self.case.root_solids,self.bases,self.offsets)]
        cloud = [t.domain.mesh.vertices for t in tasks]+[np.c_[v,np.zeros(len(v))] for v in demands]
        mandatory = [S.unpack(v).vertices for v in roots]
        installed = [(np.vstack(mandatory)-o)@b.T for b,o in zip(self.bases,self.offsets)]
        self.navigation_window = D.enlarged(B.box(cloud+installed), .05)
        if prepared is None:
            selected_reports=reports
            if selected_option_ids is not None:
                # Replay the saved plans directly. Never rerank or beam-prune a
                # Step3 witness while materializing it in Step4.
                selected_reports=[dict(r,exit_options=[next(v for v in r['exit_options']
                    if v['plan']['id']==ident)]) for r,ident in zip(reports,selected_option_ids)]
            prepared=PATHS.select(tasks,selected_reports,roots,self.bases,self.offsets,self.navigation_window,
                                  beam_width=1 if selected_option_ids is not None else 3)
        self._selection=prepared
        states,selection=prepared
        I.save(self.out/'exit_selection.json',selection)
        if not states:raise RuntimeError(selection['error'])
        if selected_option_ids is not None:
            choice=next(i for i,state in enumerate(states) if [p['plan']['id'] for p in state['selected']]==selected_option_ids)
        self.selected_state=states[choice]
        selected=self.selected_state['selected']
        self.case.exit_paths=[p['plan'] for p in selected]
        self.directions=np.asarray([p['plan']['support_withdrawal_world'] for p in selected])
        self.reference_report['placement']['directions']=self.directions.tolist()
        self.case.preview_directions=self.directions
        self.guard=A.Guard(self.case,self.reference_report['placement'])
        raw_meshes=[p['raw'] for p in selected]
        forbidden=F.union([p['padded'] for p in selected])
        self.sweep_meshes=raw_meshes
        self.search=SimpleNamespace(sweep_meshes=raw_meshes,sweep_conditioning=[],precompute=lambda:None)
        self.forbidden=forbidden
        self.mask = (F.bounded_space(self.navigation_window,self.bases,self.offsets)-forbidden)+F.union(roots)
        self.reference = S.unpack(self.mask)
        self.occupied_lo=None; self.occupied_hi=None
        self.saved_feet = old.get('construction', {}).get('feet')
        self.failure_diagnostics = []

    def roots(self):
        seen=set()
        for k,(row,contacts,b,o) in enumerate(zip(self.case.support_seeds,self.case.groups,self.bases,self.offsets)):
            for cells,contact in zip(row,contacts):
                ident=contact['candidate_id']
                if ident in seen:continue
                seen.add(ident)
                points=[v@b+o for v in cells]
                root=F.union([S.solid(G.hull_mesh(v)) for v in points])
                self.attach(root,np.vstack(points),self.case.poses[k]+':'+ident)
        if len(seen)!=len(self.case.groups[0]):raise RuntimeError('Physical head count mismatch')

    def attach(self, original, points, name):
        try:
            return super().attach(original, points, name)
        except RuntimeError:
            if ':foot' not in name:
                pose, ident = name.split(':', 1)
                k = self.case.poses.index(pose)
                contact = next(c for c in self.case.groups[k] if c['candidate_id'] == ident)
                center = np.asarray(contact['center_m'])@self.bases[k]+self.offsets[k]
                face = self.case.tasks[k].domain.mesh.face_normals[int(contact['center_face'])]@self.bases[k]
                average = np.sum(self.case.tasks[k].domain.mesh.face_normals[contact['source_faces']]
                    *contact['triangle_areas_m2'][:, None], axis=0)@self.bases[k]
                average /= np.linalg.norm(average)
                for label, normal in [('center_face', face), ('patch_area_average', average)]:
                    for depth in (.0031, .0036, .0041, .0046, .0051):
                        start = center+depth*normal
                        ball = self.sphere(start)
                        vertices = start+self.bead
                        self.failure_diagnostics.append(dict(head=name, normal=label, depth_m=depth,
                            outside_legal_mask_cm3=abs(float((ball-self.mask).volume()))*S.SCALE**3*1e6,
                            sweep_overlap_cm3=abs(float((ball^self.forbidden).volume()))*S.SCALE**3*1e6,
                            all_pose_min_heights_m=[float(((vertices-o)@b.T)[:, 2].min())
                                for b, o in zip(self.bases, self.offsets)]))
            raise

def run_trial(group, choice=0, prepared=None, output=None, proposal=False):
    out=(Path(output) if output is not None else group/'step4/data/dsl_shared_support')/'trials'/f'choice_{choice:03d}';out.mkdir(parents=True,exist_ok=True)
    for name in ('shape.obj','geometry_certificate.npz','overview.png'):
        (out/name).unlink(missing_ok=True)
    started=time.monotonic()
    grow = None
    I.save(out/'report.json',dict(complete=False,passed=False,constructed=False,status='running'))
    try:
        grow=DSLGrow(group,choice=choice,out=out,prepared=prepared,proposal=proposal)
        report=grow.run(pitch=.004)
        if not report.get('complete_cores_preserved'):raise RuntimeError('Construction did not certify full rod/foot cores')
        report.update(schema='dsl_complete_construction_witness_v4',baseline_placement_retained=False,shared_object_attached_placement=True,
            physical_head_count=len(grow.case.groups[0]),shared_head_count=len(grow.case.groups[0]) if len(grow.case.tasks)>1 else 0,
            installed_shared_solid_max_error_m=max(grow.shared_geometry_errors,default=0.),
            baseline_feet_retained=False,compactness_improvement_proved=False,
            selected_exit_paths=grow.case.exit_paths,selected_exit_option_ids=[p['id'] for p in grow.case.exit_paths],
            excluded_workspace_cm3=grow.selected_state['cost'],cross_pose_angle_penalty=True,
            minimum_branch_thickness=dict(guaranteed_inscribed_diameter_mm=grow.guaranteed_radius*2000,
                all_complete_cores_preserved=True,original_contact_transition_exceptions=True))
        report['comparison_reference'] = str((grow.source/'shape.obj').relative_to(I.ROOT))
        report['comparison_is_previous_fixture'] = grow.has_comparison_fixture
        report['construction']['beam_radius_m'] = .0026
        # DirectGrow replaces its reference with a navigation mask. Compare
        # material against the saved fixture, not against that empty search space.
        report['previous_material_volume_cm3'] = float(grow.foot_reference.volume*1e6)
        report['material_reduction_percent'] = 100*(1-report['volume_cm3']/report['previous_material_volume_cm3'])
        if not grow.has_comparison_fixture:
            report['previous_material_volume_cm3'] = None
            report['material_reduction_percent'] = None
        report['provenance']['code'].update(I.hashes([Path(__file__)]+PATHS.sources()))
        I.save(out/'report.json',report)
        result=dict(group=group.name,passed=True,constructed=True,volume_cm3=report['volume_cm3'],
                    space_budget=report['space_budget'],seconds=time.monotonic()-started)
    except Exception as error:
        result=dict(complete=True,group=group.name,passed=False,constructed=False,error=str(error),
                    traceback=traceback.format_exc(),seconds=time.monotonic()-started,
                    status='skipped_step3_not_accepted' if 'DSL Step3 has not passed' in str(error) else 'fixed_placement_construction_failed',
                    growth_start_diagnostics=grow.failure_diagnostics if grow is not None else [],
                    negative_verdict='No accepted construction for this retained path and fixed placement; not general infeasibility')
        I.save(out/'report.json',result)
        print('DSL CONSTRUCTION PROPOSAL REJECTED',group.name,str(error),flush=True)
    result.update(choice=choice,trial_folder=str(out.relative_to(I.ROOT)))
    return result,grow


def construct_witness(group,trials=3,output=None,proposal=False,render=True):
    out=Path(output) if output is not None else group/'step4/data/dsl_shared_support';out.mkdir(parents=True,exist_ok=True)
    for name in ('shape.obj','geometry_certificate.npz','overview.png'):(out/name).unlink(missing_ok=True)
    rows=[];prepared=None;success=[]
    for choice in range(trials):
        result,grow=run_trial(group,choice,prepared,output=out,proposal=proposal)
        rows.append(result)
        if grow is not None:prepared=grow._selection
        if result['passed']:success.append((result,grow))
        if prepared is None or choice+1>=len(prepared[0]):break
    I.save(out/'construction_trials.json',dict(complete=True,results=rows,
        selection_order=['actual_workstation_xy_footprint','material_volume','excluded_workspace'],cross_pose_angle_penalty=True))
    if not success:
        result=dict(rows[0],trial_count=len(rows),all_trial_failures=rows)
        I.save(out/'report.json',result)
        return result
    result,grow=min(success,key=lambda item:(item[0]['space_budget']['xy_area_cm2'],item[0]['volume_cm3'],item[1].selected_state['cost']))
    source=grow.out
    for name in ('shape.obj','geometry_certificate.npz'):shutil.copy2(source/name,out/name)
    report=json.loads((source/'report.json').read_text())
    report.update(exit_path_trials=len(rows),accepted_exit_path_trials=len(success),
        final_selection_objective='actual workstation XY footprint, then material, then workspace exclusion',
        trial_folder=str(source.relative_to(I.ROOT)))
    report['provenance']['inputs'].update(I.hashes([source/'report.json',out/'construction_trials.json']))
    I.save(out/'report.json',report)
    if render:F.draw(out/'overview.png',grow.case,trimesh.load(out/'shape.obj',force='mesh',process=False),grow.bases,grow.offsets)
    print('DSL CONSTRUCTION WITNESS DONE',group.name,'accepted',len(success),'of',len(rows),flush=True)
    return dict(result,trial_count=len(rows),accepted_trial_count=len(success))


def run(group,trials=3):
    from step3_scheculer.construction_contract import publish
    return publish(group)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--groups',nargs='+')
    parser.add_argument('--include-independent',action='store_true');parser.add_argument('--jobs',type=int,default=1)
    args=parser.parse_args()
    root=I.OUTPUTS/'B'
    groups=[root/g for g in args.groups] if args.groups else sorted(p for p in root.iterdir() if p.is_dir() and '+' in p.name)
    if args.include_independent:groups+=sorted((root/'independent_poses').glob('pose_*'))
    from step3_scheculer.construction_contract import publish
    if args.jobs>1:
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows=list(pool.map(publish,groups))
    else:rows=[publish(group) for group in groups]
    I.save(root/'pose2+9+13+15+17/step4/data/dsl_shared_support/batch_report.json',dict(complete=True,results=rows))
