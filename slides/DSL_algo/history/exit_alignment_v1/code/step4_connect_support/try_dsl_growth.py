"""Try the copied from-head Step4 with NEW DSL contacts and certified exits.

Placements and ground target locations are retained from the copied comparison.
A failure is explicitly a fixed-placement construction failure, not infeasibility.
"""
import argparse
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
from step4_connect_support import boxed_support as F, build_coupled_saddle as S
from step4_connect_support import deterministic_space as D, process_access as A
from step4_connect_support import growing_support as L, space_budget as B
from step4_connect_support.fixture_view import cells_for
from step2_local_support import geometry as G
from step2_local_support import circles as P


class DSLGrow(H.DirectGrow):
    def __init__(self, group):
        self.group = group
        self.out = group/'step4/data/dsl_support'
        self.out.mkdir(parents=True, exist_ok=True)
        dsl_path = group/'step3_scheculer/dsl/report.json'
        schedule = I.check_report(dsl_path)
        if not schedule['passed']:
            raise ValueError('DSL Step3 has not passed all loads and certified exits')
        self.source = group/'step4/data/growing_support'
        if not (self.source/'report.json').is_file():
            self.source = group/'step4/data'
        old = json.loads((self.source/'report.json').read_text())
        self.reference_report = dict(old)
        self.bases = np.asarray(old['placement']['bases'])
        self.offsets = np.asarray(old['placement']['offsets'])
        shape = self.source/'shape.obj' if (self.source/'shape.obj').exists() else group/'step4/shape.obj'
        self.reference = trimesh.load(shape, force='mesh', process=False)
        self.foot_reference = self.reference.copy()
        tasks = [saved_task(group, pose) for pose in schedule['poses']]
        reports = [I.check_report(group/'step3_scheculer/dsl'/pose/'report.json') for pose in schedule['poses']]
        contacts = [I.read_contacts(group/'step3_scheculer/dsl'/pose/f'final_contacts_{pose}.npz') for pose in schedule['poses']]
        self.directions = np.array([r['selected_withdrawal_world'] for r in reports])
        paths = [dsl_path, shape]+[p for t in tasks for p in t.inputs]
        paths.append(self.source/'report.json')
        comparison = self.out/'comparison_input'
        comparison.mkdir(exist_ok=True)
        shutil.copy2(self.source/'report.json',comparison/'report.json')
        shutil.copy2(shape,comparison/'shape.obj')
        self.source = comparison
        paths += [group/'step3_scheculer/dsl'/p/f'final_contacts_{p}.npz' for p in schedule['poses']]
        paths += [group/'step3_scheculer/dsl'/p/'report.json' for p in schedule['poses']]
        seeds = []
        for task, row in zip(tasks, contacts):
            # Match the real constructor's roots. Thin Step3 exit probes are
            # proposal checks; they cannot bridge the constructor's 0.4mm relief.
            offsets = G.vertex_offsets(task.domain.mesh, P.DEPTH_FRACTION*task.domain.mesh.extents.max())[0]
            seeds.append([cells_for(c, task.domain, offsets) for c in row])
        demands = [pressure_centers(t.targets/t.scale, t.domain.com)[0] for t in tasks]
        self.case = SimpleNamespace(name='B', pair=group, output=self.out, poses=schedule['poses'], tasks=tasks,
            groups=contacts, heads=[[list(c['triangles_m']) for c in row] for row in contacts],
            support_seeds=seeds, demands=demands, schedule=dict(passed=True, covered_counts=schedule['covered_counts']),
            paths=paths)
        self.case.root_solids = [F.union([S.solid(G.hull_mesh(v)) for cells in row for v in cells]) for row in seeds]
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
        raw_meshes = [S.swept_solid(t.domain.mesh, -S.SWEEP_LENGTH*d) for t,d in zip(tasks,self.directions)]
        padded = []
        cube = F.md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
        for mesh,b,o in zip(raw_meshes,self.bases,self.offsets):
            transformed = F.transform(S.solid(mesh),b,o)
            padded.append(transformed.minkowski_sum(cube))
        forbidden = F.union(padded)
        self.sweep_meshes = raw_meshes
        self.search = SimpleNamespace(sweep_meshes=raw_meshes, sweep_conditioning=[], precompute=lambda:None)
        self.forbidden = forbidden
        cloud = [t.domain.mesh.vertices for t in tasks]+[np.c_[v,np.zeros(len(v))] for v in demands]
        mandatory = [S.unpack(v).vertices for v in roots]
        installed = [(np.vstack(mandatory)-o)@b.T for b,o in zip(self.bases,self.offsets)]
        self.navigation_window = D.enlarged(B.box(cloud+installed), .05)
        self.mask = (F.bounded_space(self.navigation_window,self.bases,self.offsets)-forbidden)+F.union(roots)
        self.reference = S.unpack(self.mask)
        self.occupied_lo=None; self.occupied_hi=None
        self.saved_feet = old.get('construction', {}).get('feet')
        self.failure_diagnostics = []

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

    def ground(self):
        if not self.saved_feet:
            return super().ground()
        for k,(b,o,demands) in enumerate(zip(self.bases,self.offsets,self.case.demands)):
            item = next(row for row in self.saved_feet if row['pose']==self.case.poses[k])
            centers = np.asarray(item['centers_xy_m'])
            self.feet.append(dict(pose=self.case.poses[k],centers_xy_m=centers.tolist()))
            ground=[]
            for j,center in enumerate(centers):
                angle=np.arange(32)*2*np.pi/32
                disk=center+.003*np.c_[np.cos(angle),np.sin(angle)]
                vertices=np.vstack([np.c_[disk,np.full(32,z)] for z in (0.,.005)])@b+o
                ground.append(disk)
                sole=S.solid(G.hull_mesh(vertices))
                self.clip(sole)
                self.attach(sole,vertices,f'{self.case.poses[k]}:foot{j}')
            if MultiPoint(demands).convex_hull.difference(MultiPoint(np.vstack(ground)).convex_hull.buffer(1e-10)).area > 1e-12:
                raise RuntimeError('Actual saved sole hull fails original demand coverage')
        self.save_stage('seed_targets',F.union([t['solid'] for t in self.terminals]))


def run(group):
    out=group/'step4/data/dsl_support';out.mkdir(parents=True,exist_ok=True)
    for name in ('shape.obj','geometry_certificate.npz','overview.png'):
        (out/name).unlink(missing_ok=True)
    started=time.monotonic()
    grow = None
    I.save(out/'report.json',dict(complete=False,passed=False,constructed=False,status='running'))
    try:
        grow=DSLGrow(group)
        report=grow.run(pitch=.004)
        final=S.solid(trimesh.load(out/'shape.obj',force='mesh',process=False))
        cores=grow.core_solids+[(f'beam_{i}',c) for i,c in enumerate(grow.beams)]+[(f'sole_{i}',c) for i,c in enumerate(grow.original_soles)]
        missing=[abs(float((c-final).volume()))*S.SCALE**3 for _,c in cores]
        if max(missing,default=0)>8e-14:raise RuntimeError('Serialized support loses a complete rod/foot core')
        report.update(schema='dsl_contacts_fixed_placement_direct_growth_v1',baseline_placement_retained=True,
            baseline_feet_retained=True,compactness_improvement_proved=False,
            minimum_branch_thickness=dict(guaranteed_inscribed_diameter_mm=grow.guaranteed_radius*2000,
                all_complete_cores_preserved=True,original_contact_transition_exceptions=True))
        report['construction']['beam_radius_m'] = .0026
        # DirectGrow replaces its reference with a navigation mask. Compare
        # material against the saved fixture, not against that empty search space.
        report['previous_material_volume_cm3'] = float(grow.foot_reference.volume*1e6)
        report['material_reduction_percent'] = 100*(1-report['volume_cm3']/report['previous_material_volume_cm3'])
        report['provenance']['code'].update(I.hashes([Path(__file__)]))
        I.save(out/'report.json',report)
        F.draw(out/'overview.png',grow.case,trimesh.load(out/'shape.obj',force='mesh',process=False),grow.bases,grow.offsets)
        result=dict(group=group.name,passed=True,constructed=True,volume_cm3=report['volume_cm3'],
                    space_budget=report['space_budget'],seconds=time.monotonic()-started)
    except Exception as error:
        result=dict(complete=True,group=group.name,passed=False,constructed=False,error=str(error),
                    traceback=traceback.format_exc(),seconds=time.monotonic()-started,
                    status='skipped_step3_not_accepted' if 'DSL Step3 has not passed' in str(error) else 'fixed_placement_construction_failed',
                    growth_start_diagnostics=grow.failure_diagnostics if grow is not None else [],
                    negative_verdict='No accepted construction in this fixed-placement, fixed-feet trial; not general infeasibility')
        I.save(out/'report.json',result)
        print('DSL STEP4 FAILED',group.name,str(error),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--groups',nargs='+');args=parser.parse_args()
    root=I.OUTPUTS/'B'
    groups=[root/g for g in args.groups] if args.groups else sorted(p for p in root.iterdir() if p.is_dir() and '+' in p.name)
    rows=[run(group) for group in groups]
    I.save(root/'pose2+9+13+15+17/step4/data/dsl_support/batch_report.json',dict(complete=True,results=rows))
