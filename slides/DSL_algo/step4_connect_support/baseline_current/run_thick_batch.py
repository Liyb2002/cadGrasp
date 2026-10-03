"""Apply verified thick direct growth to the eight current surface-head groups.

Original historical pair and its already generated copy remain untouched.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import deterministic_space as D,boxed_support as F
from step4_connect_support.baseline_current import run_copied_reference as ADAPTER
from step4_connect_support.baseline_current import replay_compact as R,render_compact_images as V
from step4_connect_support.baseline_current.run_boxed_batch import protected_hashes
from step4_connect_support.baseline_current import build_coupled_saddle as S
from step2_local_support import geometry as G
from step4_connect_support.baseline_current import growing_support as L
import trimesh

GROUPS=['pose3+6','pose5+7','pose2+10+15','pose2+12+15','pose1+2+8+17','pose6+8+10+19','pose1+2+3+4+5','pose2+9+13+15+17']
ROOT=I.OUTPUTS/'B'

def copied_hashes():
    return {p:h for p,h in ADAPTER.tree_hash(ROOT/'pose1+3copied').items()
        if not p.startswith('step5_evaluate/') and not Path(p).name.startswith('process_sweep_') and p not in ('step4/construction_steps.png','step4/README.md','step4/data/growing_support/construction_steps_render.json')}


class BatchGrow(ADAPTER.StagedGrow):
    def __init__(self,group):
        super().__init__(group)
        sphere=trimesh.creation.icosphere(subdivisions=2,radius=.0026)
        self.bead=sphere.vertices
        self.guaranteed_radius=float(np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]))))
    def attach(self,original,points,name):
        if ':foot' in name:return super().attach(original,points,name)
        self.original_roots.append(original)
        pose,ident=name.split(':',1);k=self.case.poses.index(pose)
        contact=next(c for c in self.case.groups[k] if c['candidate_id']==ident)
        center=np.asarray(contact['center_m'])@self.bases[k]+self.offsets[k]
        normal=self.case.tasks[k].domain.mesh.face_normals[int(contact['center_face'])]@self.bases[k]
        _,nearest=self.tree.query(center,k=min(2048,len(self.nodes)))
        average=np.sum(self.case.tasks[k].domain.mesh.face_normals[contact['source_faces']]*contact['triangle_areas_m2'][:,None],axis=0)@self.bases[k]
        average/=np.linalg.norm(average)
        choices=[(label,direction,depth) for label,direction in [('center_face',normal),('patch_area_average',average)]
            for depth in (.0031,.0036,.0041,.0046,.0051)]
        for label,normal,depth in choices:
            start=center+depth*normal
            ball=S.solid(G.hull_mesh(start+self.bead))
            if abs(float((ball-self.mask).volume()))*S.SCALE**3 > 8e-14:continue
            transition=S.solid(G.hull_mesh(np.vstack([points,start+self.bead]))) ^ self.mask
            for node in np.atleast_1d(nearest):
                anchor=self.nodes[node]
                stem=S.solid(G.hull_mesh(np.vstack([start+self.bead,anchor+self.bead])))
                if abs(float((stem-self.mask).volume()))*S.SCALE**3 > 8e-14:continue
                grown=original+transition+stem
                if len(L.components(grown)) != 1:continue
                if abs(float((original-grown).volume()))*S.SCALE**3 > 8e-14:continue
                self.terminals.append(dict(name=name,node=int(node),solid=grown))
                self.core_solids.append((name,stem))
                self.thickness.append(dict(name=name,start_m=start.tolist(),anchor_m=anchor.tolist(),
                    guaranteed_core_diameter_mm=self.guaranteed_radius*2000,
                    original_contact_transition_length_mm=depth*1000,transition_normal=label,transition_may_taper=True))
                return
        raise RuntimeError('No short root transition / complete thick anchor for '+name)


def archive(group):
    source=group/'step4/data/boxed_support'
    target=group/'step4/data/history/before_thick_boxed'
    if not target.exists():
        target.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file():shutil.copy2(path,target/path.name)
    for directory in ('growing_support',):
        target=group/'step4/data/history'/('before_thick_'+directory)
        if not target.exists():
            target.mkdir(parents=True)
            for path in (group/'step4/data'/directory).iterdir():
                if path.is_file():shutil.copy2(path,target/path.name)
    return group/'step4/data/history/before_thick_boxed'


def run(name):
    started=time.monotonic();group=ROOT/name;baseline=archive(group)
    existing=json.loads((baseline/'report.json').read_text())
    search=D.Search(group)
    search.case.paths=[p for p in search.case.paths if p != group/'step4/data/report.json']
    search.before=I.hashes(search.case.paths)
    search.directions=np.asarray(existing['placement']['directions'])
    search.precompute()
    attempts=[];winner=None
    compact=(np.asarray(existing['placement']['bases']),np.asarray(existing['placement']['offsets']))
    legacy=(search.bases,search.offsets)
    for placement_name,(bases,offsets) in [('compact',compact),('legacy',legacy)]:
        lower=existing['result']['box'] if placement_name=='compact' else search.envelope(bases,offsets)
        for margin_mm in (0,4,8,12,16,24):
            bounds=D.enlarged(lower,margin_mm/1000)
            candidate=search.construct(bases,offsets,bounds)
            attempt=dict(placement=placement_name,margin_mm=margin_mm,passed=False)
            attempts.append(attempt)
            if candidate is None:
                attempt['reason']=search.attempts[-1]['reason'];continue
            mesh,cert,check=candidate
            D.export_exact_obj(mesh,search.out/'shape.obj')
            np.savez_compressed(search.out/'geometry_certificate.npz',**cert)
            search.case.preview_directions=search.directions
            F.draw(search.out/'overview.png',search.case,mesh,bases,offsets)
            boxed=dict(existing,result=check,space_budget=D.measure(search.case,mesh,bases,offsets),volume_cm3=float(mesh.volume*1e6),
                placement=dict(bases=bases.tolist(),offsets=offsets.tolist(),directions=search.directions.tolist()),
                thickening_domain_margin_mm=margin_mm,thickening_domain_seed=placement_name,
                provenance=dict(existing['provenance'],inputs=search.before))
            boxed['artifacts']={p:I.sha256(search.out/p) for p in ('shape.obj','overview.png','geometry_certificate.npz')}
            I.save(search.out/'report.json',boxed)
            try:
                growth=BatchGrow(group);report=growth.run(pitch=.004)
                winner=(growth,report);attempt['passed']=True;break
            except (RuntimeError,AssertionError) as error:
                attempt['reason']=str(error)
                print('THICK RETRY',name,placement_name,margin_mm,str(error),flush=True)
        if winner is not None:break
    out=group/'step4/data/growing_support'
    I.save(out/'thick_attempts.json',dict(attempts=attempts,passed=winner is not None))
    if winner is None:raise RuntimeError('Thick growth unresolved: '+name)
    growth,report=winner
    final=S.solid(trimesh.load(out/'shape.obj',force='mesh',process=False))
    cores=growth.core_solids+[(f'beam_{j}',c) for j,c in enumerate(growth.beams)]+[(f'sole_{j}',c) for j,c in enumerate(growth.original_soles)]
    core_checks=[dict(name=n,missing_volume_m3=abs(float((c-final).volume()))*S.SCALE**3) for n,c in cores]
    assert all(c['missing_volume_m3'] <= 8e-14 for c in core_checks)
    report['construction']['beam_radius_m']=.0026
    report['minimum_branch_thickness']=dict(minimum_required_diameter_mm=5.,guaranteed_inscribed_beam_diameter_mm=growth.guaranteed_radius*2000,
        beam_outer_diameter_mm=5.2,sole_thickness_mm=5.,foot_connection_column_height_mm=6.,unclipped_body_validation=True,
        centerline_validation='Exact surface distance + half-edge Lipschitz bound; full-solid direct connection and shortcut inclusion',
        scope='complete structural rod cores and soles; original contact edges and explicitly recorded short contact transitions (3.1 to 5.1mm) excluded',
        local_branches=growth.thickness,exported_unclipped_core_checks=core_checks,all_complete_cores_preserved=True)
    report['provenance']['code'].update(I.hashes([Path(__file__),Path(ADAPTER.__file__)]))
    I.save(out/'report.json',report)
    R.replay(group,'growing_support')
    V.render(group,'growing_support')
    I.check_report(search.out/'report.json');I.check_report(out/'report.json');I.check_report(group/'step4/data/compact_visualization.json')
    result=dict(group=name,passed=True,material_volume_cm3=report['volume_cm3'],space_budget=report['space_budget'],
        core_count=len(core_checks),maximum_missing_core_volume_m3=max(c['missing_volume_m3'] for c in core_checks),
        direct_connections=report['construction']['direct_connection_count'],removed_grid_bends=report['construction']['removed_grid_bends'],
        segments=report['construction']['grown_segment_count'],domain_margin_mm=attempts[-1]['margin_mm'],placement=attempts[-1]['placement'],seconds=time.monotonic()-started)
    print('THICK GROUP DONE',result,flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--groups',nargs='+',default=GROUPS);args=parser.parse_args()
    if any(g not in GROUPS for g in args.groups):raise ValueError('Only eight current groups permitted')
    protected=protected_hashes(ROOT);copied_before=copied_hashes()
    rows=[]
    for name in args.groups:rows.append(run(name))
    assert protected==protected_hashes(ROOT)
    assert copied_before==copied_hashes()
    output=ROOT/GROUPS[-1]/'step4/data/growing_support'
    I.save(output/'thick_batch_report.json',dict(passed=all(r['passed'] for r in rows),results=rows,protected_file_count=len(protected),original_and_copied_unchanged=True))
