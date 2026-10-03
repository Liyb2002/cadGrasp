"""Publish a fully checked whole-loft fixture and independently replay it."""
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh

from step1.needs import ROOT
from step4_connect_support import build_coupled_saddle as S, material_graph as M
from step4_connect_support import material_network as N, loft_candidates as L
from step4_connect_support import loft_growth as G, loft_acceptance as A, loft_refine as P
from step4_connect_support import loft_joints as J
from step4_connect_support import run_unified_growth as R
from step4_connect_support.run_greedy import plain, clear_previous_outputs
from step4_connect_support.fixture_view import export_viewer
from step4_connect_support.audit_fixture import audit

HERE = Path(__file__).resolve().parent
SCHEMA = 'unified_reference_style_loft_growth_v1'


def publish(out, work, case, context, placement, graph, result, design, reference):
    if not result['passed']: raise RuntimeError('Refusing to publish an unaccepted loft fixture')
    stage = work/'publish'; stage.mkdir(exist_ok=True)
    mesh = result['mesh']
    mesh.export(stage/'fixture.obj',file_type='obj',digits=17,include_normals=False)
    mm=mesh.copy();mm.apply_scale(1000)
    (stage/'fixture_mm.stl').write_text(trimesh.exchange.stl.export_stl_ascii(mm))
    raw=trimesh.load(stage/'fixture_mm.stl',force='mesh',process=False)
    vertices,inverse=np.unique(raw.vertices,axis=0,return_inverse=True)
    loaded=trimesh.Trimesh(vertices,inverse[raw.faces],process=False)
    if not loaded.is_watertight or not loaded.is_winding_consistent or len(loaded.split())!=1:
        raise RuntimeError('STL round trip changed closed connected topology')
    np.testing.assert_allclose(loaded.extents,mm.extents,atol=1e-9,rtol=0)
    np.savez_compressed(stage/'geometry.npz',vertices_m=mesh.vertices,faces=mesh.faces,
        rotations=context['bases'],local_offsets_m=context['offsets'])
    np.savez_compressed(stage/'equilibrium.npz',**result['certificate'])
    records=[]
    for head,row in zip(context['heads'],context['records']):
        hm=S.unpack(head);record=dict(row,file=f"body{row['index']}.obj",
            volume_cm3=float(hm.volume*1e6),role='immutable_contact_head')
        hm.export(stage/record['file'],file_type='obj',digits=17,include_normals=False)
        records.append(record)
    selected=result['selected']
    np.savez_compressed(stage/'loft_graph.npz',edges=graph['edges'],
        individual_candidate_cost_cm3=graph['cost'],selected=selected,head_nodes=graph['heads'])
    design.update(local_bodies=records,material_ownership_per_head=False,
        total_selected_candidate_nodes=int(selected.sum()),graph=graph['stats'],
        candidate_records=graph['records'],selected_nodes=np.flatnonzero(selected).tolist(),
        fixed_head_volume_cm3=L.volume(M.union(context['heads'])),
        actual_union_volume_cm3=float(mesh.volume*1e6),actual_external_area_cm2=float(mesh.area*1e4),
        individual_candidate_costs_are_additive=False,assembly_coordinate_snap_m=None,
        export_regularization=result['regularization'])
    S.I.save(stage/'design.json',plain(design))
    artifacts=['fixture.obj','fixture_mm.stl','geometry.npz','equilibrium.npz',
        'loft_graph.npz','design.json']+[r['file'] for r in records]
    code=[Path(module.__file__) for module in (S,M,N,L,G,A,P,R,J)]+[
        Path(__file__),HERE/'run_loft_growth.py',HERE/'audit_fixture.py',
        Path(S.swept_solid.__code__.co_filename),Path(S.surface_distances.__code__.co_filename),
        Path(S.bearing_rays.__code__.co_filename),Path(S.Q.__file__),Path(S.W.__file__),
        Path(S.G.__file__),Path(S.FLOOR.__file__),HERE/'run_greedy.py',HERE/'fixture_view.py',
        HERE/'refresh_shared_geometry_view.py',HERE/'shared_geometry_viewer.html',
        HERE/'export_shared_geometry.cjs']
    geometry=out/'reference_geometry.npz'
    assert S.I.sha256(geometry)==reference['artifacts']['geometry.npz']
    with np.load(geometry) as data:
        refmesh=trimesh.Trimesh(data['vertices_m'],data['faces'],process=False)
        np.testing.assert_allclose(data['rotations'],context['bases'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(data['local_offsets_m'],context['offsets'],atol=1e-12,rtol=0)
    area=float(mesh.area*1e4);refarea=float(refmesh.area*1e4)
    report=dict(schema=SCHEMA,object=case.name,poses=case.poses,particle=case.schedule['particle'],
        complete=True,passed=True,status='full_acceptance_passed',same_rigid_solid_in_both_poses=True,
        physical_head_definition='six_retained_contact_patches',contact_groups=5,contact_patch_count=6,
        strict_five_head_sharing_solved=False,original_active_contacts_preserved=True,
        original_task_poses_changed=False,support_mass_ignored=True,structural_strength_verified=False,
        extra_loads_added=False,source_schedule=str((case.source/'schedule.json').relative_to(ROOT)),
        placement=plain(placement),task_fixture_transforms=[dict(pose=p,rotation=b.tolist(),translation_m=(-b@o).tolist())
            for p,b,o in zip(case.poses,context['bases'],context['offsets'])],
        dimensions_mm=(mesh.extents*1000).tolist(),volume_cm3=float(mesh.volume*1e6),external_area_cm2=area,
        solid=result['solid'],checks=result['checks'],actual_ground_containment=result['floor_checks'],
        body_design=plain(design),comparison=dict(reference_method=reference['schema'],
            reference_volume_cm3=reference['volume_cm3'],reference_external_area_cm2=refarea,
            reference_passed=reference['passed'],placement_unchanged=True,
            volume_change_percent=100*(mesh.volume*1e6/reference['volume_cm3']-1),
            external_area_change_percent=100*(area/refarea-1),strength_equivalence_claimed=False,
            aesthetic_equivalence_verified=False),
        presentation_description='统一贪心直接选择整块头—脚 loft；同头同地面的扩展合成一块身体后计费。彩色保留原头，灰白是共同材料。六块接触面、五个标识，黄色仍有两份实体。',
        exports=dict(obj_units='m',stl_units='mm',stl_encoding='ASCII',stl_welding='exact_coordinate_weld',
            reloaded_stl_watertight=True),
        provenance=dict(inputs=S.I.hashes(case.paths+[out/'reference_report.json',geometry]),
            code=S.I.hashes(code)),artifacts={f:S.I.sha256(stage/f) for f in artifacts})
    S.I.save(stage/'report.json',plain(report))
    S.I.save(stage/'independent_audit.json',plain(audit(stage)))
    colors={case.schedule['shared_head']['selected_id']:'#dc9d47'}
    colors.update(zip([i for i in case.schedule['selected_ids'] if i not in colors],
        ['#ac7098','#7196c0','#50a59b','#77a76a']))
    export_viewer(stage,mesh,[(r['candidate_id'],S.unpack(h)) for r,h in zip(records,context['heads'])],
        plain(report),case.tasks,context['bases'],context['offsets'],colors)
    clear_previous_outputs(out)
    for name in ('material_graph.npz','comparison.png','grid_comparison.png','batch_summary.json',
                 'equilibrium_diagnostic.npz'):
        (out/name).unlink(missing_ok=True)
    for name in artifacts+['report.json','independent_audit.json','index.html']:
        shutil.copy2(stage/name,out/name)
    return report
