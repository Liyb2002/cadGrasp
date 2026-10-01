"""Construct and verify sparse exterior feet for the two-pose reference design."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import reseating as R,build_coupled_saddle as S,outer_feet as O


def build(name,group_name):
    began=time.perf_counter();out=I.OUTPUTS/name/group_name/'step4'
    case=R.load_case(out);plan=I.check_report(out/'data/design_inputs.json');placement=plan['placement']
    if len(case.poses)!=2:raise ValueError('Reference design currently targets two-pose cases')
    bases=np.asarray(placement['bases']);offsets=np.asarray(placement['offsets'])
    roots=R.root_records(case,bases,offsets)
    designer=O.Designer(case,placement,out/'data/construction_cache')
    menus=[designer.menu(k) for k in range(2)]
    print('FLOOR MENUS',list(map(len,menus)),flush=True)
    if not all(menus):raise RuntimeError('No clean sparse outer-foot pattern on one floor')
    selections=designer.assignments(menus)
    print('FEASIBLE ASSIGNMENTS',len(selections),flush=True)
    search=out/'data/outer_design';search.mkdir(exist_ok=True)
    attempts=[]
    for index,selection in enumerate(selections[:12]):
        work=search/f'candidate_{index:02d}';work.mkdir(exist_ok=True)
        try:
            mesh,bodies,bridges,connections=designer.join(selection)
        except RuntimeError as error:
            attempts.append(dict(index=index,error=str(error)));print('JOIN REJECTED',index,str(error),flush=True);continue
        passed=True
        try:
            checks,certificate=S.verify(case.tasks,case.groups,case.support_seeds,
                np.asarray(placement['directions']),bases,offsets,mesh,designer.sweeps,check_equilibrium=False)
        except RuntimeError as error:
            if not hasattr(error,'checks'):raise
            checks,certificate=error.checks,error.certificate;passed=False
        for check in checks:
            check.update(head_preservation_geometry='constructor_generated_support_roots_and_exact_contact_patch',
                input_head_thickness_m=0.,original_probe_solid_preservation_required=False)
        metrics=R.span_metrics(mesh.vertices,bases,offsets)
        local=[]
        for h,(patch,body) in enumerate(zip(designer.patches,bodies)):
            filename=f'body{h}.obj';m=S.unpack(body);m.export(work/filename,file_type='obj',digits=17,include_normals=False)
            local.append(dict(index=h,head_pose=case.poses[patch['pose']],candidate_id=patch['id'],
                file=filename,volume_cm3=float(m.volume*1e6)))
        if bridges:S.unpack(O.union(bridges)).export(work/'bridges.obj',file_type='obj',digits=17,include_normals=False)
        mesh.export(work/'fixture.obj',file_type='obj',digits=17,include_normals=False)
        np.savez_compressed(work/'geometry.npz',vertices_m=mesh.vertices,faces=mesh.faces,rotations=bases,local_offsets_m=offsets)
        certificate_name='geometry_certificate.npz' if passed else 'geometry_diagnostic.npz'
        np.savez_compressed(work/certificate_name,**certificate)
        landings=[]
        for edge in selection['edges']:
            k,j=edge['floor'],edge['pad'];patch=designer.patches[edge['head']]
            landings.append(dict(head_body=edge['head'],candidate_id=patch['id'],floor_index=k,floor_pose=case.poses[k],
                pad_index=j,polygon_xy_m=selection['plans'][k]['pads'][j].tolist(),connection_count=1,
                retained_blank_fraction=edge['retained_blank_fraction'],length_m=edge['length_m']))
        design=dict(method='sparse_exterior_pads_then_tapered_codesign_bodies',global_ground_ring=False,
            local_bodies=local,connections=connections,consolidated_landings=landings,max_connections_per_head_floor=1,
            floor_patterns=[p['record'] for p in selection['plans']],
            feet_xy_m=[[xy.tolist() for xy in p['pads']] for p in selection['plans']],
            interior_hollowing_applied=False,geometry_search_cost=selection['cost'])
        I.save(work/'design.json',design)
        artifacts=['fixture.obj','geometry.npz','design.json',certificate_name]+[r['file'] for r in local]
        if bridges:artifacts.append('bridges.obj')
        code=[Path(__file__),Path(O.__file__),Path(R.__file__),Path(S.__file__),Path(O.F.__file__),
            Path(S.Q.__file__),Path(S.W.__file__),Path(O.G.__file__),Path(S.surface_distances.__code__.co_filename),
            *O.ACCESS.sources()]
        construction=dict(complete=True,schema='sparse_exterior_codesign_feet_v1',object=name,poses=case.poses,
            passed=passed,checks=checks,process_access=designer.access_check,
            passed_scope='step4_geometry_only',force_torque_authority='step3',
            step4_force_torque_enforced=False,step4_force_torque_recomputed=False,
            verification_artifact=certificate_name,
            volume_cm3=float(mesh.volume*1e6),dimensions_mm=(mesh.extents*1000).tolist(),
            original_active_contacts_preserved=True,extra_loads_added=False,support_mass_ignored=True,
            structural_strength_verified=False,same_rigid_solid_in_all_poses=True,
            generated_support_roots=roots,source_schedule=str((case.source/'schedule.json').relative_to(I.ROOT)),
            solid=dict(component_count=1,one_solid=True,watertight=bool(mesh.is_watertight),
                consistently_wound=bool(mesh.is_winding_consistent),volume_m3=float(mesh.volume)),
            body_design=design,timings_seconds=dict(total=time.perf_counter()-began),
            provenance=dict(inputs=I.hashes(case.paths),code=I.hashes(code)),
            artifacts={f:I.sha256(work/f) for f in artifacts})
        I.save(work/'report.json',construction)
        report=dict(complete=True,schema='independent_seating_step4_v2',object=name,poses=case.poses,
            constructed=True,passed=bool(passed),step3_passed=case.schedule['passed'],
            passed_scope='step4_geometry_only',force_torque_authority='step3',
            step4_force_torque_enforced=False,step4_force_torque_recomputed=False,
            step3_covered_counts=case.schedule['covered_counts'],physical_head_count=len(designer.patches),shared_head_count=0,
            head_model=case.head_model,placement=placement,construction=construction,body_directory=str(work.relative_to(out)),
            construction_model='sparse_exterior_codesign_feet',process_access=designer.access_check,
            contact_and_load_inputs_unchanged=True,
            source_schedule=construction['source_schedule'],maximum_spatial_span_m=plan['maximum_spatial_span_m'],
            max_span_ratio=plan['max_span_ratio'],object_scale_m=case.scale,metrics=metrics,
            compactness_passed=None,size_limit_enforced=False,
            volume_cm3=construction['volume_cm3'],support_seed_records=roots,videos_generated=False,html_generated=False,
            status='step4_geometry_passed' if passed else 'step4_geometry_failed',
            provenance=dict(inputs=I.hashes(case.paths),code=I.hashes(code)))
        I.save(work/'outer_report.json',report)
        attempts.append(dict(index=index,work=str(work.relative_to(out)),passed=report['passed'],volume_cm3=report['volume_cm3']))
        I.save(search/'search.json',dict(complete=True,menus=list(map(len,menus)),assignments=len(selections),attempts=attempts))
        print('OUTER CANDIDATE',group_name,index,report['passed'],report['volume_cm3'],flush=True)
        if report['passed']:return work
    raise RuntimeError('No fully accepted exterior-foot candidate in the finite first twelve assignments')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('object');p.add_argument('group')
    args=p.parse_args();print(build(args.object,args.group),flush=True)
