"""Check the co-design body and its final optional material removal separately."""
import argparse
import json
from pathlib import Path
import sys

import manifold3d as md
import numpy as np
from shapely.geometry import Polygon
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import convex_foot as F,build_coupled_saddle as S,reseating as R
from step4_connect_support.baseline_current.review_reseated import review as review_physics
from step4_connect_support.baseline_current.surface_check import surface_distances,winding_number
from step2_local_support import geometry as G


def replay_subtractions(output,report,design,original,final):
    """Rebuild ONLY the recorded cuts, then compare the exported boundary.

    An A-B Boolean of almost coincident imported meshes can produce false
    positive slivers. This replay starts with A and performs only subtraction;
    its surface and volume must match B. Surface probes are numerical evidence,
    not an analytic Hausdorff certificate; keep that distinction in the report.
    """
    details=design['postprocess_hollowing'];cuts=details['pockets']
    if not cuts:
        np.testing.assert_array_equal(original.vertices,final.vertices)
        np.testing.assert_array_equal(original.faces,final.faces)
        return dict(method='unchanged_original_mesh',maximum_surface_probe_error_m=0.,volume_error_m3=0.)
    case=R.load_case(output)
    bases=np.asarray(report['placement']['bases']);offsets=np.asarray(report['placement']['offsets'])
    keep=[]
    def protect(points,margin):
        low=np.min(points,axis=0)-margin;high=np.max(points,axis=0)+margin
        keep.append(md.Manifold.cube(((high-low)/S.SCALE).tolist()).translate((low/S.SCALE).tolist()))
    for k,row in enumerate(case.support_seeds):
        for cells in row:protect(np.vstack(cells)@bases[k]+offsets[k],details['wall_m'])
    for joint in design['connections']:
        if joint.get('kind')=='carved_connection_envelope':protect(np.asarray(joint['bounds_m']),.001)
        else:
            route=np.asarray(joint['path_m'])
            for a,b in zip(route[:-1],route[1:]):protect(np.array([a,b]),joint['radius_m']+.002)
    protected=md.Manifold.batch_boolean(keep,md.OpType.Add)
    replay=S.solid(original)
    for pocket in cuts:
        k=pocket['floor_index'];xy=np.asarray(pocket['polygon_xy_m'])
        vertices=np.vstack([np.c_[xy,np.full(len(xy),z)] for z in (-.001,pocket['depth_m'])])
        tool=S.solid(G.hull_mesh(vertices@bases[k]+offsets[k]))-protected
        replay=replay-tool
    mesh=S.unpack(replay)
    volume_error=abs(float(mesh.volume-final.volume))
    assert volume_error<8e-14,volume_error
    error=max(float(surface_distances(a,np.vstack([b.vertices,b.triangles_center])).max())
        for a,b in ((mesh,final),(final,mesh)))
    assert error<1e-10,error
    return dict(method='recorded_subtractive_csg_replay_with_bidirectional_surface_probes',
        maximum_surface_probe_error_m=error,volume_error_m3=volume_error,
        analytic_subset_certificate=False,replayed_cut_count=len(cuts))


def difference_diagnostics(difference,original):
    result=[]
    for part in difference.decompose():
        volume=abs(float(part.volume()))*S.SCALE**3
        if volume<1e-16:continue
        mesh=S.unpack(part)
        points=np.vstack([mesh.vertices,mesh.triangles_center,mesh.center_mass])
        distances=surface_distances(original,points)
        ids=np.argsort(distances)[-8:]
        witnesses=[dict(point_m=points[i].tolist(),surface_distance_m=float(distances[i]),
            original_winding_number=winding_number(original,points[i])) for i in ids]
        assert all(w['surface_distance_m']<1e-10 or w['original_winding_number']>.999999 for w in witnesses),witnesses
        result.append(dict(boolean_difference_volume_m3=volume,witnesses=witnesses))
    return result


def review(output):
    output=Path(output).resolve();result=review_physics(output)
    report=I.check_report(output/'data/report.json')
    if not report['constructed']:return result
    work=output/report['body_directory'];design=json.loads((work/'design.json').read_text())
    before_path=output/'data/co_design_body/report.json';before=I.check_report(before_path)
    old_mesh=trimesh.load(before_path.parent/'fixture.obj',force='mesh',process=False)
    mesh=trimesh.load(output/'shape.obj',force='mesh',process=False)
    old,body=S.solid(old_mesh),S.solid(mesh)
    difference=body-old
    extra=abs(float(difference.volume()))*S.SCALE**3
    subtraction=replay_subtractions(output,report,design,old_mesh,mesh)
    diagnostics=difference_diagnostics(difference,old_mesh) if extra>=8e-14 else []
    assert mesh.volume<=old_mesh.volume+8e-14
    details=design['postprocess_hollowing']
    assert details['stage']=='after_complete_codesign_construction' and not details['global_ground_ring']
    assert details['removed_fraction']<=details['maximum_removed_fraction']+1e-10
    assert abs(details['removed_fraction']-(1-mesh.volume/old_mesh.volume))<1e-10
    pairs=[(r['head_body'],r['floor_index']) for r in design['consolidated_landings']]
    assert len(pairs)==len(set(pairs)) and design['max_connections_per_head_floor']==1
    checks=[]
    for k,(pose,b,o) in enumerate(zip(report['poses'],report['placement']['bases'],report['placement']['offsets'])):
        actual=F.landing(body,np.asarray(b),np.asarray(o))
        required=Polygon(details['original_ground_hulls_xy_m'][k])
        loss=required.difference(actual.convex_hull.buffer(1e-10)).area
        assert loss<1e-12,(pose,loss)
        prior=before['checks'][k];current=report['construction']['checks'][k]
        assert not prior['coupled_equilibrium_passed'] or current['coupled_equilibrium_passed']
        assert not prior['withdrawal']['clear'] or current['withdrawal']['clear']
        checks.append(dict(pose=pose,maximum_ground_hull_loss_m2=loss,
            prior_complete_load_pass_retained=not prior['coupled_equilibrium_passed'] or current['coupled_equilibrium_passed']))
    result.update(codesign_construction_before_hollowing=True,global_ground_ring=False,
        hollowing_is_only_subtraction=True,subtraction_replay=subtraction,
        imported_mesh_boolean_extra_volume_m3_diagnostic=extra,boolean_difference_diagnostics=diagnostics,
        separate_head_floor_feet=len(pairs),pocket_count=len(details['pockets']),
        removed_fraction=details['removed_fraction'],hollowing_checks=checks)
    result['provenance']['inputs'].update(I.hashes([before_path,before_path.parent/'fixture.obj',work/'design.json']))
    result['provenance']['code'].update(I.hashes([Path(__file__),Path(F.__file__),Path(R.__file__),
        Path(G.__file__),Path(sys.modules[surface_distances.__module__].__file__)]))
    I.save(output/'data/independent_review.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('outputs',type=Path,nargs='+')
    for output in p.parse_args().outputs:
        r=review(output);print(output.parent.parent.name,'pocketed-foot review passed',r['fixture_acceptance_passed'],flush=True)
