"""Audit one rigid support, its actual footprint, shared forces and whole sweep."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import trimesh
from shapely.geometry import Polygon,MultiPoint
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step5_connect_support import connect as C,layout as L,fixed_feet as H,solids as S,motion as M,routing as T
from step2_local_support import work_volume as W
from step5_connect_support.surface_check import surface_distances
from step1.cases import pose_name


def run(name):
    out=C.OUTPUTS/name/pose_name()/C.STAGE
    report=C.I.check_report(out/'connection.json')
    assert report['schema']==C.SCHEMA and report['foot_positions_frozen']
    domain,contacts,schedule,floor,directions,depth,_=C.read_inputs(name)
    assert report['footprint_source_sha256']==C.sha256(out.parent/C.F.STAGE/'floor_contact.json')
    assert report['selected_ids']==[c['candidate_id'] for c in contacts]
    assert [r['candidate_id'] for r in report['support_results']]==report['selected_ids']
    success={r['candidate_id'] for r in report['support_results'] if r['passed']}
    assert {r['candidate_id'] for r in report['supports']}==success
    assert report['support_count']==report['saved_geometry_count']==len(success)
    assert report['contact_count']==len(contacts)
    work=W.WorkVolume.read(out/report['work_volume_record']);scale=float(domain.mesh.extents.max())
    by_contact={c['candidate_id']:c for c in contacts}
    by_direction={c['candidate_id']:r for c,r in zip(contacts,directions['contacts'])}
    feet={f['candidate_id']:f for f in floor['report']['ground_footprints']}
    records=[];modules=[]
    for entry in report['supports']:
        cid=entry['candidate_id'];contact=by_contact[cid];foot=feet[cid]
        assert C.D.contains(by_direction[cid]['certified_directions'],entry['bearing_deg'])
        arrays=C.I.load_npz(out/entry['folder']/'geometry.npz');parts=S.unpack_parts(arrays)
        joined=trimesh.Trimesh(arrays['union_vertices_m'],arrays['union_faces'],process=False)
        assert joined.is_watertight and joined.is_winding_consistent and joined.volume>0 and len(joined.split())==1
        reconstructed,solid=S.union_parts(parts,scale)
        assert solid['one_solid']
        np.testing.assert_array_equal(reconstructed.vertices,joined.vertices)
        np.testing.assert_array_equal(reconstructed.faces,joined.faces)
        expected_heads=C.D.Analyzer(domain.mesh,depth*entry['backing_depth_factor']).heads(contact)
        actual_heads=[p for p,label in zip(parts,arrays['part_labels']) if str(label).startswith('contact_head_')]
        assert len(actual_heads)==len(expected_heads)
        for actual,expected in zip(actual_heads,expected_heads):
            np.testing.assert_array_equal(actual.vertices,expected.vertices)
            np.testing.assert_array_equal(actual.faces,expected.faces)
        for route in entry['routes']:
            prefix=f"routed_bar_{route['pad_index']:02d}_"
            bars=[p for p,label in zip(parts,arrays['part_labels']) if str(label).startswith(prefix)]
            points=np.asarray(route['waypoints_m']);assert len(bars)==len(points)-1
            for i,bar in enumerate(bars):
                radius=route['member_width_m']/2
                expected=T.beam(points[i],points[i+1],route['root_joint_half_width_m'] if i==0 else radius,radius)
                np.testing.assert_array_equal(expected.vertices,bar.vertices)
                np.testing.assert_array_equal(expected.faces,bar.faces)
        assert sorted(r['pad_index'] for r in entry['routes'])==list(range(len(foot['pads_xz_m'])))
        np.testing.assert_array_equal(entry['ground_polygons_xz_m'],foot['pads_xz_m'])
        assert entry['ground_height_m']==foot['height_m']
        expected_pads=H.pad_parts(foot)
        actual_pads=[p for p,label in zip(parts,arrays['part_labels']) if str(label).startswith('ground_pad_')]
        assert len(actual_pads)==len(expected_pads)
        for actual,expected,polygon in zip(actual_pads,expected_pads,foot['pads_xz_m']):
            np.testing.assert_array_equal(actual.vertices,expected.vertices)
            np.testing.assert_array_equal(actual.faces,expected.faces)
            bottom=COORD.floor(actual.vertices[np.abs(actual.vertices[:,1])<scale*1e-12])
            assert MultiPoint(bottom).convex_hull.symmetric_difference(Polygon(polygon)).area<scale**2*1e-11
        others=[p for p,label in zip(parts,arrays['part_labels']) if not str(label).startswith('ground_pad_')]
        assert all(p.vertices[:,1].min()>scale*1e-10 for p in others),'Unrecorded ground-bearing material'
        corners=COORD.lift_floor(np.concatenate(foot['pads_xz_m']))
        np.testing.assert_array_equal(corners,arrays['ground_corners_m'])
        np.testing.assert_array_equal(corners,entry['ground_corners_m'])
        a=np.asarray(entry['direction']);np.testing.assert_array_equal(a,arrays['insertion_direction'])
        np.testing.assert_allclose(a, T.G.frame(entry['bearing_deg'])[0], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(arrays['ground_polygon_offsets'],
                                      np.cumsum([0]+list(map(len,foot['pads_xz_m']))))
        trajectory=json.loads((out/entry['folder']/'trajectory.json').read_text())
        assert trajectory==entry['trajectory']
        assert M.sweep_check(domain.mesh,parts,a,entry['trajectory']['length_m'])['passed']
        np.testing.assert_allclose(entry['trajectory']['start_translation_m'],-entry['trajectory']['length_m']*a,atol=1e-15)
        assert work.check_parts(parts,arrays['part_labels'])['passed']
        points=np.unique(np.vstack([contact['triangles_m'].reshape(-1,3),contact['triangles_m'].mean(axis=1)]),axis=0)
        maximum=float(surface_distances(joined,points).max());assert maximum<=scale*1e-9
        assert surface_distances(joined,corners).max()<=scale*1e-9
        exported=trimesh.load(out/entry['folder']/'support.stl',process=False)
        np.testing.assert_array_equal(exported.triangles,joined.triangles)
        modules.append(dict(plan=dict(candidate_id=cid,direction=a),parts=parts,joined=joined,labels=arrays['part_labels']))
        records.append(dict(candidate_id=cid,passed=True,footprint_exactly_matches_step4=True,
            complete_solid_rebuilt=True,contact_surface_preserved=True,maximum_contact_distance_m=maximum,
            complete_object_sweep_verified=True,work_volume_clear=True,ascii_stl_exact=True))
    for result in report['support_results']:
        assert json.loads((out/'supports'/result['candidate_id']/'status.json').read_text())==result
        if not result['passed']:
            assert not (out/'supports'/result['candidate_id']/'geometry.npz').exists()
            assert not (out/'supports'/result['candidate_id']/'support.stl').exists()
    assembly=L.check_assembly(domain.mesh,modules,work) if modules else None
    subset,order=H.compatible_subset(assembly,len(modules)) if modules else ([],[])
    ids=[m['plan']['candidate_id'] for m in modules]
    assert report['assemblable_subset_ids']==[ids[i] for i in subset]
    assert report['assemblable_subset_order']==[ids[i] for i in order]
    if subset:
        assert L.check_assembly(domain.mesh,[modules[i] for i in subset],work)['passed']
    whole=bool(contacts) and len(modules)==len(contacts) and bool(assembly and assembly['passed'])
    assert report['geometric_assembly_verified']==whole
    verified=whole and bool(floor['report']['continuous_domain_coverage_proved']) and bool(schedule['continuous_coverage_proved'])
    assert report['passed']==report['independent_support_equilibrium_verified']==verified
    assert report['physical_supports_verified'] is False and report['subset_load_coverage_claimed'] is False
    result=dict(object=name,pose=pose_name(),complete=True,passed=True,status=report['status'],supports=records,
        failed_support_count=len(contacts)-len(modules),independent_successes_retained=True,
        geometric_assembly_verified=whole,complete_static_and_geometric_model_verified=verified,
        assembly_subset_replayed=True,foot_positions_unchanged=True,
        provenance=dict(inputs=C.I.hashes([out/'connection.json']),code=C.I.hashes([Path(__file__)])))
    C.I.save(out/'audit.json',result)
    print(name,'Step 5 audit:',len(records),'successful solids,',len(contacts)-len(records),'explicit failures',flush=True)
    return result


# Current one-body entry; earlier independent-body helpers remain for regressions.
from step5_connect_support.belt_assembly import audit as run


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or C.OBJECTS:run(name)
