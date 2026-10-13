"""Independent mesh/contact/work-ray audit of compact whole continuations."""
import _bootstrap
import argparse
from PIL import Image
from co_common import *
from run_all import saved_groups,output_root
from audit_whole_step4 import protected_files,work_audit
from whole_search.classify import classify


def audit(name,group,mode):
    root=output_root(name);source=root/group['id']/'step4/step4.2'
    out=source/'compact'/mode;report=I.check_report(out/'data/report.json')
    baseline=I.check_report(source/'data/report.json');render=I.check_report(out/'data/render.json')
    assert report['schema']=='whole_compact_volume_v1' and report['poses']==group['poses']
    assert report['original_baseline_report_sha256']==I.sha256(source/'data/report.json')
    assert report['passed'] and report['force_exit_work_passed']
    assert report['volume_cm3']<=baseline['volume_cm3']+1e-4
    states=[state(name,p) for p in group['poses']];mesh=states[0][2]
    support=trimesh.load(out/'support.obj',force='mesh',process=False)
    assert abs(abs(support.volume)*1e6-report['volume_cm3'])<=1e-4
    with np.load(out/'layout.npz') as z:
        placements=z['placements'].copy();directions=z['directions'].copy();hosts=z['hosts'].copy()
        np.testing.assert_array_equal(z['active'],np.arange(len(states)))
        np.testing.assert_allclose(z['native_world'],np.array([s[1] for s in states]),atol=1e-12,rtol=0)
    replay=[]
    for k,(task,T,body) in enumerate(states):
        world=states[int(hosts[k])][1];q=placements[k]
        np.testing.assert_allclose((world@q)[:3,:3],T[:3,:3],atol=1e-9,rtol=0)
        assert abs((world@q)[2,3]-T[2,3])<1e-9
        assert (world[:3,:3]@directions[k])[2]>=-1e-12
        np.testing.assert_allclose(np.linalg.norm(directions[k]),1.,atol=1e-12,rtol=0)
        allowed=np.setdiff1d(np.arange(len(mesh.faces)),task.domain.work_ids)
        local_d=q[:3,:3].T@directions[k]
        allowed=allowed[mesh.face_normals[allowed]@local_d<=1e-9]
        # Model deliberately uses the FIRST task's canonical mesh for every
        # configuration. Undoing a different native T incurs roundoff and can
        # change the triangulation of tiny clipped boundary contacts.
        posed=mesh.copy();posed.apply_transform(q)
        triangles,sources=contact_boundary(posed,support,allowed)
        actual=supply(task,T@np.linalg.inv(q),triangles,sources)
        with np.load(out/f'{group["poses"][k]}_force.npz') as z:
            assert z['mask'].shape==(32768,) and z['mask'].all()
            np.testing.assert_array_equal(sources,z['source_faces'])
            np.testing.assert_allclose(triangles,z['triangles_fixture_m'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(actual,z['supply_7d'],atol=1e-12,rtol=1e-12)
        # Reclassify original demands on independently reconstructed contacts.
        mask,info=classify(actual,task.targets)
        assert mask.all() and len(mask)==32768
        replay.append(dict(pose=group['poses'][k],loads=32768,classifier=info))
    diagnostics=report['diagnostics']
    assert max(diagnostics[k] for k in ['nominal_sweep_overlap_m3','body_overlap_m3',
        'padded_overlap_outside_contact_cores_m3','work_band_overlap_m3',
        'exported_boundary_max_exclusion_overlap_m3','endpoint_overlap_m3'])<=1e-10
    assert diagnostics['minimum_endpoint_projection_gap_m']>0
    rays=work_audit(mesh,states,placements,support)
    assert rays['passed']
    assert render['final_force_exit_work_verified'] and render['last_tile_uses_verified_final_geometry']
    for filename in ['process.png','final_result.png']:
        with Image.open(out/filename) as img:img.verify()
    tree=json.loads((out/'tree.json').read_text());path=json.loads((out/'selected_path.json').read_text())['tree_nodes']
    for left,right in zip(path,path[1:]):assert tree['nodes'][right]['parent']==left
    return dict(id=group['id'],method=mode,passed=True,volume_cm3=report['volume_cm3'],
        rays_tested=rays['rays_tested'],blocked_rays=rays['blocked_rays'],original_load_replays=replay,
        final_exit_checks_fingerprinted_not_rebuilt=True,report_sha256=I.sha256(out/'data/report.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=output_root(args.object);previous={}
    output=root/'data/whole_compact_audit.json'
    if args.resume and output.exists():
        previous={(r['id'],r['method']):r for r in json.loads(output.read_text())['rows']}
    protected=protected_files(root);rows=[]
    for group in saved_groups(args.object):
        for mode in ['greedy','structural-greedy','beam']:
            path=root/group['id']/'step4/step4.2/compact'/mode/'data/report.json'
            if not path.exists():continue
            old=previous.get((group['id'],mode))
            if old and old['report_sha256']==I.sha256(path):
                I.check_report(path);rows.append(old);continue
            row=audit(args.object,group,mode);rows.append(row)
            save(output,dict(rows=rows,protected_files=protected,complete=False))
            print('COMPACT AUDIT',group['id'],mode,row['volume_cm3'],row['rays_tested'],flush=True)
    save(output,dict(rows=rows,protected_files=protected,complete=True,
        passed=all(r['passed'] for r in rows),rays_tested=sum(r['rays_tested'] for r in rows),
        blocked_rays=sum(r['blocked_rays'] for r in rows),
        original_loads_replayed=sum(r['loads'] for row in rows for r in row['original_load_replays']),
        full_fixture_accepted=False,provenance=provenance([],[Path(__file__)])))
    print('COMPACT AUDIT SUMMARY',len(rows),'results',flush=True)


if __name__=='__main__':main()
