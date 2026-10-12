"""Read immutable native-pose inputs without an algorithm-search dependency."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import json
import numpy as np
from step3_scheculer import contacts as I
from step3_scheculer.pair_scoring import TaskProblem
from codes.precompute_objects.loads import ContinuousNeeds, demand


def current_task(name,pose):
    folder=I.ROOT/'objects'/name/'poses'/pose
    source=folder/'needs.json';sample_path=folder/'samples.json';snapshot=folder/'setup.npz'
    domain=ContinuousNeeds.read(source);raw=json.loads(sample_path.read_text())
    assert domain.data['object']==name and domain.data['pose_id']==pose
    assert I.sha256(source)==raw['provenance']['physical_domain_sha256']
    assert I.sha256(snapshot)==domain.data['provenance']['setup_snapshot_sha256']
    with np.load(snapshot) as z:
        assert str(z['object'])==name and str(z['pose_id'])==pose
        assert str(z['mesh_sha256'])==I.sha256(I.ROOT/'objects'/name/'mesh.stl')
        assert str(z['poses_sha256'])==I.sha256(I.ROOT/'objects'/name/'poses.json')
        np.testing.assert_array_equal(z['T_world_mesh'],domain.data['frame']['T_world_mesh'])
        np.testing.assert_array_equal(z['com_m'],domain.com)
        floor=z['floor_contact_m'].copy()
    targets=np.asarray(raw['need_wrench']);assert len(targets)==32768
    np.testing.assert_allclose(targets,demand(raw['pt_m'],raw['force_push_mg'],domain.com,domain.gravity),atol=1e-13,rtol=0)
    scale=np.r_[np.ones(3),np.ones(3)/domain.mesh.extents.max()]
    task=TaskProblem(pose,domain,floor,np.ascontiguousarray(targets*scale),scale)
    task.inputs=[source,sample_path,snapshot,I.ROOT/'objects'/name/'mesh.stl',I.ROOT/'objects'/name/'poses.json']
    task.random_sample_count=len(targets);return task
