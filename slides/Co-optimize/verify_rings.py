from co_common import *
from step33 import circle
reports=sorted((HERE/'output/B').glob('*/step3/step3.3/data/report.json'))
for path in reports:
    r=json.loads(path.read_text());I.check_report(path)
    for row in r['state_results']:
        p=np.load(ROOT/'objects/B/poses'/row['pose']/'floor_contact.npz')['floor_demands_xy_m'];c=np.array(row['center_world_xy_m']);rad=row['minimum_radius_m'];q=p[row['boundary_sample_indices']]
        weights=np.linalg.lstsq(np.vstack([q.T,np.ones(len(q))]),np.r_[c,1],rcond=None)[0]
        assert np.all(weights>=-1e-9)
        np.testing.assert_allclose(weights@q,c,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(q-c,axis=1),rad,atol=1e-10)
        assert np.linalg.norm(p-c,axis=1).max()<=rad+2e-11
print(len(reports),'saved Step3.3 reports checked; all minimum-circle boundary certificates valid')
