"""Publish certified fragment deletion even while connectivity is unresolved."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_value_guided import *
import step42_geometry_cache as cache
import shutil
if __name__=='__main__':
    cache.install();g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+6+11+13+14+17');s=ValueGuidedSearch('B',g);s.warm=np.array(json.loads((s.out/'data/value_checkpoint.json').read_text())['directions']);np.savez(s.out/'data/warm_paths.npz',directions=s.warm);began=time.monotonic()
    assert s.initialize_value();raw=s.best;values=json.loads((s.out/'data/component_values.json').read_text());parts=[p for p in raw['remaining'].decompose() if material_volume(p)>1e-12];parts.sort(key=material_volume);keep=union([parts[k] for k in values['remaining_essential_components']]);tri,src=motion_consistent_boundary(s.mesh,S.unpack(keep),s.allowed);masks,infos,supplies=s.classify(tri,src);assert all(m.all() for m in masks)
    raw_remaining=raw['remaining'];result=dict(raw,remaining=keep,removed=s.seed-keep,triangles=tri,sources=src,masks=masks,infos=infos,supplies=supplies,counts=[int(m.sum()) for m in masks],overlap=max(material_volume(keep^sweep) for sweep in raw['sweeps']),partition=abs(material_volume(s.seed)-material_volume(keep)-material_volume(s.seed-keep)))
    assert result['overlap']<1e-10 and result['partition']<1e-10
    finish(s,result,began);p=s.out/'data/report.json';r=json.loads(p.read_text());r.update(passed=False,status='force_exit_pass_connectivity_unresolved',force_exit_passed=True,connectivity_required=True,pruned_unnecessary_volume_cm3=(material_volume(raw_remaining)-material_volume(keep))*1e6,pre_prune_component_count=len(parts),component_value_policy='Sequential deletion: every original demand must pass after each deletion; two remaining components are jointly necessary in this configuration',motion_consistent_contacts_required=True,validation_policy='Construction certifies all saved loads and full continuous floor-legal exits after verified pruning; connectivity remains unresolved. No exported replay. Ground reconstruction and strength deferred.')
    r['provenance']['inputs'].update(I.hashes([s.archive/'data/report.json',s.out/'data/warm_paths.npz']));r['provenance']['code'].update(I.hashes([HERE/f for f in ['step4.2/publish_value_pending.py','step4.2/step42_connected.py','step4.2/step42_connected_safe.py','step4.2/step42_connected_local.py','step4.2/step42_connected_independent.py','step4.2/step42_contact_consistency.py','step4.2/step42_topology.py','step4.2/step42_topology_common.py','step4.2/step42_value_guided.py','helper_func/step42_geometry_cache.py']]));r['artifacts']['component_values.json']=I.sha256(s.out/'data/component_values.json');save(p,r)
    target=s.base/'step4/step4.2'
    for f in s.out.iterdir():
        if f.is_dir():
            for item in f.iterdir():shutil.copy2(item,target/'data'/item.name)
        else:shutil.copy2(f,target/f.name)
    I.check_report(target/'data/report.json');print('PENDING VALUE RESULT',r['remaining_component_count'],'components;',r['pruned_unnecessary_volume_cm3'],'cm3 removed; all original forces/exits pass',flush=True)
