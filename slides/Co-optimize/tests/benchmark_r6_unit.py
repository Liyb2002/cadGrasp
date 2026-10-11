"""Read-only comparison of the unchanged control and R6 unit-change batches.

Run after both stable_pipeline runs finish. No physics replay, mesh Boolean,
search modification or old answer substitution in the attempted comparison.
"""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'slides/Co-optimize/output/B'
OUT=BASE/'data/r6_unit_comparison_v1'
CONTROL='r6_unit_control_v1'
NEW='r6_unit_gradient_v1'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def stages(name):
    pipe=read(BASE/'data'/name/'pipeline.json')
    primary=read(BASE/'data'/name/'batch.json')
    repair=BASE/'data'/(name+'_repair')/'batch.json'
    batches=[primary]+([read(repair)] if repair.exists() else [])
    assert pipe['complete'] and all(b['complete'] for b in batches)
    assert all(b['protected_unchanged'] and b['executed_sources_unchanged'] for b in batches)
    timed={}
    for batch in batches:
        for row in batch['results']:
            data=timed.setdefault(row['id'],dict(search=0.,worker=0.,force=0.,gradient=0.,gradient_calls=0,reused=0,solved=0))
            data['search']+=row.get('search_seconds',0.);data['worker']+=row['seconds']
            data['force']+=row.get('timing',{}).get('sampled_full_load_checks_s',0.)
            data['gradient']+=row.get('timing',{}).get('r6_gradient_loss_s',0.)
            data['gradient_calls']+=row.get('timing',{}).get('r6_gradient_loss_calls',0)
            data['reused']+=row.get('timing',{}).get('r6_projection_cells_reused',0)
            data['solved']+=row.get('timing',{}).get('r6_projection_nodes_solved',0)
    return pipe,batches,timed


def main():
    old,ob,ot=stages(CONTROL);new,nb,nt=stages(NEW)
    sets=read(OUT/'sets.json');assert set(sets)==set(old['attempted'])==set(new['attempted'])
    assert ob[0]['options']['jobs']==nb[0]['options']['jobs']
    for option in ['worker_cpus','iterations','demand_nodes']:
        assert ob[0]['options'][option]==nb[0]['options'][option]
    objects=read(OUT/'protected_objects.json')
    assert all(sha(ROOT/p)==s for p,s in objects.items())
    executed=read(OUT/'unit_sources.json')
    assert all(sha(ROOT/p)==s for p,s in executed['code'].items())
    assert all(sha(ROOT/p)==s for p,s in executed['snapshots'].items())
    control_sources=read(BASE/'data'/CONTROL/'source_manifest.json')['code']
    changed=sorted(p for p,s in control_sources.items() if sha(ROOT/p)!=s)
    allowed=sorted('slides/Co-optimize/helper_func/'+p for p in [
        'continuous_support/fast_gradient.py','continuous_support/adaptive_fast_gradient.py',
        'translation/layout_update.py'])
    assert changed==allowed,changed
    rows=[]
    for label in sets:
        a,b=old['attempted'][label],new['attempted'][label]
        size=len(label.removeprefix('pose').split('+'))
        material=all(r.get('mesh_exported') and r.get('volume_measure')=='exported_nominal_mesh_material' for r in [a,b])
        av,bv=(a.get('volume_cm3'),b.get('volume_cm3')) if material else (None,None)
        row=dict(id=label,poses=size,old_passed=a['passed'],new_passed=b['passed'],
                 old_experiment=a['experiment'],new_experiment=b['experiment'],
                 old_volume_cm3=av,new_volume_cm3=bv,
                 material_change_percent=(bv/av-1)*100 if av and bv is not None else None,
                 old_search_seconds=ot[label]['search'],new_search_seconds=nt[label]['search'],
                 old_worker_seconds=ot[label]['worker'],new_worker_seconds=nt[label]['worker'],
                 old_counts=a.get('counts'),new_counts=b.get('counts'),
                 new_gradient_loss_seconds=nt[label]['gradient'],new_gradient_loss_calls=nt[label]['gradient_calls'],
                 projection_cells_reused=nt[label]['reused'],projection_nodes_solved=nt[label]['solved'],
                 retained_incumbent=new['selected'][label].get('retained_incumbent',False))
        if material and a['passed'] and b['passed']:
            pa=BASE/label/'step4/step4.2'/a['experiment']/'layout.npz'
            pb=BASE/label/'step4/step4.2'/b['experiment']/'layout.npz'
            with np.load(pa) as za,np.load(pb) as zb:
                row['layout_max_difference']={k:float(np.abs(za[k]-zb[k]).max(initial=0.))
                                              for k in ['placements','directions','hosts']}
        rows.append(row)
    common=[r for r in rows if r['old_passed'] and r['new_passed'] and r['old_volume_cm3'] is not None]
    same=[r for r in common if abs(r['material_change_percent'])<.01]
    wall_old=sum(b['seconds'] for b in ob);wall_new=sum(b['seconds'] for b in nb)
    search_old=sum(r['old_search_seconds'] for r in rows);search_new=sum(r['new_search_seconds'] for r in rows)
    summary=dict(complete=True,groups=len(rows),same_jobs=ob[0]['options']['jobs'],
        worker_cpus=ob[0]['options']['worker_cpus'],control=CONTROL,new=NEW,
        new_formula=read(ROOT/'objects/r6_quadratic_index.json')['schema'],
        old_new_search_passed=old['new_search_passed'],new_new_search_passed=new['new_search_passed'],
        old_selected_passed=old['passed'],new_selected_passed=new['passed'],
        new_retained_incumbents=new['retained_incumbents'],
        old_wall_seconds=wall_old,new_wall_seconds=wall_new,wall_speedup=wall_old/wall_new,
        old_total_search_seconds=search_old,new_total_search_seconds=search_new,search_speedup=search_old/search_new,
        old_median_search_seconds=statistics.median(r['old_search_seconds'] for r in rows),
        new_median_search_seconds=statistics.median(r['new_search_seconds'] for r in rows),
        common_passed_meshes=len(common),material_unchanged_within_point01_percent=len(same),
        old_common_material_cm3=sum(r['old_volume_cm3'] for r in common),
        new_common_material_cm3=sum(r['new_volume_cm3'] for r in common),
        source_changes=changed,unchanged_original_objects_files=len(objects),
        unchanged_executed_unit_files=len(executed['code']),unit_pilot=read(OUT/'unit_pilot.json'),
        final_full_demand_recheck_run=False,geometry_recovery_run=False,rows=rows)
    write(OUT/'comparison.json',summary)
    lines=['# R6 gradient unit comparison','',
           f"All {len(rows)} current B sets; cold Step4.1 starts, same {summary['same_jobs']} jobs / {summary['worker_cpus']} CPUs per worker, same search budgets.",
           '',f"Fresh force passes: {old['new_search_passed']}/{len(rows)} → {new['new_search_passed']}/{len(rows)}. Selected feasible results: {new['passed']}/{len(rows)}; {new['retained_incumbents']} retained incumbents.",
           f"Wall time (cold + own continuation): {wall_old/60:.2f} → {wall_new/60:.2f} min ({wall_old/wall_new:.3f}×). Sum of search time: {search_old/60:.2f} → {search_new/60:.2f} min ({search_old/search_new:.3f}×).",
           f"Material comparison includes {len(common)} common passed, exported nominal meshes; {len(same)} differ by less than 0.01%.",
           '', 'The table compares fresh attempted results, including their own continuation. Historical final selections are not attributed to the new algorithm. PASS still uses all original 32,768 demands per pose; no repeated final force check or geometry conditioning.',
           '', '| Set | Old / new PASS | Material cm³ old → new | Δ | Search seconds old → new |',
           '|---|---|---:|---:|---:|']
    for r in rows:
        volume=f"{r['old_volume_cm3']:.4f} → {r['new_volume_cm3']:.4f}" if r['old_volume_cm3'] is not None else 'not comparable'
        delta=f"{r['material_change_percent']:+.3f}%" if r['material_change_percent'] is not None else '—'
        result=f"{'PASS' if r['old_passed'] else 'FAIL'} / {'PASS' if r['new_passed'] else 'FAIL'}"
        lines.append(f"| {r['id']} | {result} | {volume} | {delta} | {r['old_search_seconds']:.1f} → {r['new_search_seconds']:.1f} |")
    lines.extend(['','Only the three existing numerical gradient files listed in comparison.json changed; the new internal evaluator is direction_choice/r6_gradient.py. Entry points, Juxtapose, step pools, Step3/4.1 and original object demands are unchanged.','',
                  '[Exact forward formulas](../../../../objects/r6_quadratic.md) · [JSON](data/r6_unit_comparison_v1/comparison.json) · [Images](r6_unit_comparison_v1_index.html)'])
    (BASE/'r6_unit_comparison_v1_results.md').write_text('\n'.join(lines)+'\n')
    html=['<!doctype html><meta charset="utf-8"><title>R6 unit comparison</title>',
          '<style>body{font:16px system-ui;margin:24px;background:#fafafa}img{width:49%;vertical-align:top}article{margin-bottom:32px}h2{font-size:18px}</style>',
          '<h1>R6 unit comparison</h1><p>Fresh attempted layouts, control on the left and new unit on the right.</p>']
    for r in rows:
        html.append(f'<article><h2>{r["id"]}</h2>')
        for experiment in [r['old_experiment'],r['new_experiment']]:
            src=f'{r["id"]}/step4/step4.2/{experiment}/final_result.png'
            if (BASE/src).exists():html.append(f'<img loading="lazy" src="{src}" alt="{experiment}">')
            else:html.append(f'<p>{experiment}: no exported result mesh.</p>')
        html.append('</article>')
    (BASE/'r6_unit_comparison_v1_index.html').write_text('\n'.join(html)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','unit_pilot']},indent=2))


if __name__=='__main__':main()
