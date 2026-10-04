"""Correct material-mask labels; retain geometry and original generator hashes."""
from pathlib import Path
import json
from step3_scheculer import contacts as I

STAGES=('dsl_cavity',)

def correct(groups):
    records={};changed=[]
    for group in groups:
        previous=group/'step3_scheculer'/('dsl_absolute_floor' if (group/'step3_scheculer/dsl_absolute_floor/report.json').exists() else 'dsl_absolute_refined')/'final'
        for stage_name in STAGES:
            stage=group/'step3_scheculer'/stage_name
            if not (stage/'report.json').exists():continue
            I.check_report(stage/'report.json')
            material=float(I.check_report(previous/'report.json')['volume_cm3'])
            for path in sorted((stage/'trials').glob('*/report.json')):
                r=json.loads(path.read_text())
                if not r.get('passed'):continue
                I.check_report(path)
                if not r.get('material_comparison_corrected'):
                    r['metadata_history']=dict(previous_material_volume_cm3=r.get('previous_material_volume_cm3'),material_reduction_percent=r.get('material_reduction_percent'))
                    r['search_space_volume_cm3']=r.get('previous_material_volume_cm3')
                    r['previous_material_volume_cm3']=material
                    r['material_reduction_percent']=100*(1-r['volume_cm3']/material)
                    r['material_comparison_corrected']=True;r['metadata_only_update']=True
                    r['comparison_reference']='qualified previous physical fixture, not navigation free space'
                    r['provenance']['code'].update(I.hashes([Path(__file__)]))
                    changed.append(path)
                records[path]=r
                metric=path.parent/'step5/report.json'
                if metric.exists():records[metric]=json.loads(metric.read_text())
            for path in (stage/'final/report.json',stage/'report.json',stage/'previous_public_step4/report.json',group/'step4/data'/(stage_name+'_support')/'report.json'):
                if path.exists():records[path]=json.loads(path.read_text())
            previous=stage/'final'
        for path in (group/'step4/report.json',group/'step4/data/report.json',group/'step5_evaluate/report.json'):
            records[path]=json.loads(path.read_text())
    # Refresh derived JSON dependencies in their DAG. Geometry, contact arrays,
    # acceptance results and original generator code hashes remain unchanged.
    for _ in range(len(records)+1):
        updates=0
        for path,r in records.items():
            for name,digest in list(r.get('provenance',{}).get('inputs',{}).items()):
                actual=I.sha256(I.ROOT/name)
                if actual!=digest:r['provenance']['inputs'][name]=actual;updates+=1
            I.save(path,r)
        if not updates:break
    else:raise RuntimeError('Cyclic metadata dependency')
    for path in records:I.check_report(path)
    return changed
