"""Correct inherited comparison/radius labels; preserve exact geometry proofs."""
from pathlib import Path
import json
from step3_scheculer import operation_dsl as F,contacts as I


def correct_all(groups):
    records={};changed=[]
    for group in groups:
        stage=group/'step3_scheculer'/F.STAGE
        top=I.check_report(stage/'report.json')
        previous=I.check_report(group/'step3_scheculer/dsl_feasible/final/report.json')
        previous_volume=float(previous['volume_cm3'])
        witnesses={I.ROOT/top['witness']}
        witnesses.update(I.ROOT/e['witness'] for e in top['events'] if 'witness' in e)
        for path in witnesses:
            report=I.check_report(path)
            if report.get('comparison_metadata_corrected'):continue
            construction=report['construction']
            original=dict(previous_material_volume_cm3=report.get('previous_material_volume_cm3'),material_reduction_percent=report.get('material_reduction_percent'),beam_radius_m=construction.get('beam_radius_m'))
            # The adapter's bead radius is explicitly 2.6 mm. DirectGrow's
            # copied-group name heuristic is not the adapter's geometry input.
            construction['beam_radius_m']=.0026
            report['search_space_volume_cm3']=report.get('previous_material_volume_cm3')
            report['previous_material_volume_cm3']=previous_volume
            report['material_reduction_percent']=100*(1-report['volume_cm3']/previous_volume)
            report['comparison_reference']='qualified V5 final physical fixture; free-space mask is not a support'
            report['comparison_metadata_corrected']=True
            report['metadata_correction']=dict(original_values=original,geometry_and_acceptance_unchanged=True,reason='DirectGrow inherited labels used the navigation mask volume and a copied-group radius heuristic')
            report['provenance']['code'].update(I.hashes([Path(__file__)]))
            I.save(path,report);changed.append(str(path.relative_to(I.ROOT)))
        dependencies=[stage/'final/report.json',stage/'report.json',group/'step4/data'/F.BODY_STAGE/'report.json']
        records.update({path:json.loads(path.read_text()) for path in dependencies})
    # Refresh only derived JSON dependencies. Physical bytes and original
    # generator hashes are preserved; the normal audit/publisher run afterwards.
    for _ in range(len(records)+1):
        updates=0
        for path,record in records.items():
            for name,digest in list(record.get('provenance',{}).get('inputs',{}).items()):
                actual=I.sha256(I.ROOT/name)
                if actual!=digest:
                    record['provenance']['inputs'][name]=actual;updates+=1
            I.save(path,record)
        if not updates:break
    else:raise RuntimeError('Cyclic report dependency')
    return changed
