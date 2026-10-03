"""Summarize the completed frozen direct-growth batch without altering solids."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current.run_thick_batch import GROUPS,ROOT
from step4_connect_support.baseline_current.run_boxed_batch import protected_hashes

out=ROOT/GROUPS[-1]/'step4/data/growing_support'
old_protected=json.loads((out/'before_straight_batch_protected.json').read_text())
current=protected_hashes(ROOT)
assert old_protected==current,'Frozen Step3 / original historical pair changed'
rows=[]
for name in GROUPS:
    group=ROOT/name;folder=group/'step4/data/growing_support'
    new=I.check_report(folder/'report.json')
    I.check_report(group/'step4/data/boxed_support/report.json')
    I.check_report(group/'step4/data/compact_visualization.json')
    assert json.loads((folder/'replay.json').read_text())['passed']
    old=json.loads((group/'step4/data/history/before_thick_growing_support/report.json').read_text())
    old_c=old['construction'];new_c=new['construction'];thick=new['minimum_branch_thickness']
    trial=json.loads((folder/'thick_attempts.json').read_text())['attempts'][-1]
    row=dict(group=name,passed=True,old_segments=old_c['grown_segment_count'],new_segments=new_c['grown_segment_count'],
        direct_connections=new_c['direct_connection_count'],removed_grid_bends=new_c['removed_grid_bends'],
        old_material_volume_cm3=old['volume_cm3'],new_material_volume_cm3=new['volume_cm3'],
        old_space_budget=old['space_budget'],new_space_budget=new['space_budget'],
        occupied_volume_change_percent=100*(new['space_budget']['box_volume_cm3']/old['space_budget']['box_volume_cm3']-1),
        minimum_core_diameter_mm=thick['guaranteed_inscribed_beam_diameter_mm'],
        minimum_sole_thickness_mm=thick['sole_thickness_mm'],
        maximum_contact_transition_mm=max(t['original_contact_transition_length_mm'] for t in thick['local_branches'] if ':foot' not in t['name']),
        maximum_missing_core_volume_m3=max(t['missing_volume_m3'] for t in thick['exported_unclipped_core_checks']),
        construction_and_internal_replay_seconds=new['seconds'],successful_domain_margin_mm=trial['margin_mm'],
        upstream_step3_passed=new['step3_passed'],code=new['provenance']['code'])
    rows.append(row)
report=dict(passed=True,groups=rows,protected_file_count=len(current),protected_inputs_unchanged=True,
    final_models_fresh_sweep_replayed=True,full_core_preservation=True,
    original_and_copied_physical_models_unchanged=True,source_hash=I.sha256(Path(__file__).with_name('run_thick_batch.py')),
    guarantee_scope='complete rod cores and soles; explicitly recorded original contact margins and short root transitions excluded',
    global_minimum_claim=False,structural_strength_claim=False,forces_recomputed=False)
I.save(out/'thick_batch_report.json',report)
lines=['# All eight groups: thick direct growth','',
'All eight exported supports pass fresh continuous-sweep replay, actual ground coverage, working faces, roots/contacts, floors and connectivity. Frozen Step3 and the original historical pair remain unchanged. Prior boxed domains and sparse bodies are retained under each group’s `step4/data/history/before_thick_*`.','',
'Complete rod cores have a certified 5.108 mm inscribed diameter (5.2 mm template outer diameter). Soles remain at least 5 mm thick and connect through full 6 mm high columns, without tapered foot necks. Original contact edge margins and explicitly recorded 3.1–5.1 mm contact-to-core transitions are excluded; this is not an every-boundary-point thickness or strength claim. Whole core preservation is verified on serialized OBJ.','',
'| Group | Rod segments old → new | Direct | Grid bends removed | Material cm³ | Occupied box cm³ | Box change | Successful extra margin mm | Construction + internal replay s |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['group']} | {r['old_segments']} → {r['new_segments']} | {r['direct_connections']} | {r['removed_grid_bends']} | {r['new_material_volume_cm3']:.2f} | {r['new_space_budget']['box_volume_cm3']:.2f} | {r['occupied_volume_change_percent']:+.1f}% | {r['successful_domain_margin_mm']} | {r['construction_and_internal_replay_seconds']:.1f} |")
lines += ['',f"Protected files: {len(current)}. Physical source code was frozen before the final eight-group rerun. Force/torque acceptance remains the original Step3 verdict; an incomplete Step3 remains incomplete.",'',
'Each group’s `data/growing_support/stages.json` supplies the actual cumulative bodies, every installed object in fixture coordinates, the ideal/accepted/actual boxes and camera. Public overview, support and all pose PNGs display the new bodies. Models are `data/growing_support/shape.obj` with that directory’s placement report. Public legacy `shape.obj` remains the comparison source.']
(out/'thick_batch_report.md').write_text('\n'.join(lines)+'\n')
print('FINAL EIGHT-GROUP LEDGER PASSED',len(current),flush=True)
