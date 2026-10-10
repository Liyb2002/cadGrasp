"""Summarize all selected groups and saved figures without running a solver."""
import _bootstrap
from collections import Counter
from co_common import *
from codes.precompute_objects.dataset import read_selected_pose_groups


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def collect():
    rows=[];execution_scope=['B']
    for manifest in sorted((ROOT/'objects').glob('*/selected_pose_sets.json')):
        name=manifest.parent.name;root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
        progress={r['id']:r for r in read(root/'data/selected_stage3_progress.json',[])}
        if progress and name not in execution_scope:execution_scope.append(name)
        progress41={r['id']:r for r in read(root/'data/selected_stage41_progress.json',[])}
        batch41=read(root/'data/step41_batch.json',{})
        progress41.update({r['id']:r for r in batch41.get('results',[])})
        for group in read_selected_pose_groups(name):
            base=root/group['id'];stage3=base/'step3';stage41=base/'step4/step4.1'
            row=dict(object=name,id=group['id'],category=group['category'],poses=group['poses'],terminal_stage=None,status='pending',step42_run=False,full_fixture_accepted=False,figures=[])
            r31=read(stage3/'step3.1/data/report.json')
            r32=read(stage3/'step3.2/data/report.json');r33=read(stage3/'step3.3/data/report.json');r41=read(stage41/'data/report.json')
            if r31 and r31.get('schema')=='whole_registered_work_cones_v1':
                row.update(terminal_stage='step3.2' if r32 and r32.get('presentation_only') else 'step3.1',
                    status=r31['status'],initialization_schema=r31['schema'],
                    initial_force_status=r31.get('initial_force_status'),
                    stage3_initialized=r31.get('passed',False),
                    step3_working_surfaces_clear=r31.get('working_surfaces_clear'),
                    downstream_requires_rebuild=True,terminal_complete=False)
                # Old Step4 reports still exist, but their seed/contact inputs
                # are historical. Advance only through verified current files.
                for path,stage in [(stage3/'step3.3/data/report.json','step3.3'),(stage41/'data/report.json','step4.1')]:
                    try:current=I.check_report(path)
                    except (OSError,RuntimeError,AssertionError,KeyError):continue
                    row.update(terminal_stage=stage,status=current['status'])
                    if stage=='step4.1':row.update(downstream_requires_rebuild=False,terminal_complete=True)
                stage42=base/'step4/step4.2'
                try:current42=I.check_report(stage42/'data/report.json')
                except (OSError,RuntimeError,AssertionError,KeyError):current42=None
                if current42 and current42.get('schema')=='whole_step4_v1':
                    row.update(terminal_stage='step4.2',status=current42['status'],step42_run=True,
                        terminal_complete=True,downstream_requires_rebuild=False,
                        force_exit_work_passed=current42['force_exit_work_passed'],
                        support_material_volume_cm3=current42['volume_cm3'])
                paths=[stage3/'step3.1/overview.png',stage3/'step3.2/overview.png']
                if not row['downstream_requires_rebuild']:paths += [stage41/f for f in ['exit_directions.png','exit_sweeps.png','final_result.png']]
                if row['step42_run']:paths += [stage42/'process.png',stage42/'final_result.png']
                for path in paths:
                    if path.exists():row['figures'].append(str(path.relative_to(HERE)))
                rows.append(row)
                continue
            executed=name=='B' or group['id'] in progress
            if executed and r32:
                row.update(terminal_stage='step3.2',status=r32['status'],step3_force_gate_passed=r32.get('force_gate_passed'),step3_working_surfaces_clear=r32.get('working_surfaces_clear'))
                if r32.get('step4_ready') and r33:
                    row.update(terminal_stage='step3.3',status=r33.get('status','pass' if r33['passed'] else 'unresolved'))
                    current41=bool(r41 and group['id'] in r41.get('initialization',{}).get('groups',{}))
                    completed41=current41 if name=='B' else group['id'] in progress41
                    if r33.get('passed') and completed41:
                        result=progress41.get(group['id'],{})
                        if result.get('error'):
                            row.update(terminal_stage=result.get('stage','step4.1'),status=result.get('status','unresolved'),error=result['error'])
                        elif r41 and current41:
                            row.update(terminal_stage='step4.1',status=r41['status'],force_and_path_initialization_gate=r41.get('force_and_path_initialization_gate'),geometry_partition_resolved=r41.get('geometry_partition_resolved'),initialization=r41.get('initialization'))
            if name not in execution_scope:row['status']='not_run_in_this_batch'
            stage3_result=progress.get(group['id'],{})
            if stage3_result.get('error'):
                row.update(terminal_stage='step3.2' if r32 else 'step3.1',status='unresolved',error=stage3_result['error'])
            elif stage3_result.get('step33_error'):
                row.update(terminal_stage='step3.3',status='unresolved',error=stage3_result['step33_error'])
            if executed:
                for path in [stage3/'step3.2/overview.png',stage3/'step3.3/overview.png',stage41/'exit_directions.png',stage41/'final_result.png',stage41/'exit_sweeps.png']:
                    if path.exists() and (path.parent!=stage41 or row['terminal_stage']=='step4.1'):
                        row['figures'].append(str(path.relative_to(HERE)))
            row['terminal_complete']=bool(row['terminal_stage'] and (row['terminal_stage'].startswith('step4.1') or row['status'] in ('fail','unresolved')))
            rows.append(row)
    counts=Counter((r['terminal_stage'] or 'pending',r['status']) for r in rows)
    result=dict(complete=all(r['terminal_complete'] for r in rows if r['object'] in execution_scope),all_selected_groups_executed=all(r['terminal_complete'] for r in rows),execution_scope_objects=execution_scope,selected_sets=len(rows),object_count=len({r['object'] for r in rows}),category_counts=dict(Counter(r['category'] for r in rows)),stage_status_counts=[dict(stage=s,status=t,count=c) for (s,t),c in sorted(counts.items())],full_fixture_accepted=False,step42_run=any(r['step42_run'] for r in rows),results=rows)
    result['execution_summary']=[]
    for name in execution_scope:
        actual=[r for r in rows if r['object']==name]
        result['execution_summary'].append(dict(object=name,selected_sets=len(actual),stage_status_counts=[dict(stage=s,status=t,count=n) for (s,t),n in sorted(Counter((r['terminal_stage'],r['status']) for r in actual).items())],category_status_counts={k:dict(Counter(r['status'] for r in actual if r['category']==k)) for k in result['category_counts']}))
    save(HERE/'data/selected_to41_status.json',result)
    lines=['# Selected pose-set stage status','',f"{len(rows)} selected groups across {result['object_count']} objects. Actual algorithm batch scope: {', '.join(execution_scope)}. Other objects have database selections only. Step4.2 is reported only after checking its current source/input/artifact fingerprints.", '',
        'B uses the merged whole initializer and production whole Step4. Old Step4 outputs are archived; current Direction/Juxtapose/Translation results and figures are in [the B Step4 table](../output/B/step4_results.md). All42 existing B sets include the current30 selections. Full installed-fixture acceptance remains deferred.', '',
        '| Object | Set | Category | Last stage | Status | Figures |','|---|---|---|---|---|---|']
    for row in rows:
        links=' · '.join(f"[{Path(p).stem}]({p.removeprefix('data/') if p.startswith('data/') else '../'+p})" for p in row['figures'])
        lines.append(f"| {row['object']} | {row['id']} | {row['category']} | {row['terminal_stage'] or 'pending'} | {row['status']} | {links} |")
    (HERE/'data/selected_to41_status.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    result=collect();print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
