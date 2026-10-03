"""Publish every accepted DSL fixture at the public Step4 entry.

Preserve byte-exact initialization inputs before replacing public files. Migrate
input references and their dependent hashes; retain original generator hashes.
No construction search or exported-model geometry replay is performed here.
"""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I


def relative(p):return str(p.resolve().relative_to(I.ROOT))


def migrate(value,mapping):
    if isinstance(value,dict):return {mapping.get(k,k):migrate(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [migrate(v,mapping) for v in value]
    if isinstance(value,str):return mapping.get(value,value)
    return value


def publish_all():
    groups=sorted((I.OUTPUTS/'B').glob('pose*+*'));mapping={};rows=[]
    # Validate the completed experiment before any public replacement.
    for group in groups:
        r=I.check_report(group/'step3_scheculer/dsl_feasible/report.json')
        assert r['passed'] and r['constructed']
        body=group/'step4/data/dsl_feasible_support'
        assert I.check_report(body/'report.json')['passed']
        assert I.sha256(body/'shape.obj')==I.sha256(group/'step3_scheculer/dsl_feasible/final/shape.obj')
        snapshot=group/'step3_scheculer/dsl_feasible/original_seed/public_step4_inputs'
        snapshot.mkdir(parents=True,exist_ok=True)
        for source,name in ((group/'step4/shape.obj','shape.obj'),(group/'step4/data/report.json','report.json')):
            if not source.exists():continue
            digest=I.sha256(source)
            target=snapshot/(Path(name).stem+"_"+digest[:12]+Path(name).suffix)
            if not target.exists():shutil.copy2(source,target)
            assert I.sha256(target)==digest, "Initialization snapshot must preserve exact input bytes"
            mapping[relative(source)]=relative(target)
    # Only active V5 records are relocated. Historical algorithm reports retain
    # their original definitions; the immutable input snapshots preserve bytes.
    files=[]
    for group in groups:
        files.extend(p for p in (group/'step3_scheculer/dsl_feasible').rglob('*.json') if 'public_step4_inputs' not in p.parts)
        files.append(group/'step4/data/dsl_feasible_support/report.json')
    records={p:json.loads(p.read_text()) for p in files}
    changed=[]
    for p,data in records.items():
        updated=migrate(data,mapping)
        if updated!=data:
            records[p]=updated;I.save(p,updated);changed.append(relative(p))
    # Update the dependency DAG from leaves toward reports and audit summaries.
    for _ in range(len(records)+1):
        updates=0
        for p,data in records.items():
            inputs=data.get('provenance',{}).get('inputs',{})
            for name,digest in list(inputs.items()):
                target=I.ROOT/name
                if not target.exists():raise RuntimeError('Missing migrated input: '+name)
                actual=I.sha256(target)
                if actual!=digest:
                    if target not in records:raise RuntimeError("Unexpected changed physical input: "+name)
                    inputs[name]=actual;updates+=1
            if updates:I.save(p,data)
        if not updates:break
    else:raise RuntimeError('Cyclic active report dependencies')
    for group in groups:
        root=group/'step4';body=root/'data/dsl_feasible_support'
        r=I.check_report(body/'report.json')
        for source,target in (('shape.obj','shape.obj'),('geometry_certificate.npz','geometry_certificate.npz'),
                              ('overview.png','overview.png'),('construction.png','construction_steps.png')):
            shutil.copy2(body/source,root/target)
            assert I.sha256(body/source)==I.sha256(root/target)
        report=dict(r,step4_executed=True,public_step4_published=True,
            construction_search_rerun=False,exported_geometry_replay=False,
            publication_policy='publish the exact fully accepted Step3 fixture; no second construction search',
            placement=dict(r['placement'],directions=[p['support_withdrawal_world'] for p in r['selected_exit_paths']]),
            provenance=dict(inputs=I.hashes([body/'report.json']),code=I.hashes([Path(__file__)])),
            artifacts={name:I.sha256(root/name) for name in ('shape.obj','geometry_certificate.npz','overview.png','construction_steps.png')})
        I.save(root/'report.json',report)
        data_report=dict(report,artifacts={'../'+k:v for k,v in report['artifacts'].items()})
        I.save(root/'data/report.json',data_report)
        (root/'README.md').write_text(f"# Current Step4: feasible DSL fixture\n\n{group.name}: **{r['physical_head_count']} physical heads, {r['shared_head_count']} shared**. All saved poses pass all 32768 original loads and their own continuous exits. This is the complete connected fixture accepted in Step3.\n\n[Model](shape.obj) · [Overview](overview.png) · [Construction illustration](construction_steps.png) · [Report](report.json)\n\nStep4 publishes the identical accepted fixture, with one construction acceptance and no additional geometry replay. Initial public input files are preserved byte-for-byte under `../step3_scheculer/dsl_feasible/original_seed/public_step4_inputs/`.\n")
        # Validate both public report entry points after replacing old files.
        assert I.check_report(root/'report.json')['passed']
        assert I.check_report(root/'data/report.json')['passed']
        rows.append(dict(group=group.name,passed=True,physical_heads=r['physical_head_count'],shared_heads=r['shared_head_count'],
                         shape_sha256=I.sha256(root/'shape.obj'),step3_step4_identical=True))
        print('PUBLIC STEP4',group.name,r['physical_head_count'],r['shared_head_count'],'PASS',flush=True)
    # Validate relocated force/identity audits and every active final record.
    for group in groups:
        for p in (group/'step3_scheculer/dsl_feasible/report.json',group/'step3_scheculer/dsl_feasible/audit.json'):
            I.check_report(p)
    out=groups[[g.name for g in groups].index('pose2+9+13+15+17')]/'step4/data'
    I.save(out/'dsl_publication.json',dict(complete=True,passed=True,passed_count=len(rows),results=rows,
        input_path_migrations=mapping,relocated_reports=changed,original_generator_hashes_retained=True,
        new_construction_claimed=False,provenance=dict(inputs=I.hashes([g/'step4/report.json' for g in groups]),code=I.hashes([Path(__file__)]))))

if __name__=='__main__':publish_all()
