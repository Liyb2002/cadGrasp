"""Preserve an accepted legal witness's doubles when saving an older worker.

This adapter only changes OBJ formatting and the associated artifact hashes.
It never queries the Oracle, constructs material or evaluates demands.
"""
from pathlib import Path
import hashlib
import json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path=Path(path)
    temporary=path.with_suffix('.exact-witness-writing.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def write_exact_witness(folder, witness, report, *, export, origin):
    """Write the same mesh with round-trip coordinates; refresh hash references."""
    folder=Path(folder)
    path=folder/'oracle_connected_witness.obj'
    old=digest(path)
    export(witness,path)
    exact=digest(path)
    evidence=dict(origin=origin,previous_decimal16_sha256=old,
        exact_float64_sha256=exact,mesh_coordinates_modified=False,
        search_run=False,force_solver_run=False)
    pre_path=folder/'preconstructed_base.json'
    if report.get('solid_constructed_during_search'):
        pre=json.loads(pre_path.read_text())
        pre['source_sha256'][path.name]=exact
        pre['legal_witness_export']=evidence
        write_json(pre_path,pre)
        report['artifacts'][pre_path.name]=digest(pre_path)
    report['artifacts'][path.name]=exact
    report['legal_witness_export']=evidence
    write_json(folder/'report.json',report)
    data_path=folder/'data/report.json'
    if data_path.exists():
        data=dict(report)
        data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
        write_json(data_path,data)
    return evidence


def install():
    """Adapt the frozen save method after the original scope has been checked."""
    from oracle_search import OracleSearch
    from co_common import D
    previous=OracleSearch.save
    def save(self,current,*args,**kwargs):
        report=previous(self,current,*args,**kwargs)
        selected=getattr(self,'material_selected',None)
        if (report.get('base_oracle_passed') and selected is not None
                and selected.layout.key()==current['layout'].key()
                and selected.witness is not None):
            write_exact_witness(self.out,selected.witness,report,
                export=D.export_exact_obj,origin='accepted_in_memory_witness_at_save')
        return report
    OracleSearch.save=save
