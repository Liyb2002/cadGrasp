"""Retry numerical sweep-preparation failures using a separate initializer."""
import _bootstrap
import argparse
import copy
from co_common import *
from codes.precompute_objects.dataset import read_selected_pose_groups
import step41


def retry(name, group_id, update_ledgers=True):
    root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
    folder=root/'data/step41';source=folder/'initialization.json'
    original=json.loads(source.read_text());groups=read_selected_pose_groups(name)
    group=next(g for g in groups if g['id']==group_id)
    initialization=copy.deepcopy(original);record=copy.deepcopy(initialization['groups'][group_id]);record.pop('preparation_error',None)
    gid,artifacts,error,record=step41._prepare_group_sweeps((name,group,record,original['full_length_m'],folder))
    if error:
        raise RuntimeError(error)
    initialization['groups'][gid]=record;initialization['artifacts'].update(artifacts)
    destination=folder/('recovery_'+group_id.replace('/','_')+'.json')
    initialization['source_path']=str(destination)
    initialization['original_initialization_provenance']=initialization['provenance']
    initialization['provenance']=provenance([source,*[ROOT/p for p in original['provenance']['inputs']]],[Path(__file__),HERE/'step4.1/step41.py',HERE/'helper_func/optimization/initial_directions.py'])
    save(destination,initialization)
    row=step41.run(name,group,initialization)
    save(root/group_id/'step4/step4.1/data/recovery_result.json',row)
    for path,key in ([(root/'data/selected_stage41_progress.json',None),(root/'data/step41_batch.json','results')] if update_ledgers else []):
        if not path.exists():continue
        data=json.loads(path.read_text());rows=data if key is None else data[key]
        rows[:]=[r for r in rows if r['id']!=group_id]+[row]
        save(path,data)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('object');parser.add_argument('group');parser.add_argument('--no-ledger',action='store_true');args=parser.parse_args()
    print(retry(args.object,args.group,update_ledgers=not args.no_ledger),flush=True)
