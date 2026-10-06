"""Check one algorithm tree against every published input; write only temporary stages."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('tree',choices=['baseline_algo']);args=p.parse_args()
sys.path.insert(0,str(ROOT/'slides'/args.tree));sys.path.insert(0,str(ROOT))
from step1.registry import active_cases, active_objects, task_poses
from step1.needs import build, sha256
from step1.cases import selected_pose
from step3_scheculer.pair_tasks import read_task, task_folder
from step3_scheculer import contacts as I
from step0_pose_selection.select_poses import select_many, TaskCache, materialize_step1
from step3_scheculer import joint_tasks
import numpy as np
for name in active_objects():
    assert len(task_poses(name))==30
    for pose in task_poses(name):
        folder=task_folder(name,pose)
        assert folder==ROOT/'objects'/name/'poses'/pose
        task=read_task(name,pose)
        assert len(task.targets)==32768
    print('READ',args.tree,name,'30 inputs',flush=True)
with tempfile.TemporaryDirectory(prefix='.consumer_check_',dir=ROOT) as tmp:
    out=Path(tmp)
    with selected_pose('pose_30'):
        build('B',output_folder=out/'step1')
    source=ROOT/'objects/B/poses/pose_30'
    assert sha256(source/'samples.json')==sha256(out/'step1/samples.json')
    with patch.object(I,'OUTPUTS',out/'output'),patch.object(joint_tasks,'OUTPUTS',out/'output'):
        selections,record=select_many('B',6,20261003,1)
        assert record['pose_search_repeated'] is False and record['attempted_count']==0
        tasks=materialize_step1('B',selections[0].report['selected_poses'],selections[0].check_path)
        assert len(tasks)==6
        for task in tasks:
            assert sha256(task.inputs[1])==sha256(ROOT/'objects/B/poses'/task.pose/'samples.json')
            pickle.loads(pickle.dumps(task.domain))
    # Old algorithm inputs must fail instead of mixing their heads with new poses.
    old=ROOT/'slides'/args.tree/'output/B/pose1+3/step_1_needs/pose_1'
    rejected=False
    try: read_task('B','pose_1',folder=old)
    except (ValueError,FileNotFoundError): rejected=True
    assert rejected
print('CONSUMER PASS',args.tree,'630 inputs / exact copies / six-pose adapter / stale-input rejection',flush=True)
