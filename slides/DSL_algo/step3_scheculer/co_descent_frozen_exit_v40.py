"""Matched-budget ablation: update heads/seating but freeze the initial exit."""
import json,time,argparse
from pathlib import Path
from dataclasses import replace
from step3_scheculer import co_descent_v39 as C,contacts as I

C.STAGE='co_descent_frozen_exit_v40'

class FrozenExit(C.Optimizer):
    def joint(self,state,angle,azimuth,i,j,sign=1):
        trial=super().joint(state,angle,azimuth,i,j,sign)
        if trial is None:return None
        return replace(trial,paths=C.paths(self.initial_state.paths[0]['initial_object_exit_world'],len(self.tasks)))
    def publish(self):
        summary=super().publish();out=self.base/'step4'/C.STAGE;step5=self.base/'step5_evaluate'/C.STAGE
        report=I.check_report(out/'report.json');report.update(direction_is_continuous=False,exit_frozen=True,ablation='same original incumbent, head/rewrite/seating budgets and seeds; initial world exit held fixed')
        report['provenance']['code'].update(I.hashes([Path(__file__)]));I.save(out/'report.json',report)
        metric=json.loads((step5/'report.json').read_text())
        metric['provenance']['inputs'].update(I.hashes([out/'report.json']));metric['provenance']['code'].update(I.hashes([Path(__file__)]));I.save(step5/'report.json',metric)
        I.check_report(out/'report.json');I.check_report(step5/'report.json');return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+',default=['pose5+7','pose3+6']);args=parser.parse_args();began=time.monotonic()
    config=dict(device='cuda',rounds=2,joint_trials=4,rewrite_trials=2,rewrite_local_steps=1,placement_attempts=2,angle_degrees=3.7,max_heads=10,seed=20261004)
    results=[FrozenExit(g,config).run() for g in args.sets]
    out=I.OUTPUTS/'B'/args.sets[-1]/'step5_evaluate'/C.STAGE;I.save(out/'batch.json',dict(complete=True,groups=results,wall_seconds=time.monotonic()-began))
