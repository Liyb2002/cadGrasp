"""Clarify recovered timing scope without changing search or geometry evidence."""
from pathlib import Path
import json
from step3_scheculer import contacts as I

def main():
    for p in (I.OUTPUTS/'B'/'independent_poses_gpu_v12').glob('pose_*/step3_scheculer/schedule.json'):
        d=json.loads(p.read_text())
        if not d.get('recovered_interrupted_run'):continue
        d['timing_scope']='Sum of recorded search-round timings, INCLUDING candidate preparation inside rounds; excludes constructor setup, final rechecks, export/rendering and interruptions'
        d['timing_metadata_correction']=dict(search_seconds_unchanged=True,geometry_changed=False)
        d['provenance']['code'].update(I.hashes([Path(__file__)]));I.save(p,d);I.check_report(p)
    print('Recovered timing scope clarified')

if __name__=='__main__':main()
