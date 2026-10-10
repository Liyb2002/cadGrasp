"""The normal pipeline never runs post-search geometry recovery or replay."""
import json,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
import stable_pipeline


class StablePipelineTests(unittest.TestCase):
    def run_simulated(self,folder,labels,simulate):
        with patch.object(stable_pipeline,'HERE',Path(folder)), \
             patch.object(stable_pipeline,'invoke',side_effect=simulate) as invoke, \
             patch.object(sys,'argv',['run.py','--sets',*labels,'--output-name','new_run']):
            stable_pipeline.main()
        report=json.loads((Path(folder)/'output/B/data/new_run/pipeline.json').read_text())
        return report,invoke.call_count

    def test_force_pass_with_mesh_failure_never_enters_geometry_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/'output/B'
            def simulate(main,args):
                self.assertIs(main,stable_pipeline.run_stable_gradient.main)
                data=base/'data/new_run';data.mkdir(parents=True)
                (data/'batch.json').write_text(json.dumps({'results':[
                    dict(id='passed',passed=True,force_passed=True,mesh_exported=False,
                         volume_cm3=10.,volume_measure='nominal_material_occupancy_estimate')]}))
                (base/'passed/step4/step4.2/new_run').mkdir(parents=True)
            report,calls=self.run_simulated(folder,['passed'],simulate)
            self.assertEqual(calls,1);self.assertEqual(report['passed'],1)
            self.assertFalse(report['geometry_recovery_run']);self.assertFalse(report['final_full_demand_recheck_run'])
            self.assertEqual(report['conditioning_guests'],[])

    def test_only_uncovered_loads_get_own_state_gradient_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/'output/B'
            def simulate(main,args):
                name=args[args.index('--output-name')+1];data=base/'data'/name;data.mkdir(parents=True)
                repair=main is stable_pipeline.run_fast_state_seat_gradient.main
                if repair:
                    self.assertEqual(args[args.index('--sets')+1:args.index('--jobs')],['failed'])
                    self.assertEqual(args[args.index('--resume-experiment')+1],'new_run')
                    rows=[dict(id='failed',passed=True,volume_cm3=9.)]
                else:
                    rows=[dict(id='passed',passed=True,volume_cm3=10.),dict(id='failed',passed=False)]
                    for label in ['passed','failed']:(base/label/'step4/step4.2'/name).mkdir(parents=True)
                    out=base/'failed/step4/step4.2'/name
                    (out/'sampled_layout.npz').touch()
                    (out/'search_report.json').write_text(json.dumps({'counts':{'p1':32767}}))
                (data/'batch.json').write_text(json.dumps({'results':rows}))
            report,calls=self.run_simulated(folder,['passed','failed'],simulate)
            self.assertEqual(calls,2);self.assertEqual(report['passed'],2)
            self.assertEqual(report['gradient_repair_guests'],['failed'])
            self.assertEqual(report['conditioning_guests'],[])
            self.assertEqual(report['selected']['passed']['experiment'],'new_run')

    def test_material_selection_uses_existing_pass_record_without_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/'output/B'
            old=dict(id='passed',passed=True,volume_cm3=10.,experiment='old')
            def simulate(main,args):
                data=base/'data/new_run';data.mkdir(parents=True)
                (data/'batch.json').write_text(json.dumps({'results':[
                    dict(id='passed',passed=True,force_passed=True,volume_cm3=9.,
                         volume_measure='exported_nominal_mesh_material')]}))
                (base/'passed/step4/step4.2/new_run').mkdir(parents=True)
            with patch.object(stable_pipeline,'read_incumbents',return_value=({'passed':old},[])):
                report,calls=self.run_simulated(folder,['passed'],simulate)
            self.assertEqual(calls,1);self.assertEqual(report['selected']['passed']['volume_cm3'],9.)
            self.assertEqual(report['improved_incumbents'],1)


if __name__=='__main__':unittest.main()
