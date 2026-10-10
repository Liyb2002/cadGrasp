"""Beam isolation, real-volume acceptance and Boolean-free whole search."""
import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from compact_volume import CompactSearch,Node,distinct_frontier,load_incumbent
from whole_search.model import Layout,Model
from whole_search.fast_search import FastModel
from whole_search.common import C


def candidate(index,volume,passed=True,host=0):
    layout=Layout(np.eye(4)[None],np.array([[np.sin(index*.02),0.,np.cos(index*.02)]]),
        np.array([host]),(0,))
    return dict(layout=layout,serial=index,volume_cm3=volume,
        masks={0:np.array([passed,True])},counts={'0':int(passed)+1},
        maximum_projected_footprint_m2=.01,actual_work_surface_checks=[dict(passed=True)])


class CompactTests(unittest.TestCase):
    def test_repair_frontier_cannot_displace_feasible_incumbents(self):
        nodes=[Node(0,None,candidate(0,100), 'baseline',{},0),
            Node(1,0,candidate(1,1,False),'restore-reuse',{},1),
            Node(2,0,candidate(2,80,True,1),'juxtapose',{},1),
            Node(3,0,candidate(3,79,True,1),'translation',{},1)]
        chosen=distinct_frontier(nodes,2,1)
        self.assertEqual([n.index for n in chosen],[3,0,1])
        self.assertTrue(all(n.result['masks'][0].all() for n in chosen[:2]))

    def test_finalists_keep_distinct_hosts_and_include_multiple_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            search=CompactSearch(SimpleNamespace(),Path(directory))
            nodes=[Node(0,None,candidate(0,100),'baseline',{},0),
                Node(1,0,candidate(1,80,True,0),'direction',{},1),
                Node(2,0,candidate(2,81,True,0),'direction',{},1),
                Node(3,0,candidate(3,90,True,1),'juxtapose',{},1)]
            self.assertEqual([n.index for n in search.actual_finalists(nodes,2)],[1,3])

    def test_actual_growth_or_failed_load_keeps_checked_baseline(self):
        baseline=candidate(0,100);optimistic=candidate(1,70)
        for actual in [dict(optimistic,volume_cm3=110),candidate(1,60,False)]:
            model=SimpleNamespace(poses=['pose_1'],exact=lambda q:actual,timing={})
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory);search=CompactSearch(model,out)
                search.search_seconds=0.;search.export_path=lambda n,r:None
                node=Node(1,0,optimistic,'direction',{},1)
                with patch.object(Model,'save',side_effect=lambda m,r,o,e:dict(result=r,provenance=dict(inputs={},code={}),**e)),\
                     patch('compact_volume.I.sha256',return_value='checked'),\
                     patch('compact_volume.I.hashes',return_value={}),\
                     patch('compact_volume.save'):
                    report=search.validate_and_save([node],baseline,{},out,1)
                self.assertTrue(report['baseline_retained'])
                self.assertEqual(report['result']['volume_cm3'],100)
                self.assertEqual(report['material_saved_cm3'],0)

    def test_export_path_follows_parent_chain_not_other_beam_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            search=CompactSearch(SimpleNamespace(),Path(directory));seen=[]
            search.nodes=[Node(0,None,candidate(0,100),'baseline',{},0),
                Node(1,0,candidate(1,90),'direction',{},1),
                Node(2,0,candidate(2,88),'translation',{},1),
                Node(3,2,candidate(3,80),'direction',{},2)]
            search.process_snapshot=lambda r,p,d:seen.append(d['tree_node'])
            search.export_path(search.nodes[3],search.nodes[3].result)
            self.assertEqual(seen,[0,2,3])

    def test_whole_compaction_search_never_calls_boolean_or_exact(self):
        model=FastModel(['pose_1','pose_2'])
        layout,_=model.initial();original=[task.targets.copy() for task in model.tasks]
        with tempfile.TemporaryDirectory() as directory:
            search=CompactSearch(model,Path(directory),rounds=1,screen_budget=12,full_budget=2)
            with patch.object(model,'exact',side_effect=AssertionError('real finalist during search')),\
                 patch.object(C,'union',side_effect=AssertionError('Boolean during search')):
                nodes=search.run(dict(layout=layout))
            self.assertEqual(search.full_checks,2)
            for node in search.nodes:
                self.assertEqual(node.result['layout'].active,(0,1))
                self.assertEqual(len(node.result['masks'][0]),32768)
            for k,task in enumerate(model.tasks):np.testing.assert_array_equal(task.targets,original[k])

    def test_checked_baseline_can_actually_be_published_without_duplicate_fields(self):
        import json
        base=Path(__file__).resolve().parents[1]/'output/B/pose19+28'
        source=base/'step4/step4.2'
        if not (source/'data/report.json').exists():self.skipTest('Production B reference unavailable')
        record=json.loads((source/'data/report.json').read_text())
        model=FastModel(record['poses'],'B',initialization_report=base/'step3/step3.1/data/report.json')
        baseline,report=load_incumbent(model,source)
        with tempfile.TemporaryDirectory() as directory:
            search=CompactSearch(model,Path(directory));search.search_seconds=0
            search.export_path=lambda n,r:None
            with patch.object(model,'exact',side_effect=AssertionError('Incumbent Boolean repeated')):
                saved=search.validate_and_save([],baseline,report,source)
            self.assertTrue(saved['force_exit_work_passed'])
            self.assertTrue(saved['baseline_retained'])
            self.assertEqual(saved['volume_cm3'],baseline['volume_cm3'])


if __name__=='__main__':unittest.main()
