import unittest
import numpy as np
import manifold3d as md
from whole_task_2.operations.construction import CompactConstructionModel
from whole_task_2.operations.state import Layout
from whole_task_2.step3 import independent_geometry as G


class LocalMaterialTests(unittest.TestCase):
    def test_new_contact_island_is_joined_to_parent_before_component_selection(self):
        def build(keep):
            m=CompactConstructionModel.__new__(CompactConstructionModel)
            m.ids=list('ABC');m.policy=dict(plate_thickness_m=.008,plate_edge_margin_m=.006);m.tolerance=1e-10
            q=Layout(np.tile(np.eye(4),(3,1,1)),np.tile(np.eye(4),(3,1,1)),np.tile([0.,0.,1.],(3,1)))
            obstacle=G.box([.45,-.03,0.],[.55,.03,.1]);rows=[];stocks=[]
            for name,x in zip(m.ids,(1.,.2,0.)):
                stock=G.box([x-.04,-.04,0.],[x+.04,.04,.06]);stocks.append(stock)
                points=np.array([[x+u,v,z] for u in (-.05,.05) for v in (-.05,.05) for z in (0.,.008)])
                rows.append(dict(id=name,material=np.eye(4),stock=stock,work=obstacle,sweep=obstacle,
                    patch_points=points,body=G.mesh(G.box([x-.03,-.03,.06],[x+.03,.03,.08]))))
            # This old path connects through y=.1; the newly generated
            # straight connection at y=0 is cut into two components.
            detour=G.union([G.box([-.003,.097,0.],[1.003,.103,.008]),
                G.box([-.003,-.01,0.],[.003,.11,.008]),G.box([.997,-.01,0.],[1.003,.11,.008])])
            m.prepare=lambda layout:rows
            m.family_before_support=G.union([stocks[1],stocks[2],detour])
            m.family_edit_ids={'A','B'};m.family_refit_mode='local'
            m.family_protected_contact_regions=stocks[2]
            m.family_previous_regions=[G.box([.15,-.05,0.],[.25,.05,.061])]
            m.keep_intermediate_material_components=keep
            return G.solid(m.build(q)['support']),stocks[0]
        old,target=build(False);actual,_=build(True)
        self.assertGreater(G.volume(target-old),.9*G.volume(target))
        self.assertLess(G.volume(target-actual),1e-18)
        self.assertEqual(sum(p.volume()*G.SCALE**3>1e-15 for p in actual.decompose()),1)

    def test_old_connected_detour_and_independent_foot_survive_a_new_cut_mst(self):
        m=CompactConstructionModel.__new__(CompactConstructionModel)
        m.ids=list('ABC');m.policy=dict(plate_thickness_m=.008,plate_edge_margin_m=.006);m.tolerance=1e-10
        q=Layout(np.tile(np.eye(4),(3,1,1)),np.tile(np.eye(4),(3,1,1)),np.tile([0.,0.,1.],(3,1)))
        obstacle=G.box([.45,-.03,0.],[.55,.03,.1]);rows=[];stocks=[]
        for name,x in zip(m.ids,(0.,.2,1.)):
            stock=G.box([x-.04,-.04,0.],[x+.04,.04,.06]);stocks.append(stock)
            points=np.array([[x+u,v,z] for u in (-.05,.05) for v in (-.05,.05) for z in (0.,.008)])
            rows.append(dict(id=name,material=np.eye(4),stock=stock,work=obstacle,sweep=obstacle,
                patch_points=points,body=G.mesh(G.box([x-.03,-.03,.06],[x+.03,.03,.08]))))
        detour=G.union([G.box([-.003,.097,0.],[1.003,.103,.008]),
            G.box([-.003,-.01,0.],[.003,.11,.008]),G.box([.997,-.01,0.],[1.003,.11,.008])])
        feet=[md.Manifold.hull_points(r['patch_points']/G.SCALE) for r in rows]
        parent=G.union([*stocks,*feet,detour])
        self.assertLess(G.volume(parent^obstacle),1e-18)
        m.prepare=lambda layout:rows
        m.family_before_support=parent;m.family_edit_ids={'A','B'};m.family_refit_mode='local'
        m.family_protected_contact_regions=feet[2]
        m.family_previous_regions=[G.box([x-.05,-.05,0.],[x+.05,.05,.061]) for x in (0.,.2)]
        candidate=m.build(q);actual=G.solid(candidate['support'])
        self.assertLess(G.volume(feet[2]-actual),1e-18)
        self.assertLess(G.volume(detour-actual),1e-18)
        self.assertEqual(sum(p.volume()*G.SCALE**3>1e-15 for p in actual.decompose()),1)
        self.assertLess(G.volume(actual^obstacle),m.tolerance)


if __name__=='__main__':unittest.main()
