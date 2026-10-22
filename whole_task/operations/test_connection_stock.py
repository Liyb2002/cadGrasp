import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
from whole_task.step3 import independent_geometry as G
from .state import Layout
from .connection_stock import connecting_stock,positive_parts


class ConnectionStockTests(unittest.TestCase):
    def test_redundant_columns_are_removed_while_whole_contact_skin_and_floor_remain(self):
        body=trimesh.creation.box([.08,.08,.06]);body.apply_translation([0.,0.,.17])
        dense=G.mesh(G.box([-.04,-.04,0.],[.04,.04,.205]))
        model=SimpleNamespace(source=body,policy=dict(plate_thickness_m=.008),records=[{}],
            world_data=lambda record,pose:dict(stock_world=dense),stock_grid=(.035,.008))
        contact=G.box([-.045,-.045,.135],[.045,.045,.205])-G.solid(body)
        floor=G.box([-.07,-.07,0.],[.07,.07,.008])
        axis=np.linspace(-.036,.036,4)
        columns=[G.box([x-.004,y-.004,.004],[x+.004,y+.004,.138]) for x in axis for y in axis]
        solid=contact+floor+G.union(columns)
        q=Layout(np.eye(4)[None],np.eye(4)[None],np.array([[0.,0.,1.]]))
        before=G.mesh(solid)
        after,proof=connecting_stock(model,dict(layout=q,support=before))
        actual=G.solid(after)
        self.assertTrue(proof['applied'])
        self.assertLess(proof['selected_connector_count'],len(columns))
        self.assertLess(after.volume,.9*before.volume)
        self.assertLess(G.volume(contact-actual),1e-18)
        self.assertLess(G.volume(floor-actual),1e-18)
        self.assertLess(G.volume(actual-solid),1e-18)
        self.assertEqual(len(positive_parts(actual)),1)


if __name__=='__main__':unittest.main()
