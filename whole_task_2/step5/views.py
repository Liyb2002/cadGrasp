"""Text-free views of the real final fixture and its actual ground contacts."""
import numpy as np
from whole_task_2.step3.legacy import kernel as K
from whole_task_2.step3.render import COLORS
from whole_task_2.step4.render_mobile import rod
from whole_step4_render import layers_for_pose,sheet
from work_access_render import Panel,render_scene
from .geometry import transform,execution_rows


def render_views(out,model,layout,support):
    rows=execution_rows(model,layout);scenes=[];bounds=[]
    for i,k in enumerate(layout.active):
        row=rows[i];working,body,fixture=[m for m,c in layers_for_pose(model,layout,support,k,display_transform=row['fixture_transform'])]
        working.vertices+=working.vertex_normals*1e-6
        low=np.vstack([fixture.vertices,body.vertices]).min(0);high=np.vstack([fixture.vertices,body.vertices]).max(0)
        floor=K.trimesh.creation.box(extents=[.31,.31,.002]);floor.apply_translation([*(.5*(low+high))[:2],-.0011])
        hull=row['hull'];gold=[rod(hull[j]+[0,0,.0005],hull[(j+1)%len(hull)]+[0,0,.0005],.00022,8) for j in range(len(hull))]
        scene=[(floor,'#edf0f2'),(body,'#bfc5ca'),(fixture,'#2a91d2'),(working,COLORS[i])]+[(g,'#caa353') for g in gold]
        scenes.append(scene);bounds.extend([m.vertices for m,c in scene])
    panel=Panel(None,size=(760,700),points=np.vstack(bounds),elevation=28,azimuth=65)
    sheet([render_scene(panel,s) for s in scenes],out/'execution_overview.png')
    panel=Panel(None,size=(1000,900),points=support.vertices,elevation=24,azimuth=55)
    render_scene(panel,[(support,'#2a91d2')]).save(out/'fixture.png')
