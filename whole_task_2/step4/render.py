"""Actual saved support installations, viewed from the worker side."""
import numpy as np
from whole_task_2.step3.render import COLORS
from whole_step4_render import layers_for_pose, sheet
from work_access_render import Panel, render_scene


def render_support(model,layout,support,path):
    tiles=[]
    for i,k in enumerate(layout.active):
        layers=layers_for_pose(model,layout,support,k)
        work,body,fixture=[row[0]for row in layers]
        work.vertices+=1e-6*work.vertex_normals  # Display-only depth separation.
        points=np.vstack([body.vertices,fixture.vertices])
        panel=Panel(None,size=(760,700),points=points,elevation=38,azimuth=80)
        tiles.append(render_scene(panel,[(body,'#b8b8b8'),(fixture,'#2a91d2'),(work,COLORS[i%len(COLORS)])]))
    sheet(tiles,path)
