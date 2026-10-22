"""Four centered execution views; actual saved fixture, tool and work faces."""
import argparse
import json
from pathlib import Path
import numpy as np

from .mobile import MobileTaskModel
from whole_task_2.step1.common import DEFAULT_OUTPUT
from whole_task_2.step3.legacy import kernel as K
from whole_task_2.step3.render import COLORS
from whole_search.model import Layout
from whole_step4_render import layers_for_pose, sheet
from work_access_render import Panel, render_scene


def rod(start, end, radius, sections=16):
    return K.trimesh.creation.cylinder(radius=radius, segment=np.array([start,end]),sections=sections)


def render(folder):
    folder=Path(folder)
    report=json.loads((folder/'report.json').read_text())
    model=MobileTaskModel(report['poses'],report['object'],report['output_root'])
    second=folder/'step4.2'
    with np.load(second/'layout.npz')as data:layout=Layout.from_arrays(data)
    support=K.trimesh.load(second/'complete_support.obj',force='mesh',process=False)
    scenes=[];points=[]
    for i,k in enumerate(layout.active):
        execution=model.execution(layout,k)
        T=np.array(execution['world_from_support'])
        working,body,fixture=[mesh for mesh,_ in layers_for_pose(model,layout,support,k,display_transform=T)]
        working.vertices+=1e-6*working.vertex_normals
        target=np.array(execution['work_window_center_m'])
        axis=model.tasks[k].domain.axis
        flange=target-model.tool_length*axis
        needle=rod(target,flange,.0012)
        barrel=rod(flange,flange-.035*axis,.006)
        # Reference window boundary is a drawing only; the saved finite work
        # exclusion, not these lines, supplies the collision certification.
        apex=np.array(execution['head_world_m'])
        u=np.array([1.,0.,0.]);v=np.cross(axis,u)
        radius=.12*np.tan(np.radians(model.station['workspace']['spatial_cone']['usable_half_angle_deg']))
        ring=np.array([target+radius*(np.cos(a)*u+np.sin(a)*v) for a in np.linspace(0,2*np.pi,65)])
        gold=[rod(ring[j],ring[j+1],.00035,8) for j in range(64)]
        gold += [rod(apex,ring[j],.00035,8) for j in [0,16,32,48]]
        scene=[(body,'#c4c8cd'),(fixture,'#5c849e'),(working,COLORS[i]),
               (needle,'#505862'),(barrel,'#333b46')]+[(g,'#d0a24f')for g in gold]
        bounds=np.vstack([m.vertices for m,_ in scene])
        points.append(bounds)
        scenes.append(scene)
    all_points=np.vstack(points)
    lower,upper=all_points.min(0),all_points.max(0)
    center=(lower+upper)/2
    floor=K.trimesh.creation.box(extents=[max(.4,upper[0]-lower[0]+.08),max(.4,upper[1]-lower[1]+.08),.005])
    floor.apply_translation([center[0],center[1],-.0025])
    # Every panel uses identical camera and scale in the same world coordinates.
    panel=Panel(None,size=(760,700),points=np.vstack([all_points,floor.vertices]),elevation=32,azimuth=65)
    tiles=[render_scene(panel,[(floor,'#e9e8e4')]+scene)for scene in scenes]
    sheet(tiles,folder/'execution_overview.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',nargs='?',type=Path,default=DEFAULT_OUTPUT/'B/step4/movable_station')
    render(p.parse_args().folder)
