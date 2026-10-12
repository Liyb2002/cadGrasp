"""World-space force application and persistent projected arrow helpers."""
import mujoco
import numpy as np


def project(scene,point,width=800,height=640):
    a,b=scene.camera
    origin=(a.pos.astype(float)+b.pos.astype(float))/2
    forward=a.forward.astype(float);up=a.up.astype(float);right=np.cross(forward,up)
    v=np.asarray(point)-origin;depth=float(v@forward)
    if depth<=0:raise ValueError('Force annotation behind camera')
    focal=height*a.frustum_near/(a.frustum_top-a.frustum_bottom)
    return np.array([width/2+focal*(v@right)/depth,height/2-focal*(v@up)/depth])


def annotate(ink,start,end):
    color=(242,75,20);delta=end-start;length=np.linalg.norm(delta)
    ink.line([tuple(start),tuple(end)],fill='white',width=9)
    ink.line([tuple(start),tuple(end)],fill=color,width=5)
    if length>2:
        unit=delta/length;side=np.array([-unit[1],unit[0]])
        base=end-unit*min(14.,.5*length)
        ink.polygon([tuple(end),tuple(base+5*side),tuple(base-5*side)],fill=color)
    ink.ellipse((end[0]-4,end[1]-4,end[0]+4,end[1]+4),fill=color,outline='white',width=1)


def apply_load(model, data, body, point, force):
    """Constant force at a fixed world point, including its current moment arm."""
    data.qfrc_applied[:] = 0
    mujoco.mj_applyFT(model, data, force, np.zeros(3), point, body, data.qfrc_applied)

