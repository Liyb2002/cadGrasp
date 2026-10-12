"""Measure where the simulated fingers actually touch after each pickup."""
import mujoco
import numpy as np
import grasp as G


def measured_grasps(model, history, events):
    data=mujoco.MjData(model)
    rows=[]
    for event in events:
        if event['event']!='new_grasp':continue
        sample_time=event.get('contact_sample_time_s',event['end_time_s'])
        frame=min(len(history)-1,int(np.floor(sample_time/.033))-1)
        q,pos,quat=history[frame]
        data.qpos[:]=q;data.mocap_pos[:]=pos;data.mocap_quat[:]=quat
        mujoco.mj_forward(model,data)
        groups={'left_finger':[],'right_finger':[]}
        for contact in data.contact:
            names=[model.geom(int(g)).name for g in (contact.geom1,contact.geom2)]
            if contact.dist>.0002 or not any(n.startswith('object_') for n in names):continue
            for geom in (contact.geom1,contact.geom2):
                body=model.body(model.geom_bodyid[geom]).name
                if body in groups:groups[body].append(contact.pos.copy())
        if not all(groups.values()):
            raise ValueError(f'No bilateral measured contact at pickup frame {frame}')
        inverse=np.linalg.inv(G.transform(data,model.body('object').id))
        hand=inverse@G.transform(data,model.body('hand').id)
        points=np.array([np.mean(v,axis=0) for v in groups.values()])@inverse[:3,:3].T+inverse[:3,3]
        rows.append(dict(contact_points_object_m=points.tolist(),
            approach_object=hand[:3,2].tolist(),closing_axis_object=hand[:3,1].tolist(),
            frame=frame,time_s=(frame+1)*.033,
            contact_measurement='Mean geometric contact position for each finger in the saved free-object state'))
    return rows
