"""Finite rigid-motion tree; every accepted edge has a continuous solid sweep.

Fixed workpiece and floor. Saved vertices remain in world metres. No clearance,
contact deletion, object lifting, or animation-only acceptance is introduced.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from step5_connect_support import rigid_path as P


def transform(points, origin, pose):
    pose=np.asarray(pose)
    return (np.asarray(points)-origin)@pose[:3,:3].T+pose[:3,3]+origin


def moved_parts(parts, origin, pose):
    result=[]
    for part in parts:
        p=part.copy();p.vertices=transform(p.vertices,origin,pose);result.append(p)
    return result


def edge(scene, parts, origin, first, last, budget=160):
    moved=moved_parts(parts,origin,first)
    relative=last[:3,:3]@first[:3,:3].T
    motion=np.r_[(last[:3,3]-first[:3,3])/scene.scale,Rotation.from_matrix(relative).as_rotvec()]
    center=origin+first[:3,3]
    state=dict(remaining=budget,checked=0)
    return P.segment(scene,moved,center,motion,0.,1.,state),state['checked']


def interpolate(first,last,t):
    out=np.eye(4)
    rot=Rotation.from_matrix(last[:3,:3]@first[:3,:3].T).as_rotvec()
    out[:3,:3]=Rotation.from_rotvec(rot*t).as_matrix()@first[:3,:3]
    out[:3,3]=(1-t)*first[:3,3]+t*last[:3,3]
    return out


def search(scene, contacts, parts, origin, iterations=1800):
    vertices=np.vstack([p.vertices for p in parts]); floor=vertices[np.abs(vertices[:,1])<=scene.scale*1e-10]
    motions,local=P.local_motions(scene.mesh,contacts,floor,origin,scene.scale)
    report=dict(passed=False,status='no_piecewise_path_in_finite_search',local_motion=local,
        method='seeded rigid-motion tree with continuously checked SE3 edges',
        random_seed=417,iteration_budget=iterations,global_impossibility_claimed=False,
        path_kind='piecewise_rigid',origin_m=np.asarray(origin).tolist(),length_scale_m=scene.scale,
        continuous_sweep_verified=False,rotation_allowed=True)
    if not motions:
        report.update(status='no_nonzero_first_order_exit_found',tree_nodes=1,iterations=0);return report
    rng=np.random.default_rng(417);nodes=[np.eye(4)];parents=[-1];checked=0
    def accept(parent,target):
        nonlocal checked
        moved=transform(vertices,origin,target)
        if moved[:,1].min() < -scene.scale*1e-10:return False
        if np.max(np.abs(target[:3,3]))>3*scene.scale:return False
        if any(not scene.clear(p,ground=True) for p in moved_parts(parts,origin,target)):return False
        okay,count=edge(scene,parts,origin,nodes[parent],target);checked+=count
        if okay:nodes.append(target);parents.append(parent)
        return okay
    for motion in motions:
        for amount in (.001,.005,.02,.06):
            pose=np.eye(4);pose[:3,:3]=Rotation.from_rotvec(amount*motion[3:]).as_matrix();pose[:3,3]=amount*scene.scale*motion[:3]
            accept(0,pose)
    if len(nodes)==1:
        report.update(status='no_continuous_first_step_in_menu',tree_nodes=1,iterations=0,checked_intervals=checked);return report
    directions=np.vstack([np.eye(3),-np.eye(3)])
    for iteration in range(iterations):
        parent=int(rng.integers(len(nodes)))
        if rng.random()<.7:
            motion=np.asarray(motions[int(rng.integers(len(motions)))])*rng.choice([-1,1])
            amount=float(rng.uniform(.005,.04));delta=amount*motion
        else:
            delta=rng.normal(size=6)*np.r_[np.full(3,.018),np.full(3,.1)]
        pose=nodes[parent].copy();pose[:3,:3]=Rotation.from_rotvec(delta[3:]).as_matrix()@pose[:3,:3];pose[:3,3]+=delta[:3]*scene.scale
        if not accept(parent,pose):continue
        end=len(nodes)-1
        for direction in directions:
            goal=pose.copy();goal[:3,3]+=direction*scene.scale*2.5
            points=transform(vertices,origin,goal)
            if points[:,1].min() < -scene.scale*1e-10 or not P.separated(scene,points):continue
            if accept(end,goal):
                ids=[len(nodes)-1]
                while parents[ids[-1]]>=0:ids.append(parents[ids[-1]])
                path=[nodes[i] for i in ids[::-1]]
                # Shortcut only with the same continuous solid checker.
                simplified=[path[0]];i=0
                while i<len(path)-1:
                    j=len(path)-1
                    while j>i+1:
                        okay,count=edge(scene,parts,origin,path[i],path[j]);checked+=count
                        if okay:break
                        j-=1
                    simplified.append(path[j]);i=j
                report.update(passed=True,status='piecewise_rigid_trajectory_verified',
                    continuous_sweep_verified=True,poses=[p.tolist() for p in simplified],
                    tree_nodes=len(nodes),iterations=iteration+1,checked_intervals=checked)
                return report
    report.update(tree_nodes=len(nodes),iterations=iterations,checked_intervals=checked)
    return report


def replay(scene,parts,report):
    assert report['passed'] and report['continuous_sweep_verified']
    origin=np.asarray(report['origin_m']);poses=np.asarray(report['poses'])
    np.testing.assert_allclose(poses[0],np.eye(4),atol=1e-12,rtol=0)
    for pose in poses:
        np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-12,rtol=0)
        assert np.linalg.det(pose[:3,:3])>0
    for first,last in zip(poses[:-1],poses[1:]):assert edge(scene,parts,origin,first,last,budget=1000)[0]
    points=transform(np.vstack([p.vertices for p in parts]),origin,poses[-1])
    assert P.separated(scene,points)
    return True
