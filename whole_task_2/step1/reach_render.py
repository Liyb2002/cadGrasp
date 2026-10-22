"""Text-free Step1 figures: the station, direction tasks, every task at the station.

Station views follow robots_config: the KUKA at its fixed base and slanted needle direction,
with the object in six axis-aligned placements resting on one tabletop.
"""
from pathlib import Path
import math
import shutil
import time

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .common import ROOT, sha256, write_json

# One colour per working area, in order of size.
PALETTE = np.array([[.93, .42, .62], [.25, .55, .85], [.40, .78, .85], [.38, .74, .48], [.62, .48, .82], [.95, .66, .22],
                    [.83, .42, .28], [.68, .68, .30], [.98, .86, .36], [.50, .50, .58], [.30, .35, .78], [.80, .60, .70]])
WORKING_POSES_ZOOM = 1.5


def contact(mesh, faces, W):
    """A face to show the head on: in the largest piece, nearest its centre seen from above."""
    inside = np.zeros(len(mesh.faces), bool); inside[faces] = True
    pairs = mesh.face_adjacency[inside[mesh.face_adjacency].all(1)]
    graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(mesh.faces),)*2)
    _, label = connected_components(graph, directed=False)
    pieces = [faces[label[faces] == c] for c in np.unique(label[faces])]
    piece = max(pieces, key=lambda f: mesh.area_faces[f].sum())
    C = mesh.triangles_center[piece] @ W[:3, :3].T + W[:3, 3]
    middle = np.average(C, axis=0, weights=mesh.area_faces[piece])
    return int(piece[np.argmin(np.linalg.norm(C[:, :2]-middle[:2], axis=1)-1e-3*C[:, 2])])


def tiles(images, columns):
    """Arrange views without titles, labels or reserved text space."""
    width, height = images[0].size; rows = math.ceil(len(images)/columns); margin = 12
    canvas = Image.new('RGB', (columns*width+2*margin, rows*height+2*margin), 'white')
    for i, image in enumerate(images):
        x = margin+(i % columns)*width; y = margin+(i//columns)*height
        canvas.paste(image, (x, y))
    return canvas


def picture(scene, W, colors, tools, fit=True):
    if fit: scene.fit(scene.points(W, tools))
    return Image.frombytes('RGB', scene.size, scene.draw(W, colors, tools)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def station(area):
    """Object to world as the direction places it at the work point."""
    W = np.eye(4); W[:3, :3] = np.asarray(area['placement']['rotation_world_object']); W[:3, 3] = area['placement']['translation_world_m']
    return W


def tabletop(scene, setup):
    """The same finite tabletop used by the arm's collision checks."""
    if 'table_bounds_world_m' not in setup:return []
    import trimesh
    mesh=trimesh.creation.box(bounds=np.asarray(setup['table_bounds_world_m']))
    shape=scene.r.Shape(scene.ctx,scene.shader,mesh.vertices,mesh.faces,(.91,.91,.91),roughness=.85)
    return [(shape,np.eye(4))]


def at_work(scene, arm, W, colors, tip, axis, ghosts=(), zoom=1., shapes=(), joints=None):
    """KUKA holding the head on the object; zoomed views center the work and crop the arm."""
    from .reach_video import arm_links
    if joints is None:links, _ = arm_links(arm, tip, axis)
    else:
        from whole_task_2.step5.video.prepare import matrices
        links=matrices(arm.forward(joints,geometry=True)[2])
    tools = [(tip, axis, False), *ghosts]
    points = scene.points(W, tools)
    focus = (points.min(axis=0)+points.max(axis=0))/2 if zoom != 1 else None
    scene.fit(np.vstack([points, links[:, :3, 3]]), 1.06/zoom, target=focus)
    return Image.frombytes('RGB', scene.size, scene.draw(W, colors, tools, links,shapes=shapes)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def working_poses(mesh, report, arrays, output):
    """Render the exclusive task partition in pink at 150% scale."""
    from .reach_video import Scene, kuka, STATION_VIEW
    from whole_task_2.step5.video import raster as r
    tasks = report['tasks']; arm = kuka(report['setup']); images = []
    scene = Scene(mesh, report['setup'], 640, 560, arm=True); scene.view = STATION_VIEW
    shapes=tabletop(scene,report['setup'])
    try:
        for j, task in enumerate(tasks):
            assigned = arrays['face_task'] == j
            colors = np.tile(r.GRAY, (len(mesh.faces), 1)); colors[assigned] = r.PINK
            W = station(task)
            faces = np.flatnonzero(arrays['reach'][:,j]&assigned)
            if not len(faces):
                # Retain a redundant input pose without inventing a working contact.
                images.append(picture(scene, W, colors, []))
                continue
            f = contact(mesh, faces, W)
            tip = mesh.triangles_center[f] @ W[:3, :3].T + W[:3, 3]; axis = W[:3, :3] @ arrays['reach_axis'][f,j]
            images.append(at_work(scene, arm, W, colors, tip, axis, zoom=WORKING_POSES_ZOOM,shapes=shapes,joints=arrays['joints'][f,j,-1]))
    finally:
        scene.close()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    tiles(images, 3 if len(tasks)==6 else min(5, len(tasks))).save(output)
    return output


def coverage_figures(mesh, report, arrays, output):
    """Two matching six-view figures of assigned task faces and their complement."""
    from .reach_video import Scene
    from whole_task_2.step5.video import raster as r
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=np.asarray(arrays['target'],bool)
    membership = arrays['face_task'][:,None] == np.arange(len(report['tasks']))
    counts = membership.sum(axis=1)
    if np.any(counts>1):raise ValueError('Step1.3 tasks overlap')
    covered=target&(counts>0);uncovered=target&~covered
    colors={name:np.tile(r.GRAY,(len(mesh.faces),1)) for name in ('covered','uncovered')}
    colors['covered'][covered]=r.PINK;colors['uncovered'][uncovered]=(.94,.12,.16)
    # Original +Y is the bunny's up direction. Keep it upright for the side views.
    rotation=np.array([[1.,0,0],[0,0,-1.],[0,1.,0]])
    W=np.eye(4);W[:3,:3]=rotation;W[:3,3]=-rotation@mesh.bounds.mean(axis=0)
    views=np.array([[1,.15,.20],[.15,1,.10],[.15,.15,1],
                    [-1,.15,-.20],[.15,-1,-.10],[-.15,.15,-1]],float)
    views/=np.linalg.norm(views,axis=1)[:,None]
    images={name:[] for name in colors}
    scene=Scene(mesh,report['setup'],800,800)
    try:
        for view in views:
            scene.view=rotation@view;scene.fit(scene.points(W,[]),1.10)
            for name in images:images[name].append(picture(scene,W,colors[name],[],fit=False))
    finally:
        scene.close()
    files={}
    for name,panels in images.items():
        path=output/f'{name}.png';tiles(panels,3).save(path)
        files[str(path.relative_to(ROOT))]=sha256(path)
    area=mesh.area_faces;total=float(area[target].sum())
    result=dict(schema='step1_exclusive_task_union_v1',object=report['object'],
        source_arrays_sha256=report['arrays_sha256'],pose_ids=[p['id'] for p in report['tasks']],
        covered_rule='target & (face_task >= 0)',uncovered_rule='target & (face_task < 0)',
        definition='exclusive task assignment shared by Step1.2 and Step1.3; each source face belongs to at most one task',
        task_face_counts=membership.sum(axis=0).tolist(),
        task_area_fractions=[float(area[mask].sum()/total) for mask in membership.T],
        active_task_count=int(membership.any(axis=0).sum()),
        empty_task_ids=[task['id'] for task,mask in zip(report['tasks'],membership.T) if not mask.any()],
        duplicate_face_count=int((counts>1).sum()),duplicate_area_m2=float(area[counts>1].sum()),
        pairwise_overlap_face_counts=(membership.astype(int).T@membership.astype(int)-np.diag(membership.sum(axis=0))).tolist(),
        target_area_m2=total,covered_area_m2=float(area[covered].sum()),uncovered_area_m2=float(area[uncovered].sum()),
        covered_fraction=float(area[covered].sum()/total),uncovered_fraction=float(area[uncovered].sum()/total),
        covered_face_ids=np.flatnonzero(covered).tolist(),uncovered_face_ids=np.flatnonzero(uncovered).tolist(),
        view_directions_object=views.tolist(),image_size_px=[2424,1624],
        covered_color=list(r.PINK),uncovered_color=[.94,.12,.16],files=files)
    write_json(output/'coverage.json',result)
    return result


def publish(mesh, report, root):
    """Replace step1.1-1.3 under root with the plan's setup, figures and per-area views."""
    from .reach_video import Scene
    from whole_task_2.step5.video import raster as r
    start = time.perf_counter(); root = Path(root)
    work = ROOT/'whole_task_2/data'/report['object']/'step1_reach'
    with np.load(work/'plan.npz') as z: arrays = {k: z[k] for k in z.files}
    for stale in [root/'poses', *(root/f'step1.{k}' for k in (1, 2, 3))]:
        if stale.exists(): shutil.rmtree(stale)
    folders = {k: root/f'step1.{k}' for k in (1, 2, 3)}
    for folder in folders.values(): folder.mkdir(parents=True)
    s = report['setup']; tasks = report['tasks']; face_task = arrays['face_task']
    gray = np.asarray(r.GRAY)
    def colors(j):
        c = np.tile(gray, (len(mesh.faces), 1)); c[face_task == j] = PALETTE[j % len(PALETTE)]; return c
    every = np.tile(gray, (len(mesh.faces), 1))
    for j in range(len(tasks)): every[face_task == j] = PALETTE[j % len(PALETTE)]

    # step1.1: the inputs, and the head on one face with its two extreme tilts.
    write_json(folders[1]/'setup.json', dict(object=report['object'], target=s['target'], robot=s['robot'], robot_name=s['robot_name'],
        base_position_world_m=s['base_position_world_m'], working_direction_world=s['working_direction_world'],
        work_window_center_m=s['work_window_center_m'], work_point_m=s['work_point_m'],
        work_point_forward_of_window_m=s['work_point_forward_of_window_m'], work_point_rule=s['work_point_rule'],
        table_height_m=s['table_height_m'],table_bounds_world_m=s['table_bounds_world_m'],pose_rule=s['pose_rule'],
        config_incidence_deg_not_applied=s['config_incidence_deg_not_applied'],
        tool=dict(source=s['tool_source'], needle_length_m=s['needle_length_m'], needle_radius_m=s['needle_radius_m'],
                  head_radius_m=s['head_radius_m']), max_tilt_deg=s['max_tilt_deg'],
        arm_motion=report['reach_definition']['arm_motion'], reach_definition=report['reach_definition']))
    from .reach_video import kuka, STATION_VIEW
    arm = kuka(s); view = STATION_VIEW
    scene = Scene(mesh, s, 1200, 1000, arm=True); scene.view = view
    shapes=tabletop(scene,s)
    try:
        W = station(tasks[0]); f = contact(mesh, np.flatnonzero(arrays['reach'][:,0]), W)
        tip = mesh.triangles_center[f] @ W[:3, :3].T + W[:3, 3]; axis = W[:3, :3] @ arrays['reach_axis'][f,0]
        a = np.asarray(s['working_direction_world']); tilt = np.radians(s['max_tilt_deg'])
        across = np.cross(a, scene.view); across /= np.linalg.norm(across)
        # The two extreme tilts about the slanted nominal direction, across the picture.
        ghosts = [(tip, np.cos(tilt)*a+side*np.sin(tilt)*across, True) for side in (-1, 1)]
        at_work(scene, arm, W, colors(0), tip, axis, ghosts,shapes=shapes,joints=arrays['joints'][f,0,-1]).save(folders[1]/'station.png')
    finally:
        scene.close()

    # step1.2: direction tasks in colour from six sides and alone.
    scene = Scene(mesh, s, 560, 560); views = []; images = []
    try:
        center = np.eye(4); center[:3, 3] = -mesh.bounds.mean(axis=0)
        for d in ([.3, -1, .4], [-.3, 1, .4], [1, .2, .3], [-1, -.2, .3], [.1, -.2, 1], [.1, .2, -1]):
            scene.view = np.asarray(d, float)/np.linalg.norm(d); views.append(picture(scene, center, every, []))
        # Each direction seen from where its needle comes: against the slanted nominal direction.
        v = -np.asarray(s['working_direction_world']) + np.array([.25, 0, 0]); scene.view = v/np.linalg.norm(v)
        for j, task in enumerate(tasks):
            images.append(picture(scene, station(task), colors(j), []))
    finally:
        scene.close()
    tiles(views, 3).save(folders[2]/'areas.png')
    tiles(images, 3 if len(tasks)==6 else min(5, len(tasks))).save(folders[2]/'overview.png')
    # step1.3: zoom in on each task's object and head, allowing the arm to leave the frame.
    working_poses(mesh, report, arrays, folders[3]/'working_poses.png')
    coverage=coverage_figures(mesh,report,arrays,folders[3])
    files = sorted(p for folder in folders.values() for p in folder.iterdir())
    return dict(files={str(p.relative_to(ROOT)): sha256(p) for p in files}, elapsed_s=time.perf_counter()-start,
                working_poses_zoom=WORKING_POSES_ZOOM, working_poses_color='pink',
                working_poses_region='exclusive face_task partition: every pink source face belongs to exactly one input pose; empty poses remain gray',
                coverage_definition=coverage['definition'],covered_fraction=coverage['covered_fraction'],uncovered_fraction=coverage['uncovered_fraction'],
                view='text-free images: one colour per direction task from six sides; overview and working poses per task')
