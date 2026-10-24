"""Two needle contacts on the processable region's 3D boundary.

The whole head and needle move together. The gold cone shows the permitted
orientation about the nominal axis; pink is the angle/3D-box intersection. The camera frames the
object and tool poses, leaving most of the real arm outside the image.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ROOT, load_shape, sha256, write_json
from .reach import frame
from .arm import Check
from .reach_video import Scene, tool_frame, GOLD, kuka
from .workspace import box_edges
from .task_boundary import boundary_loops,candidate_region

VIEW = np.array([1., .65, .55]); VIEW /= np.linalg.norm(VIEW)
CONE_OPACITY = .50
POSE_INDEX = 0
OUTLINE_COLOR = (.28, .07, .17)
OUTLINE_WIDTH_PX = 2.
OUTLINE_OFFSET_M = .00005
WORKSPACE_COLOR = (.08, .48, .62)


def task_outline(mesh, mask, W):
    """Only edges separating retained task faces from the rest of the surface."""
    adjacent = mesh.face_adjacency
    edges = mesh.face_adjacency_edges[mask[adjacent[:, 0]] != mask[adjacent[:, 1]]]
    if not len(edges):return []
    vertices = mesh.vertices+OUTLINE_OFFSET_M*mesh.vertex_normals
    points = vertices[edges].reshape(-1, 3)@W[:3, :3].T+W[:3, 3]
    return [(points, OUTLINE_COLOR)]


def placement(report, mesh):
    """Pose1's saved tabletop placement, translated with the whole station for rendering."""
    record = report['areas'][POSE_INDEX]
    if record['id'] != '+X':raise ValueError('Pose1 must be the +X object placement')
    actual = np.eye(4)
    actual[:3, :3] = record['placement']['rotation_world_object']
    actual[:3, 3] = record['placement']['translation_world_m']
    shift = np.asarray(report['setup']['work_point_m'])
    W = actual.copy(); W[:3, 3] -= shift
    bottom = (mesh.vertices@actual[:3, :3].T+actual[:3, 3])[:, 2].min()
    if not np.isclose(bottom, report['setup']['table_height_m'], atol=1e-9):
        raise ValueError('Pose1 does not rest on the saved table')
    return actual, W, shift


def tabletop(scene, setup, shift):
    """Render the collision-check tabletop in the same translated coordinates."""
    import trimesh
    bounds = np.asarray(setup['table_bounds_world_m'])-shift
    mesh = trimesh.creation.box(bounds=bounds)
    shape = scene.r.Shape(scene.ctx, scene.shader, mesh.vertices, mesh.faces,
                          (.91, .91, .91), roughness=.85)
    return [(shape, np.eye(4))]


def combine(meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(v); faces.append(f+offset); offset += len(v)
    return np.vstack(vertices), np.vstack(faces)


def tube(start, end, radius, end_radius=None):
    from whole_task_2.step5.video.raster import tool_mesh
    start, end = np.asarray(start), np.asarray(end); length = np.linalg.norm(end-start)
    v, f = tool_mesh(0., length, radius, radius if end_radius is None else end_radius)
    W = tool_frame(start, (start-end)/length)
    return v@W[:3, :3].T+W[:3, 3], f


def tubes(segments, radius):
    return combine([tube(a, b, radius) for a, b in segments if np.linalg.norm(np.asarray(b)-a) > 1e-9])


def workspace_box(scene, setup, shift=(0.,0.,0.)):
    """Draw twelve true 3D edges, with the object occluding the rear edges."""
    edges=box_edges(setup['surface_workspace']['bounds_world_m'],shift)
    points=np.asarray(edges).reshape(-1,3)
    scene.workspace_wireframe=(points,WORKSPACE_COLOR)
    return [],points


def cone(root, axis, length, angle, ribs=8):
    """Forward cone: apex at the needle root, rim one fixed needle length away."""
    u, v = frame(axis); beta = np.radians(angle); t = np.linspace(0, 2*np.pi, 97)
    rim = root+length*(np.cos(beta)*axis+np.sin(beta)*(np.cos(t)[:, None]*u+np.sin(t)[:, None]*v))
    return list(zip(rim[:-1], rim[1:]))+[(root, rim[k]) for k in range(0, 96, 96//ribs)]


def dashed(segments, on=.004, off=.003):
    """World-length dashes, preserving dash phase around the segmented cone rim."""
    result=[];phase=0.;last=None;period=on+off
    for start,end in segments:
        start,end=np.asarray(start),np.asarray(end)
        if last is None or not np.allclose(start,last,atol=1e-10,rtol=0):phase=0.
        length=np.linalg.norm(end-start);direction=(end-start)/length;t=0.
        while t<length-1e-12:
            lit=phase<on;step=min(length-t,(on if lit else period)-phase)
            if lit and step>1e-12:result.append((start+t*direction,start+(t+step)*direction))
            t+=step;phase=(phase+step)%period
        last=end
    return result


class Opacity:
    """Reusable GPU group-alpha composition for the figure and its animation."""
    def __init__(self,scene):
        self.scene=scene;ctx=scene.ctx;r=scene.r
        self.textures=[ctx.texture(scene.size,3) for _ in range(4)]
        self.program=ctx.program(vertex_shader=r.SCREEN_VERTEX,fragment_shader='''#version 330
in vec2 uv;out vec4 pixel;
uniform sampler2D base_image;uniform sampler2D tool_image;uniform sampler2D cone_image;uniform sampler2D combined_image;
uniform float opacity;uniform float cone_opacity;
void main(){vec3 base=mix(texture(base_image,uv).rgb,texture(cone_image,uv).rgb,cone_opacity);
vec3 tool=mix(texture(tool_image,uv).rgb,texture(combined_image,uv).rgb,cone_opacity);
pixel=vec4(mix(base,tool,opacity),1);}
''')
        self.vao=ctx.vertex_array(self.program,[]);self.output=ctx.simple_framebuffer(scene.size,components=3)

    def draw(self,W,colors,shapes,links,ghost,opacity=.30,cones=None,cone_opacity=CONE_OPACITY,segments=()):
        scene=self.scene;ctx=scene.ctx
        if segments:ctx.line_width=OUTLINE_WIDTH_PX*scene.studio.size[0]/scene.size[0]
        base=scene.draw(W,colors,[],links=links,shapes=shapes,segments=segments)
        full=scene.draw(W,colors,[],links=links,shapes=shapes+ghost,segments=segments) if ghost else base
        with_cones=scene.draw(W,colors,[],links=links,shapes=shapes+cones,segments=segments) if cones else base
        combined=scene.draw(W,colors,[],links=links,shapes=shapes+ghost+cones,segments=segments) if ghost and cones else (with_cones if cones else full)
        units={}
        for name,data in zip(('base_image','tool_image','cone_image','combined_image'),(base,full,with_cones,combined)):
            key=id(data)
            if key not in units:
                unit=len(units);self.textures[unit].write(data);self.textures[unit].use(unit);units[key]=unit
            self.program[name].value=units[key]
        self.program['opacity'].value=opacity
        self.program['cone_opacity'].value=cone_opacity if cones else 0.
        self.output.use();ctx.viewport=(0,0,*scene.size);ctx.disable(scene.gl.DEPTH_TEST|scene.gl.BLEND)
        self.vao.render(scene.gl.TRIANGLES,vertices=3)
        return self.output.read(components=3,alignment=1)

    def close(self):
        for item in [self.vao,self.program,self.output,*self.textures]:item.release()


def picture(scene, W, colors, shapes, points, links, ghost, opacity,cones=None,segments=()):
    """Render the tool as one group at exact opacity, retaining scene occlusion."""
    scene.fit(np.asarray(points).reshape(-1, 3), 1.12)
    compositor=Opacity(scene)
    try:data=compositor.draw(W,colors,shapes,links,ghost,opacity,cones,segments=segments)
    finally:compositor.close()
    return Image.frombytes('RGB',scene.size,data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def select_contacts(mesh, arrays, j, W, length=.1):
    """Two certified contacts on the actual 3D task boundary, high and low."""
    can = candidate_region(arrays,j); d = arrays['area_axes'][j]
    tilt = np.degrees(np.arccos(np.clip(arrays['reach_axis'][:, j]@d, -1, 1)))
    loop=boundary_loops(mesh,can)[0];ids=loop['face_ids']
    boundary_points=mesh.vertices[loop['vertex_ids'][:-1]]
    C = boundary_points@W[:3, :3].T+W[:3, 3]
    axes = arrays['reach_axis'][ids, j]@W[:3, :3].T
    right = np.cross([0,0,1],VIEW); right /= np.linalg.norm(right); up = np.cross(VIEW,right)
    head_centers = C-(.5*.07+length)*axes
    nominal = tilt[ids] < 5.
    if not nominal.any(): nominal[:] = True
    middle = np.average(head_centers[nominal],axis=0,weights=mesh.area_faces[ids[nominal]])
    high = np.quantile(head_centers[nominal]@up,.93)
    cost = np.abs(head_centers@up-high)+.25*np.abs((head_centers-middle)@right)
    cost[~nominal] = np.inf
    chosen = [int(np.argmin(cost))]
    # Display landmark only: the bunny is Y-up, its hindquarters towards +X.
    aim=mesh.bounds[0]+mesh.extents*np.array([.88,.34,.48])
    score=np.linalg.norm(boundary_points-aim,axis=1)
    valid=(tilt[ids]>=9.)&(tilt[ids]<=20.);valid[chosen[0]]=False
    if not valid.any():valid=np.arange(len(ids))!=chosen[0]
    score[~valid]=np.inf;chosen.append(int(np.argmin(score)))
    return [(int(ids[k]), C[k], W[:3, :3]@arrays['reach_axis'][ids[k], j]) for k in chosen]


def render(object_name='B', output=ROOT/'whole_task_2/idea/find_tasks.png', height=1400):
    output = Path(output); work = ROOT/'whole_task_2/data'/object_name/'step1_reach'
    report = json.loads((work/'plan.json').read_text()); s = report['setup']
    with np.load(work/'plan.npz') as z: arrays = {k: z[k] for k in z.files}
    mesh = load_shape(ROOT/'objects'/object_name/'mesh.stl')
    a = np.asarray(s['working_direction_world'], float)
    L, beta = s['needle_length_m'], s['max_tilt_deg']
    j = POSE_INDEX
    actual, W, world_center = placement(report, mesh)
    T = mesh.vertices@W[:3, :3].T+W[:3, 3]
    contacts = select_contacts(mesh, arrays, j, W, L)
    checker = Check(mesh, s)
    for f, _, _ in contacts:
        if not checker.replay(f, arrays['reach_axis'][f,j], actual, arrays['joints'][f,j]):
            raise ValueError(f'Pose1 contact {f} no longer matches its certified witness')
    arm = kuka(s); black_face,black_tip,black_axis = contacts[0]
    q=checker.solve(black_tip+world_center,black_axis,actual,arrays['joints'][black_face,j,3])
    if q is None or q is False:raise ValueError('Arm cannot hold the boundary contact')
    tcp, _, transforms = arm.forward(q,geometry=True)
    links = np.repeat(np.eye(4)[None],len(transforms),axis=0)
    for M,(rotation,point) in zip(links,transforms):M[:3,:3]=rotation;M[:3,3]=point-world_center
    assert np.linalg.norm(tcp-world_center-contacts[0][1]) < 1e-4
    scene = Scene(mesh, s, int(height*1.1), height, arm=True); r = scene.r
    scene.view = VIEW.copy()
    region=candidate_region(arrays,j)
    colors = np.tile(r.GRAY, (len(mesh.faces), 1)); colors[region] = r.PINK
    outline = task_outline(mesh, region, W)
    def shape(geometry, color, metal=.1):
        return r.Shape(scene.ctx, scene.shader, *geometry, color, metallic=metal, roughness=.5)
    try:
        box,box_points=workspace_box(scene,s,world_center)
        shapes = tabletop(scene, s, world_center)+box; ghost = []; cones = []; fit = list(T)+list(box_points)
        for k, (f, tip, d) in enumerate(contacts):
            parts = [(part,tool_frame(tip,d)) for part in scene.tools[False]]
            if k==0:shapes+=parts
            else:ghost+=parts
            fit += [tip-(L+.07)*d,tip]
            wire=cone(tip-L*d,a,L,beta,8)
            cones.append((shape(tubes(dashed(wire),.00065),GOLD),np.eye(4)))
            fit += [p for segment in wire for p in segment]
        result = picture(scene,W,colors,shapes,fit,links,ghost,.30,cones,segments=outline)
        vertices = np.vstack([v@M[:3,:3].T+M[:3,3] for v,M in zip(arm.collisions,links)])
        clip = np.c_[vertices,np.ones(len(vertices))]@scene.vp.T
        visible = (np.abs(clip[:,:3]/clip[:,3:]) <= 1).all(1)
        arm_cropped_fraction = float(1-visible.mean())
    finally:
        scene.close()
    output.parent.mkdir(parents=True, exist_ok=True); result.save(output)
    record = dict(output=str(output), sha256=sha256(output), plan_sha256=sha256(work/'plan.json'),
                  source_sha256=sha256(Path(__file__)),
                  plan_arrays_sha256=sha256(work/'plan.npz'),
                  task=report['areas'][j]['id'],pose_number=1,pose_index=j,
                  placement=report['areas'][j]['placement'],table_height_m=s['table_height_m'],
                  width=result.width,height=result.height,needle_length_m=L, max_tilt_deg=beta,
                  layout='black head and transparent head at two actual 3D task-boundary contacts',
                  head_opacities=[1.,.30],ghost_location='task boundary',cone_styles=['dashed','dashed'],cone_opacities=[CONE_OPACITY,CONE_OPACITY],
                  cone='one direction cone at each moving head root, opening along the nominal needle axis',
                  motion='entire heads and needles translate and rotate together within max_tilt_deg',
                  arm='only the black head uses the real arm; arm excluded from camera fitting',
                  arm_cropped_vertex_fraction=arm_cropped_fraction,
                  black_head_joints=q.tolist(),
                  surface='pose1 candidate region: angle reachable surface intersect 3D work box',
                  candidate_face_count=int(region.sum()),
                  candidate_area_fraction_of_target=float(mesh.area_faces[region].sum()/mesh.area_faces[arrays['target']].sum()),
                  downstream_tasks='exclusive face_task partition unchanged',
                  surface_workspace=s['surface_workspace'],workspace_box_color=list(WORKSPACE_COLOR),
                  workspace_box_edges=12,surface_contact_points_inside_box=True,
                  workspace_rendering='3D inward surface workspace; depth-tested edges, rear edges occluded by the object; reference object AABB hidden',
                  task_outline=dict(color_rgb=OUTLINE_COLOR,width_px=OUTLINE_WIDTH_PX,
                                    boundary_edges=sum(len(points)//2 for points,_ in outline),
                                    vertex_normal_offset_m=OUTLINE_OFFSET_M,visible_surfaces_only=True),
                  task_area_fraction_of_target=report['areas'][j]['area_fraction_of_target'],
                  contact_face_ids=[f for f, _, _ in contacts],
                  contacts_in_displayed_certified_candidate_region=True,
                  contact_axes_world=[d.tolist() for _, _, d in contacts],
                  axis_provenance=['stored pose1 Step1 arm witness']*len(contacts),
                  tilts_deg=np.degrees(np.arccos(np.clip(np.asarray([d for _,_,d in contacts])@a,-1,1))).tolist())
    write_json(work/'find_tasks_figure.json', record)
    return output


if __name__ == '__main__':
    print(render())
