"""Two text-free counterexamples using the saved six-pose proof geometry.

This presentation does not modify or run the production optimizer. An empty
fixture is lifted during its illustrative transfer; the ground obstruction
belongs to the saved final installed state. The work obstruction uses an
original visible processing ray, not an invented tool direction.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/cadgrasp-failure-video-numba')
FOLDER = Path(__file__).resolve().parent
PROOF = FOLDER.parent
COOPT = PROOF.parent
ROOT = COOPT.parents[1]
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'vis_func'), str(COOPT/'step5.2')]
import _bootstrap
import numpy as np
import trimesh
import manifold3d as md
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from shapely.geometry import Polygon
from trimesh.ray.ray_pyembree import RayMeshIntersector
from deterministic_base import solid, unpack, volume, moved, SCALE
from work_access import WorkAccess
import step51_render as V

LABEL = 'pose1+2+3+4+5+6'
ORANGE = '#DF9B3E'
RED = '#DC344E'
GREEN = '#718B67'


def transform(points, matrix):
    return np.asarray(points)@matrix[:3, :3].T+matrix[:3, 3]


def posed(mesh, matrix):
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def smooth(value):
    value = np.clip(value, 0., 1.)
    return float(value*value*(3.-2.*value))


def line(points, radius=.00045, closed=True):
    points = np.asarray(points)
    pairs = zip(points, np.roll(points, -1, axis=0)) if closed else zip(points[:-1], points[1:])
    meshes = [trimesh.creation.cylinder(radius=radius, segment=np.array([a, b]), sections=8)
              for a, b in pairs if np.linalg.norm(a-b)>1e-10]
    return trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()


def piece(mesh, color, opacity=1.):
    return V.piece(mesh, color, opacity) if len(mesh.faces) and opacity>0. else None


class SavedInputs:
    def __init__(self):
        self.hashes = {}
        self.base = PROOF/'output/B'/LABEL
        self.demo = self.read_json(PROOF/'demo/report.json')
        self.fixture = self.mesh(PROOF/'demo/saved_step4_support.obj')
        self.ground_base = self.mesh(PROOF/'demo/illustrative_base_common.obj')
        self.body = self.mesh(ROOT/'objects/B/mesh.stl')
        selection = self.read_json(self.base/'data/selection.json')['cases'][0]
        self.experiment = selection['experiment']
        stage = self.base/'step4/step4.2'/self.experiment
        report = self.read_json(stage/'report.json')
        if not report['passed']:
            raise ValueError('Expected the historical force-passed counterexample')
        self.poses = report['poses']
        self.layout = self.arrays(stage/'layout.npz')
        self.rows = []
        for k, pose in enumerate(self.poses):
            data = self.arrays(self.base/f'step5/step5.1/data/{pose}.npz')
            setup = self.arrays(ROOT/f'objects/B/poses/{pose}/setup.npz')
            ids = setup['work_faces']
            if ids.dtype == bool:
                ids = np.flatnonzero(ids)
            T, U = data['T_world_fixture'], data['T_world_object']
            np.testing.assert_allclose(T@self.layout['placements'][k], U, atol=1e-12, rtol=0)
            self.rows.append(dict(pose=pose, T=T, U=U, hull=data['floor_demands_world_m'][data['hull_indices']],
                                  setup=setup, work_ids=ids.astype(int), index=k))

    def protect(self, path):
        path = Path(path)
        self.hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(self, path):
        return json.loads(self.protect(path).read_text())

    def arrays(self, path):
        with np.load(self.protect(path), allow_pickle=False) as saved:
            return {k: saved[k].copy() for k in saved.files}

    def mesh(self, path):
        return trimesh.load(self.protect(path), force='mesh', process=False)

    def verify(self):
        for relative, digest in self.hashes.items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Presentation input changed: {relative}')


def split_ground(value, matrix):
    above = value.trim_by_plane(tuple(matrix[2, :3]), float(-matrix[2, 3]/SCALE))
    return posed(unpack(above), matrix), posed(unpack(value-above), matrix), volume(value-above)*1e6


class GroundScene:
    def __init__(self, inputs):
        self.inputs = inputs
        proof = inputs.demo['ground_certificate']
        self.first, self.last = [inputs.rows[proof[key]] for key in ['task_index', 'other_index']]
        self.base = solid(inputs.ground_base)
        self.fixture = solid(inputs.fixture)
        self.common = np.vstack([inputs.fixture.vertices, inputs.ground_base.vertices])
        self.rotation = Slerp([0., 1.], Rotation.from_matrix(np.array([self.first['T'][:3, :3], self.last['T'][:3, :3]])))
        matrices = [self.interpolate(x) for x in np.linspace(0., 1., 61)]
        self.lift = max(.07, .02-min(transform(self.common, T)[:, 2].min() for T in matrices))
        endpoint_points = [np.vstack([transform(self.common, r['T']), transform(inputs.body.vertices, r['U'])])
                           for r in [self.first, self.last]]
        self.span = max(V.camera_for(p)[2] for p in endpoint_points)*1.12
        self.focus = [V.camera_for(p)[0] for p in endpoint_points]
        floor_points = np.vstack(endpoint_points)
        low, high = floor_points[:, :2].min(0)-.06, floor_points[:, :2].max(0)+.06
        self.floor = trimesh.Trimesh([[*low, 0.], [high[0], low[1], 0.], [*high, 0.], [low[0], high[1], 0.]],
                                     [[0, 1, 2], [0, 2, 3]], process=False)
        self.floor_border = line(self.floor.vertices, radius=.0003)
        self.ground_checks = []
        for row in [self.first, self.last]:
            _, _, below = split_ground(self.base, row['T'])
            _, _, support_below = split_ground(self.fixture, row['T'])
            self.ground_checks.append(dict(pose=row['pose'], base_minimum_z_mm=float(transform(inputs.ground_base.vertices, row['T'])[:, 2].min()*1000),
                                           base_below_ground_cm3=below,
                                           nominal_support_minimum_z_mm=float(transform(inputs.fixture.vertices, row['T'])[:, 2].min()*1000),
                                           nominal_support_below_ground_cm3=support_below))
        # The historical OBJ used eight decimal places. Keep its exact saved
        # vertices; tolerate only its nanometre boundary rounding here.
        assert self.ground_checks[0]['base_minimum_z_mm']>-.0001
        assert self.ground_checks[0]['base_below_ground_cm3']<1e-5
        assert self.ground_checks[1]['base_below_ground_cm3']>1e-4
        np.testing.assert_allclose(self.ground_checks[1]['base_minimum_z_mm'], inputs.demo['illustrated_base_after']['base_minimum_z_mm'], atol=1e-5)
        self.metadata = dict(failure='saved final installed base penetrates the ground',
            starting_pose=self.first['pose'], rejected_pose=self.last['pose'], installed_states=self.ground_checks,
            ground_certificate_gap_mm=proof['separation_gap_m']*1000,
            base_source='saved illustrative_base_common.obj, rigidly attached to saved support',
            object_removed_before_empty_fixture_transfer=True, illustrative_transfer_lift_m=self.lift,
            transfer_path_is_not_a_design_constraint=True, rejected_state_is_counterfactual=True)

    def interpolate(self, fraction):
        T = np.eye(4)
        T[:3, :3] = self.rotation([fraction]).as_matrix()[0]
        T[:3, 3] = (1.-fraction)*self.first['T'][:3, 3]+fraction*self.last['T'][:3, 3]
        return T

    def frame(self, time):
        fraction = smooth((time-3.)/2.)
        lift = self.lift*smooth((time-2.4)/.6)*(1.-smooth((time-5.)/1.))
        T = self.interpolate(fraction)
        T[2, 3] += lift
        focus = (1.-fraction)*self.focus[0]+fraction*self.focus[1]+np.array([0., 0., lift*.75])
        angle = -np.pi/4 + .12*smooth((time-6.5)/3.5)
        direction = np.array([np.cos(angle), np.sin(angle), .7])
        direction /= np.linalg.norm(direction)
        right = np.array([-np.sin(angle), np.cos(angle), 0.])
        camera = (focus, np.array([right, np.cross(direction, right), direction]), self.span)
        above, below, below_volume = split_ground(self.base, T)
        if below_volume<1e-5:
            above, below = posed(self.inputs.ground_base, T), trimesh.Trimesh()
        support_above, support_below, _ = split_ground(self.fixture, T)
        parts = [piece(self.floor, '#E6E9ED', .3), piece(self.floor_border, '#AEB5BD', .6),
                 piece(support_above, V.BLUE, .82), piece(support_below, '#E98B9A', .9),
                 piece(above, ORANGE), piece(below, RED)]
        if time<2.4:
            alpha = 1.-smooth((time-1.9)/.5)
            row = self.first
        else:
            alpha = smooth((time-6.)/.7)
            row = self.last
        parts += [piece(posed(self.inputs.body, row['U']), V.GRAY, .36*alpha),
                  piece(line(row['hull']+np.array([0., 0., .0007])), ORANGE, alpha)]
        return parts, camera


def illustrative_ring(inputs, row):
    """Ground-covering illustration only: deliberately omit work exclusions."""
    inner = Polygon(row['hull'][:, :2])
    outer = inner.buffer(.002, join_style=2)
    contours = [np.asarray(p.exterior.coords)[:-1]/SCALE for p in [outer, inner]]
    ring = md.CrossSection(contours, md.FillRule.EvenOdd).extrude(.003/SCALE)
    ring = moved(ring, np.linalg.inv(row['T']))
    vertices = unpack(ring).vertices
    top = vertices[transform(vertices, row['T'])[:, 2]>.0029]
    support = inputs.fixture.vertices
    support = support[transform(support, row['T'])[:, 2]>.003]
    distances, ids = cKDTree(support).query(top)
    k = int(distances.argmin())
    offsets = np.array([[x, y, z] for x in [-.00075, .00075] for y in [-.00075, .00075] for z in [-.00075, .00075]])
    connector = solid(trimesh.convex.convex_hull(np.vstack([top[k]+offsets, support[ids[k]]+offsets])))
    return ring+connector


class WorkScene:
    def __init__(self, inputs):
        self.inputs = inputs
        required = inputs.demo['work_certificate']['pose']
        target = next(r for r in inputs.rows if r['pose']==required)
        self.base = illustrative_ring(inputs, target)
        base_mesh = unpack(self.base)
        ray = RayMeshIntersector(base_mesh)
        candidates = []
        per_pose = []
        for row in inputs.rows:
            k = row['index']
            Q = inputs.layout['placements'][k]
            access = WorkAccess(inputs.body, row['work_ids'], float(row['setup']['cone_half_deg']))
            local_points = transform(base_mesh.vertices, np.linalg.inv(Q))
            full = moved(access.solid(access.required_length(local_points, .001)), Q)
            intersection = self.base^full
            overlap = volume(intersection)*1e6
            angle = int(float(row['setup']['cone_half_deg']))
            arrays = inputs.arrays(ROOT/f'objects/B/poses/{row["pose"]}/angle_{angle}/samples.npz')
            common_from_native = Q@np.linalg.inv(row['setup']['T_world_mesh'])
            origins = transform(arrays['pt_m'], common_from_native)
            directions = -arrays['force_push_mg']@common_from_native[:3, :3].T
            norms = np.linalg.norm(directions, axis=1)
            valid_ids = np.flatnonzero(norms>1e-8)
            directions[valid_ids] /= norms[valid_ids, None]
            locations, ray_ids, _ = ray.intersects_location(origins[valid_ids]+1e-7*directions[valid_ids],
                                                           directions[valid_ids], multiple_hits=False)
            sample_ids = valid_ids[ray_ids]
            distances = np.linalg.norm(locations-origins[sample_ids], axis=1)
            blocked_ray_count = int(np.count_nonzero(distances>1e-7))
            keep = (distances>.003)&(distances<.3)
            sample_ids, locations, distances = sample_ids[keep], locations[keep], distances[keep]
            per_pose.append(dict(pose=row['pose'], work_intersection_cm3=overlap,
                                 blocked_original_ray_count=blocked_ray_count,
                                 readable_blocked_ray_count=len(sample_ids), original_sample_count=32768))
            if len(sample_ids) and overlap>1e-5:
                # A longer visible segment makes the real obstruction readable.
                index = int(np.argmax(distances))
                sample = int(sample_ids[index])
                face = int(row['work_ids'][int(arrays['work_face_index'][sample])])
                candidates.append((len(sample_ids), row, full, intersection, sample, face,
                                   origins[sample], directions[sample], locations[index], float(distances[index])))
        if not candidates:
            raise ValueError('No original visible processing ray is blocked by the illustrative base')
        _, row, full, intersection, sample, face, origin, direction, location, distance = max(candidates, key=lambda x:x[0])
        self.row = row
        self.T = row['T']
        Q = inputs.layout['placements'][row['index']]
        self.intersection = posed(unpack(intersection), self.T)
        self.unblocked_base = posed(unpack(self.base-full), self.T)
        self.complete_base = posed(base_mesh, self.T)
        self.fixture = posed(inputs.fixture, self.T)
        self.body = posed(inputs.body, row['U'])
        self.work_faces = trimesh.Trimesh(inputs.body.vertices, inputs.body.faces[row['work_ids']], process=False)
        self.work_faces = posed(self.work_faces, row['U'])
        self.origin, self.hit = transform(origin, self.T), transform(location, self.T)
        self.direction = self.T[:3, :3]@direction
        self.distance = distance
        self.stop = distance+.002
        # Show the exact blocking triangle's cone, capped only beyond a sphere.
        single = WorkAccess(inputs.body, [face], float(row['setup']['cone_half_deg']))
        center = (self.body.bounds.mean(0)+self.hit)/2.
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=.15)
        sphere.apply_translation(center)
        local = transform(sphere.vertices, np.linalg.inv(row['U']))
        self.cone = unpack(moved(single.solid(single.required_length(local, .001)), row['U'])^solid(sphere))
        normal = row['U'][:3, :3]@inputs.body.face_normals[face]
        assert self.direction@normal>=np.cos(np.radians(float(row['setup']['cone_half_deg'])))-1e-10
        tangent = np.cross(normal, np.array([0., 0., 1.]))
        if np.linalg.norm(tangent)<1e-6:
            tangent = np.array([1., 0., 0.])
        tangent /= np.linalg.norm(tangent)
        sight = normal+.65*tangent+.35*np.array([0., 0., 1.])
        sight /= np.linalg.norm(sight)
        right = np.cross(np.array([0., 0., 1.]), sight)
        right /= np.linalg.norm(right)
        basis = np.array([right, np.cross(sight, right), sight])
        start = self.origin+self.direction*(distance+.065)
        cloud = np.vstack([self.body.vertices, self.fixture.vertices, self.complete_base.vertices,
                           start, start+self.direction*.035])
        projected = cloud@basis.T
        low, high = projected.min(0), projected.max(0)
        self.camera = (((low+high)/2.)@basis, basis, max(high[1]-low[1], (high[0]-low[0])*9/16)*1.22)
        self.ray_line = line([self.origin, start], .0004, closed=False)
        self.marker = trimesh.creation.icosphere(subdivisions=2, radius=.0023)
        self.marker.apply_translation(self.hit)
        self.metadata = dict(failure='ground-covering illustrative base blocks original work access',
            base_designed_for_pose=target['pose'], shown_work_pose=row['pose'],
            base_width_mm=2., base_height_mm=3., base_is_counterexample_not_optimizer_output=True,
            original_processing_sample_index=sample, original_work_face_id=face,
            blocked_ray_from_original_saved_visible_load=True, processing_ray_distance_to_base_mm=distance*1000,
            processing_point_world_m=self.origin.tolist(), outward_processing_ray_world=self.direction.tolist(),
            first_base_hit_world_m=self.hit.tolist(), work_intersection_cm3=volume(intersection)*1e6,
            per_pose_work_intersections=per_pose, display_cone='full original blocking triangle cone intersected with display sphere',
            display_radius_m=.15, display_cap_is_not_a_physical_limit=True,
            ground_work_certificate_pose=target['pose'],
            certificate_gap_mm=inputs.demo['work_certificate']['separator']['separation_gap_m']*1000)

    def frame(self, time):
        reveal = smooth((time-1.7)/1.)
        travel = smooth((time-4.)/2.)
        tip = self.origin+self.direction*((1.-travel)*(self.distance+.065)+travel*self.stop)
        shaft = trimesh.creation.cylinder(radius=.0018, segment=np.array([tip, tip+self.direction*.037]), sections=12)
        tip_mesh = trimesh.creation.icosphere(subdivisions=2, radius=.002)
        tip_mesh.apply_translation(tip)
        tool_alpha = smooth((time-3.3)/.5)
        work_piece = piece(self.work_faces, GREEN)
        work_piece['bias'] = .000002
        parts = [piece(self.cone, '#E8B267', .09*reveal), piece(self.fixture, V.BLUE),
                 piece(self.body, V.GRAY), work_piece,
                 piece(self.complete_base, ORANGE, 1.-reveal), piece(self.unblocked_base, ORANGE, reveal),
                 piece(self.intersection, RED, reveal), piece(self.ray_line, '#6F777E', .65*tool_alpha),
                 piece(shaft, '#535E69', tool_alpha), piece(tip_mesh, RED if travel>.99 else '#535E69', tool_alpha),
                 piece(self.marker, RED, smooth((time-6.)/.35))]
        return parts, self.camera


def render_frame(renderer, scene, time, width=1600, height=900):
    parts, camera = scene.frame(time)
    parts = [V.CPU.placed(p) for p in parts if p is not None]
    focus, basis, span = camera
    image = renderer.render(parts, (focus, basis, span*width/height), width)
    top = (width-height)//2
    return image.crop((0, top, width, top+height)).convert('RGB')


def encode(renderer, scene, number, fps, seconds):
    output = FOLDER/f'video{number}.mp4'
    temporary = FOLDER/f'.video{number}.rendering.mp4'
    count = int(round(fps*seconds))
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1600x900', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
               '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for k in range(count):
            frame = render_frame(renderer, scene, k/fps)
            process.stdin.write(frame.tobytes())
            if k == count-1:
                frame.save(FOLDER/f'video{number}.png')
            if k % fps == 0:
                print(f'VIDEO{number} {k}/{count}', flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Video encoder failed')
        temporary.replace(output)
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    return dict(seconds=count/fps, fps=fps, frames=count, width=1600, height=900,
                text=False, audio=False, codec='H.264 Main',
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', nargs='+', type=int, choices=[2, 3], default=[2, 3])
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--seconds', type=float, default=10.)
    args = parser.parse_args()
    if args.fps<=0 or args.seconds<8.:
        parser.error('A positive frame rate and at least eight seconds are required')
    inputs = SavedInputs()
    renderer = V.Renderer()
    classes = {2: GroundScene, 3: WorkScene}
    for number in args.videos:
        scene = classes[number](inputs)
        print(json.dumps(dict(video=number, **scene.metadata), ensure_ascii=False), flush=True)
        if args.preview_only:
            out = Path('/tmp/cadgrasp-failure-previews')
            out.mkdir(exist_ok=True)
            for time in [0., 5., 8.]:
                render_frame(renderer, scene, time).save(out/f'video{number}_{int(time)}.png')
        else:
            record = encode(renderer, scene, number, args.fps, args.seconds)
            inputs.verify()
            record.update(pose_set=LABEL, historical_step4_experiment=inputs.experiment,
                          **scene.metadata, inputs_sha256=inputs.hashes,
                          optimizer_run=False, force_solver_run=False, saved_layout_changed=False,
                          original_inputs_unchanged=True,
                          render_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            (FOLDER/f'video{number}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    inputs.verify()


if __name__ == '__main__':
    main()
