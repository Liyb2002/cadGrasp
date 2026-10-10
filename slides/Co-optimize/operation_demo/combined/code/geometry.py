"""Real layout states after Juxtapose, with Translation and Direction refinement.

Only the current configuration's exits are carved. The lateral design motion
is never turned into a sliding channel or into a swept outer wrapping.
"""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DEMO = Path(__file__).resolve().parents[2]
OUT = DEMO/'combined/vis'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


G = module('juxtapose_demo_geometry', DEMO/'juxtapose/code/geometry.py')
# The original renderer imports its local geometry by the short module name.
# Bind that name only while loading, without changing any production imports.
previous_geometry = sys.modules.get('geometry')
sys.modules['geometry'] = G
R = module('juxtapose_demo_render', DEMO/'juxtapose/code/render.py')
if previous_geometry is None:
    del sys.modules['geometry']
else:
    sys.modules['geometry'] = previous_geometry

TRANSLATION_STEPS = 48
DIRECTION_STEPS = 18
DIRECTION_ANGLE = np.radians(30.)


class BooleanUnresolved(RuntimeError):
    pass


def moved(mesh, offset):
    transform = np.eye(4)
    transform[:3, 3] = offset
    return G.mesh_in(mesh, transform)


class Sequence:
    def __init__(self):
        self.specimen = G.Specimen()
        s = self.specimen
        self.direction = s.direction.copy()
        # Horizontal camera-right is a fixed ground-tangent direction. Define
        # one body width by the actual guest projection along that direction.
        azimuth = np.radians(-55.)
        self.translation_axis = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
        self.body_width = float(np.ptp(s.guest.vertices @ self.translation_axis))
        self.offset = self.translation_axis*self.body_width/3
        tangent = self.translation_axis-self.direction*(self.translation_axis @ self.direction)
        self.direction_tangent = tangent/np.linalg.norm(tangent)
        old_work = np.unique(np.concatenate([s.states[i][0].domain.work_ids for i in [0, 2, 3]]))
        self.host_work = G.work_exclusion(s.host, old_work)
        self.guest_work = G.work_exclusion(s.guest, s.guest_work)
        self.host_body = G.C.S.solid(s.host)
        self.guest_body = G.C.S.solid(s.guest)
        # Start from the actual already-carved Juxtapose support. Add only the
        # current moved fitted wrapping; do not union a relocation trail or
        # force restoration of every historical cut from an uncarved shell.
        self.fixed_seed = s.final
        self.baseline = s.final
        self.meshes, self.records = {}, {}

    def direction_at(self, fraction):
        angle = DIRECTION_ANGLE*fraction
        return np.cos(angle)*self.direction+np.sin(angle)*self.direction_tangent

    def state(self, name, fraction, direction, previous=None, check_work=False):
        s = self.specimen
        offset = self.offset*fraction
        guest = moved(s.guest, offset)
        body = self.guest_body.translate((offset/G.C.S.SCALE).tolist())
        guest_wrap = s.guest_wrap.translate((offset/G.C.S.SCALE).tolist())
        work = self.host_work+self.guest_work.translate((offset/G.C.S.SCALE).tolist())
        guest_exit = G.C.S.solid(G.C.S.swept_solid(guest, s.sweep_length*direction, fan_in=8))
        exits = s.exits[:3]+[guest_exit]
        bodies = [self.host_body, body]
        forbidden = G.C.union(exits+bodies+[work])
        if name == 'direction_18':
            # Pin actual playback positions on this SAME exit, with no padding.
            # These bodies are already part of the analytic continuous sweep.
            anchors = [body.translate((t*direction/G.C.S.SCALE).tolist())
                       for t in np.linspace(0, .07, 14)]
            forbidden = forbidden+G.C.union(anchors)
        seed = self.fixed_seed+guest_wrap
        final = seed-forbidden
        # Resolve only genuine exclusion intersections, never arbitrary scraps.
        for _ in range(3):
            overlaps = [G.C.material_volume(final ^ value) for value in exits+bodies+[work]]
            if max(overlaps) <= G.TOL:
                break
            for value, overlap in zip(exits+bodies+[work], overlaps):
                if overlap > G.TOL:
                    final = final-value
        overlaps = [G.C.material_volume(final ^ value) for value in exits+bodies+[work]]
        if max(overlaps) > G.TOL:
            raise BooleanUnresolved((name, overlaps))
        kept = final ^ self.baseline
        added = final-self.baseline
        deleted = self.baseline-final
        # Old BLUE support disappears only inside a current real exclusion.
        deletion_outside = G.C.material_volume(deleted-forbidden)
        if deletion_outside > G.TOL:
            raise BooleanUnresolved((name, deletion_outside))
        fresh = deleted if previous is None else (previous ^ self.baseline)-final
        self.meshes[name] = dict(final=G.C.S.unpack(final), kept=G.C.S.unpack(kept),
                                 added=G.C.S.unpack(added), fresh=G.C.S.unpack(fresh), guest=guest)
        row = dict(translation_fraction=float(fraction), translation_world_m=offset.tolist(),
                   direction_world=direction.tolist(), final_volume_cm3=G.C.material_volume(final)*1e6,
                   new_volume_vs_juxtapose_cm3=G.C.material_volume(added)*1e6,
                   lost_volume_vs_juxtapose_cm3=G.C.material_volume(deleted)*1e6,
                   full_exit_overlap_m3=overlaps[:4], body_overlap_m3=overlaps[4:6],
                   work_band_overlap_m3=overlaps[6],
                   deleted_original_material_outside_current_exclusions_m3=deletion_outside)
        if check_work:
            checks = []
            for index, (task, transform, _) in enumerate(s.states):
                material = self.meshes[name]['final'].copy()
                if index == 1:
                    material.apply_translation(-offset)
                else:
                    material.apply_transform(transform @ np.linalg.inv(s.host_transform))
                result = G.C.WORK.check(material, task)
                replay_cycle = 0
                # HiGHS can return an unknown status at an exact shared edge.
                # Retry ONLY unresolved LPs by cyclically ordering each triangle's
                # vertices: identical geometry, winding, constraints and tolerance.
                if not result['passed'] and result['forbidden_intersections'] and all(
                        c['classification'] == 'solver_unresolved'
                        for c in result['forbidden_intersections']):
                    for cycle in [1, 2]:
                        replay = material.copy()
                        replay.faces = np.roll(material.faces, cycle, axis=1)
                        result = G.C.WORK.check(replay, task)
                        replay_cycle = cycle
                        if result['passed']:
                            break
                checks.append(dict(pose=task.pose, passed=result['passed'],
                                   forbidden_intersections=result['forbidden_intersection_count'],
                                   equivalent_triangle_cyclic_replay=replay_cycle))
            assert all(c['passed'] for c in checks), (name, checks)
            allowed = np.setdiff1d(np.arange(len(guest.faces)), s.guest_work)
            triangles, _ = G.C.contact_boundary(guest, self.meshes[name]['final'], allowed)
            row['guest_contact_area_mm2'] = G.area(triangles)*1e6
            row['actual_work_surface_checks'] = checks
            row['component_volumes_cm3'] = sorted([G.C.material_volume(p)*1e6 for p in final.decompose()], reverse=True)
        self.records[name] = row
        return final

    def build(self, reuse_translation=False):
        previous = None
        if reuse_translation:
            cached_meshes, cached_report = load()
            assert np.allclose(cached_report['final_translation_world_m'], self.offset)
            assert np.allclose(cached_report['direction_initial_world'], self.direction)
            names = [f'translation_{i}' for i in range(TRANSLATION_STEPS+1)]
            self.meshes = {name: cached_meshes[name] for name in names}
            self.records = {name: cached_report['states'][name] for name in names}
            previous = G.C.S.solid(self.meshes[names[-1]]['final'])
            self.reused_translation = dict(
                geometry_sha256=hashlib.sha256((OUT/'data/geometry.json').read_bytes()).hexdigest(),
                states_sha256=hashlib.sha256((OUT/'data/states.npz').read_bytes()).hexdigest(),
                provenance=cached_report['provenance'])
            print('REUSED', len(names), 'unchanged translation states', flush=True)
        for index in range(0 if reuse_translation else TRANSLATION_STEPS+1):
            fraction = R.smooth(index/TRANSLATION_STEPS)
            # For illustrative intermediate layouts, a sub-micron neighboring
            # position can resolve a coplanar Mesh64 ambiguity. Every accepted
            # frame must still pass the ORIGINAL tolerance; endpoints stay exact.
            deltas = [0.] if index in [0, TRANSLATION_STEPS] else [0., -1e-6, 1e-6, -1e-5, 1e-5, -1e-4, 1e-4]
            for delta in deltas:
                try:
                    previous = self.state(f'translation_{index}', fraction+delta, self.direction, previous,
                                          check_work=index in [0, TRANSLATION_STEPS])
                    self.records[f'translation_{index}']['nominal_translation_fraction'] = fraction
                    self.records[f'translation_{index}']['fraction_adjustment_for_boolean_resolution'] = delta
                    break
                except BooleanUnresolved:
                    if delta == deltas[-1]:
                        raise
            if index % 8 == 0:
                print('TRANSLATION', index, '/', TRANSLATION_STEPS, flush=True)
        for index in range(1, DIRECTION_STEPS+1):
            fraction = R.smooth(index/DIRECTION_STEPS)
            # Resolve the same coplanar ambiguity as for intermediate layouts
            # using a tiny neighboring angle. The final angle remains exact.
            deltas = [0.] if index == DIRECTION_STEPS else [0., -1e-6, 1e-6, -1e-5, 1e-5, -1e-4, 1e-4]
            for delta in deltas:
                try:
                    previous = self.state(f'direction_{index}', 1., self.direction_at(fraction+delta), previous,
                                          check_work=index == DIRECTION_STEPS)
                    self.records[f'direction_{index}']['nominal_direction_change_degrees'] = float(
                        np.degrees(DIRECTION_ANGLE)*fraction)
                    self.records[f'direction_{index}']['direction_fraction_adjustment_for_boolean_resolution'] = delta
                    break
                except BooleanUnresolved:
                    if delta == deltas[-1]:
                        raise
            if index % 6 == 0:
                print('DIRECTION', index, '/', DIRECTION_STEPS, flush=True)
        self.save()

    def save(self):
        data = OUT/'data'
        data.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for name, meshes in self.meshes.items():
            for kind, mesh in meshes.items():
                arrays[f'{name}_{kind}_vertices'] = mesh.vertices
                arrays[f'{name}_{kind}_faces'] = mesh.faces
        np.savez_compressed(data/'states.npz', **arrays)
        report = dict(complete=True, demo_only=True, force_acceptance_run=False,
                      full_fixture_accepted=False, object='B', host_pose='pose_1', guest_pose='pose_2',
                      translation_axis_world=self.translation_axis.tolist(),
                      body_width_m=self.body_width, final_translation_world_m=self.offset.tolist(),
                      translation_distance_m=self.body_width/3, fraction_of_body_width=1/3,
                      direction_change_degrees=float(np.degrees(DIRECTION_ANGLE)),
                      direction_initial_world=self.direction.tolist(),
                      direction_final_world=self.direction_at(1.).tolist(),
                      translation_step_optimized=False, direction_step_optimized=False,
                      relocation_sweep_carved=False, fixture_moves=False,
                      full_exit_length_m=self.specimen.sweep_length, volume_tolerance_m3=G.TOL,
                      seed='Verified Juxtapose shared support plus current moved fitted wrap',
                      construction='Subtract all current bodies, complete exits and work exclusions; no lateral sliding tunnel',
                      states=self.records,
                      provenance=G.C.provenance(self.specimen.inputs, [Path(__file__), Path(G.__file__), Path(R.__file__)]))
        if hasattr(self, 'reused_translation'):
            report['reused_translation_cache'] = self.reused_translation
        (data/'geometry.json').write_text(json.dumps(report, indent=2)+'\n')


def load():
    data = OUT/'data'
    report = json.loads((data/'geometry.json').read_text())
    meshes = {}
    with np.load(data/'states.npz') as source:
        for name in report['states']:
            meshes[name] = {kind: G.C.trimesh.Trimesh(source[f'{name}_{kind}_vertices'],
                source[f'{name}_{kind}_faces'], process=False)
                for kind in ['final', 'kept', 'added', 'fresh', 'guest']}
    return meshes, report


if __name__ == '__main__':
    Sequence().build()
