"""Merge the first adjacent Step3 tasks by translating their actual saved shells."""
import json
from pathlib import Path

import numpy as np
import trimesh

from subdivide_task.step1.common import ROOT, sha256, write_json
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step3.tool_membranes import load_model, stem
from subdivide_task.step4 import fast_checks  # noqa: F401; same continuous exit test, faster broad phase
from subdivide_task.step4.keepout import load as load_keepouts
from .geometry import moved_mesh
from .state import Layout
from .render_refinements import condition_geometry, component_volumes, connected_shell


DATA = ROOT / 'subdivide_task/data/B/operation_videos/juxtapose_step3'
SOURCE = ROOT / 'subdivide_task/data/B/step3_tool_membranes'
TASKS = ['+X', '+Y']


def shifted(mesh, offset):
    result = mesh.copy()
    result.apply_translation(offset)
    return result


class Pair:
    def __init__(self):
        base, original, self.report = load_model(SOURCE)
        assert base.ids[:2] == TASKS
        proof = json.loads((SOURCE / 'validation.json').read_text())
        assert proof['status'] == 'passed' and proof['report_sha256'] == sha256(SOURCE / 'report.json')
        self.model = base.subset([0, 1])
        keepouts = load_keepouts('B')
        # Construction also excludes all stored KUKA link hulls. The final
        # audit additionally replays the original exact arm collision test.
        self.model.tools = {name: keepouts[name][2] for name in TASKS}
        self.layout = Layout(original.poses[:2], original.material[:2], original.exits[:2])
        self.shifts = np.asarray(self.report['connected']['installation_shifts_m'])[:2]
        self.layout.material = np.tile(np.eye(4), (2, 1, 1))
        self.layout.material[:, :3, 3] = self.shifts
        self.shells = []
        self.inputs = [SOURCE / name for name in ('report.json', 'validation.json', 'initialization.npz')]
        for name, shift in zip(TASKS, self.shifts):
            path = SOURCE / (stem(name) + '.npz')
            with np.load(path) as saved:
                mesh = trimesh.Trimesh(saved['vertices'], saved['faces'], process=False)
            self.shells.append(shifted(mesh, shift))
            self.inputs.append(path)
        keep = ROOT / 'subdivide_task/data/B/step4_keepout'
        self.inputs += [keep / name for name in ('report.json', 'validation.json', 'PX.npz', 'PY.npz')]
        self.solids = [G.solid(mesh) for mesh in self.shells]
        self.tasks = self.model.prepare(self.layout)

    def layout_at(self, distance):
        q = self.layout.copy()
        q.material[1, 0, 3] -= distance
        return q

    def tasks_at(self, distance):
        tasks = [dict(row) for row in self.tasks]
        delta = np.array([-distance, 0., 0.])
        tasks[1]['body'] = shifted(tasks[1]['body'], delta)
        for key in ('stock', 'work', 'sweep'):
            tasks[1][key] = tasks[1][key].translate(delta / G.SCALE)
        tasks[1]['material'] = self.layout_at(distance).material[1]
        return tasks

    def construct(self, distance, connected=False):
        tasks = self.tasks_at(distance)
        guest = self.solids[1].translate(np.array([-distance, 0., 0.]) / G.SCALE)
        seed = G.union([self.solids[0], guest])
        bounds = np.vstack([self.shells[0].vertices, self.shells[1].vertices - [distance, 0., 0.]])
        box = G.box(bounds.min(0) - 1e-5, bounds.max(0) + 1e-5)
        forbidden = {f'{i}_{key}': (G.solid(t['body']) if key == 'body' else t[key] ^ box)
                     for i, t in enumerate(tasks) for key in ('body', 'work', 'sweep')}
        raw = seed - G.union(list(forbidden.values()))
        volumes = component_volumes(raw)
        if connected:
            if not volumes:
                return None
            raw = connected_shell(raw)
        support, overlaps = condition_geometry(raw, forbidden)
        shared = self.solids[0] ^ guest
        return dict(layout=self.layout_at(distance), support=support, tasks=tasks,
                    volume_cm3=float(support.volume * 1e6), forbidden_overlaps_m3=overlaps,
                    shell_overlap_cm3=G.volume(shared) * 1e6,
                    retained_shared_material_cm3=G.volume(G.solid(support) ^ shared) * 1e6,
                    construction=dict(method='union of the actual two Step3 shells minus both current body/tool/arm/exit exclusions',
                                      support_support_overlap_allowed=True,
                                      support_support_intersection_subtracted=False,
                                      source_shells_regenerated=False, guest_translation_m=[-distance, 0., 0.]),
                    raw_components_cm3=volumes, seed=seed)

    def bindings(self):
        return {str(path.relative_to(ROOT)): sha256(path) for path in self.inputs}


def build():
    DATA.mkdir(parents=True, exist_ok=True)
    pair = Pair()
    proof_path = DATA / 'validation.json'
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if (proof.get('status') == 'passed' and proof.get('inputs_sha256') == pair.bindings()
                and proof.get('construction_source_sha256') == sha256(Path(__file__))
                and proof['geometry_sha256'] == sha256(DATA / 'geometry.npz')):
            print('STEP3 PAIR REUSE', TASKS, flush=True)
            return pair, proof
    gap = max(0., pair.shells[1].bounds[0, 0] - pair.shells[0].bounds[1, 0])
    trace, candidates = [], []
    # Explore well past first contact. Support/support overlap is the useful
    # shared material, never a forbidden volume or a reason to stop packing.
    aligned_centers = float(pair.shells[1].bounds.mean(0)[0] - pair.shells[0].bounds.mean(0)[0])
    for index, distance in enumerate(np.arange(gap + .002, aligned_centers + .061, .006)):
        candidate = pair.construct(float(distance), connected=True)
        if candidate is None:
            trace.append(dict(distance_m=float(distance), status='separate_material_components'))
            continue
        if candidate['retained_shared_material_cm3'] < 1.:
            trace.append(dict(distance_m=float(distance), status='insufficient_shared_material',
                              shell_overlap_cm3=candidate['shell_overlap_cm3']))
            continue
        candidates.append((float(distance), candidate))
        if index % 5 == 0:
            print('STEP3 OVERLAP CANDIDATE', round(float(distance), 4),
                  'volume', round(candidate['volume_cm3'], 3),
                  'shared', round(candidate['retained_shared_material_cm3'], 3), flush=True)
    candidates.sort(key=lambda item: item[1]['volume_cm3'])
    for distance, candidate in candidates:
        try:
            pair.model.refine_exits(candidate)
            audit = pair.model.audit(candidate)
        except (ValueError, RuntimeError) as error:
            trace.append(dict(distance_m=float(distance), status='rejected', reason=str(error)))
            print('STEP3 PAIR TRY', round(float(distance), 4), str(error)[:180], flush=True)
            continue
        final = candidate['support']
        guest = pair.solids[1].translate([-distance / G.SCALE, 0., 0.])
        shared = pair.solids[0] ^ guest
        retained_shared = G.volume(G.solid(final) ^ shared) * 1e6
        assert retained_shared >= 1., 'The merge must retain substantial shared support material'
        np.savez_compressed(DATA / 'geometry.npz', source_vertices=pair.model.source.vertices,
            source_faces=pair.model.source.faces, poses=pair.layout.poses, initial_shifts=pair.shifts,
            exits=pair.layout.exits, host_vertices=pair.shells[0].vertices, host_faces=pair.shells[0].faces,
            guest_vertices=pair.shells[1].vertices, guest_faces=pair.shells[1].faces,
            final_vertices=final.vertices, final_faces=final.faces, translation_m=float(distance),
            work_faces_0=pair.model.records[0]['working_face_ids'], work_faces_1=pair.model.records[1]['working_face_ids'])
        proof = dict(status='passed', task_ids=TASKS, adjacent_step3_task_indices=[0, 1],
            input_shells_used_directly=True, guest_translation_m=[-float(distance), 0., 0.],
            support_support_overlap_allowed=True, support_support_intersection_subtracted=False,
            shell_overlap_cm3=G.volume(shared) * 1e6, retained_shared_material_cm3=retained_shared,
            selection='lowest-volume fully accepted layout from translations continuing past first contact',
            evaluated_layouts=[dict(distance_m=d, volume_cm3=c['volume_cm3'],
                                   shell_overlap_cm3=c['shell_overlap_cm3'],
                                   retained_shared_material_cm3=c['retained_shared_material_cm3'])
                               for d, c in candidates],
            source_volumes_cm3=[float(mesh.volume * 1e6) for mesh in pair.shells],
            final_volume_cm3=float(final.volume * 1e6), final_connected=True,
            audit=audit, trace=trace, inputs_sha256=pair.bindings(),
            geometry_sha256=sha256(DATA / 'geometry.npz'), construction_source_sha256=sha256(Path(__file__)))
        write_json(proof_path, proof)
        print('STEP3 PAIR PASS', TASKS, 'translation', round(float(distance), 4),
              'volume', round(float(final.volume * 1e6), 3), flush=True)
        return pair, proof
    write_json(DATA / 'search.json', dict(status='search_failed', trace=trace, inputs_sha256=pair.bindings()))
    raise RuntimeError('No accepted connected juxtaposition in the tested translations')


if __name__ == '__main__':
    build()
