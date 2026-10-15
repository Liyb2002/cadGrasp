"""BBX trimming, short solid feet, then bounded finishing of a complete solid.

Every accepted candidate uses the original geometry and full-demand checks.
The runner keeps an immutable complete checkpoint outside this worker.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time

COOPT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COOPT / name) for name in
               ['helper_func', 'step5.2', 'step5.3', 'step4.2', 'evaluation']]
import stage5_shared as C
import numpy as np
from scipy.sparse import coo_matrix, diags
from prune_support import ForceCertificate, Pruner, material_components
from deterministic_base import solid, unpack, volume, union, moved, enclosure, SCALE
from continuous_legs import Builder as LegBuilder, capsule, no_tunnels
from whole_search.model import Layout


def validate(mesh, witness, body, placements, rows, checker, *, final=True,
             stock=None, bounds=None, mandatory=None):
    """Check the actual mesh; do not repair geometry or relax a tolerance."""
    if (not np.isfinite(mesh.vertices).all() or not mesh.is_watertight
            or not mesh.is_winding_consistent):
        raise ValueError('Nonfinite or open material boundary')
    value = solid(mesh)
    workpiece = body.copy()
    workpiece.merge_vertices(digits_vertex=15)
    objects = union([moved(solid(workpiece), Q) for Q in placements])
    cover = enclosure(mesh, rows) if final else []
    actual_bounds = C.occupied_bounds(body, mesh, rows)
    checks = dict(components=len(material_components(value)),
                  material_cm3=abs(float(mesh.volume))*1e6,
                  outside_m3=volume(value-solid(witness)),
                  object_overlap_m3=volume(value ^ objects),
                  added_m3=volume(value-solid(stock)) if stock is not None else 0.,
                  mandatory_loss_m3=volume(solid(mandatory)-value)
                  if mandatory is not None else 0.,
                  bbx_l=C.volume(actual_bounds)*1000,
                  bbx_expansion_m=0. if bounds is None else float(max(
                      0., np.max(bounds[0]-actual_bounds[0]),
                      np.max(actual_bounds[1]-bounds[1]))),
                  floor_passed=all(row['passed'] for row in cover))
    if (checks['components'] != 1 or checks['outside_m3'] > 1e-10
            or checks['object_overlap_m3'] > 1e-10 or checks['added_m3'] > 1e-10
            or checks['mandatory_loss_m3'] > 1e-10
            or checks['bbx_expansion_m'] > 1e-8 or not checks['floor_passed']
            or (bounds is not None and checks['bbx_l'] > C.volume(bounds)*1000+1e-8)):
        raise ValueError('Original solid checks rejected candidate: '+str(checks))
    proof = checker.evaluate_candidate(mesh)
    if not proof['passed']:
        raise ValueError('Original demands rejected candidate: '+str(proof['counts']))
    # A passing early-rejection query must have checked every saved demand.
    if (len(proof['masks']) != len(rows)
            or any(len(mask) != 32768 or not mask.all() for mask in proof['masks'].values())):
        raise ValueError('Incomplete original-demand certificate')
    checks.update(passed=True, watertight=True, winding_consistent=True,
                  serialized_geometry_checked=True, counts=proof['counts'],
                  pressure_enclosure=cover)
    return proof, checks


def serialize(mesh, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'support.obj'
    mesh.export(path, digits=16)
    return C.trimesh.load(path, force='mesh', process=False)


class ShortLegBuilder(LegBuilder):
    """Keep needed floor ends and try short round rods; failure retains old feet."""
    def required_targets(self):
        super().required_targets()
        # Previous BBX extrema are an upper bound, not new material targets.
        self.targets = [r for r in self.targets if r['kind'] == 'ground_terminal']

    def attach(self, target):
        from scipy.spatial import cKDTree
        point = target['point']
        mesh = unpack(self.body)
        _, ids = cKDTree(mesh.vertices).query(point, k=min(160, len(mesh.vertices)))
        roots = []
        for index in np.atleast_1d(ids):
            root = mesh.vertices[index]
            if not roots or min(np.linalg.norm(root-r) for r in roots) > .002:
                roots.append(root)
            if len(roots) == 8:
                break
        import manifold3d as md
        probe = md.Manifold.sphere(.0002/SCALE, 12).translate(tuple(point/SCALE))
        for radius in [.004, .003, .005]:
            for root in roots:
                rod = capsule(point, root, radius, root_radius=radius*1.35) ^ self.domain
                for part in material_components(rod):
                    if (volume(part ^ probe) <= 1e-14 or volume(part ^ self.body) <= 1e-13
                            or not no_tunnels(part)):
                        continue
                    candidate = self.body+part
                    if (len(material_components(candidate)) != 1
                            or volume(candidate) > volume(self.previous)+1e-10):
                        continue
                    self.history.append(dict(operation='short_solid_leg',
                        tip_m=point.tolist(), root_m=root.tolist(), radius_m=radius,
                        added_cm3=(volume(candidate)-volume(self.body))*1e6))
                    self.body = candidate
                    return
        raise ValueError('No short legal solid leg; retain certified complete support')

    def run(self):
        self.required_targets()
        for target in self.targets:
            self.attach(target)
        mesh = unpack(self.body)
        return mesh, dict(history=self.history, leg_profile='short_round_solid',
                         large_box_bridge_used=False, routing_search_used=False)


class CompletePruner(Pruner):
    """A deletion needs actual floor coverage as well as original object balance."""
    def accept(self, candidate, operation):
        if not all(row['passed'] for row in enclosure(unpack(candidate), self.rows)):
            return False
        previous = (self.body, self.proof, self.best_mesh)
        if not super().accept(candidate, operation):
            return False
        if not all(row['passed'] for row in enclosure(self.best_mesh, self.rows)):
            self.body, self.proof, self.best_mesh = previous
            self.history[-1].update(accepted=False, error='Serialized floor coverage rejected deletion')
            return False
        return True


def smooth_free_surface(mesh, body, placements, rows, displacement=.00055):
    """One small Laplacian proposal, with contacts and actual floor ends pinned.

    No global offset, Minkowski sum, curved refinement or fillet knife.
    The result is intersected with the input material before certification.
    """
    if len(mesh.faces) > 40000:
        raise ValueError('Large surface retained; bounded finishing skips refinement')
    surface = unpack(solid(mesh).refine(2))
    surface.merge_vertices(digits_vertex=12)
    original = surface.vertices.copy()
    pinned = np.zeros(len(original), bool)
    for Q in placements:
        workpiece = C.posed(body, Q)
        for start in range(0, len(original), 128):
            _, distances, _ = C.trimesh.proximity.closest_point(workpiece, original[start:start+128])
            pinned[start:start+128] |= distances < .0003
    for row in rows:
        T = row['fixture_transform']
        pinned |= abs((original @ T[:3, :3].T+T[:3, 3])[:, 2]) < .0004
    edges = surface.edges_unique
    indices = np.r_[edges[:, 0], edges[:, 1]]
    neighbors = np.r_[edges[:, 1], edges[:, 0]]
    matrix = coo_matrix((np.ones(len(indices)), (indices, neighbors)),
                        shape=(len(original), len(original))).tocsr()
    matrix = diags(1/np.maximum(np.asarray(matrix.sum(1)).ravel(), 1)) @ matrix
    vertices = original.copy()
    for _ in range(6):
        delta = vertices+.35*(matrix @ vertices-vertices)-original
        lengths = np.linalg.norm(delta, axis=1)
        delta *= np.minimum(1., displacement/np.maximum(lengths, 1e-12))[:, None]
        vertices = original+delta
        vertices[pinned] = original[pinned]
    surface.vertices = vertices
    return unpack(solid(surface) ^ solid(mesh))


def worker(payload):
    began = time.monotonic()
    args = SimpleNamespace(**payload)
    inputs = C.Inputs(args.object, args.selection_summary, args.step51_experiment)
    data = inputs.load(args.case)
    witness_path = data['selected']/'oracle_connected_witness.obj'
    inputs.protected[C.relative(witness_path)] = C.digest(witness_path)
    witness = C.trimesh.load(witness_path, force='mesh', process=False)
    with np.load(data['selected']/'layout.npz') as saved:
        layout = Layout.from_arrays(saved)
    placements = layout.placements[list(layout.active)]
    checker = ForceCertificate(data['record']['poses'], layout, data['body'], data['rows'], args.object)
    executed_paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(C.ROOT) and not path.is_relative_to(C.ROOT/'.venv') and path.is_file():
                executed_paths.add(path)
    code = {C.relative(path):C.digest(path) for path in executed_paths}
    reference_path, reference = C.constructed_reference(inputs, data, SimpleNamespace(reference_stage='step4'))
    previous = C.trimesh.load(reference_path, force='mesh', process=False)
    bounds = C.occupied_bounds(data['body'], previous, data['rows'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    core = data['fixture']
    full = previous
    record = dict(operation=args.operation, history=[], accepted=False)
    context = dict(witness=witness, body=data['body'], placements=placements,
                   rows=data['rows'], checker=checker)
    if args.operation == 'baseline':
        import shutil
        # First copy the exact Step4 files; any legacy cleanup is explicit below.
        shutil.copy2(reference_path, output/'support.obj')
        shutil.copy2(data['selected']/'support.obj', output/'core.obj')
        if getattr(args,'complete_reference_experiment',None):
            path, saved = C.constructed_reference(inputs,data,SimpleNamespace(
                reference_stage='step5.3',reference_experiment=args.complete_reference_experiment))
            if saved['selected_step4_source'] != C.relative(data['selected']):
                raise ValueError('Explicit complete reference belongs to another Step4 layout')
            for name, expected in saved['artifacts'].items():
                if C.digest(path.parent/name) != expected:
                    raise ValueError('Explicit complete reference artifact changed: '+name)
                inputs.protected[C.relative(path.parent/name)] = expected
            full = C.trimesh.load(path,force='mesh',process=False)
            proof, checks = validate(full, **context)
            if checks['bbx_l'] > C.volume(bounds)*1000+1e-8:
                raise ValueError('Explicit legacy reference increases the original full-support BBX volume')
            shutil.copy2(path,output/'support.obj')
            record.update(explicit_same_layout_reference=C.relative(path),
                          original_step4_full_bbx_l=C.volume(bounds)*1000,
                          byte_identical_step4_full=False)
        else:
            try:
              proof, checks = validate(full, **context)
              record.update(byte_identical_step4_full=True)
            except ValueError as error:
              # Older Step4 construction checked its saved witness, which can
              # differ slightly from the actual object boundary. Clean only this
              # new copy and certify it; never silently accept the old overlap.
              record['original_step4_revalidation_error'] = str(error)
              record['byte_identical_step4_full'] = False
              workpiece = data['body'].copy()
              workpiece.merge_vertices(digits_vertex=15)
              objects = [moved(solid(workpiece), Q) for Q in placements]
              original = full
              for cycle in range(4):
                  value = solid(full)
                  if volume(value-solid(witness)) > 1e-10:
                      value = value ^ solid(witness)
                  for obj in objects:
                      value = value-obj
                  full = serialize(unpack(value), output)
                  try:
                      proof, checks = validate(full, **context, stock=original, bounds=bounds)
                      record['boundary_cleanup_cycles'] = cycle+1
                      break
                  except ValueError:
                      if cycle == 3:
                          raise
        record.update(accepted=True)
    elif args.operation == 'trim':
        pruner = Pruner(core, witness, checker, grids=(), budget=24,
                        rows=data['rows'], object_vertices=data['body'].vertices,
                        require_connected=True)
        core, pruning, _ = pruner.run()
        core = serialize(core, output)
        proof, checks = validate(core, **context, final=False, stock=data['fixture'],
                                bounds=C.occupied_bounds(data['body'], data['fixture'], data['rows']))
        core.export(output/'core.obj', digits=16)
        full = core
        record.update(accepted=True, pruning=pruning, complete_support=False)
    else:
        checkpoint = Path(args.checkpoint)
        full = C.trimesh.load(checkpoint/'support.obj', force='mesh', process=False)
        core = C.trimesh.load(checkpoint/'core.obj', force='mesh', process=False)
        if args.operation == 'legs':
            # A declared complete reference may already be smaller than the
            # raw Step4 full solid. Grow inside the certified checkpoint box.
            previous = full
            bounds = C.occupied_bounds(data['body'], previous, data['rows'])
            pending = Path(args.trim)/'core.obj'
            core = C.trimesh.load(pending, force='mesh', process=False)
            builder = ShortLegBuilder(core, previous, witness, data['body'], data['rows'], bounds)
            full, construction = builder.run()
            full = serialize(full, output)
            proof, checks = validate(full, **context, bounds=bounds, mandatory=core)
            record.update(accepted=True, construction=construction)
        elif args.operation == 'finish':
            original = full
            original_bounds = C.occupied_bounds(data['body'], original, data['rows'])
            pruner = CompletePruner(full, witness, checker, grids=(2,), budget=12,
                                   rows=data['rows'], object_vertices=data['body'].vertices,
                                   require_connected=True)
            try:
                candidate, pruning, _ = pruner.run()
                candidate = serialize(candidate, output/'pruning_candidate')
                validate(candidate, **context, stock=original, bounds=original_bounds)
                full = candidate
                record['history'].append(dict(operation='whole_block_deletion', accepted=True,
                                              pruning=pruning))
            except (ValueError, RuntimeError) as error:
                record['history'].append(dict(operation='whole_block_deletion', accepted=False, error=str(error)))
            try:
                candidate = smooth_free_surface(full, data['body'], placements, data['rows'])
                candidate = serialize(candidate, output/'rounding_candidate')
                validate(candidate, **context, stock=full,
                         bounds=C.occupied_bounds(data['body'], full, data['rows']))
                full = candidate
                record['history'].append(dict(operation='bounded_free_surface_rounding', accepted=True))
            except (ValueError, RuntimeError) as error:
                record['history'].append(dict(operation='bounded_free_surface_rounding', accepted=False, error=str(error)))
            full = serialize(full, output)
            proof, checks = validate(full, **context, stock=original, bounds=original_bounds)
            record.update(accepted=True, contact_geometry_recertified=True)
        else:
            raise ValueError('Unknown operation: '+args.operation)
        core.export(output/'core.obj', digits=16)
    np.savez_compressed(output/'force_certificate.npz',
                        **{'mask_'+str(k): mask for k, mask in proof['masks'].items()})
    inputs.verify()
    if any(C.digest(C.ROOT/path) != expected for path, expected in code.items()):
        raise ValueError('Executed geometry source changed during the operation')
    full_value, core_value = solid(full), solid(core)
    record.update(passed=True, complete=True, checks=checks,
                  added_to_core_cm3=volume(full_value-core_value)*1e6,
                  core_loss_cm3=volume(core_value-full_value)*1e6,
                  protected=inputs.protected, seconds=time.monotonic()-began,
                  executed_code=code,
                  support_sha256=C.digest(output/'support.obj'),
                  core_sha256=C.digest(output/'core.obj'))
    C.save(output/'report.json', record)
    return record


if __name__ == '__main__':
    payload = json.loads(Path(sys.argv[1]).read_text())
    try:
        worker(payload)
    except Exception as error:
        import traceback
        C.save(Path(payload['output'])/'failure.json',
               dict(passed=False, error=str(error), traceback=traceback.format_exc()))
        raise
