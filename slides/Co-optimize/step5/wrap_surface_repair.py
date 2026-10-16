"""A small additive finish at narrow connections of the already-pruned solid.

The grid only locates candidate necks. Actual additions are a 1 mm dilation
of small patches of the original boundary, clipped to the original legal
solid, all workpiece exclusions and the already-achieved system BBX.
"""
import itertools
import numpy as np
import manifold3d as md
from scipy import ndimage
from skimage.morphology import skeletonize

from wrap_geometry import solid, unpack, volume, union, box, SCALE, C
from wrap_dimensions import physical_component_count
from continuous_legs import raster
from exact_mesh import save_mesh

SURFACE_EXPANSION_M = .001


def neck_regions(mesh, pitch=.00075, radius=.0015):
    """Find thin skeleton links between interior lobes; no width acceptance gate."""
    origin = np.floor(mesh.bounds[0]/pitch)*pitch-2*pitch
    shape = tuple(np.ceil((mesh.bounds[1]-origin)/pitch).astype(int)+3)
    inside = raster(solid(mesh), origin, shape, pitch)
    clearance = ndimage.distance_transform_edt(inside, sampling=pitch)
    skeleton = skeletonize(inside, method='lee')
    structure = np.ones((3,3,3), bool)
    cores, _ = ndimage.label(clearance > radius, structure=structure)
    sizes = np.bincount(cores.ravel())
    thin, _ = ndimage.label(skeleton & (clearance <= radius), structure=structure)
    offsets = np.array(list(itertools.product([-1,0,1], repeat=3)))
    regions = []
    for index, section in enumerate(ndimage.find_objects(thin), 1):
        ids = np.argwhere(thin[section] == index)+np.array([s.start for s in section])
        if len(ids) < 2:
            continue
        neighbors = (ids[:,None,:]+offsets[None,:,:]).reshape(-1,3)
        neighbors = neighbors[np.all((neighbors >= 0)&(neighbors < shape), axis=1)]
        tags = np.unique(cores[tuple(neighbors.T)])
        tags = tags[tags != 0]
        # Keep the tiny intermediate lobes: a very short thin connection can
        # otherwise be missed immediately beside a larger pad.
        if len(tags) < 2 or sizes[tags].max()*pitch**3*1e6 < .03:
            continue
        points = origin+ids*pitch
        regions.append(dict(bounds_m=[points.min(0).tolist(), points.max(0).tolist()],
            skeleton_points=len(ids), adjacent_lobe_volumes_cm3=(sizes[tags]*pitch**3*1e6).tolist()))
    return regions, dict(pitch_m=pitch,thin_link_radius_m=radius,
        candidate_regions=len(regions), grid_shape=[int(n) for n in shape],
        width_or_strength_certificate=False, hard_minimum_width_enforced=False)


def local_expansion(value, regions, domain, expansion=SURFACE_EXPANSION_M, padding=.003,
                    geometry_check=None, replay=False,return_mesh=False):
    """Only grow near the existing surface; no new long bridges or replacement solid."""
    ball = md.Manifold.sphere(expansion/SCALE, 8)
    grown = value
    last_checked_mesh=unpack(value)
    for row in regions:
        if replay and not row.get('accepted',True):
            continue
        bounds = np.asarray(row['bounds_m'])
        patch = value ^ box(bounds[0]-padding, bounds[1]+padding)
        addition = (patch.minkowski_sum(ball) ^ domain)-value
        row['added_cm3_before_overlap'] = volume(addition)*1e6
        if volume(addition) <= 1e-13:
            row['accepted']=False
            continue
        candidate=grown+addition
        # Round-trip the actual float64 boundary before counting components.
        # Discard only disconnected proposed chips, preserving all prior stock.
        for _ in range(3):
            actual=solid(unpack(candidate))
            parts=[part for part in actual.decompose() if float(part.volume())>0.]
            candidate=max(parts,key=lambda part:float(part.volume())) if len(parts)>1 else actual
            if len(parts)==1:
                break
        try:
            mesh=unpack(candidate)
            if geometry_check is not None:
                candidate=geometry_check(mesh)
            if (physical_component_count(candidate)!=1 or volume(grown-candidate)>1e-10
                    or volume(value-candidate)>1e-10):
                raise ValueError('Expansion failed to preserve the complete connected input')
            grown=candidate
            last_checked_mesh=mesh
            row['accepted']=True
        except (ValueError,RuntimeError) as error:
            row['accepted']=False;row['reason']=str(error)
    if physical_component_count(grown) != 1 or volume(value-grown) > 1e-10:
        raise ValueError('Surface expansion must preserve the complete connected input')
    result=(grown,grown-value)
    return (*result,last_checked_mesh) if return_mesh else result


def bounded_domain(validator, bounds):
    domain = validator.witness-validator.objects
    for T in {np.round(r['fixture_transform'],11).tobytes():r['fixture_transform']
              for r in validator.rows}.values():
        for axis in range(3):
            domain = domain.trim_by_plane(tuple(T[axis,:3]), float((bounds[0,axis]-T[axis,3])/SCALE))
            domain = domain.trim_by_plane(tuple(-T[axis,:3]), float((T[axis,3]-bounds[1,axis])/SCALE))
    return domain


def repair(mesh, validator, out):
    out.mkdir(parents=True, exist_ok=True)
    value = solid(mesh)
    bounds = C.occupied_bounds(validator.data['body'], mesh, validator.rows)
    domain = bounded_domain(validator, bounds)
    regions, detection = neck_regions(mesh)
    def geometry_check(candidate):
        return validator.check(candidate,bounds=bounds,forces=False)[0]
    grown, allowed, checked_mesh = local_expansion(value, regions, domain,
        geometry_check=geometry_check,return_mesh=True)
    changed = volume(grown-value) > 1e-10
    candidate = save_mesh(checked_mesh if changed else mesh, out/'support.obj')
    actual, proof, checks = validator.check(candidate, bounds=bounds)
    if volume(value-actual) > 1e-10:
        raise ValueError('Reinforcement removed existing material')
    # Save an exact independently checkable geometric allowance for this
    # additive operation. Every old deletion checkpoint remains unchanged.
    allowance = actual-value
    if changed:
        save_mesh(unpack(allowance), out/'added_material.obj')
    record = dict(complete=True,passed=True,changed=changed,operation='local_surface_expansion',
        surface_expansion_m=SURFACE_EXPANSION_M,local_patch_padding_m=.003,
        original_pruned_material_loss_m3=volume(value-actual),
        added_material_cm3=volume(allowance)*1e6,regions=regions,detection=detection,
        before_material_cm3=volume(value)*1e6,after_material_cm3=checks['material_cm3'],
        original_pruning_preserved=True,global_uniform_thickening=False,
        original_legal_and_force_guards_passed=True,original_system_bbx_preserved=True)
    C.save(out/'report.json',record)
    return candidate, actual, proof, checks, record
