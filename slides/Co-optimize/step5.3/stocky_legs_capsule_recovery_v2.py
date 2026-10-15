"""Broad rebuilt legs with bounded reinforcement of an accepted solid fallback."""
import time
import numpy as np

import stocky_legs_resumed_profiles as R
from deterministic_base import volume, unpack, union, enclosure
from prune_support import material_components
from continuous_legs import capsule, no_tunnels
def flared_path(path, shaft_radius, root_radius, flare_length=.020, minimum_envelope=None):
    path=np.asarray(path, dtype=float)
    if len(path)==2:
        return capsule(path[-1],path[0],shaft_radius,root_radius)
    # Closed convex segment solids avoid folded loft rings at sharp bends.
    return union([capsule(b,a,shaft_radius,root_radius if i==0 else shaft_radius)
                  for i,(a,b) in enumerate(zip(path[:-1],path[1:]))])


class Builder(R.Builder):
    def run(self):
        rebuild_rejection = 'Independent recovery of the retained sharp-bend case rejected by stocky_legs_all52_20261013_02' 
        # Preserve the certified old solid while adding legal round material.
        # This avoids moving an existing thin branch to an expensive new route.
        for shaft, root, reach in [(.0045, .0065, .016), (.00425, .006, .012),
                                    (.004125, .00575, .008), (.0040625, .00555, .006)]:
            current = self.previous
            history = []
            for index, record in enumerate(self.recipes):
                path = R.recipe_path(record)
                old_radius = float(record['radius_m'])
                old_root = float(record.get('root_radius_m', old_radius * 1.35))
                straight = record['operation'] == 'solid_rounded_leg'
                old = (capsule(path[-1], path[0], old_radius, old_root) if straight else
                       union([capsule(a, b, old_radius) for a, b in zip(path[:-1], path[1:])]))
                old = old ^ self.domain
                # Only saved actual leg material is used as the retained envelope.
                retained = old
                actual_shaft = max(shaft, old_radius * 1.025)
                actual_root = max(root, (old_root if straight else old_radius) * 1.05)
                proposal = flared_path(path, actual_shaft, actual_root, reach,
                    minimum_envelope=(old_radius, old_root) if straight else (old_radius, old_radius))
                parts = material_components(proposal ^ self.domain)
                options = []
                for part in parts:
                    if volume(part ^ retained) <= 1e-14 or not no_tunnels(part):
                        continue
                    candidate = current + part
                    if len(material_components(candidate)) != 1:
                        continue
                    options.append((volume(part ^ retained), part, candidate))
                if not options:
                    print('RETAINED PROFILE REJECT', shaft, 'leg', index, 'no solid options',
                        [(unpack(p).euler_number,volume(p)*1e6) for p in parts], flush=True)
                    break
                _, piece, candidate = max(options, key=lambda option: option[0])
                widths = R.cross_section_widths(piece, path)
                old_widths = self.old_reference[index]
                if widths and old_widths and min(widths) < min(old_widths) - 1e-7:
                    print('RETAINED PROFILE REJECT', shaft, 'leg', index, 'width', min(widths), min(old_widths), flush=True)
                    break
                added = volume(candidate - current) * 1e6
                history.append(dict(operation='retained_flared_solid_leg', kind=record['kind'],
                    tip_m=path[-1].tolist(), root_m=path[0].tolist(), path_m=path[::-1].tolist(),
                    radius_m=actual_shaft, root_radius_m=actual_root, flare_length_m=reach,
                    added_cm3=added, leg_tunnels=0, sampled_section_widths_m=widths,
                    previous_sampled_section_widths_m=old_widths,
                    sampled_min_width_m=min(widths) if widths else None,
                    previous_sampled_min_width_m=min(old_widths) if old_widths else None,
                    clipping_limited=bool(widths and min(widths) < .004),
                    retained_previous_leg_material=True, nominal_thickness_is_not_actual_minimum=True))
                current = candidate
            if len(history) != len(self.recipes) or volume(current) > self.limit + 1e-10:
                print('RETAINED PROFILE FINISH', shaft, 'legs', len(history), 'material %',
                    (volume(current)/volume(self.previous)-1)*100, flush=True)
                continue
            mesh = unpack(current)
            cover = enclosure(mesh, self.rows)
            from stage5_shared import occupied_bounds
            bounds = occupied_bounds(self.object, mesh, self.rows)
            if not all(row['passed'] for row in cover) or np.max(abs(bounds - self.bounds)) > 1e-8:
                continue
            if abs(np.prod(bounds[1]-bounds[0])-np.prod(self.bounds[1]-self.bounds[0]))*1000 > 1e-8:
                continue
            if volume(current - self.previous) <= 1e-12:
                continue
            return mesh, dict(passed=True, connected=True, watertight=bool(mesh.is_watertight),
                material_cm3=volume(current)*1e6, added_material_cm3=volume(current-self.core)*1e6,
                retained_legal_step4_material_loss_cm3=volume(self.core-current)*1e6,
                per_pose=cover, history=history, leg_profile='stocky_flared_solid',
                contact_core_preserved=True, large_box_bridge_used=False,
                shaft_radius_m=shaft, root_radius_m=root, flare_length_m=reach,
                removed_redundant_leg_indices=[], material_growth_limit_fraction=.20,
                retained_previous_solid_fallback=True, closed_convex_segment_fallback=True, rebuild_rejection=rebuild_rejection,
                seconds=time.monotonic()-self.began)
        raise ValueError('Rebuild and bounded retained-leg reinforcement failed: ' + rebuild_rejection)
