"""B-only experimental contour fingers. No object modification or output sidecars.

This is a geometry/dynamics prototype, not a manufacturing-ready Franka finger.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

import grasp as G

TURN_AXIS = np.array([-.9799247046208297, -.19936793441719694, 0.])


def contour_boxes(mesh, initial, hand, pitch=.003, jaw_at_contact=.006, approach='side'):
    local = mesh.copy()
    local.apply_transform(np.linalg.inv(hand) @ initial)
    xs = np.arange(-.04, .04, pitch) + pitch/2
    zs = np.arange(.085, .110, pitch) + pitch/2
    if approach == 'top':
        xs = np.arange(-.035, .035, pitch) + pitch/2
        zs = np.arange(.110, .180, pitch) + pitch/2
    xx, zz = np.meshgrid(xs, zs)
    grid = np.column_stack([xx.ravel(), zz.ravel()])
    result = {}
    for side, name in ((1, 'left_finger'), (-1, 'right_finger')):
        rays = []
        for x, z in grid:
            for dx, dz in ((0,0),(-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)):
                rays.append([x+dx*pitch, side*.15, z+dz*pitch])
        rays = np.asarray(rays)
        directions = np.tile([0., -side, 0.], (len(rays),1))
        hit, ray, _ = local.ray.intersects_location(rays, directions, multiple_hits=False)
        heights = np.full(len(rays), np.nan)
        heights[ray] = side*hit[:,1]
        heights = heights.reshape(-1,5)
        boxes = []
        outer = .085
        for (x,z), h in zip(grid, heights):
            if not np.isfinite(h).all():
                continue
            inner = float(h.max()) + .0003
            minimum = .046 if approach == 'top' else .010
            if inner < minimum or inner > outer-.004:
                continue
            center = [side*x, (inner+outer)/2-jaw_at_contact, z-.0584]
            boxes.append((center, [pitch*.501,(outer-inner)/2,pitch*.501]))
        # Back plate, offset bridge, and stem connect the contact blocks to the joint.
        if approach == 'top':
            boxes += [([0, outer+.003-jaw_at_contact, .077], [.037,.003,.048]),
                      ([0, .0425, .026], [.010,.0405,.004]),
                      ([0, .006, .014], [.008,.006,.014])]
        else:
            boxes += [([0, outer+.003-jaw_at_contact, .033], [.042,.003,.021]),
                      ([0, .0425, .014], [.010,.0405,.003]),
                      ([0, .006, .007], [.008,.006,.007])]
        # The mirrored jaws must remain disjoint even at zero joint opening.
        assert min(center[1]-size[1] for center,size in boxes) >= -1e-12
        result[name] = boxes
    return result


def prototype(approach='side'):
    mesh = trimesh.load(G.ROOT/'objects/B/mesh.stl', force='mesh')
    saved = json.loads((G.ROOT/'objects/B/poses.json').read_text())
    # New inputs keep their preparation pose separately from robot-held targets.
    pose = saved['rest'] if saved.get('schema') == 'cadgrasp_sequence_v1' else saved['poses'][4]
    initial = np.array(pose['T_world_mesh'])
    initial[2,3] -= trimesh.transform_points(mesh.vertices, initial)[:,2].min()
    hand = np.eye(4)
    hand[:3,:3] = [[0,1,0],[0,0,1],[1,0,0]]
    hand[:3,3] = [0,-.14,.065]
    if approach == 'top':
        hand[:3,:3] = [[0,1,0],[1,0,0],[0,0,-1]]
        hand[:3,3] = [0,0,.205]
    boxes = contour_boxes(mesh, initial, hand, approach=approach)
    return mesh, initial, dict(hand=hand, width=.015), boxes


def target_clearance(model, mesh, initial, candidate):
    """Check seated target geometries and reversible hand approach/withdrawal.

    These are proposed new setup states, not replacements for saved baseline poses.
    This is a sampled geometry replay, not a dynamic placement simulation.
    """
    com = trimesh.transform_points(mesh.center_mass[None], initial)[0]
    relative_hand = np.linalg.inv(initial) @ candidate['hand']
    rows = []
    for angle in (40,50,60):
        rotation = Rotation.from_rotvec(TURN_AXIS*np.radians(angle)).as_matrix()
        pose = initial.copy()
        pose[:3,:3] = rotation @ initial[:3,:3]
        pose[:3,3] = com + rotation @ (initial[:3,3]-com) + [.03,0,0]
        pose[2,3] -= trimesh.transform_points(mesh.vertices, pose)[:,2].min()
        vertices = trimesh.transform_points(mesh.vertices, pose)
        ground = vertices[np.abs(vertices[:,2]) < 1e-8]
        mass_center = trimesh.transform_points(mesh.center_mass[None], pose)[0]
        clear = G.geometry_check(model, pose, dict(hand=pose@relative_hand, width=candidate['width']))
        rows.append(dict(angle_deg=angle, T_world_mesh=pose.tolist(),
            hand_approach_open_close_clear=bool(clear), floor_vertex_count=len(ground),
            com_distance_to_ground_vertex_m=float(np.linalg.norm(mass_center[:2]-ground[0,:2]))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exact', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--approach', choices=('side', 'top'), default='side')
    parser.add_argument('--turn', action='store_true', help='Test additional 40/50/60 degree reorientation')
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    if args.video and args.video.suffix.lower() != '.mp4':
        parser.error('--video must be an MP4')
    mesh, initial, candidate, boxes = prototype(args.approach)
    model = G.build('B', exact=args.exact, finger_boxes=boxes)
    clear = G.geometry_check(model, initial, candidate)
    print('approach_and_closure_samples_clear', clear, flush=True)
    if not clear:
        raise SystemExit('Contour prototype does not pass geometry screening')
    if args.turn:
        targets = target_clearance(model, mesh, initial, candidate)
        print(json.dumps(dict(proposed_seated_targets_geometry_only=targets)), flush=True)
        if not all(row['hand_approach_open_close_clear'] for row in targets):
            raise SystemExit('Hand cannot approach/withdraw at a proposed seated target')
    axis = TURN_AXIS if args.turn else None
    result, history = G.simulate(model, initial, candidate, capture=bool(args.video), turn_axis=axis)
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit('Contour prototype does not pass pickup/carry')
    if args.video:
        G.render(model, history, args.video)
        print(args.video, flush=True)


if __name__ == '__main__':
    main()
