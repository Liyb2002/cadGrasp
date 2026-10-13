"""Independently check illustrated directions by solid swept-volume intersections."""
import argparse
from pathlib import Path
import numpy as np
import manifold3d as M
import study as R
import angles as A


def solid(mesh, origin, scale):
    result = M.Manifold(M.Mesh64(np.asarray((mesh.vertices-origin)/scale, np.float64),
                                np.asarray(mesh.faces, np.uint64)))
    if result.status() != M.Error.NoError:
        raise ValueError(f'Manifold rejected mesh: {result.status()}')
    return result


def run(name):
    out = R.OUTPUT/name
    record = R.I.check_report(out/'angles.json')
    domain, contacts, _ = R.inputs(name)
    heads, owners = R.build_heads(domain.mesh, contacts, record['normal_depth_m'])
    scale = float(domain.mesh.extents.max());origin = domain.mesh.bounds.mean(axis=0)
    obstacle = solid(domain.mesh, origin, scale)
    checked=[]
    geometry={}
    for patch, row in zip(contacts, record['contacts']):
        assert patch['candidate_id'] == row['candidate_id']
        cells = [h for h, owner in zip(heads, owners) if owner['candidate_id']==patch['candidate_id']]
        cases=[dict(example_index=i,**sample) for i,sample in enumerate(row['example_directions'])]
        # These extra probes challenge the claimed interval interiors, independently
        # of the five directions chosen for visual explanation.
        for low, high in row['clear_intervals_deg']:
            for fraction in [.1,.9]:
                cases.append(dict(example_index=None,angle_deg=low+fraction*(high-low),clear=True))
        for case in cases:
            maximum=0.;hit_geometry=None
            a=A.direction(case['angle_deg'])
            for head in cells:
                sweep=A.hull_mesh(np.vstack([head.vertices,head.vertices-row['length_m']*a]))
                hit=obstacle ^ solid(sweep,origin,scale)
                volume=abs(hit.volume())*scale**3
                if volume>maximum:
                    maximum=volume
                    data=hit.to_mesh64()
                    hit_geometry=(np.asarray(data.vert_properties[:,:3])*scale+origin)[np.asarray(data.tri_verts)]
            tolerance=1e-11*scale**3
            if case['clear'] and maximum>tolerance:
                raise AssertionError((name,patch['candidate_id'],case['angle_deg'],'claimed clear',maximum,tolerance))
            if not case['clear'] and maximum<=tolerance:
                raise AssertionError((name,patch['candidate_id'],case['angle_deg'],'blocked direction lacks independent volume',maximum,tolerance))
            if case['example_index'] is not None:
                key=f"{patch['candidate_id']}_{case['example_index']}"
                geometry[key]=hit_geometry if hit_geometry is not None else np.empty((0,3,3))
            checked.append(dict(candidate_id=patch['candidate_id'],**case,
                maximum_cell_intersection_m3=maximum,volume_tolerance_m3=tolerance))
        print(name,patch['candidate_id'],'independent sweep checks passed',flush=True)
    np.savez_compressed(out/'example_intersections.npz',**geometry)
    result=dict(object=name,passed=True,angles_sha256=R.P.sha256(out/'angles.json'),
        audit_code_sha256=R.P.sha256(__file__),checks=checked,
        intersection_geometry_sha256=R.P.sha256(out/'example_intersections.npz'),
        method='Independent manifold3d float64 solid intersections of complete convex-cell sweeps',
        scope='All five illustrative directions per contact and additional interior probes; this audit does not replace the angular envelope certificate.')
    R.I.save(out/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or R.P.OBJECTS:run(name)
