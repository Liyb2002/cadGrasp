"""Study five reproducibly selected contact heads per object, without a floor base."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent/'baseline_algo'))
from step3_scheculer import contacts as I
from step2_local_support import circles as P, surface
from step5_connect_support import solids as S
import angles as A

SEED = 20260908
OUTPUT = HERE/'output'


def build_heads(mesh,contacts,depth):
    """Original continuous offset skin, retaining all tiny clipped source cells."""
    offsets,valid=S.G.vertex_offsets(mesh,depth)
    heads=[];owners=[]
    for contact in contacts:
        for face in np.unique(contact['source_faces']):
            face=int(face);which=contact['source_faces']==face
            if not valid[mesh.faces[face]].all():raise ValueError('Invalid outward skin')
            vertices=np.unique(contact['triangles_m'][which].reshape(-1,3),axis=0)
            source=mesh.triangles[face];x=source[1]-source[0];x/=np.linalg.norm(x)
            basis=np.array([x,np.cross(mesh.face_normals[face],x)])
            xy=(vertices-source[0])@basis.T
            polygon=vertices[ConvexHull(xy).vertices]
            np.testing.assert_allclose(surface.area(polygon),contact['triangle_areas_m2'][which].sum(),rtol=1e-7,atol=1e-24)
            heads.append(A.hull_mesh(S.G.head_cell(mesh,polygon,face,offsets)))
            owners.append(dict(candidate_id=contact['candidate_id'],source_face=face))
    return heads,owners


def inputs(name):
    domain, data, report = P.read(name)
    chosen = np.random.default_rng(SEED).choice(np.flatnonzero(data.valid), 5, replace=False)
    contacts = []
    for index in chosen:
        a, b = data.offsets[index:index+2]
        triangles = data.triangles[a:b]
        contacts.append(dict(candidate_id=report['patches'][index]['id'],candidate_index=int(index),
            center_m=data.centers_m[index],center_face=int(data.center_faces[index]),
            radius_m=float(data.radius_m[index]),source_faces=data.source_faces[a:b],
            triangles_m=triangles,triangle_areas_m2=surface.areas(triangles)))
    return domain, contacts, report


def run(name):
    started = time.monotonic()
    domain, contacts, source = inputs(name)
    heads, owners = build_heads(domain.mesh, contacts, source['normal_depth_m'])
    out = OUTPUT/name
    out.mkdir(parents=True, exist_ok=True)
    I.save_contacts(out/'contacts.npz', contacts)
    results = []
    print(name, 'selected', [p['candidate_id'] for p in contacts], flush=True)
    for contact in contacts:
        cells = [h for h, owner in zip(heads, owners) if owner['candidate_id'] == contact['candidate_id']]
        study = A.AngleStudy(domain.mesh, cells)
        result = study.classify(domain.mesh.face_normals[np.unique(contact['source_faces'])])
        result.update(candidate_id=contact['candidate_id'],center_m=contact['center_m'].tolist(),
            radius_m=contact['radius_m'],contact_area_m2=float(contact['triangle_areas_m2'].sum()),
            head_cells=len(cells))
        # Five explicit directions accompany each angular set, selected after classification.
        examples = []
        clear = result['clear_intervals_deg']
        if clear:
            low, high = max(clear, key=lambda x:x[1]-x[0])
            examples.extend([low+.2*(high-low), (low+high)/2, low+.8*(high-low)])
        blocked = result['geometry_blocked_intervals_deg']
        if blocked:
            low, high = max(blocked, key=lambda x:x[1]-x[0])
            examples.append((low+high)/2)
        normal = domain.mesh.face_normals[contact['center_face']]
        examples.append(float(np.rad2deg(np.arctan2(normal[1], normal[0]))%360))
        for angle in [0., 72., 144., 216., 288.]:
            if len(examples) >= 5: break
            if all(abs((angle-x+180)%360-180)>1e-5 for x in examples): examples.append(angle)
        result['example_directions'] = [study.test_angle(x) for x in examples[:5]]
        results.append(result)
        print(name,contact['candidate_id'],'clear',result['clear_intervals_deg'],
              'unresolved degrees',sum(b-a for a,b in result['unresolved_intervals_deg']),
              'queries',study.queries,flush=True)
        I.save(out/'progress.json', dict(object=name, complete=False, contacts=results))
    paths = [P.OUTPUTS/name/'step_1_needs/needs.json',
             P.OUTPUTS/name/'step2_local_support/circles.json',P.OUTPUTS/name/'step2_local_support/circles.npz']
    codes = [Path(__file__),Path(A.__file__),Path(S.__file__),Path(S.G.__file__),Path(P.__file__),Path(surface.__file__)]
    record = dict(object=name,complete=True,selection_seed=SEED,
        selection='Uniformly choose five valid Step 2 contacts before examining insertion outcomes',
        frame='Existing Step 1 world frame; metres. +X=0 deg, +Y=90 deg. a is insertion direction.',
        motion='Head(s)=Head_final-s*a; s decreases from L to 0. Fixed orientation, horizontal straight translation.',
        obstacle='Complete object mesh only; floor, base, connections and other contacts excluded',
        normal_depth_m=source['normal_depth_m'],contacts=results,
        elapsed_seconds=time.monotonic()-started,
        artifacts={'contacts.npz':P.sha256(out/'contacts.npz')},
        provenance=dict(inputs=I.hashes(paths),code=I.hashes(codes)))
    I.save(out/'angles.json',record)
    (out/'progress.json').unlink(missing_ok=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or P.OBJECTS:
        run(name)
