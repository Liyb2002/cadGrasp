"""B / pose 2: individually withdrawable heads can block one rigid translation.

Run with the cadgrasp environment. This reads the current baseline snapshot and
writes only head_sweep.* in this directory; it never changes baseline outputs.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace
import numpy as np
import trimesh
from scipy.optimize import linprog
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools'))
import slide_scene as SC
BASE = SC.BASE
sys.path.insert(0, str(BASE))
from step1.needs import ContinuousNeeds
from step5_connect_support import solids as S
from step2_local_support import insertion_directions as ID, withdrawal as W, render as R

CASE = BASE / 'output/B/pose_2'
IDS = ('C104', 'C118', 'C015')
VIEW = SC.VIEW.tolist()
COLORS = (np.array([229., 135., 53.]), np.array([52., 137., 180.]), np.array([164., 92., 173.]))
BELT = np.array([57., 158., 143.])
INK = '#111111'
PAPER = (255, 255, 255)


def sources():
    paths = [CASE / 'step_1_needs/needs.json',
             CASE / 'step2_local_support/circles.json',
             CASE / 'step2_local_support/circles.npz',
             CASE / 'step2_local_support/insertion_directions.json']
    domain = ContinuousNeeds.read(paths[0])
    circles = json.loads(paths[1].read_text())
    data = SimpleNamespace(**dict(np.load(paths[2])))
    record = json.loads(paths[3].read_text())
    rows = [next(r for r in record['candidates'] if r['candidate_id'] == key) for key in IDS]
    contacts = [ID.candidate(data, circles, r['candidate_index']) for r in rows]
    for c,row in zip(contacts,rows):
        assert not np.intersect1d(c['source_faces'], domain.work_ids).size
        assert np.array_equal(np.unique(domain.mesh.face_normals[c['source_faces']],axis=0),
                              np.asarray(row['local_constraint_normals']))
    return domain, record, rows, contacts, paths


def exact_positive_span(normals, target):
    """Replay a 3-normal certificate exactly for the stored IEEE float values."""
    assert normals.shape == (3, 3)
    source = [[Fraction.from_float(float(v)) for v in row] for row in normals.T]
    wanted = [Fraction.from_float(float(v)) for v in target]
    augmented = [row + [rhs] for row, rhs in zip(source, wanted)]
    for col in range(3):
        pivot = next(row for row in range(col, 3) if augmented[row][col] != 0)
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(3):
            if row == col:
                continue
            multiplier = augmented[row][col]
            augmented[row] = [a - multiplier * b
                              for a, b in zip(augmented[row], augmented[col])]
    weights = [augmented[row][-1] for row in range(3)]
    assert all(weight > 0 for weight in weights)
    reconstructed = [sum(a * w for a, w in zip(row, weights)) for row in source]
    assert reconstructed == wanted
    return dict(arithmetic='Exact Fraction.from_float of stored mesh-normal values',
        exact_weights=[str(weight) for weight in weights],
        exact_reconstruction=[str(value) for value in reconstructed],
        strictly_positive=True, exact_equality_verified=True)


def prove_blocked(rows):
    """Dual certificates positively span R^3, so N d >= 0 implies d = 0.

    This uses all actual outward patch-face normals. It is stronger than an
    empty sampled direction intersection, and does not need work/floor locks.
    """
    N = np.vstack([r['local_constraint_normals'] for r in rows])
    certificates = []
    for target in np.vstack([np.eye(3), -np.eye(3)]):
        dual = linprog(np.ones(len(N)), A_eq=N.T, b_eq=target,
                       bounds=(0, None), method='highs')
        assert dual.success
        ids = np.flatnonzero(dual.x > 1e-12)
        weights = dual.x[ids]
        exact = exact_positive_span(N[ids], target)
        residual = np.max(np.abs(weights @ N[ids] - target))
        assert residual < 1e-10 and np.all(weights >= 0)
        # Independent primal extrema over a bounded cube must all equal zero.
        primal = linprog(-target, A_ub=-N, b_ub=np.zeros(len(N)),
                         bounds=[(-1, 1)] * 3, method='highs')
        assert primal.success and abs(primal.fun) < 1e-10
        certificates.append(dict(target=target.tolist(), normal_indices=ids.tolist(),
            nonnegative_weights=weights.tolist(), reconstruction_error=float(residual),
            primal_maximum=float(-primal.fun), exact_rational_replay=exact))
    return dict(constraint='For every contacting face, n_out dot d >= 0',
        conclusion='Only d=0 satisfies all three heads; no nonzero straight translation',
        applies_to='Finite patch geometry at the installed pose; fixed orientation; exactly stored floating-point normal values',
        needs_floor_or_work_constraints=False, excludes_rotation=False,
        solver='scipy.optimize.linprog(method=highs)', verification_tolerance=1e-10,
        normal_rows=N.tolist(), positive_span_certificates=certificates)


def tube(a, b, radius):
    return trimesh.creation.cylinder(radius=radius, segment=np.array([a, b]), sections=16)


def ribbon(a, b, width=.010, thickness=.005):
    """A broad rectangular strap, with its wide face toward the camera."""
    along = b-a; length=np.linalg.norm(along); along/=length
    across=np.cross([1.,0.,0.],along); across/=np.linalg.norm(across)
    normal=np.cross(along,across)
    transform=np.eye(4);transform[:3,:3]=np.column_stack([normal,across,along])
    transform[:3,3]=(a+b)/2
    return trimesh.creation.box([thickness,width,length],transform=transform)


def build_connector(domain, contacts, heads, rows, record, analyzer):
    """Coloured broad straps join the three heads outside the workpiece."""
    depth=record['normal_depth_m']; vectors=np.asarray(record['direction_catalogue']['vectors'])
    ends=[];meshes=[];segments=[];attachment_records=[]
    x=domain.mesh.bounds[1,0]+.016
    for c,row,head_cells,col in zip(contacts,rows,heads,COLORS):
        n=domain.mesh.face_normals[c['center_face']]
        start=c['center_m']+.82*depth*n
        ids=np.array(row['certified_directions']['ids'])
        d=vectors[ids[np.argmax(vectors[ids,0])]]
        end=start+((x-start[0])/d[0])*d
        ends.append(end);segments.append((start,end))
        # Extend the entire head cross-section; a thin centreline pin can miss
        # the actual offset skin. Each rib overlaps the original head volume.
        length=np.linalg.norm(end-start)+.004
        rib_start=len(meshes)
        for head in head_cells:
            collar=W.H.engine.hull_mesh(np.vstack([head.vertices,head.vertices+.008*d]))
            collar.metadata['figure_color']=col.tolist()
            meshes.append(collar)
            meshes.append(W.H.engine.hull_mesh(np.vstack([head.vertices+.007*d,head.vertices+length*d])))
        attachment_records.append(dict(candidate_id=c['candidate_id'],
            construction='Full head cross-section swept to overlap the broad belt',
            sweep_length_m=float(length),direction=d.tolist(),
            colored_contact_end_length_m=.008,collar_rib_overlap_m=.001,
            rib_part_indices=list(range(rib_start,len(meshes)))))
    # A continuous broad loop, with rounded overlapping bends.
    for a,b in zip(ends,ends[1:]+ends[:1]):
        meshes.append(ribbon(a,b));segments.append((a,b))
    for point in ends:
        cap=trimesh.creation.cylinder(radius=.006,height=.005,sections=32)
        cap.apply_transform(trimesh.geometry.align_vectors([0,0,1],[1,0,0]))
        cap.apply_translation(point);meshes.append(cap)
    volumes=[abs(float((analyzer.obstacle ^ W.solid(m,analyzer.origin,analyzer.scale)).volume()))*analyzer.scale**3 for m in meshes]
    assert max(volumes)<1e-14,volumes
    _,connectivity=S.union_parts([cell for group in heads for cell in group]+meshes,
                                float(domain.mesh.extents.max()))
    assert connectivity['one_solid'],connectivity
    return meshes,dict(interpretation='Broad coloured rigid-belt illustration',
        centerlines_m=[[a.tolist(),b.tolist()] for a,b in segments],
        belt_width_m=.010,belt_thickness_m=.005,
        full_width_attachments=attachment_records,assembled_connectivity=connectivity,
        installed_object_intersection_volumes_m3=volumes,work_faces_not_contacted=True)


def arrow(draw, a, b, color, width=8, head=22):
    a = np.array(a, float); b = np.array(b, float)
    v = b-a; v /= np.linalg.norm(v); side = np.array([-v[1], v[0]])
    draw.line([tuple(a), tuple(b)], fill=tuple(map(int, color)), width=width)
    draw.polygon([tuple(b), tuple(b-head*v+.45*head*side), tuple(b-head*v-.45*head*side)],
                 fill=tuple(map(int, color)))


def text(draw, xy, s, size=28, anchor='mm'):
    draw.text(xy, s, font=R.font(size), fill=INK, anchor=anchor)


def sparse_direction_cover(row, vectors, normal, max_angle_degrees=27.):
    """Cover every certified 3-D ray, without selecting a screen-side fan."""
    ids=np.asarray(row['certified_directions']['ids'],int)
    directions=vectors[ids]
    chosen=[int(np.argmax(directions@normal))]
    # Farthest-angle sampling keeps normal, oblique and tangent directions together.
    while True:
        nearest=np.max(directions@directions[chosen].T,axis=1)
        farthest=int(np.argmin(nearest))
        if nearest[farthest]>=np.cos(np.deg2rad(max_angle_degrees)):
            break
        chosen.append(farthest)
    coverage=np.rad2deg(np.arccos(np.clip(nearest,-1.,1.)))
    return ids[chosen].tolist(),dict(
        method='Greedy farthest-angle cover of the entire certified 3-D direction set',
        candidate_count=len(ids),shown_count=len(chosen),
        max_nearest_arrow_angle_degrees=float(coverage.max()),
        requested_max_angle_degrees=max_angle_degrees,
        screen_side_filter=False,projection_length_filter=False,
        completeness_scope='Every certified catalogue ray is represented within the angular bound; this does not assert unsampled rays are clear')


def surface_arrow_starts(heads, directions, basis):
    """Emit rays from different actual surface vertices around the whole head."""
    points=np.unique(np.concatenate([h.vertices for h in heads]),axis=0)
    projected=points@basis[:2].T
    center=projected.mean(axis=0);relative=projected-center
    scale=max(np.linalg.norm(relative,axis=1).max(),1e-12)
    starts=[];used=[]
    for direction in directions:
        outward=direction@basis[:2].T
        outward/=max(np.linalg.norm(outward),1e-12)
        score=relative@outward+.12*(points@basis[2])
        if used:
            separation=np.linalg.norm(projected[:,None,:]-projected[used][None,:,:],axis=2).min(axis=1)
            score-=.5*scale*np.exp(-separation/(.3*scale))
            score[used]=-np.inf
        pick=int(np.argmax(score));used.append(pick);starts.append(points[pick])
    starts=np.asarray(starts)
    assert len(np.unique(starts,axis=0))==len(starts)
    return starts


def draw_scene(domain, heads, contacts, directions, starts, connector, linked, depth, size=1000):
    basis=R.axes(VIEW)
    extra=np.concatenate([m.vertices for group in heads for m in group]+[m.vertices for m in connector])
    cam=SC.camera(domain,size=size,extra=extra)
    focus,width=cam.focus,cam.width
    tris=[SC.floor(domain),domain.mesh.triangles]
    colors=[np.tile(SC.FLOOR,(2,1)), np.tile(SC.GREY,(len(domain.mesh.faces),1))]
    colors[1][domain.work_ids]=SC.GREEN
    for hs,col in zip(heads,COLORS):
        tt=np.concatenate([h.triangles for h in hs]); tris.append(tt);colors.append(np.tile(col,(len(tt),1)))
    if linked:
        for m in connector:
            tris.append(m.triangles);colors.append(np.tile(m.metadata.get('figure_color',BELT),(len(m.faces),1)))
    unlit=[0,1]
    if not linked:
        # Render actual 3-D arrows with the same depth buffer as the bunny.
        # A ray behind the work region must not be painted over its surface.
        count=sum(len(t) for t in tris)
        for roots,ds,col in zip(starts,directions,COLORS):
            for a,d in zip(roots,ds):
                projected_length=np.linalg.norm(d@basis[:2].T)
                length=.040/max(.55,projected_length)
                # Reverse the checked withdrawal ray: insert toward the head.
                tail=a+length*d;tip=a;neck=tip+.005*d
                stem=tube(tail,neck,.0007)
                # A camera-facing triangle retains a readable arrow tip even
                # when the 3-D direction points nearly toward the viewer.
                screen=-d@basis[:2].T
                screen/=np.linalg.norm(screen)
                along=screen@basis[:2]
                across=np.array([-screen[1],screen[0]])@basis[:2]
                arrow_tip=trimesh.Trimesh(
                    vertices=[tip,tip-.0055*along+.0023*across,
                              tip-.0055*along-.0023*across],
                    faces=[[0,1,2]],process=False)
                for m in (stem,arrow_tip):
                    tris.append(m.triangles);colors.append(np.tile(col,(len(m.faces),1)))
                    unlit.extend(range(count,count+len(m.faces)));count+=len(m.faces)
    im,_=R.raster(np.concatenate(tris),np.concatenate(colors),focus,basis,width,size,unlit=unlit)
    return im


def direction_chart(canvas, rows, vectors):
    """Finite certified rays shown as points; no coloured angular-area claim."""
    draw=ImageDraw.Draw(canvas)
    x0,x1,y0,y1=550,1590,1110,1310
    draw.rectangle([x0,y0,x1,y1],outline='#929292',width=2)
    for x,lab in zip(np.linspace(x0,x1,5),[-180,-90,0,90,180]):
        draw.line([(x,y0),(x,y1)],fill='#deded8',width=1)
        text(draw,(x,y1+24),str(lab),19)
    for e in [0,-20,-40,-60,-80]:
        y=y0-e/90*(y1-y0)
        draw.line([(x0,y),(x1,y)],fill='#deded8',width=1)
        text(draw,(x0-15,y),str(e),19,'rm')
    for row,col in zip(rows,COLORS):
        v=-vectors[row['certified_directions']['ids']]
        az=np.rad2deg(np.arctan2(v[:,1],v[:,0]));el=np.rad2deg(np.arcsin(v[:,2]))
        for a,e in zip(az,el):
            x=x0+(a+180)/360*(x1-x0);y=y0-e/90*(y1-y0)
            draw.ellipse([x-4,y-4,x+4,y+4],fill=tuple(col.astype(int)))
    text(draw,((x0+x1)/2,y1+57),'Insertion azimuth (degrees)',22)
    text(draw,(x0-18,y0-22),'Elevation (degrees)',20,'lm')
    text(draw,(270,1140),'Individually clear:',27)
    for j,(row,col) in enumerate(zip(rows,COLORS)):
        y=1190+j*47
        draw.ellipse([140,y-7,154,y+7],fill=tuple(col.astype(int)))
        text(draw,(168,y),f"{len(row['certified_directions']['ids'])} tested rays",24,'lm')
    text(draw,(1850,1180),'Common rays: 0',28)
    text(draw,(1850,1230),'No common direction',24)
    text(draw,(1850,1270),'also proved continuously',22)


def main():
    domain,record,rows,contacts,paths=sources()
    catalogue=record['direction_catalogue'];vectors=np.asarray(catalogue['vectors'])
    analyzer=W.Analyzer(domain.mesh,record['normal_depth_m'],catalogue)
    connector_rows=rows
    baseline_counts=[len(row['certified_directions']['ids']) for row in rows]
    # This figure illustrates geometric escape, rather than Step2's additional
    # area-averaged work-side preference. Check the entire above-floor catalogue.
    allowed=W.normalize(np.flatnonzero(vectors[:,2]>=-W.NORMAL_TOL))
    rows=[analyzer.analyze(c,allowed=allowed) for c in contacts]
    proof=prove_blocked(rows)
    assert not set.intersection(*(set(r['certified_directions']['ids']) for r in rows))
    heads=[analyzer.heads(c) for c in contacts]
    directions=[];checks=[];selected=[];coverage=[]
    basis=R.axes(VIEW)
    for row,hs,c in zip(rows,heads,contacts):
        normal=domain.mesh.face_normals[c['center_face']]
        chosen,cover=sparse_direction_cover(row,vectors,normal)
        ds=vectors[chosen]
        replay=[analyzer.test(hs,d) for d in ds]
        assert all(c['clear'] for c in replay)
        directions.append(ds);checks.append(replay);selected.append(chosen);coverage.append(cover)
    starts=[surface_arrow_starts(hs,ds,basis) for hs,ds in zip(heads,directions)]
    connector,connector_record=build_connector(domain,contacts,heads,connector_rows,record,analyzer)
    canvas=Image.new('RGB',(2160,1480),PAPER);draw=ImageDraw.Draw(canvas)
    text(draw,(1080,48),'A shared insertion direction is required',40)
    text(draw,(1080,96),'B / pose 2  |  Three surface-matched heads  |  Fixed orientation',25)
    text(draw,(560,155),'(a) Separate heads',32)
    text(draw,(1600,155),'(b) Heads joined rigidly',32)
    canvas.paste(draw_scene(domain,heads,contacts,directions,starts,connector,False,record['normal_depth_m']),(60,180))
    canvas.paste(draw_scene(domain,heads,contacts,directions,starts,connector,True,record['normal_depth_m']),(1100,180))
    draw=ImageDraw.Draw(canvas)
    # Place conclusions on a clean band below the geometry.
    draw.rectangle([0,1000,2160,1480],fill=PAPER)
    text(draw,(560,1025),'Individual insertion directions.',27)
    text(draw,(1600,1025),'One rigid belt must use the same direction.',29)
    direction_chart(canvas,rows,vectors)
    text(draw,(1080,1410),'No shared straight withdrawal; therefore no reverse straight insertion.',31)
    text(draw,(1080,1454),'Continuous check: the three patch-normal cones meet only at zero.',22)
    out=HERE/'head_sweep.png';canvas.save(out)
    evidence=dict(object='B',pose='pose_2',candidate_ids=list(IDS),
        candidate_indices=[r['candidate_index'] for r in rows],
        direction_counts=[len(r['certified_directions']['ids']) for r in rows],
        baseline_direction_counts=baseline_counts,
        direction_scope='Full head/object/floor withdrawal checks; no area-averaged work-side hemisphere prefilter',
        direction_analysis=rows,
        arrows_drawn_panel_a=True,arrows_drawn_panel_b=False,
        arrow_convention='Insertion toward each head, reversing the certified withdrawal ray',
        direction_chart_convention='Insertion vectors: negatives of the certified withdrawal vectors',
        arrow_rendering='3-D stems with camera-facing triangular tips, all depth-tested against the workpiece',
        upward_orange_check=analyzer.test(heads[0],np.array([0.,0.,1.])),
        horizontal_catalogue_counts=[int(np.sum(abs(vectors[r['certified_directions']['ids'],2])<1e-12)) for r in rows],
        direction_coverage=coverage,
        display_note='Sparse arrows cover the full certified 3-D direction set, including normal-like, oblique and sideways directions. Lengths are adjusted for legibility, preserving the actual 3-D directions.',
        displayed_direction_ids=selected,displayed_withdrawal_directions=[d.tolist() for d in directions],
        displayed_insertion_directions=[(-d).tolist() for d in directions],
        independent_full_ray_checks=checks,continuous_local_translation_proof=proof,
        rigid_link=connector_record, camera_vector=VIEW, blocked_direction_crosses=False,
        arrows_per_head=[len(d) for d in directions], arrow_surface_end_points_m=[s.tolist() for s in starts],
        arrow_origin_convention='Arrows end at distinct real head-surface vertices; tails lie outside along the certified withdrawal rays',
        head_locations=['Chest, facing approximately opposite the blue contact','Forward lower chest','Broad rear face of the upright ear, below the tip'],
        labels_in_figure='No candidate IDs; colours identify heads',
        inputs={str(p.relative_to(HERE.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (HERE/'head_sweep.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(out)
    print(IDS, selected, 'continuous local cone = {0}; all displayed rays replayed clear')

if __name__=='__main__':main()
