"""Render one parameter slide with the current reuse fixture and approved formulas."""
from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import json
import sys
from types import SimpleNamespace
import numpy as np
import trimesh
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from scipy.spatial import ConvexHull

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(OUT.parent/'tools'))
import slide_scene as S
from step0_pose_selection.whole_assembly import pressure_centers

INK,MUTED='#20303e','#667580'
TEAL,BLUE=(41,145,133),(37,112,188)
TASK_COLORS=[(209,116,43),(153,90,179)]
VIEW=np.array([1.05,-1.25,.8])
SIZE=1040
STROKE=.09
FIGURE=OUT/'optimization_overview.png'
RECORD=OUT/'optimization_overview.json'
APPROVED_PANELS=OUT/'approved_panels.npz'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(draw,xy,label,size=34,color=INK,anchor='la'):
    S.text(draw,xy,label,size,color,anchor)


def formula(canvas,xy,source,size=26,color=INK):
    buf=BytesIO()
    math_to_image('$'+source+'$',buf,prop=FontProperties(size=size),dpi=144,format='png',color=color)
    buf.seek(0);stamp=Image.open(buf).convert('RGBA')
    canvas.paste(stamp,xy,stamp)


def domain_for(case, work=True):
    return SimpleNamespace(mesh=case['object'], work_ids=case['work_ids'] if work else np.array([], int))


def camera(case, extra=(), size=SIZE, view=VIEW):
    return S.camera(domain_for(case), size=size, extra=extra, ground_points=extra, view=view)


def layer(pic, cam, meshes, colors, opacity=1., flat=False):
    triangles=np.concatenate([m.triangles if hasattr(m,'triangles') else m for m in meshes])
    palette=np.concatenate([np.tile(c,(len(m.faces) if hasattr(m,'faces') else len(m),1)) for m,c in zip(meshes,colors)])
    image,ids=S.R.raster(triangles,palette,cam.focus,cam.basis,cam.width,cam.size,
                        unlit=range(len(triangles)) if flat else [])
    if opacity<1:
        image=Image.blend(pic,image,opacity)
    pic.paste(image,(0,0),Image.fromarray(np.uint8(ids>=0)*255))


def reuse_ground_data():
    """Reuse saved physical demands after checking the two displayed task poses.

    The reuse mesh is rounded to seven decimals. Validate its rigid translation
    against the exact saved mesh and retain exact coordinates for the mechanics.
    No baseline search or reuse builder is executed.
    """
    import trimesh
    source=OUT.parent/'reuse/data.js'
    layout_path=OUT.parent/'reuse/layout.json'
    artwork=json.loads(source.read_text().removeprefix('window.REUSE_DATA=').strip().removesuffix(';'))
    layout=json.loads(layout_path.read_text())
    source_paths={source,layout_path}
    cases=[];records=[]
    for index,(pose,placement) in enumerate(zip(artwork['poses'][:2],layout['poses'][:2])):
        name=pose['source'].split(' / ')[1]
        folder=OUT.parent/'baseline_algo/output/B'/name
        needs_path=folder/'step_1_needs/needs.json'
        floor_path=folder/'step0_pose_selection/floor_contact.npz'
        setup_path=ROOT/'objects/B/tasks'/name/'setup.npz'
        contact_path=folder/'step3_scheculer/final_contacts.npz'
        source_paths.update((needs_path,floor_path,setup_path,contact_path))
        data=json.loads(needs_path.read_text())
        vertices=np.asarray(data['geometry']['vertices_m'])
        faces=np.asarray(data['geometry']['faces'])
        work=np.asarray(data['geometry']['work_face_ids'])
        com=np.asarray(data['frame']['moment_origin_m'])
        transform=np.asarray(data['frame']['T_world_mesh'])
        with np.load(setup_path) as setup:
            np.testing.assert_allclose(transform,setup['T_world_mesh'],atol=1e-12)
            np.testing.assert_allclose(com,setup['com_m'],atol=1e-12)
            np.testing.assert_array_equal(work,np.flatnonzero(setup['work_faces']))
            np.testing.assert_allclose([data['load']['K'],data['load']['cone_half_deg']],
                                       [setup['K'],setup['cone_half_deg']],atol=1e-12)
        np.testing.assert_allclose(pose['objectR'],transform[:3,:3],atol=1e-12)
        np.testing.assert_array_equal(faces,np.asarray(pose['object']['f']).reshape(-1,3))
        np.testing.assert_array_equal(work,pose['work'])
        offset=np.r_[np.asarray(placement['xy'])-vertices.mean(0)[:2],0.]
        placed=vertices+offset
        np.testing.assert_allclose(placed,np.asarray(pose['object']['v']).reshape(-1,3),atol=5.1e-8,rtol=0)
        with np.load(floor_path) as saved:
            np.testing.assert_allclose(saved['moment_origin_m'],com,atol=1e-12)
            points=saved['floor_demands_xy_m'].copy()+offset[:2]
            replay,_=pressure_centers(saved['load_wrenches'],com+offset)
        np.testing.assert_allclose(points,replay,atol=1e-12)
        hull=ConvexHull(points)
        shown=np.unique(np.r_[np.linspace(0,len(points)-1,min(500,len(points)),dtype=int),hull.vertices])
        with np.load(contact_path) as saved:
            contact_triangles=saved['triangles_m'].copy()
            source_faces=saved['source_faces'].copy()
            offsets=saved['offsets'].copy()
        assert source_faces.min()>=0 and source_faces.max()<len(faces)
        # The saved contact regions are clipped triangles, not entire mesh faces.
        parents=vertices[faces[source_faces]]
        for corner in range(3):
            weights=trimesh.triangles.points_to_barycentric(parents,contact_triangles[:,corner])
            np.testing.assert_allclose((parents*weights[:,:,None]).sum(1),contact_triangles[:,corner],atol=1e-10)
            assert weights.min()>-1e-7 and weights.max()<1+1e-7
        cases.append(dict(pose=name,object=trimesh.Trimesh(vertices=placed,faces=faces,process=False),
            work_ids=work,fixture_R=np.asarray(pose['fixtureR']),fixture_t=np.asarray(pose['fixtureT']),
            contact_triangles=contact_triangles+offset,contact_source=str(contact_path.relative_to(ROOT)),
            contact_source_face_ids=source_faces.tolist(),contact_region_count=len(offsets)-1))
        records.append(dict(task=index+1,pose=name,kind='saved_physical_load_samples',
            source=str(floor_path.relative_to(ROOT)),needs_source=str(needs_path.relative_to(ROOT)),
            setup_source=str(setup_path.relative_to(ROOT)),translation_to_reuse_world_m=offset.tolist(),
            reuse_rotation_work_area_floor_height_and_com_checked=True,
            pressure_centers_replay_checked=True,point_count=len(points),display_count=len(shown),
            displayed_xy_m=points[shown].tolist(),hull_xy_m=points[hull.vertices].tolist()))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(source_paths)}
    return cases,records,hashes



def paste_row(pic,cell,index,symbol):
    occupied=np.any(np.asarray(cell)<250,axis=2)
    ys,xs=np.nonzero(occupied)
    cell=cell.crop((max(0,xs.min()-12),max(0,ys.min()-12),min(SIZE,xs.max()+13),min(SIZE,ys.max()+13)))
    height=SIZE//2
    cell.thumbnail((SIZE-125,height-32),Image.Resampling.LANCZOS)
    pic.paste(cell,((SIZE-cell.width)//2,index*height+(height-cell.height)//2))
    formula(pic,(16,index*height+14),symbol,25,color='#bd343d')


def contact_panel(cases):
    pic=Image.new('RGB',(SIZE,SIZE),'white')
    for i,c in enumerate(cases):
        cam=camera(c)
        cell=Image.new('RGB',(SIZE,SIZE),'white')
        layer(cell,cam,[S.floor(domain_for(c))],[S.FLOOR],.6,True)
        layer(cell,cam,[c['object']],[S.GREY],.30)
        layer(cell,cam,[c['contact_triangles']],[BLUE],1.,True)
        paste_row(pic,cell,i,rf'A_{{\rm obj}}^{i+1}')
    return pic


def ground_panel(cases,records):
    pic=Image.new('RGB',(SIZE,SIZE),'white')
    for i,(c,record) in enumerate(zip(cases,records)):
        points=np.asarray(record['displayed_xy_m'])
        hull=np.asarray(record['hull_xy_m'])
        cloud=np.c_[np.vstack([points,hull]),np.zeros(len(points)+len(hull))]
        cam=camera(c,cloud)
        cell=Image.new('RGB',(SIZE,SIZE),'white')
        layer(cell,cam,[S.floor(domain_for(c),cloud)],[S.FLOOR],.7,True)
        layer(cell,cam,[c['object']],[S.GREY],.27)
        outline=cam.project(np.c_[hull,np.zeros(len(hull))])[:,:2]
        overlay=Image.new('RGBA',cell.size,(0,0,0,0));pen=ImageDraw.Draw(overlay)
        color=TASK_COLORS[i]
        pen.polygon([tuple(p) for p in outline],fill=(*color,25))
        pen.line([tuple(p) for p in np.vstack([outline,outline[0]])],fill=(*color,255),width=3)
        cell=Image.alpha_composite(cell.convert('RGBA'),overlay).convert('RGB')
        pen=ImageDraw.Draw(cell)
        uv=cam.project(np.c_[points,np.zeros(len(points))])[:,:2]
        for x,y in uv:pen.ellipse((x-3,y-3,x+3,y+3),fill=color)
        paste_row(pic,cell,i,rf'A_{{\rm floor}}^{i+1}')
    return pic


def fixture_meshes():
    source=OUT.parent/'reuse/data.js'
    data=json.loads(source.read_text().removeprefix('window.REUSE_DATA=').strip().removesuffix(';'))
    packed=[data['fixture']]
    return [trimesh.Trimesh(vertices=np.asarray(m['v']).reshape(-1,3),
            faces=np.asarray(m['f']).reshape(-1,3),process=False) for m in packed]


def current_reuse_cases():
    """Read the new fixture placements against the pair's frozen task inputs."""
    source=OUT.parent/'reuse/data.js'
    data=json.loads(source.read_text().removeprefix('window.REUSE_DATA=').strip().removesuffix(';'))
    assert data['schema']=='saved_pose1_3_fixture_v1'
    assert [p['source'] for p in data['poses']]==['B / pose_1','B / pose_3']
    hashes={str(source.relative_to(ROOT)):sha(source)}
    shape_path=ROOT/data['sourceShape']
    assert sha(shape_path)==data['sourceShapeSha256']
    hashes[str(shape_path.relative_to(ROOT))]=sha(shape_path)
    cases=[]
    for pose in data['poses'][:2]:
        name=pose['source'].split(' / ')[1]
        needs_path=OUT.parent/'baseline_algo/output/B/pose1+3/step_1_needs'/name/'needs.json'
        saved=json.loads(needs_path.read_text())
        np.testing.assert_allclose(pose['objectR'],np.asarray(saved['frame']['T_world_mesh'])[:3,:3],atol=1e-12)
        np.testing.assert_array_equal(pose['work'],saved['geometry']['work_face_ids'])
        np.testing.assert_array_equal(np.asarray(pose['object']['f']).reshape(-1,3),saved['geometry']['faces'])
        np.testing.assert_allclose(np.asarray(pose['object']['v']).reshape(-1,3),saved['geometry']['vertices_m'],atol=5.1e-11,rtol=0)
        hashes[str(needs_path.relative_to(ROOT))]=sha(needs_path)
        cases.append(dict(pose=name,
            object=trimesh.Trimesh(vertices=np.asarray(pose['object']['v']).reshape(-1,3),
                faces=np.asarray(pose['object']['f']).reshape(-1,3),process=False),
            work_ids=np.asarray(pose['work'],int),fixture_R=np.asarray(pose['fixtureR']),
            fixture_t=np.asarray(pose['fixtureT']),insertion_direction=-np.asarray(pose['withdrawalDirection'])))
    return cases,hashes


def approved_panels(cases,hashes):
    """Keep approved columns 1/2 reproducible after their historical data is removed.

    Cache only the two diagram cells, with their original provenance. Formula
    pixels are still rendered below from the unchanged expressions. Once saved,
    this cache can regenerate the slide without any existing PNG or report.
    """
    if not APPROVED_PANELS.exists():
        if FIGURE.exists() and RECORD.exists():
            metadata=json.loads(RECORD.read_text())
            metadata['task_poses']=metadata.get('approved_columns_1_2_task_poses',metadata['task_poses'])
            metadata['source_sha256']=metadata.get('approved_panels_historical_source_sha256',metadata['source_sha256'])
            with Image.open(FIGURE) as saved:
                if saved.size!=(4400,1660):
                    raise ValueError('Approved slide must have its original 4400 x 1660 layout')
                panels=[np.asarray(saved.convert('RGB').crop((x,350,x+SIZE,350+SIZE)))
                        for x in (30,1130)]
        else:
            original,ground,original_hashes=reuse_ground_data()
            panels=[np.asarray(contact_panel(original)),np.asarray(ground_panel(original,ground))]
            metadata=dict(task_poses=[c['pose'] for c in original],source_sha256=original_hashes,
                ground_demands=ground,object_contacts=[dict(task=i+1,pose=c['pose'],
                    source=c['contact_source'],region_count=c['contact_region_count'],
                    source_face_ids=c['contact_source_face_ids'],
                    source_triangles_checked_against_current_mesh=True,
                    scope='Actual clipped regions selected by the independent single-pose baseline; not solved for V.')
                    for i,c in enumerate(original)])
        metadata={key:metadata[key] for key in
                  ('task_poses','source_sha256','ground_demands','object_contacts')}
        np.savez_compressed(APPROVED_PANELS,contacts=panels[0],ground=panels[1],
                            metadata=np.array(json.dumps(metadata)))
    with np.load(APPROVED_PANELS,allow_pickle=False) as saved:
        panels=[Image.fromarray(saved[key]) for key in ('contacts','ground')]
        metadata=json.loads(str(saved['metadata']))
    # Columns 1/2 deliberately retain their approved historical pose_2/pose_3
    # pixels; columns 3/4 now illustrate the current pose_1/pose_3 fixture.
    for source,digest in metadata['source_sha256'].items():
        if source.startswith('objects/'):
            assert sha(ROOT/source)==digest, 'Approved task geometry changed'
    hashes[str(APPROVED_PANELS.relative_to(ROOT))]=sha(APPROVED_PANELS)
    return *panels,metadata


def structure_panel():
    """One illustrative rigid fixture, without a separate base or dock.

    Reuse artwork supplies the shape only, not a solution for columns 1/2.
    Reading data.js avoids executing the geometry builder or changing its assets.
    """
    meshes=fixture_meshes()
    vertices=np.vstack([m.vertices for m in meshes])
    basis=S.R.axes(np.array([-1.,-1.,.82]))
    projected=vertices@basis.T
    bounds=np.array([projected.min(0),projected.max(0)])
    cam=S.Camera(bounds.mean(0)@basis,basis,1.3*float(np.max(bounds[1,:2]-bounds[0,:2])),SIZE)
    pic=Image.new('RGB',(SIZE,SIZE),'white')
    layer(pic,cam,meshes,[BLUE]*len(meshes))
    return pic


def structure_formulas(canvas):
    formula(canvas,(2250,1370),r'A_{\rm obj}^{(k)}\subseteq\partial V,\quad\forall k',26)
    formula(canvas,(2250,1475),r'A_{\rm floor}^{(k)}\subseteq\partial V,\quad\forall k',26)
    formula(canvas,(2250,1580),r'\operatorname{int}(V)\cap\operatorname{int}(\mathrm{Obj}_k)=\varnothing,\quad\forall k',20)


def direction_panel(cases):
    pic=Image.new('RGB',(SIZE,SIZE),'white')
    shared=fixture_meshes()
    records=[]
    for i,c in enumerate(cases):
        rotation,translation=c['fixture_R'],c['fixture_t']
        transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=translation
        parts=[m.copy().apply_transform(transform) for m in shared]
        direction=c['insertion_direction']
        np.testing.assert_allclose(direction[2],0,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(direction),1,atol=1e-12)
        # Begin with complete object/fixture separation along the saved axis.
        stroke=max(STROKE,float((c['object'].vertices@direction).max()
                   -min((m.vertices@direction).min() for m in parts)+.012))
        pre=c['object'].copy().apply_translation(-stroke*direction)
        assert (pre.vertices@direction).max() < min((m.vertices@direction).min() for m in parts)
        end=c['object'].bounds.mean(0)
        end[2]=max(c['object'].bounds[1,2],*(m.bounds[1,2] for m in parts))+.02
        arrow_world=np.array([end-stroke*direction,end])
        cloud=np.vstack([c['object'].vertices,pre.vertices,arrow_world]+[m.vertices for m in parts])
        # Look into the opening, following reuse's consistent fixture-relative
        # azimuth; the old fixed world camera hides the new seated object.
        yaw=np.arctan2(direction[1],direction[0])
        view=np.array([np.cos(yaw+2.1),np.sin(yaw+2.1),.5])
        cam=camera(c,cloud,view=view)
        cell=Image.new('RGB',(SIZE,SIZE),'white')
        layer(cell,cam,[S.floor(domain_for(c),cloud)],[S.FLOOR],.55,True)
        layer(cell,cam,[pre],[S.GREY],.26)
        # One final depth test draws the seated object and the complete fixed V.
        final,_,ids=S.render(domain_for(c,work=False),parts=[(m,BLUE) for m in parts],cam=cam,ground=False)
        cell.paste(final,(0,0),Image.fromarray(np.uint8(ids>=0)*255))
        pen=ImageDraw.Draw(cell)
        arrow=cam.project(arrow_world)[:,:2]
        geometry=np.vstack([c['object'].vertices,pre.vertices]+[m.vertices for m in parts])
        top=cam.project(geometry)[:,1].min()
        arrow[:,1]+=top-28-arrow[:,1].max()
        S.arrow(pen,arrow[0],arrow[1],'white',11,24)
        S.arrow(pen,arrow[0],arrow[1],BLUE,6,19)
        paste_row(pic,cell,i,rf'd_{i+1}')
        records.append(dict(task=i+1,pose=c['pose'],fixture_rotation_world=rotation.tolist(),
            fixture_translation_world_m=translation.tolist(),direction_world=direction.tolist(),
            direction_fixture=(rotation.T@direction).tolist(),preinsertion_translation_world_m=(-stroke*direction).tolist(),
            stroke_m=stroke,fixed_body='complete V',moving_body='object only',
            direction_source='negative saved reuse withdrawalDirection (original Step3/Step5 direction)',horizontal_world=True,
            camera_view_world=view.tolist(),
            sweep_collision_free_verified=False,fully_separated_start_verified=True))
    return pic,records


def main():
    cases,hashes=current_reuse_cases()
    first,second,approved=approved_panels(cases,hashes)
    ground_records=approved['ground_demands']
    third=structure_panel()
    fourth,directions=direction_panel(cases)
    entries=[('1   Object contacts',r'A_{\rm obj}^{k}',BLUE,first),
             ('2   Ground contacts',r'A_{\rm floor}^{k}',TEAL,second),
             ('3   Shared support geometry',r'V',BLUE,third),
             ('4   Insertion directions',r'd_k',BLUE,fourth)]
    canvas=Image.new('RGB',(4400,1660),'white');ink=ImageDraw.Draw(canvas)
    for i,(title,symbol,color,pic) in enumerate(entries):
        x=1100*i
        text(ink,(x+50,200),title,40,color)
        formula(canvas,(x+50,265 if i<2 else 285),symbol,28 if i<2 else 30,color='#bd343d')
        canvas.paste(pic,(x+30,350))
        if i<3:ink.line([(x+1090,200),(x+1090,1620)],fill='#e1e6e8',width=2)
    # Keep every approved force, geometry and insertion formula unchanged.
    formula(canvas,(50,1405),r'\int_{\rm supp\_obj}\; \mathbf{F}_{\rm supp}\,dA=mg\,\hat z-\mathbf{F}_{\rm push}',24)
    formula(canvas,(50,1510),r'\int_{\rm supp\_obj}\; r_{\rm supp}\times\mathbf{F}_{\rm supp}\,dA=-r_{\rm push}\times\mathbf{F}_{\rm push}',22)
    formula(canvas,(1150,1405),r'\int_{\rm sys\_floor}\; F_{\rm supp}\,r_{\rm supp}\times\hat z\,dA=-r_{\rm push}\times\mathbf{F}_{\rm push}',22)
    structure_formulas(canvas)
    formula(canvas,(3350,1405),r'\operatorname{Sweep}(\mathrm{Obj}_k,d_k)\cap\operatorname{int}(V)=\varnothing',22)
    formula(canvas,(3350,1510),r'\forall k',24)
    compared=FIGURE.exists()
    if compared:
        with Image.open(FIGURE) as saved:
            before=np.asarray(saved.convert('RGB'))
        after=np.asarray(canvas)
        # Only the two geometry cells may change; all formulas and columns 1/2 stay exact.
        unchanged=np.ones(before.shape[:2],bool)
        unchanged[:180]=False  # Retired page title; column headings start at y=200.
        for x in (2230,3330):unchanged[350:1370,x:x+SIZE]=False
        np.testing.assert_array_equal(before[unchanged],after[unchanged])
    assert hashes=={p:sha(ROOT/p) for p in hashes}
    source=OUT.parent/'reuse/data.js'
    result=dict(schema='shared_fixture_two_pose_design_groups',object='B',task_count=2,
        task_poses=[c['pose'] for c in cases],design_unknowns=['A_obj^k','A_floor^k','V','d_k'],
        approved_columns_1_2_task_poses=approved['task_poses'],
        updated_columns_3_4_task_poses=[c['pose'] for c in cases],
        source_sha256=hashes,source_inputs_unchanged=True,
        parameters_are_solution_objects_not_algorithm_tuning=True,
        equilibrium_equation_source='slides/tools/combined_equations.py',
        approved_formulas_preserved_pixel_exact=compared,
        equilibrium_formulas_preserved_pixel_exact=compared,
        approved_panels_source=str(APPROVED_PANELS.relative_to(ROOT)),
        approved_panels_historical_source_sha256=approved['source_sha256'],
        object_contacts=approved['object_contacts'],
        ground_demands=ground_records,ground_common_hull_used=False,
        sampled_hull_is_not_actual_ground_contact_region=True,
        ground_calculation='Approved historical panel preserved; its original pressure_centers replay and source hashes are retained.',
        structure_geometry_source=str(source.relative_to(ROOT)),structure_geometry_sha256=sha(source),
        shared_fixture_geometry_unchanged_between_poses=True,
        structure_geometry_solves_displayed_contacts=False,
        structure_geometry_solves_displayed_demands=False,
        structure_nonpenetration_verified=False,support_bearing_verified=False,
        direction_figure='Two poses of the same complete fixed V; only the object translates horizontally from pale start to seated state.',
        insertion=directions,insertion_feasibility_verified=False,
        sweep_scope='Finite object sweep from seated geometry minus stroke*d to the seated geometry; common fixture coordinates; boundary contact allowed.',
        sweep_constraint='Sweep(Obj_k,d_k) intersect int(V) is empty for every task k; a design requirement, not a certificate.',
        material_efficiency_assumed=False,
        union_baseline='Solve each pose independently, align and merge geometries, then optimize and revalidate.',
        notation=dict(V='One shared rigid support solid in fixture coordinates.',
            A_obj_k='Selected object contact region for task k; expressed in fixture coordinates in column 3.',
            A_floor_k='Selected actual floor contact region for task k, not the demand cloud; expressed in fixture coordinates in column 3.',
            Obj_k='Task-k object solid already mapped into the common fixture frame.',
            d_k='Insertion direction in fixture coordinates; world direction is fixture_R @ d_k.',
            r_supp='Contact position minus object COM.',r_push='Process force application position minus object COM.',
            supp_obj='All object contacts, including original object-floor contacts.',
            sys_floor='All whole-assembly floor contacts, including original object-floor contacts.'))
    canvas.save(FIGURE)
    with Image.open(FIGURE) as saved:saved.verify()
    RECORD.write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'overview.png').unlink(missing_ok=True)
    print(FIGURE,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
