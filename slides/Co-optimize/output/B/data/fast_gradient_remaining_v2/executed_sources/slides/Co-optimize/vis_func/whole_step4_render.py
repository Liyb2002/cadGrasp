"""Reuse Step4.1 drawing; render saved whole layouts as triangle meshes."""
import _bootstrap
from collections import OrderedDict
from co_common import *
from whole_search.common import transform_mesh,transform_solid,unpack_solid,result_mesh
from whole_search.model import Layout
from step41_render import draw_pose
from work_access_render import Panel,render_scene,Image
from PIL import ImageOps


def sheet(tiles,path,columns=None):
    columns=columns or (2 if len(tiles)==4 else min(3,len(tiles)))
    size=(760,700);rows=(len(tiles)+columns-1)//columns
    picture=Image.new('RGB',(columns*size[0],rows*size[1]),'white')
    for i,tile in enumerate(tiles):
        tile=ImageOps.contain(tile,(size[0]-20,size[1]-20),Image.Resampling.LANCZOS)
        picture.paste(tile,((i%columns)*size[0]+(size[0]-tile.width)//2,(i//columns)*size[1]+(size[1]-tile.height)//2))
    picture.save(path)


def render_initialization(model,actual,out):
    layout=actual['layout'];seed=model.registered_seed;final=result_mesh(actual)
    direction_tiles=[];sweep_tiles=[];final_tiles=[];rows=[]
    for k in layout.active:
        d=layout.directions[k];T=model.native[k]
        nominal=model.clearance.sweep(model.length*d,padded=False)
        kept=seed-nominal;removed=seed-kept
        display=unpack_solid(model.clearance.sweep(.10*d,padded=False))
        direction_tiles.append(draw_pose(model.mesh,unpack_solid(kept),unpack_solid(removed),T,arrow=True,direction=d))
        sweep_tiles.append(draw_pose(model.mesh,unpack_solid(kept),unpack_solid(removed),T,arrow=True,sweep=display,direction=d))
        final_tiles.append(draw_pose(model.mesh,final,trimesh.Trimesh(),T))
        rows.append(dict(pose=model.poses[k],direction_fixture=d.tolist(),direction_world=(T[:3,:3]@d).tolist(),
            own_nominal_removed_cm3=material_volume(removed)*1e6,full_length_m=model.length))
    sheet(direction_tiles,out/'exit_directions.png',len(direction_tiles))
    sheet(sweep_tiles,out/'exit_sweeps.png',len(sweep_tiles))
    sheet(final_tiles,out/'final_result.png')
    record=dict(complete=True,presentation_only=True,states=rows,text_or_labels=False,
        wheel='existing step41_render.draw_pose',support_source='new Step3.1',
        own_red_cut_policy='this pose\'s complete nominal body sweep from the immutable new Step3.1 seed',
        final_support_policy='actual all-pose contact-core/1%-clearance geometry',
        display_sweep_length_m=.10,mechanics_rerun=False,geometry_changed=False,
        provenance=provenance([model.initialization_report,out/'support.obj'],
            [Path(__file__),HERE/'vis_func/step41_render.py']),
        artifacts={'../'+file:I.sha256(out/file) for file in ['exit_directions.png','exit_sweeps.png','final_result.png']})
    save(out/'data/render.json',record);I.check_report(out/'data/render.json')


def read_layout(path):
    with np.load(path) as z:
        return Layout(z['placements'].copy(),z['directions'].copy(),z['hosts'].copy(),tuple(map(int,z['active'])))


class NominalBuilder:
    """Same whole layout/raw material domains; no force claim for process tiles."""
    def __init__(self,model):self.model=model;self.parts=OrderedDict();self.shapes={}
    def part(self,layout,k,kind):
        m=self.model;q=layout.placements[k];d=q[:3,:3].T@layout.directions[k]
        cap=m.work_length(layout,k) if kind=='work' else None
        key=(kind,k if kind in ['wrap','work'] else None,q.tobytes(),d.tobytes() if kind=='sweep' else cap)
        if key not in self.parts:
            raw=m.wraps[k] if kind=='wrap' else m.work_solid(layout,k) if kind=='work' else m.clearance.sweep(m.length*d,padded=False)
            self.parts[key]=transform_solid(raw,q)
        return self.parts[key]
    def solid(self,layout):
        if layout.key() not in self.shapes:
            seed=(self.model.registered_seed if all(np.allclose(layout.placements[k],np.eye(4),atol=1e-12,rtol=0) for k in layout.active)
                  else union([self.part(layout,k,'wrap') for k in layout.active]))
            self.shapes[layout.key()]=seed-union([self.part(layout,k,kind) for k in layout.active for kind in ['work','sweep']])
        return self.shapes[layout.key()]


def layers_for_pose(model,layout,support,k,display_transform=None):
    q=layout.placements[k];T=model.native[layout.hosts[k]] if display_transform is None else display_transform
    body=transform_mesh(model.mesh,q);body.apply_transform(T)
    fixture=transform_mesh(support,T)
    working=trimesh.Trimesh(model.mesh.vertices.copy(),model.mesh.faces[model.tasks[k].domain.work_ids].copy(),process=False)
    working.apply_transform(T@q)
    return [(working,'#FFBD66'),(body,'#A5ADB5'),(fixture,'#2A91D2')]


def raster_tile(layers,size=(760,700)):
    points=np.vstack([mesh.vertices for mesh,color in layers if len(mesh.vertices)])
    panel=Panel(None,size=size,points=points,elevation=35.26438968,azimuth=-45)
    return render_scene(panel,layers)


def render_search(model,out,final_result):
    rows=json.loads((out/'process.json').read_text());builder=NominalBuilder(model)
    paths=[];tiles=[];metadata=[];directory=out/'mesh_states';directory.mkdir(exist_ok=True)
    previous=None;reference=model.native[0]
    for row in rows:
        layout=read_layout(out/row['layout']);solid=builder.solid(layout);mesh=unpack_solid(solid)
        path=directory/f'{row["index"]:03d}.npz';np.savez_compressed(path,vertices=mesh.vertices,faces=mesh.faces)
        k=layout.active[0]
        layers=layers_for_pose(model,layout,mesh,k,display_transform=reference)
        changes={}
        if previous is not None:
            for kind,value,color in [('added',solid-previous,'#40B989'),('removed',previous-solid,'#EE6A61')]:
                changed_mesh=unpack_solid(value)
                changed_path=directory/f'{row["index"]:03d}_{kind}.npz'
                np.savez_compressed(changed_path,vertices=changed_mesh.vertices,faces=changed_mesh.faces)
                # Coplanar current additions are placed first in depth ties.
                layers.insert(0,(transform_mesh(changed_mesh,reference),color))
                changes[kind+'_cm3']=material_volume(value)*1e6
        tiles.append(raster_tile(layers))
        metadata.append(dict(index=row['index'],pose=model.poses[k],layout=row['layout'],
            mesh=str(path.relative_to(out)),mesh_sha256=I.sha256(path),nominal_material_volume_cm3=material_volume(solid)*1e6,
            process_force_evaluation='sampled contacts',full_clearance_acceptance=False,**changes))
        paths.append(out/row['layout'])
        previous=solid
    if final_result is not None:
        layout=read_layout(out/'layout.npz');final=trimesh.load(out/'support.obj',force='mesh',process=False)
        # The final tile is the actual accepted final geometry, separately
        # recorded from sampled/nominal construction states.
        tiles.append(raster_tile(layers_for_pose(model,layout,final,layout.active[0],display_transform=reference)))
        final_tiles=[raster_tile(layers_for_pose(model,layout,final,k)) for k in layout.active]
        final_source='support.obj';verified=True
    else:
        layout=read_layout(out/'sampled_layout.npz');final=unpack_solid(builder.solid(layout))
        D.export_exact_obj(final,out/'diagnostic_support.obj')
        final_tiles=[raster_tile(layers_for_pose(model,layout,final,k)) for k in layout.active]
        final_source='diagnostic_support.obj';verified=False
    sheet(tiles,out/'process.png',min(4,len(tiles)))
    sheet(final_tiles,out/'final_result.png')
    record=dict(complete=True,presentation_only=True,text_or_labels=False,
        fixed_isometric_view=True,elevation_deg=35.26438968,azimuth_deg=-45,
        process_states=metadata,final_support_source=final_source,final_force_exit_work_verified=verified,
        process_camera_frame='fixed first-reference fixture frame; same pose and view in every tile',
        added_material_color='green',removed_material_color='red',
        work_forbidden_overlay=False,original_working_faces_colored=True,geometry_changed=False,
        last_tile_uses_verified_final_geometry=verified,
        provenance=provenance(paths+[out/final_source],[Path(__file__),HERE/'vis_func/work_access_render.py']),
        artifacts={'../'+file:I.sha256(out/file) for file in ['process.png','final_result.png']})
    save(out/'data/render.json',record);I.check_report(out/'data/render.json')
    print('WHOLE FIGURES',out.parent.parent.name,len(rows),'chosen states',flush=True)
