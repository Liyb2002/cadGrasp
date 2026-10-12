"""Build display triangle solids from saved layouts, without force acceptance."""
import argparse
from collections import OrderedDict
import json
import time
from common import *
from model import Model,Layout
from case_sets import CASES,case_directory


def load_layout(path):
    with np.load(path) as data:
        return Layout(data['placements'].copy(),data['directions'].copy(),
                      data['hosts'].copy(),tuple(map(int,data['active'])))


def read_mesh(path):
    with np.load(path) as data:
        return C.trimesh.Trimesh(data['vertices'].copy(),data['faces'].copy(),process=False)


def save_mesh(path,mesh):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,vertices=mesh.vertices,faces=mesh.faces)


class MeshBuilder:
    def __init__(self,model,fan=8):
        self.model,self.fan=model,fan
        self.parts=OrderedDict()
        self.shapes=OrderedDict()

    def part(self,layout,k,kind):
        m=self.model;q=layout.placements[k]
        d=q[:3,:3].T @ layout.directions[k]
        cap=m.work_length(layout,k) if kind=='work' else None
        key=(kind,k if kind in ['wrap','work'] else None,
             np.round(q,11).tobytes(),np.round(d,11).tobytes() if kind=='sweep' else cap)
        if key not in self.parts:
            local=(m.wraps[k] if kind=='wrap' else m.work_solid(layout,k) if kind=='work' else
                   m.body if kind=='body' else m.clearance.sweep(m.length*d,fan=self.fan,padded=False))
            self.parts[key]=transform_solid(local,q)
        self.parts.move_to_end(key)
        while len(self.parts)>256:
            self.parts.popitem(last=False)
        return self.parts[key]

    def solid(self,layout):
        if layout.key() in self.shapes:
            return self.shapes[layout.key()]
        seed=C.union([self.part(layout,k,'wrap') for k in layout.active])
        cuts=C.union([self.part(layout,k,kind) for k in layout.active
                      for kind in ['body','work','sweep']])
        support=seed-cuts
        if support.status()!=C.F.md.Error.NoError:
            raise RuntimeError('Display mesh Boolean unresolved')
        self.shapes[layout.key()]=support
        while len(self.shapes)>48:
            self.shapes.popitem(last=False)
        return support


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=list(CASES),required=True)
    parser.add_argument('--root',type=Path,default=HERE/'output/B')
    parser.add_argument('--method',choices=['whole','incremental','both'],default='both')
    parser.add_argument('--fan',type=int,default=8)
    args=parser.parse_args()
    model=Model([f'pose_{number}' for number in CASES[args.case]])
    builder=MeshBuilder(model,args.fan)
    directory=case_directory(args.root,args.case)
    methods=['whole','incremental'] if args.method=='both' else [args.method]
    for method in methods:
        out=directory/method
        rows=json.loads((out/'process.json').read_text())
        records=[]
        previous=None
        began=time.monotonic()
        for row in rows:
            path=out/'mesh_states'/f'{row["index"]:03d}.npz'
            stage=time.monotonic()
            layout=load_layout(out/row['layout'])
            delta_paths={kind:out/'mesh_states'/f'{row["index"]:03d}_{kind}.npz' for kind in ['kept','added','removed']}
            if path.exists() and all(p.exists() for p in delta_paths.values()):
                mesh=read_mesh(path)
                solid=builder.solid(layout)
            else:
                solid=builder.solid(layout)
                mesh=unpack_solid(solid)
                save_mesh(path,mesh)
                if previous is None:
                    additions=solid
                    removals=solid-solid
                    kept=removals
                else:
                    additions=solid-previous
                    removals=previous-solid
                    kept=solid^previous
                for kind,value in [('kept',kept),('added',additions),('removed',removals)]:
                    save_mesh(delta_paths[kind],unpack_solid(value))
            records.append(dict(index=row['index'],phase=row['phase'],layout=row['layout'],
                mesh=str(path.relative_to(out)),material_volume_cm3=abs(float(mesh.volume))*1e6,
                vertices=len(mesh.vertices),triangles=len(mesh.faces),seconds=time.monotonic()-stage))
            records[-1]['delta_meshes']={kind:str(p.relative_to(out)) for kind,p in delta_paths.items()}
            previous=solid
            C.save(out/'mesh_progress.json',records)
            print('MESH',directory.name,method,row['index'],'/',len(rows)-1,
                  len(mesh.faces),'faces',round(time.monotonic()-stage,2),'s',flush=True)
        final=read_mesh(out/records[-1]['mesh'])
        C.D.export_exact_obj(final,out/'support.obj')
        C.save(out/'mesh_geometry.json',dict(complete=True,geometry_representation='triangle Boolean boundary',
            constructed_from_saved_layouts=True,voxel_smoothing=False,search_rerun=False,
            force_acceptance_run=False,final_acceptance_run=False,
            shape_rule='fitted wraps union minus current bodies, full outward30deg working cones and full nominal exit sweeps',
            work_access=[work.definition() for work in model.work_rays],
            exit_length_m=model.length,fan=args.fan,steps=records,
            final_material_volume_cm3=abs(float(final.volume))*1e6,seconds=time.monotonic()-began))


if __name__=='__main__':
    main()
