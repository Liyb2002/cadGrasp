"""Presentation-only reader for unchanged saved Step4/Step5 artifacts.

Keep this video independent of removed baseline algorithm packages.
"""
from pathlib import Path
from types import SimpleNamespace,ModuleType
import hashlib,importlib.util,json,sys,shutil
import numpy as np
import trimesh
import manifold3d as md

ROOT=Path('/home/yli581/Desktop/cadGrasp')
PROOF=ROOT/'slides/Co-optimize/seperate_base_support_proof'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path):return str(Path(path).resolve().relative_to(ROOT))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def transform_points(points,T):return np.asarray(points)@T[:3,:3].T+T[:3,3]

cpu_path=ROOT/'slides/reuse/code/render_cpu.py'
spec=importlib.util.spec_from_file_location('video_saved_cpu_primitives',cpu_path)
CPU=importlib.util.module_from_spec(spec);sys.modules[spec.name]=CPU;spec.loader.exec_module(CPU)
def piece(mesh,color,opacity=1.):
    p=CPU.piece(dict(v=mesh.vertices,f=mesh.faces),color);p['opacity']=opacity;return p
V=ModuleType('step51_render');V.CPU=CPU;V.piece=piece;V.BLUE='#2A91D2';V.GRAY='#A5ADB5'
sys.modules['step51_render']=V
hull_path=ROOT/'slides/Co-optimize/step5.1/hull_render.py'
spec=importlib.util.spec_from_file_location('video_saved_hull_primitives',hull_path)
H=importlib.util.module_from_spec(spec);sys.modules[spec.name]=H;spec.loader.exec_module(H)

class Inputs:
    def __init__(self,name,selection):
        self.name=name;self.base=PROOF/'output'/name;self.selection_path=Path(selection).resolve()
        self.cases=json.loads(self.selection_path.read_text())['cases']
        self.mesh_path=ROOT/'objects'/name/'mesh.stl';self.mesh=trimesh.load(self.mesh_path,force='mesh',process=False)
        self.protected={relative(p):digest(p) for p in [self.selection_path,self.mesh_path]}
    def load(self,case):
        out=self.base/case['id']/'step5/step5.1';path=out/'data/report.json';record=json.loads(path.read_text())
        selected=ROOT/record['selected_step4_source']
        assert case['passed'] and record['complete'] and case['experiment']==record['selected_step4_experiment']
        self.protected.update(record['provenance']['inputs'])
        for p in [path,selected/'support.obj',selected/'layout.npz',selected/'data/report.json']:self.protected[relative(p)]=digest(p)
        fixture=trimesh.load(selected/'support.obj',force='mesh',process=False);rows=[]
        for previous in record['per_pose']:
            p=out/previous['arrays'];self.protected[relative(p)]=digest(p)
            with np.load(p,allow_pickle=False) as arrays:
                cloud=arrays['floor_demands_world_m'].copy();indices=arrays['hull_indices'].copy()
                ft=arrays['T_world_fixture'].copy();ot=arrays['T_world_object'].copy()
            vertices=transform_points(self.mesh.vertices,ot)
            ground=vertices[np.abs(vertices[:,2])<=1e-9,:2]
            rows.append(dict(pose=previous['pose'],color=previous['color'],fixture_transform=ft,
                object_transform=ot,cloud=cloud,hull=cloud[indices],object_ground=ground))
        assert len(rows)==case['pose_count']
        return dict(case=case,record=record,fixture=fixture,body=self.mesh,rows=rows,selected=selected)
    def verify(self):
        changed=[p for p,h in self.protected.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]
        if changed:raise ValueError('Saved input differs: '+str(changed[:5]))

def freeze(directory):
    paths=[cpu_path,hull_path];records={}
    for p in paths:
        target=directory/'executed_sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);records[relative(p)]=digest(p);assert digest(target)==records[relative(p)]
    return records
C=SimpleNamespace(Inputs=Inputs,digest=digest,relative=relative,save=save,F=SimpleNamespace(transform_points=transform_points),V=V,H=H,freeze=freeze)
D=SimpleNamespace(BASE=PROOF/'output/B')

def solid(mesh):
    result=md.Manifold(md.Mesh64(np.ascontiguousarray(mesh.vertices*1000,dtype=np.float64),np.ascontiguousarray(mesh.faces,dtype=np.uint32)))
    if result.status()!=md.Error.NoError:raise ValueError(str(result.status()))
    return result
def unpack(value):
    data=value.to_mesh64();return trimesh.Trimesh(np.asarray(data.vert_properties[:,:3])/1000,np.asarray(data.tri_verts),process=False)
def volume(value):
    assert value.status()==md.Error.NoError;return abs(float(value.volume()))*1e-9
def transform(value,matrix):
    affine=np.asarray(matrix)[:3].copy();affine[:,3]*=1000;return value.transform(affine.tolist())
def combine(parts):
    value=md.Manifold.batch_boolean(parts,md.OpType.Add) if parts else md.Manifold();volume(value);return value
G=SimpleNamespace(md=md,solid=solid,unpack=unpack,volume=volume,transform=transform,combine=combine)
