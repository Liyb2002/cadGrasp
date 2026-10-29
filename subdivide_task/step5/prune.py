"""Port of slides/Co-optimize/step5/wrap_geometry.py::Deleter.

Same finite proposal family, original tolerances and retained-volume heuristic.
Current computed loads, continuous cone cover and centered execution are supplied
by our Validator; contact faces and ground feet are free to be removed.
"""
import itertools
import time
import numpy as np
import manifold3d as md
from deterministic_base import volume, unpack, union, box, transform, SCALE
from .geometry import physical_component_count, save_mesh
from subdivide_task.step1.common import write_json, sha256

class Deleter:
    def __init__(self, mesh, value, proof, checks, validator, out):
        self.mesh=mesh; self.value=value; self.proof=proof; self.checks=checks
        self.validator=validator; self.out=out; self.history=[]; self.states=[]
        self.initial=value; self.mask=md.Manifold()
        from .geometry import RETAINED_VOLUME_FRACTION
        self.minimum_retained_volume=volume(value)*RETAINED_VOLUME_FRACTION
        self.save('connected_sparse_foot_stock', {})

    def save(self, operation, detail):
        index=len(self.states); path=self.out/f'checkpoints/{index:04d}.obj'
        mesh=save_mesh(self.mesh,path)
        # save_mesh is bit-exact; record and check bytes rather than claiming
        # that a different visualization mesh has the certificate.
        if not np.array_equal(mesh.vertices,self.mesh.vertices) or not np.array_equal(mesh.faces,self.mesh.faces):
            raise ValueError('Exact serialization changed checkpoint geometry')
        maskpath=path.with_suffix('.npz')
        np.savez_compressed(maskpath, **{f'pose_{k}':v for k,v in self.proof['masks'].items()}, **{f'endpoint_{k}':v for k,v in self.proof['endpoint_masks'].items()})
        row=dict(index=index, operation=operation, detail=detail, mesh=str(path.resolve()),
                 mesh_sha256=sha256(path), force_masks=str(maskpath.resolve()),
                 force_masks_sha256=sha256(maskpath), **self.checks)
        self.states.append(row)
        write_json(self.out/'states.json',dict(complete=False,states=self.states))
        print('ACCEPT',index,operation,round(row['material_cm3'],3),round(row['bbx_l'],6),flush=True)

    def attempt(self, candidate, operation, detail, mask=None):
        before=volume(self.value)
        rebase_mask=False
        if not candidate.is_empty():
            parts=[part for part in candidate.decompose() if float(part.volume())>0.]
            if len(parts)>1:
                main=max(range(len(parts)),key=lambda i:float(parts[i].volume()))
                discarded=union([part for i,part in enumerate(parts) if i!=main])
                # Propose one larger deletion of the cut plus dependent
                # branches. Keep cavities by subtracting the discarded parts,
                # never by rebuilding a filled outer shell. The final one-piece
                # proposal still needs every original physical/load guard.
                candidate=candidate-discarded
                if mask is not None:rebase_mask=True
                operation+='_and_dependent_branches'
                detail=dict(detail,proposed_components=len(parts),
                    dependent_material_removed_cm3=volume(discarded)*1e6)
        if candidate.is_empty() or before-volume(candidate)<1e-10:
            return False
        record=dict(operation=operation,detail=detail,accepted=False)
        began=time.monotonic()
        try:
            if volume(candidate)<self.minimum_retained_volume:
                raise ValueError('Conservative pruning material budget reached')
            if physical_component_count(candidate) != 1:
                raise ValueError('Deletion disconnects actual material')
            mesh=save_mesh(unpack(candidate.simplify(0.)),self.out/'candidate.obj')
            value,proof,checks=self.validator.check(mesh,self.value,np.asarray(self.checks['bounds_world_m']))
            self.mesh,self.value,self.proof,self.checks=mesh,value,proof,checks
            if rebase_mask:
                # Continue from the accepted actual solid. Unioning discarded
                # mesh components into a box mask creates nearly coincident
                # surfaces and can make later Boolean operations explode.
                self.initial=value;self.mask=md.Manifold()
            elif mask is not None:self.mask=mask
            record.update(accepted=True,removed_cm3=(before-volume(value))*1e6,state=len(self.states))
            self.save(operation,detail)
        except (ValueError,RuntimeError) as error:
            record['reason']=str(error)
        record['seconds']=time.monotonic()-began
        self.history.append(record)
        if record['accepted'] or len(self.history)%25==0:
            write_json(self.out/'deletions_progress.json',dict(complete=False,attempts=self.history))
        if not record['accepted'] and operation.startswith('remove_narrow'):
            print('REJECT',operation,detail.get('feature',detail.get('features')),record.get('reason','No change'),flush=True)
        return record['accepted']

    def run(self, grids=(3,5)):
        rows=self.validator.rows
        frames=list({np.round(r['fixture_transform'],11).tobytes():r['fixture_transform'] for r in rows}.values())
        objects=np.vstack([transform(self.validator.data['body'].vertices,r['object_transform']) for r in rows])
        target=np.array([objects.min(0),objects.max(0)])
        # Exterior global BBX faces first, then all cells at each declared
        # resolution. The finite proposal family is exhausted; no time cutoff.
        for fraction in (1.,.5,.25,.125):
            bounds=np.asarray(self.checks['bounds_world_m'])
            proposals=[]
            for axis in range(3):
                for side in range(2):
                    gap=(bounds[side,axis]-target[side,axis])*(1 if side else -1)
                    if gap>1e-8:proposals.append((-gap,axis,side))
            for _,axis,side in sorted(proposals):
                bounds=np.asarray(self.checks['bounds_world_m'])
                edge=bounds[side,axis];limit=edge+fraction*(target[side,axis]-edge)
                candidate=self.value
                for T in frames:
                    normal=-T[axis,:3] if side else T[axis,:3]
                    offset=(T[axis,3]-limit if side else limit-T[axis,3])/SCALE
                    candidate=candidate.trim_by_plane(tuple(normal),float(offset))
                self.attempt(candidate,'outside_bbx_face',dict(axis=axis,side=side,fraction=fraction))
        # Start a single cumulative subtraction mask from the BBX-trimmed
        # solid, preventing successive coplanar subdivisions of the stock.
        self.initial=self.value;self.mask=md.Manifold()
        low,high=self.mesh.bounds;center=(low+high)/2;extent=np.maximum((high-low)/2,1e-9)
        for grid in grids:
            cells=[];width=(high-low)/grid
            for index in itertools.product(range(grid),repeat=3):
                a=low+np.array(index)*width;b=a+width
                distance=float(np.linalg.norm(((a+b)/2-center)/extent))
                cells.append((-distance,index,a,b))
            for _,index,a,b in sorted(cells,key=lambda x:(x[0],x[1])):
                cut=box(a-1e-12,b+1e-12)
                if volume(self.value ^ cut)<1e-10:continue
                mask=self.mask+cut
                self.attempt(self.initial-mask,'outside_cell',dict(grid=grid,cell=list(index)),mask)
        final=save_mesh(self.mesh,self.out/'support.obj')
        self.value,self.proof,self.checks=self.validator.check(final,self.value,np.asarray(self.checks['bounds_world_m']))
        self.mesh=final
        write_json(self.out/'states.json',dict(complete=True,states=self.states,
            all_states_original_full_demands=True,ring_perimeter_protected=False,
            proposal_grids=list(grids),wall_clock_cutoff_used=False))
        write_json(self.out/'deletions.json',dict(complete=True,attempts=self.history))
