"""Retain broad, rounded bands around a certified load-bearing shape.

The input is still the original Step4 material. The saved pruning result only
guides which original stock to retain and fixes the all-state BBX envelope.
No added material, moved contacts, voxel acceptance or decorative bridge.
"""
import time
import numpy as np
import manifold3d as md

from prune_support import Pruner,material_components
from deterministic_base import solid,unpack,volume,SCALE,transform,VOLUME_TOL


def clip_world_box(value,bounds,frames):
    """Intersect every installation with the same measured world envelope."""
    for T in frames:
        for axis in range(3):
            value=value.trim_by_plane(tuple(T[axis,:3]),float((bounds[0,axis]-T[axis,3])/SCALE))
            value=value.trim_by_plane(tuple(-T[axis,:3]),float((T[axis,3]-bounds[1,axis])/SCALE))
    return value


def anchor_component(value,anchor):
    parts=material_components(value)
    for part in parts:
        if volume(anchor-part)<=1e-10:return part
    raise ValueError('Rounded trimming removed required reference material')


class RoundedPruner(Pruner):
    def __init__(self,*args,reference=None,rounding_radius=.006,**kwargs):
        kwargs['grids']=()
        super().__init__(*args,**kwargs)
        self.reference=reference
        self.rounding_radius=rounding_radius

    def run(self):
        if self.reference is None:
            # New cases without an earlier certified scaffold keep their
            # BBX-trimmed broad stock; never fall back to local grid nibbling.
            mesh,record,proof=super().run()
            record.update(pruning_style='rounded_block',reference_scaffold_used=False,
                grid_cell_deletion_used=False,material_strategy='retain broad BBX-trimmed stock')
            return mesh,record,proof
        anchor=solid(self.reference)
        if len(material_components(anchor))!=1 or volume(anchor-self.original)>1e-10:
            raise ValueError('Shape reference must be a connected subset of this Step4 core')
        self.proof=self.certificate.evaluate(self.fixture);self.best_mesh=self.fixture
        if not self.proof['passed']:raise ValueError('Original Step4 core fails original demands')
        target_cloud=np.vstack([self.object_cloud]+[transform(self.reference.vertices,T)for T in self.frames])
        bounds=np.array([target_cloud.min(0),target_cloud.max(0)])
        initial_bbx=self.bbx(self.original)
        stock=anchor_component(clip_world_box(self.body,bounds,self.frames),anchor)
        if volume(self.body-stock)>VOLUME_TOL:
            if not self.accept(stock,'bbx_locked_whole_plane_trim'):
                raise ValueError('Original-demand check rejected the saved BBX envelope')
        stock=self.body
        guide=anchor.simplify(.0004/SCALE)
        selected_radius=None
        for radius in [self.rounding_radius,self.rounding_radius*4/3,self.rounding_radius*2]:
            expanded=guide.minkowski_sum(md.Manifold.sphere(radius/SCALE,16))
            # Exact original/reference boundaries carry all contact proofs.
            # The rounded offset only selects a broad retained stock region.
            candidate=anchor_component(anchor+(stock^expanded),anchor)
            if volume(stock-candidate)<=VOLUME_TOL:
                selected_radius=radius;break
            if self.accept(candidate,'rounded_whole_stock_pockets'):
                selected_radius=radius;break
        actual=solid(self.best_mesh)
        if volume(anchor-actual)>1e-10 or volume(actual-self.original)>1e-10:
            raise ValueError('Rounded export violates exact contact-stock retention')
        cloud=np.vstack([self.object_cloud]+[transform(self.best_mesh.vertices,T)for T in self.frames])
        actual_bounds=np.array([cloud.min(0),cloud.max(0)])
        if not np.allclose(actual_bounds,bounds,atol=1e-8,rtol=0):
            raise ValueError('Rounded pruning changed saved BBX bounds')
        if abs(self.bbx(actual)-float(np.prod(bounds[1]-bounds[0])*1000))>1e-8:
            raise ValueError('Rounded pruning changed saved BBX volume')
        record=dict(passed=True,stage_type='support_pruning',pruning_style='rounded_block',
            bbx_first=True,input_bbx_l=initial_bbx,retained_bbx_l=self.bbx(actual),
            locked_bbx_bounds_m=bounds.tolist(),locked_bbx_unchanged=True,
            input_material_cm3=volume(self.original)*1e6,retained_material_cm3=volume(actual)*1e6,
            removed_material_cm3=volume(self.original-actual)*1e6,
            added_material_cm3=volume(actual-self.original)*1e6,
            initial_component_count=1,component_count=len(material_components(actual)),connected=True,
            force_required_isolated_components=0,watertight=bool(self.best_mesh.is_watertight),
            all_original_demands_passed=True,counts=self.proof['counts'],contact_triangles=self.proof['contact_triangles'],
            outside_saved_legal_component_cm3=volume(actual-self.allowed)*1e6,
            retained_reference_loss_cm3=volume(anchor-actual)*1e6,
            reference_scaffold_used=True,rounding_radius_m=selected_radius,
            requested_rounding_radius_m=self.rounding_radius,grid_cell_deletion_used=False,
            original_legal_stock_boundaries_preserved=True,
            stock_limited_features_may_remain=True,
            material_strategy='large rounded pockets; retain broad stock bands around certified scaffold',
            attempts=self.attempts,candidate_budget=self.budget,history=self.history,
            certificate_calls=self.certificate.calls,force_classifications=self.certificate.classifications,
            reused_identical_reaction_cones=self.certificate.reused,force_solver_run=True,
            layout_changed=False,demand_resampling=False,global_minimum_claimed=False,
            seconds=time.monotonic()-self.began)
        return self.best_mesh,record,self.proof
