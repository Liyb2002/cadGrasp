"""Check a necessary force condition for perfect frictionless horizontal insertion.

This is a model diagnostic, not a change to candidate ranking or a search stop.
"""
import argparse
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step3_scheculer.stage_imports import load_stage
from step3_scheculer import contacts as I
C=load_stage('score','contribution')

from step1.cases import pose_name


def positive_triangle(points):
    points=np.asarray(points,float)
    if len(points)<3 or np.linalg.matrix_rank(points-points[0])<2:return None
    vertices=ConvexHull(points).vertices
    best=None
    for ids in combinations(vertices,3):
        matrix=np.vstack([points[list(ids)].T,np.ones(3)])
        try:weights=np.linalg.solve(matrix,[0.,0.,1.])
        except np.linalg.LinAlgError:continue
        if weights.min()>0 and (best is None or weights.min()>best[0]):best=(weights.min(),ids)
    if best is None:return None
    ids=list(best[1])
    matrix=[[Fraction(float(x)) for x in row]+[Fraction(rhs)] for row,rhs in
            zip(np.vstack([points[ids].T,np.ones(3)]),[0,0,1])]
    for col in range(3):
        pivot=max(range(col,3),key=lambda i:abs(matrix[i][col]))
        matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
        divisor=matrix[col][col];matrix[col]=[x/divisor for x in matrix[col]]
        for i in range(3):
            if i!=col:
                factor=matrix[i][col];matrix[i]=[x-factor*y for x,y in zip(matrix[i],matrix[col])]
    weights=[row[-1] for row in matrix]
    assert min(weights)>0 and sum(weights)==1
    assert all(sum(w*Fraction(float(points[i,j])) for i,w in zip(ids,weights))==0 for j in range(2))
    return dict(sample_indices=list(map(int,ids)),horizontal_required_forces_mg=points[ids].tolist(),
                positive_weights_exact=list(map(str,weights)),weighted_horizontal_force_exact=['0','0'],
                triangle_nondegenerate=True)


def run(name):
    problem=C.Problem(name)
    proof=positive_triangle(COORD.floor(problem.targets[:,:3]))
    report=dict(object=name,complete=True,
        status='all_loads_incompatible_with_ideal_frictionless_horizontal_insertion' if proof else 'not_established',
        proof=proof,
        derivation=['Perfect insertion along a requires n_out dot a <= 0 at every retained contact.',
            'Every nonnegative frictionless reaction -lambda*n_out therefore has dot product with a >= 0.',
            'The original floor normal reaction has zero horizontal projection.',
            'Thus every supported requirement must satisfy required_force_xy dot a >= 0.',
            'Three required horizontal forces have an exact strictly positive zero combination and span the plane.',
            'No nonzero horizontal a can have nonnegative dot products with all three forces.'],
        assumptions=dict(one_rigid_support=True,fixed_orientation_horizontal_translation=True,
            perfect_nonpenetrating_contact=True,object_support_friction=False,original_object_floor_reaction='vertical normal only'),
        conclusion='Adding more perfectly insertable frictionless contacts cannot cover all stored demands under these assumptions.',
        scope='Necessary ideal-model condition. Numerical tangent tolerances do not provide a physical force mechanism; connection geometry and sampled coverage are still computed separately.',
        scheduler_stop_changed=False,
        provenance=dict(inputs=I.hashes(problem.inputs),code=I.hashes([Path(__file__)])))
    I.save(I.OUTPUTS/name/pose_name()/'step3_scheculer/insertion_force_check.json',report)
    print(name,'insertion/force model diagnostic:',report['status'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or C.OBJECTS:run(name)
