"""Necessary shared-seat floor compatibility, with free installation heights.

Registered tasks have Q=I and the same common-coordinate object. That object
is above all registered floors, so its own ground contacts cannot evade these
cross-floor pressure halfspaces. XY installation translations cancel. Work,
exits, guests and construction are omitted: feasibility is only a PRIOR.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import linprog
from system_floor import system_floor_demands,transform_points


def solve_registered_heights(fields,registered,tolerance=2e-8,max_rounds=32):
    n=len(registered);index={k:i for i,k in enumerate(registered)}
    delta=np.zeros(n);A=[];b=[];tags=[];seen=set()
    pairs=[(i,j,*fields[i,j]) for i in registered for j in registered if i!=j]
    options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
    for _ in range(max_rounds):
        added=False
        for i,j,base,slope in pairs:
            values=base+slope*delta[index[i]]+delta[index[j]];k=int(values.argmin())
            if values[k]>=-tolerance or (i,j,k) in seen:continue
            seen.add((i,j,k));added=True
            row=np.zeros(n);row[index[i]]=-slope[k];row[index[j]]=-1
            A.append(row);b.append(float(base[k]+tolerance));tags.append((i,j,k))
        if not added:
            gap=min((float((base+slope*delta[index[i]]+delta[index[j]]).min())
                    for i,j,base,slope in pairs),default=0.)
            return dict(status='necessary_feasible' if gap>=-2*tolerance else 'unresolved',
                        heights_m=np.maximum(0.,delta).tolist(),minimum_gap_m=gap,cut_count=len(A))
        solution=linprog(np.ones(n),A_ub=np.asarray(A),b_ub=np.asarray(b),
                        bounds=(-1e-9,None),options=options)
        if solution.success:delta=solution.x;continue
        if solution.status!=2:return dict(status='unresolved',message=solution.message)
        mat=np.asarray(A);rhs=np.asarray(b);certificate=None
        for size in range(2,n+1):
            for active in combinations(range(n),size):
                unused=np.setdiff1d(np.arange(n),active)
                eligible=np.flatnonzero(~np.any(mat[:,unused]!=0,axis=1))
                if not len(eligible):continue
                small=mat[eligible][:,active]
                dual=linprog(rhs[eligible],A_ub=-small.T,b_ub=-np.ones(size)*1e-7,
                    A_eq=np.ones((1,len(eligible))),b_eq=[1.],bounds=(0,None),options=options)
                if not dual.success:continue
                y=np.asarray(dual.x,np.longdouble);M=np.asarray(mat[eligible],np.longdouble)
                coeff=y@M;bound=y@np.asarray(rhs[eligible],np.longdouble)
                # Include the original numerical nonpenetration tolerance.
                lower=-np.longdouble(1e-9)*coeff.sum()
                if y.min()>=0 and coeff.min()>=0 and bound<lower-np.longdouble(1e-7):
                    ids=np.flatnonzero(dual.x>0)
                    certificate=dict(weights=dual.x[ids].tolist(),
                        rows=[dict(owner=tags[eligible[k]][0],floor=tags[eligible[k]][1],
                            original_demand_index=tags[eligible[k]][2],coefficients=mat[eligible[k]].tolist(),
                            rhs_relaxed_m=rhs[eligible[k]]) for k in ids],
                        combined_coefficients=coeff.astype(float).tolist(),
                        combined_bound_m=float(bound),verified_in_longdouble=True,
                        height_upper_bound_imposed=False)
                    break
            if certificate:break
        return dict(status='certified_infeasible' if certificate else 'numerical_lp_infeasible',
                    certificate=certificate,cut_count=len(A))
    return dict(status='unresolved',cut_count=len(A))


class RegisteredFloorPrior:
    def __init__(self,model):
        self.model=model;self.fields={};self.cache={}
        for i,T in enumerate(model.native):
            loads=model.base_oracle.loads[i]
            com=transform_points(model.mesh.center_mass[None],T)[0]
            points=system_floor_demands(loads,com)['floor_demands_world_m']
            inverse=np.linalg.inv(T)
            common=transform_points(points,inverse)
            slopes=-np.c_[loads[:,:2]/loads[:,2,None],np.ones(len(loads))]@inverse[:3,:3].T
            for j,B in enumerate(model.native):
                if i==j:continue
                if float((model.mesh.vertices@B[2,:3]+B[2,3]).min()) < -1e-9:
                    raise ValueError('Registered prior requires the original floor-aligned object')
                self.fields[i,j]=(common@B[2,:3]+B[2,3],slopes@B[2,:3])

    def assess(self,registered):
        key=tuple(sorted(map(int,registered)))
        if key not in self.cache:self.cache[key]=solve_registered_heights(self.fields,key)
        return self.cache[key]

    def summary(self):
        return dict(original_demand_count_per_pose=32768,
            original_force_model_changed=False,force_solver_run=False,
            subsets=[dict(registered=list(key),**value) for key,value in self.cache.items()])
