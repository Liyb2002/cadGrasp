"""Affine floor-pressure projection for a complete shared-material layout."""
import numpy as np
from scipy.optimize import linprog
from fixture_state import installed_state
from system_floor import transform_points,system_floor_demands
from whole_search.reuse_first import registered

def project_floor_layout(m,q,fixed_ground=(),maximum=32,protect_ports=False,fixed_seats=()):
 """L1 seat / installation proposal on all original pressure demands.

 Fixed native rotations make these halfspaces affine in seat XYZ and
 installation heights. A failed proposal is never an infeasibility claim.
 Original all-pose force, geometry and construction still judge its output.
 """
 hosts=sorted(set(int(q.hosts[k]) for k in q.active));hi={h:j for j,h in enumerate(hosts)};guests=[k for k in q.active if not registered(q,k)];gi={k:len(hosts)+3*j for j,k in enumerate(guests)};abs0=len(hosts)+3*len(guests);dim=abs0+3*len(guests);fields=[]
 bounds=[]
 for h in hosts:
  if h in fixed_ground:lower=-m.workpiece_height(q,h);bounds.append((lower,lower))
  elif h in q.active and registered(q,h):bounds.append((-m.workpiece_height(q,h),None))
  else:bounds.append((None,None))
 bounds+=[(None,None)]*(3*len(guests))+[(0,None)]*(3*len(guests));A=[];rhs=[]
 for k in fixed_seats:
  if k in gi:
   for axis in range(3):bounds[gi[k]+axis]=(0.,0.)
 state=m.contact_delta.state(q) if protect_ports else None
 for k in q.active:
  T=installed_state(m.native,q,q.hosts[k]);com=transform_points(m.mesh.center_mass[None],T@q.placements[k])[0];loads=m.base_oracle.loads[k];ratios=np.c_[loads[:,:2]/loads[:,2,None],np.ones(len(loads))];pressure=transform_points(system_floor_demands(loads,com)['floor_demands_world_m'],np.linalg.inv(T));row=np.zeros(dim);row[hi[int(q.hosts[k])]]=-1
  if k in gi:row[gi[k]:gi[k]+3]=-T[2,:3]
  A.append(row);rhs.append(m.workpiece_height(q,k))
  if protect_ports and m.floor_signature(q,k):
   # Keep the original object-ground reaction columns available.
   A.append(-row);rhs.append(-m.workpiece_height(q,k))
  if protect_ports:
   contact=transform_points(m.points[state.available[k]],q.placements[k])
   if len(contact):
    for host in hosts:
     B=installed_state(m.native,q,host);row=np.zeros(dim);row[hi[host]]=-1
     if k in gi:row[gi[k]:gi[k]+3]=-B[2,:3]
     A.append(row);rhs.append(float((contact@B[2,:3]+B[2,3]).min()))
  for h in hosts:
   if h==int(q.hosts[k]):continue
   B=installed_state(m.native,q,h);normal=B[2,:3];nw=normal@T[:3,:3].T;rates=normal[None,:]-(ratios@nw)[:,None]*T[2,:3][None,:];slopes=-ratios@nw;fields.append((k,hi[int(q.hosts[k])],hi[h],pressure@normal+B[2,3],rates,slopes))
 for j,k in enumerate(guests):
  for a in range(3):
   row=np.zeros(dim);row[gi[k]+a]=1;row[abs0+3*j+a]=-1;A.append(row);rhs.append(0);row=row.copy();row[gi[k]+a]=-1;A.append(row);rhs.append(0)
 x=np.zeros(dim)
 for h in fixed_ground:x[hi[h]]=-m.workpiece_height(q,h)
 seen=set();cost=np.r_[np.ones(len(hosts)),np.zeros(3*len(guests)),np.full(3*len(guests),.5)]
 for _ in range(maximum):
  added=False
  for f,(k,o,j,base,rates,slopes) in enumerate(fields):
   values=base+slopes*x[o]+x[j]
   if k in gi:values=values+rates@x[gi[k]:gi[k]+3]
   i=int(values.argmin())
   # Seed every pressure field on the first pass. A currently feasible
   # layout still needs the LP optimisation, otherwise its original height
   # is returned before cheaper seat/installation combinations are tried.
   if (_==0 or values[i]<-1e-9) and (f,i) not in seen:
    seen.add((f,i));row=np.zeros(dim);row[o]-=slopes[i];row[j]-=1
    if k in gi:row[gi[k]:gi[k]+3]-=rates[i]
    A.append(row);rhs.append(float(base[i]));added=True
  if not added:
   for k,o,j,base,rates,slopes in fields:
    values=base+slopes*x[o]+x[j]
    if k in gi:values=values+rates@x[gi[k]:gi[k]+3]
    if float(values.min()) < -1e-8:return None
   t=q.copy();t.state_offsets[hosts,2]+=x[:len(hosts)]
   for k in guests:t.placements[k,:3,3]+=x[gi[k]:gi[k]+3]
   try:
    for k in t.active:m.validate_placement(t,k)
   except ValueError:return None
   return t
  r=linprog(cost,A_ub=np.array(A),b_ub=np.array(rhs),bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  if not r.success:return None
  x=r.x
 return None

