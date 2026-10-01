"""Ideal snug rectangular peg/socket, separate-body reactions, fixed-base sweeps.

Dimensions follow belt_test. Zero nominal clearance is the declared ideal
contact model; manufacturing clearance/play is not certified. The end stop
resists further insertion; no latch or tensile withdrawal reaction is added.
"""
from itertools import product
import numpy as np
import trimesh
from scipy.optimize import linprog
from step2_local_support import insertion as H, withdrawal as W
from step0_pose_selection import equilibrium as Q
from step5_base import bearing as M

WIDTH, HEIGHT, LENGTH = .018, .008, .026
WALL, ENGAGEMENT, STROKE = .003, .021, .030
UP = np.array([0.,0.,1.])
CORNERS = np.array(list(product((-1.,1.), repeat=3)))


def box(center, size):
    return trimesh.creation.box(extents=size).apply_translation(center)


def joint(q):
    q=np.asarray(q,float)
    peg=box(q-UP*LENGTH/2,[WIDTH,HEIGHT,LENGTH])
    # Sleeves cover z in [-26,-5] mm; solid end stop is below z=-26 mm.
    sleeve=[box(q-UP*(LENGTH+WALL/2),[WIDTH+2*WALL,HEIGHT+2*WALL,WALL])]
    for axis,size,other in ((0,WIDTH,HEIGHT),(1,HEIGHT,WIDTH)):
        for sign in (-1,1):
            center=q-UP*(LENGTH-ENGAGEMENT/2);center[axis]+=sign*(size+WALL)/2
            dims=[WIDTH+2*WALL,HEIGHT+2*WALL,ENGAGEMENT+1e-5]
            dims[axis]=WALL
            sleeve.append(box(center,dims))
    # Forces ON BLUE, at actual coincident mating surfaces.
    points=[];normals=[]
    for axis,half,otherhalf in ((0,WIDTH/2,HEIGHT/2),(1,HEIGHT/2,WIDTH/2)):
        for sign in (-1,1):
            normal=np.zeros(3);normal[axis]=-sign
            for other,z in product((-otherhalf,otherhalf),(-LENGTH,-LENGTH+ENGAGEMENT)):
                p=q.copy();p[axis]+=sign*half;p[1-axis]+=other;p[2]+=z
                points.append(p);normals.append(normal.copy())
    for x,y in product((-WIDTH/2,WIDTH/2),(-HEIGHT/2,HEIGHT/2)):
        points.append(q+[x,y,-LENGTH]);normals.append(UP.copy())
    return peg,sleeve,np.asarray(points),np.asarray(normals)


def description(q):
    return dict(type='rectangular_peg_sleeve_end_stop',port_m=np.asarray(q).tolist(),
        peg_cross_section_m=[WIDTH,HEIGHT],peg_length_m=LENGTH,engagement_m=ENGAGEMENT,
        wall_m=WALL,nominal_clearance_m=0.,clearance_model='ideal snug contact; manufacturing tolerance not certified',
        withdrawal_direction=UP.tolist(),insertion_direction=(-UP).tolist(),docking_stroke_m=STROKE,
        anti_withdrawal_lock=False,interface_friction=0.,source_reference='slides/belt_test/code/source_geometry.py')


def port(anchor, mesh, u, planes, side, reach):
    """Nearest L1 port in an exterior side region; whole peg/root has clearance."""
    anchor=np.asarray(anchor);u=np.asarray(u);scale=float(mesh.extents.max())
    offsets=np.vstack([CORNERS*.0045, joint(np.zeros(3))[0].vertices])
    normals=[u,*[np.asarray(p[:3]) for p in planes]]
    lower=[np.max(mesh.vertices@u)+.025*scale-np.min(offsets@u)]
    for p in planes:lower.append(.002-p[3]-np.min(offsets@np.asarray(p[:3])))
    axis,sign=side;normal=np.zeros(3);normal[axis]=sign
    normals.append(normal)
    lower.append(np.max(mesh.vertices@normal)+.035-np.min(offsets@normal))
    matrix=[np.r_[-v,[0.,0.,0.]] for v in normals];rhs=list(-np.asarray(lower))
    for axis in range(3):
        e=np.eye(3)[axis]
        matrix += [np.r_[e,-e],np.r_[-e,-e]];rhs += [anchor[axis],-anchor[axis]]
    result=linprog([0,0,0,1,1,1],A_ub=matrix,b_ub=rhs,
        bounds=[(anchor[i]-reach,anchor[i]+reach) for i in range(3)]+[(0,None)]*3,method='highs')
    return result.x[:3] if result.success else None


def collision_volume(obstacles, parts, shift=None):
    """Continuous convex-cell translation against complete stationary solids."""
    origin=np.vstack([m.vertices for m in obstacles]).mean(axis=0)
    scale=max(float(m.extents.max()) for m in obstacles)
    fixed=[W.solid(m,origin,scale) for m in obstacles]
    maximum=0.
    for part in parts:
        swept=part if shift is None else H.engine.hull_mesh(np.vstack([part.vertices,part.vertices+shift]))
        moving=W.solid(swept,origin,scale)
        for target in fixed:
            maximum=max(maximum,abs(float((moving^target).volume()))*scale**3)
    return maximum


def docking_check(obj, blue, base_parts):
    # Relative motion: moving object+blue upwards equals moving the fixed base
    # downwards. Base convex cells give an exact translational swept union.
    bottom=min(obj.bounds[0,2],blue.bounds[0,2])
    length=max(STROKE,max(p.bounds[1,2] for p in base_parts)-bottom+.04)
    volume=collision_volume([obj,blue],base_parts,-UP*length)
    tol=1e-11*float(obj.extents.max())**3
    return dict(passed=bool(bottom>=-1e-9 and volume<=tol),base_stationary=True,
        moving_bodies=['object','contact_module'],withdrawal_direction=UP.tolist(),
        local_stroke_m=STROKE,checked_withdrawal_length_m=float(length),
        continuous_sweep_verified=True,maximum_intersection_m3=volume,volume_tolerance_m3=tol,
        method='convex fixed-base cells swept by inverse assembly translation; full object and blue solids checked')


def matrix(domain, contacts, floor, footprint, interface, mu):
    """18 rows: object, blue, orange. Every action/reaction shares one column."""
    scale=np.r_[np.ones(3),np.ones(3)/domain.mesh.extents.max()]
    points,normals,owners=M.bearing_rays(domain,contacts,floor['original_pivot_m'],mu)
    columns=[]
    for p,n,owner in zip(points,normals,owners):
        w=Q.wrench(p,n,domain.com)*scale;c=np.zeros(18);c[:6]=w
        if owner>=0:c[6:12]=-w
        columns.append(c)
    _,_,pts,forces=joint(interface['port_m'])
    for p,f in zip(pts,forces):
        w=Q.wrench(p,f,domain.com)*scale;c=np.zeros(18);c[6:12]=w;c[12:]=-w;columns.append(c)
    ground=M.floor_vertices(footprint)
    for xy in ground:
        for f in ([mu,0,1],[-mu,0,1],[0,mu,1],[0,-mu,1]):
            c=np.zeros(18);c[12:]=Q.wrench(np.r_[xy,0.],np.asarray(f),domain.com)*scale;columns.append(c)
    return np.asarray(columns).T,scale


def bearing(domain,contacts,floor,footprint,interface,mu):
    mat,scale=matrix(domain,contacts,floor,footprint,interface,mu)
    solver=Q.BatchSolver(mat);results={}
    report=dict(passed=False,sampled_passed=False,continuous_passed=False,
        body_count=3,shared_interface_reactions=True,interface_friction=0.,
        anti_withdrawal_lock=False,base_anchored=False,sufficient_friction_coefficient=mu)
    for group,key,certified in [('continuous','continuous_outer_load_wrenches',True),('sample','load_wrenches',False)]:
        targets=Q.padded_targets(floor[key],scale,18)
        probe=solver.solve(targets[M.seed_probes(floor[key])],certified=certified)
        if not probe['passed']:
            return dict(report,status='interface_or_floor_bearing_not_certified',failed_group=group,
                        diagnostics=probe['diagnostics']),{}
        result=solver.solve(targets,certified=certified)
        if not result['passed']:
            return dict(report,status='interface_or_floor_bearing_not_certified',failed_group=group,
                        diagnostics=result['diagnostics']),{}
        results[group]=result
    arrays=dict(equilibrium_matrix=mat,scale=scale)
    for group,result in results.items():
        arrays[group+'_assignment']=result['assignment'];arrays[group+'_coefficients_mg']=result['weights']
        arrays[group+'_basis_indices']=np.asarray(solver.bases,int).reshape(-1,18)
    return dict(report,passed=True,sampled_passed=True,continuous_passed=True,status='three_body_equilibrium_verified'),arrays


def replay_bearing(domain,contacts,floor,footprint,interface,report,arrays):
    mat,scale=matrix(domain,contacts,floor,footprint,interface,report['sufficient_friction_coefficient'])
    np.testing.assert_allclose(mat,arrays['equilibrium_matrix'],rtol=0,atol=0)
    for group,key in [('sample','load_wrenches'),('continuous','continuous_outer_load_wrenches')]:
        targets=Q.padded_targets(floor[key],scale,18)
        assignment=arrays[group+'_assignment'];weights=arrays[group+'_coefficients_mg']
        assert len(assignment)==len(targets) and np.all(assignment>=0)
        for k,ids in enumerate(arrays[group+'_basis_indices']):
            rows=np.flatnonzero(assignment==k)
            if not len(rows):continue
            np.testing.assert_allclose(weights[rows]@mat[:,ids].T,targets[rows],atol=Q.TOL,rtol=0)
            assert weights[rows].min()>=-Q.TOL
            if group=='continuous':assert Q.membership(mat,ids,targets[rows],certified=True)[0].all()
    return True
