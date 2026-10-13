"""Synthetic derivative check: geometry guidance, NOT a B solver trajectory."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parents[1]
CO = HERE.parents[1]
sys.path.insert(0, str(CO / 'helper_func/optimization'))
from physics_guided_cone_iterative import cone_projection, missing_values


def retract(d, frame, x):
    raw = d + np.einsum('kij,kj->ki', frame, x)
    return raw / np.linalg.norm(raw, axis=1)[:, None]


def frames(d):
    axes = np.eye(3)[np.argmin(np.abs(d), axis=1)]
    u = np.cross(d, axes); u /= np.linalg.norm(u, axis=1)[:, None]
    return np.stack([u, np.cross(d, u)], axis=2)


def guidance(d, normals, areas, values, width):
    """Smooth complement of ALL lock unions; area is reward, never capacity."""
    unlocked = expit(-(normals @ d.T) / width)
    available = np.prod(unlocked, axis=1)
    rewards = areas * values
    loss = -float(rewards @ available)
    # Stable derivative of product: da/dd_j = -a*(1-u_j)*n / width.
    gradient = np.einsum('i,ij,ic->jc', rewards*available, 1-unlocked, normals)/width
    return loss, available, gradient


def main():
    out = HERE / 'vis'; out.mkdir(exist_ok=True)
    d = np.array([[.1,0.,1.],[-.06,.07,1.]])
    d /= np.linalg.norm(d, axis=1)[:,None]
    normal = np.array([[1.,0.,.03],[-1.,0.,.04],[0.,1.,.02],[0.,-1.,.03]])
    normal /= np.linalg.norm(normal, axis=1)[:,None]
    p = np.array([[0.,0.,.4],[0.,0.,-.4],[.2,0.,0.],[-.2,0.,0.]])
    force = -normal
    # Force/moment already dimensionless here; seventh head coordinate is Fz.
    potential = np.c_[force, np.cross(p,force), force[:,2]]
    # Synthetic upward floor force, floor seventh coordinate 0; no-uplift slack -1.
    floor = np.array([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,-1.]])
    loads = np.array([[-1.,0.,.1,0.,-.4,0.,0.],[.2,-.1,1.,0.,0.,0.,0.]])
    projections = [cone_projection(floor, w) for w in loads]
    worst = int(np.argmax([v['loss'] for v in projections]))
    values = missing_values(potential, projections[worst], normalize=True)
    areas = np.array([1.,.8,.6,.7])*1e-4
    B = frames(d); width=.08
    f,a,g = guidance(d,normal,areas,values,width)
    tangent = np.einsum('kij,ki->kj',B,g)
    eps=1e-6; finite=np.zeros_like(tangent)
    for k in range(len(d)):
        for j in range(2):
            x=np.zeros_like(tangent); x[k,j]=eps
            fp=guidance(retract(d,B,x),normal,areas,values,width)[0]
            fm=guidance(retract(d,B,-x),normal,areas,values,width)[0]
            finite[k,j]=(fp-fm)/(2*eps)
    error=float(np.max(np.abs(tangent-finite)))
    assert error < 1e-10, error
    # Positive area scaling cannot change the nonnegative unbounded cone.
    rays=np.vstack([floor,potential]); target=loads[worst]
    base=cone_projection(rays,target)['loss']
    scaled=cone_projection(rays*np.linspace(.01,3,len(rays))[:,None],target)['loss']
    assert abs(base-scaled)<1e-10
    descent=-tangent/max(np.linalg.norm(tangent),1e-30)*np.tan(np.deg2rad(1.))
    fractions=np.array([0.,.125,.25,.5,1.])
    losses=[]; released=[]; killed=[]
    for alpha in fractions:
        trial=retract(d,B,alpha*descent)
        assert np.all(trial[:,2]>=0)
        value,availability,_=guidance(trial,normal,areas,values,width)
        losses.append(value)
        released.append(float(areas@np.maximum(availability-a,0))*1e6)
        killed.append(float(areas@np.maximum(a-availability,0))*1e6)
    assert losses[-1]<losses[0]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    axes[0].plot(tangent.ravel(),finite.ravel(),'o'); axes[0].set(xlabel='Analytic tangent derivative',ylabel='Central difference',title='Derivative check')
    axes[1].plot(fractions,losses,'o-');axes[1].set(xlabel='Fraction of 1-degree step',ylabel='Guidance loss',title='Local descent (synthetic)')
    axes[2].plot(fractions,released,'o-',label='Released');axes[2].plot(fractions,killed,'o-',label='Newly locked');axes[2].legend();axes[2].set(xlabel='Step fraction',ylabel='Smoothed area change (mm²)',title='All-blocker union')
    fig.tight_layout();fig.savefig(out/'derivative_check.png',dpi=180);plt.close(fig)
    record=dict(synthetic=True,original_B_solver_run=False,exact_force_acceptance=False,worst_load_index=worst,worst_squared_distance=projections[worst]['loss'],residual_prediction_minus_target=projections[worst]['residual'].tolist(),analytic_tangent_derivative=tangent.tolist(),central_difference=finite.tolist(),max_derivative_error=error,positive_ray_scaling_loss_difference=float(abs(base-scaled)),step_fractions=fractions.tolist(),proxy_losses=losses,released_smoothed_area_mm2=released,newly_locked_smoothed_area_mm2=killed,maximum_step_degrees=1.,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'derivative_check.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__': main()
