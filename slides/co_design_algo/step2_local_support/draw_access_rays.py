"""Plot sampled outer access rays, each checked against the actual object.

This is a sampled illustration, not a certificate of the continuous boundary.
No convex shell or interpolated volume is filled.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.cases import selected_pose, pose_name
from step1.needs import frame
from step2_local_support import circles as P, render as R
from step4_floor_contact.draw_work_volume import layer


def sample(domain):
    triangles = domain.mesh.triangles[domain.work_ids]
    normals = -domain.normals
    e1, e2 = frame(normals)
    bary = np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[1/3]*3])
    points = np.einsum('bv,fvj->fbj', bary, triangles)
    alpha = np.deg2rad(domain.data['load']['cone_half_deg'])
    polar = np.r_[0., np.full(32, alpha/2), np.full(128, alpha)]
    azimuth = np.r_[0., np.arange(32)*2*np.pi/32, np.arange(128)*2*np.pi/128]
    vectors = np.cos(polar)[None,:,None]*normals[:,None]+np.sin(polar)[None,:,None]*(
        np.cos(azimuth)[None,:,None]*e1[:,None]+np.sin(azimuth)[None,:,None]*e2[:,None])
    q = np.broadcast_to(points[:,:,None,:], (len(triangles),len(bary),len(polar),3)).reshape(-1,3).copy()
    u = np.broadcast_to(vectors[:,None,:,:], (len(triangles),len(bary),len(polar),3)).reshape(-1,3).copy()
    owners = np.repeat(np.arange(len(triangles)), len(bary)*len(polar))
    theta = np.tile(polar, len(triangles)*len(bary))
    clear = np.empty(len(q), bool)
    for start in range(0, len(q), 16000):
        end = min(start+16000, len(q))
        clear[start:end] = ~domain.mesh.ray.intersects_any(
            q[start:end]+domain.ray_offset*normals[owners[start:end]], u[start:end])
    assert np.all(np.einsum('ij,ij->i', u, normals[owners]) >= np.cos(alpha)-1e-12)
    print(f'{clear.sum()} / {len(clear)} sampled rays pass the Step1 visibility predicate', flush=True)
    return q[clear], u[clear], domain.work_ids[owners[clear]], theta[clear], len(clear)


def draw(name):
    domain, data, _ = P.read(name)
    out = P.OUTPUTS/name/pose_name()/'step2_local_support'
    q, u, face, theta, count = sample(domain)
    length = .60*float(domain.mesh.extents.max())
    end = q+length*u
    views = [('Oblique', R.axes([.6, -.9, .55])), ('Side', R.axes([1, 0, .15])),
             ('Top', R.axes([0, 0, 1]))]
    paper = Image.new('RGB', (2700, 1190), R.PAPER)
    ink = ImageDraw.Draw(paper)
    ink.text((30,20), f'{name} / {pose_name()} / outer sampled access rays', font=R.font(40), fill=R.INK)
    ink.text((30,80), f'Green: work surface. Purple: object-clear rays within the 30-degree cone. Shown length: {length*1000:.1f} mm; rays continue.',
             font=R.font(26), fill=R.INK)
    records=[]
    for j,(label,basis) in enumerate(views):
        angle=np.arange(48)*2*np.pi/48
        direction=np.cos(angle)[:,None]*basis[0]+np.sin(angle)[:,None]*basis[1]
        selected=np.unique(np.argmax(end@direction.T,axis=0))
        points=np.vstack([domain.mesh.vertices,q[selected],end[selected]])
        local=points@basis.T;lo,hi=local.min(axis=0),local.max(axis=0)
        focus=((lo+hi)/2)@basis;width=1.12*max(hi[:2]-lo[:2]);size=880
        colors=np.tile(R.GREY,(len(domain.mesh.faces),1));colors[domain.work_ids]=R.GREEN
        picture,depth=layer(domain.mesh.triangles,colors,focus,width,size,basis=basis)
        pen=ImageDraw.Draw(picture)
        for k in selected:
            t=np.linspace(0,1,150)
            projected=R.project(q[k]+t[:,None]*length*u[k],focus,basis,width,size)
            for a,b in zip(projected[:-1],projected[1:]):
                mid=(a+b)/2;x,y=np.rint(mid[:2]).astype(int)
                hidden=0<=x<size and 0<=y<size and depth[y,x]>mid[2]+width*1e-6
                pen.line([tuple(a[:2]),tuple(b[:2])],fill='#c5b6d5' if hidden else '#70439a',width=1 if hidden else 2)
            start,tip=projected[0,:2],projected[-1,:2]
            delta=tip-projected[-3,:2];delta/=max(np.linalg.norm(delta),1e-12)
            side=np.array([-delta[1],delta[0]])
            pen.polygon([tuple(tip),tuple(tip-9*delta+3*side),tuple(tip-9*delta-3*side)],fill='#70439a')
            pen.ellipse((start[0]-3,start[1]-3,start[0]+3,start[1]+3),fill='#397c35')
        ink.text((30+900*j,145),f'{label} / {len(selected)} silhouette rays',font=R.font(27),fill=R.INK)
        paper.paste(picture,(10+900*j,195))
        records.append(dict(view=label,basis=basis.tolist(),ray_indices=selected.tolist(),
            rays=[dict(source_face=int(face[k]),source_m=q[k].tolist(),direction=u[k].tolist(),
                       angle_deg=float(np.degrees(theta[k]))) for k in selected]))
    ink.text((30,1090),'Ray selection: extreme endpoints in 48 screen directions, among the visibility-checked samples. No volume is filled.',
             font=R.font(26),fill=R.INK)
    ink.text((30,1135),'Pale segments are behind the object in that camera. Sampled extremes are not a proof of the exact continuous boundary.',
             font=R.font(26),fill=R.INK)
    paper.save(out/'access_boundary_rays.png')
    np.savez_compressed(out/'access_boundary_rays.npz',source_m=q,direction=u,source_face=face,angle_rad=theta)
    P.save(out/'access_boundary_rays.json',dict(object=name,pose=pose_name(),complete=True,
        total_sampled=count,clear_count=len(q),shown_length_m=length,source_points_per_face=7,
        cap_rim_azimuths=128,inner_ring_azimuths=32,normal_included=True,
        exact_continuous_boundary_proved=False,ray_origin_offset_m=domain.ray_offset,views=records,
        provenance=dict(inputs=P.W.I.hashes([out.parent/'step_1_needs/needs.json']),
                        code=P.W.I.hashes([Path(__file__)])),
        artifacts={f:P.sha256(out/f) for f in ['access_boundary_rays.png','access_boundary_rays.npz']}))
    print(out/'access_boundary_rays.png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object');parser.add_argument('--pose',default='pose_2')
    args=parser.parse_args()
    with selected_pose(args.pose):draw(args.object)
