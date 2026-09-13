"""Audit true-area point allocation and the equal-area multilayer example."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import circles as P, surface as S, render as D
from step1.cases import pose_name


def visible(domain,data,basis):
    return ~domain.mesh.ray.intersects_any(
        data.centers_m+1e-6*domain.mesh.face_normals[data.center_faces],
        np.tile(basis[2],(len(data.centers_m),1)))


def normal_groups(normals):
    axis=np.abs(normals).argmax(axis=1)
    return 2*axis+(normals[np.arange(len(normals)),axis]<0)


def equal_area_layers():
    """Same layer areas; only their spacing changes. No camera or ray filter."""
    results=[]
    for layers in (3,5):
        for gap in (.01,.02,1.):
            polygons={}
            for i in range(layers):
                z=1.+i*gap
                vertices=np.array([[0.,0.,z],[1.,0.,z],[1.,1.,z],[0.,1.,z]])
                polygons[2*i]=vertices[[0,1,2]]
                polygons[2*i+1]=vertices[[0,2,3]]
            _,faces=S.surface_centers(polygons)
            counts=np.bincount(faces//2,minlength=layers)
            assert np.abs(counts-S.CENTER_COUNT/layers).max()<1
            results.append(dict(layer_count=layers,layer_area_m2=1.,gap_over_side_length=gap,
                                target_points_per_layer=S.CENTER_COUNT/layers,
                                final_centers_per_layer=counts.tolist()))
    return results


def chair_regions(domain,data,polygons):
    """Explicit diagnostic bins, not a semantic chair segmentation."""
    # C5's original CAD mesh has its long axis along local Y. This is a
    # body-coordinate diagnostic; the installed world and floor remain Z-up.
    T=np.asarray(domain.data['frame']['T_world_mesh']);rotation=T[:3,:3]
    normals=domain.mesh.face_normals@rotation
    groups=normal_groups(normals)
    centers=(data.centers_m-T[:3,3])@rotation
    total=sum(S.area(p) for p in polygons.values())
    result=[];empty_pieces=[]
    for low,high in ((0.,.018),(.018,.03),(.03,.04),(.04,.061)):
        bound0=low+T[:3,3]@rotation[:,1];bound1=high+T[:3,3]@rotation[:,1]
        for group,label in enumerate(('+X','-X','+Y','-Y','+Z','-Z')):
            clipped=[]
            for f,p in polygons.items():
                if groups[f]!=group:continue
                p=S.clip(p,rotation[:,1],bound0,upper=False)
                if len(p):p=S.clip(p,rotation[:,1],bound1,upper=True)
                if len(p)>=3 and S.area(p)>0:clipped.append(p)
            area=sum(S.area(p) for p in clipped)
            membership=(centers[:,1]>=low)&(centers[:,1]<high)&(groups[data.center_faces]==group)
            count=int(membership.sum())
            result.append(dict(local_height_mm=[1000*low,1000*high],dominant_normal=label,
                               eligible_area_percent=100*area/total,center_count=count,
                               accepted_center_count=int((membership&data.valid).sum())))
            if (low==0. and group==0) or (low==.03 and group==1):
                empty_pieces.extend(piece for p in clipped for piece in S.fan(p))
    return result,np.asarray(empty_pieces).reshape(-1,3,3)


def draw_chair(domain,data,report,highlight):
    T=np.asarray(domain.data['frame']['T_world_mesh'])
    views=[D.axes([.6, -.9, .55]),D.axes([-.6, .9, .65]),
           D.axes(T[:3,:3]@np.array([1.,.25,.12])),D.axes(T[:3,:3]@np.array([-1.,.25,-.12]))]
    titles=['Existing view 1','Existing view 2','Opposite side A','Opposite side B']
    paper=Image.new('RGB',(1800,1990),D.PAPER);ink=ImageDraw.Draw(paper)
    ink.text((35,25),f'C5 / {data.valid.sum()} accepted centers / sampled by surface area',font=D.font(38),fill=D.INK)
    ink.text((35,82),'Filled blue: visible center. Hollow blue: hidden center shown through the object.',font=D.font(25),fill=D.INK)
    selected=[r for r in report['local_height_normal_groups'] if
              (r['local_height_mm'][0]==0 and r['dominant_normal']=='+X') or
              (r['local_height_mm'][0]==30 and r['dominant_normal']=='-X')]
    counts=' and '.join(str(r['accepted_center_count']) for r in selected)
    ink.text((35,125),f'Orange: previously missed groups, now showing {counts} accepted centers.',font=D.font(25),fill=D.INK)
    floor=D.floor_triangles(domain);base=domain.mesh.triangles
    colors=np.tile(D.GREY,(len(base),1));colors[domain.work_ids]=D.GREEN
    triangles=np.concatenate([floor,base,highlight])
    colors=np.concatenate([np.tile(D.FLOOR,(len(floor),1)),colors,np.tile(D.ORANGE,(len(highlight),1))])
    for i,basis in enumerate(views):
        focus,width=D.overall_camera(domain,basis);size=820
        picture,_=D.raster(triangles,colors,focus,basis,width,size,
                           overlay=np.arange(len(floor)+len(base),len(triangles)),unlit=range(len(floor)))
        paint=ImageDraw.Draw(picture);xy=D.project(data.centers_m,focus,basis,width,size)[:,:2]
        seen=visible(domain,data,basis)
        for point,shown in zip(xy[data.valid],seen[data.valid]):
            x,y=point;r=4
            if shown:paint.ellipse((x-r,y-r,x+r,y+r),fill='#267fcb',outline='#fff9e7',width=1)
            else:paint.ellipse((x-r,y-r,x+r,y+r),outline='#649cbf',width=2)
        x=40+900*(i%2);y=220+850*(i//2)
        ink.text((x+410,y-26),f'{titles[i]} / {(seen&data.valid).sum()} visible',font=D.font(25),fill=D.INK,anchor='mm')
        paper.paste(picture,(x,y))
    ink.text((35,1910),'Orange groups are defined by local height and normal direction; they need not be connected.',font=D.font(22),fill=D.INK)
    ink.text((35,1948),'Green: work surface, excluded from sampling. This audits center placement, not force coverage.',font=D.font(22),fill=D.INK)
    paper.save(P.OUTPUTS/'C5'/pose_name()/'step2_local_support/sampling_audit.png')


def run(name):
    domain,data,source=P.read(name);mesh=domain.mesh
    _,polygons=S.eligible_polygons(domain)
    eligible=sum(S.area(p) for p in polygons.values())
    sampling=source['sampling'];chart_ids=np.asarray(sampling['point_charts'])
    assert len(data.centers_m)==S.CENTER_COUNT==sampling['count']
    assert len(np.unique(data.centers_m,axis=0))==S.CENTER_COUNT
    all_faces=[];quota_errors=[]
    for chart in sampling['charts']:
        ids=chart['source_faces'];all_faces.extend(ids)
        actual=sum(S.area(polygons[f]) for f in ids)
        np.testing.assert_allclose(actual,chart['area_m2'],rtol=1e-11)
        which=chart_ids==chart['index'];count=int(which.sum());quota=S.CENTER_COUNT*actual/eligible
        assert count==chart['point_count'] and abs(count-quota)<1+1e-10
        assert np.isin(data.center_faces[which],ids).all()
        assert chart['max_cell_relative_area_error']<1e-8
        if count:
            np.testing.assert_allclose(np.asarray(sampling['cell_areas_m2'])[which],actual/count,rtol=1e-8)
        quota_errors.append(abs(count-quota))
    assert sorted(all_faces)==sorted(polygons)
    # Independent barycentric containment on each original source triangle.
    for point,face in zip(data.centers_m,data.center_faces):
        assert face in polygons and point[2]>=S.FLOOR_CLEARANCE_M-1e-12
        tri=mesh.triangles[face];scale=mesh.extents.max()
        uv=np.linalg.lstsq((tri[1:]-tri[0]).T/scale,(point-tri[0])/scale,rcond=None)[0]
        bary=np.r_[1-uv.sum(),uv]
        assert bary.min()>=-1e-9
        assert np.linalg.norm(bary@tri-point)<scale*1e-10
    views=[D.axes([.6, -.9, .55]),D.axes([-.6, .9, .65])]
    seen=[visible(domain,data,b) for b in views]
    work=float(mesh.area_faces[domain.work_ids].sum())
    report=dict(object=name,center_count=len(data.centers_m),
                eligible_percent_of_total_area=100*eligible/mesh.area,
                excluded_work_percent_of_total_area=100*work/mesh.area,
                excluded_floor_percent_of_total_area=100*(mesh.area-work-eligible)/mesh.area,
                visible_in_existing_views=[int(v.sum()) for v in seen],
                hidden_in_both_existing_views=int((~np.logical_or.reduce(seen)).sum()),
                displayed_center_indices=np.flatnonzero(data.valid).tolist(),rejected_centers_displayed=False,
                method=sampling['method'],charts_checked=len(sampling['charts']),
                maximum_chart_quota_error=max(quota_errors),
                unsampled_small_chart_area_percent=100*sampling['unsampled_chart_area_fraction'],
                interpretation='Chart quotas use actual surface area with less than one point rounding error per chart. Chart interiors use equal-area cells. Point allocation has no visibility or inter-layer distance term. Small charts with quota below one may receive no point.',
                provenance=dict(circles_sha256=source['provenance']['circles_sha256'],
                                sampler_sha256=P.sha256(S.__file__),audit_code_sha256=P.sha256(__file__)))
    if name=='C5':
        report['synthetic_equal_area_layer_experiment']=equal_area_layers()
        report['local_height_normal_groups'],highlight=chair_regions(domain,data,polygons)
        assert abs(sum(r['eligible_area_percent'] for r in report['local_height_normal_groups'])-100)<1e-8
        draw_chair(domain,data,report,highlight)
    P.save(P.OUTPUTS/name/pose_name()/'step2_local_support/sampling_audit.json',report)
    print(name,'audited',len(data.centers_m),'area-allocated centers;',len(sampling['charts']),
          'charts; largest quota rounding error',round(max(quota_errors),4),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or P.OBJECTS:run(name)
