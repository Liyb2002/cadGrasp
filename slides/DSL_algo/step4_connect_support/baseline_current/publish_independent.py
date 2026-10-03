"""Publish only the final connected shape, keeping physical results unchanged."""
import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I


def publish(output):
    report_path=output/'data/report.json'
    report=I.check_report(report_path)
    if report['schema']!='independent_pose_connected_shape_v1' or not report['constructed']:
        raise ValueError('Publish an actual independent connected shape only')
    model=trimesh.load(output/'shape.obj',force='mesh',process=False)
    if not model.is_watertight or not model.is_winding_consistent:
        raise ValueError('Saved indexed shape is not closed and consistently oriented')
    with np.load(output/'data/independent_body/geometry.npz') as z:
        np.testing.assert_allclose(model.vertices,z['vertices_m'],atol=1e-14,rtol=0)
        np.testing.assert_array_equal(model.faces,z['faces'])
    page=(output/'data/independent_body/index.html').read_text()
    page=page.replace('<a href="fixture_mm.stl" download>STL · mm</a>','')
    page=page.replace('href="fixture.obj"','href="shape.obj"')
    page=page.replace(";setMode('overview');",";setMode('fixture');")
    old=" : (DATA.report.checks||[]).map(c=>c.pose+': '+c.verified_sample_count.toLocaleString()+'/'+c.original_sample_count.toLocaleString()).join(' · ');"
    new=" : (DATA.report.passed ? '一个闭合连接实体 · 完整验收通过' : '一个闭合连接实体 · 完整验收未通过')+' · '+(DATA.report.checks||[]).map(c=>c.pose+': '+(c.coupled_equilibrium_passed?'载荷通过':'载荷未通过')).join(' · ');"
    if old not in page:raise ValueError('Viewer status template changed')
    page=page.replace(old,new)
    pos=page.index('const DATA=')+len('const DATA=')
    data,length=json.JSONDecoder().raw_decode(page[pos:])
    data['report']['presentation_description']=(f'{report["physical_head_count"]} 个独立头已连接成同一件支撑；不共享头。'
        '各 pose 使用不同的相对摆放。拖动查看最终 shape，点击 pose 查看使用位置。')
    page=page[:pos]+json.dumps(data,separators=(',',':'))+page[pos+length:]
    (output/'shape.html').write_text(page)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,8),facecolor='white')
    ax=fig.add_subplot(111,projection='3d')
    light=np.array([-.35,-.5,.8]);light/=np.linalg.norm(light)
    shade=.62+.28*np.maximum(0,model.face_normals@light)
    colors=np.c_[shade*.83,shade*.89,shade*.93,np.ones(len(shade))]
    ax.add_collection3d(Poly3DCollection(model.triangles,facecolors=colors,edgecolors='none',zsort='average'))
    center=model.bounds.mean(axis=0);span=model.extents.max()
    # Equal spatial scale without letting a long layout shrink vertically.
    ranges=np.maximum(model.extents*1.08,span*.14)
    ax.set(xlim=(center[0]-ranges[0]/2,center[0]+ranges[0]/2),
        ylim=(center[1]-ranges[1]/2,center[1]+ranges[1]/2),
        zlim=(center[2]-ranges[2]/2,center[2]+ranges[2]/2))
    ax.set_box_aspect(ranges);ax.set_axis_off();ax.view_init(elev=28,azim=-50)
    ax.set_proj_type('ortho')
    label=output.parent.name
    status='ACCEPTANCE PASS' if report['passed'] else 'ACCEPTANCE NOT PASSED'
    fig.suptitle(f'{label} | One connected shape',fontsize=21,y=.96)
    fig.text(.5,.04,f'{status}  |  {report["volume_cm3"]:.1f} cm³  |  '+
        ' × '.join(f'{v:.0f}' for v in model.extents*1000)+' mm',ha='center',fontsize=13,
        color='#287754' if report['passed'] else '#b5413d')
    fig.subplots_adjust(left=0,right=1,bottom=.06,top=.93)
    fig.savefig(output/'shape.png',dpi=150);plt.close(fig)
    report['presentation']=dict(final_shape_only=True,deleted_preview_categories=['heads','head_details','overview'],
        mode='indexed OBJ with actual connected shape render',physical_geometry_unchanged=True,
        inputs=dict(shape_sha256=I.sha256(output/'shape.obj')),
        code=I.hashes([Path(__file__)]))
    report['artifacts'].update({f'../{n}':I.sha256(output/n) for n in ('shape.obj','shape.html','shape.png')})
    I.save(report_path,report);I.save(output/'data/independent_report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('outputs',type=Path,nargs='+')
    for folder in p.parse_args().outputs:
        r=publish(folder);print(folder.parent.name,'published',r['passed'])
