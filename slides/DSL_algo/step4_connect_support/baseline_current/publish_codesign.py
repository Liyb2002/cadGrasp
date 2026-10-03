"""Publish the copied Step5 CAD views of a common-object construction."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageColor, ImageDraw
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current.run_codesign import read_case, registration, head_meshes, PALETTE
from step4_connect_support.baseline_current.codesign_port import visual_details as V
from step4_connect_support.baseline_current.codesign_port.refresh_shared_geometry_view import refresh

SUPPORT_COLOR = '#8fc9e8'


def panel(body, heads, colors, obj, size=850):
    points = np.vstack([body.vertices]+([] if obj is None else [obj.vertices]))
    view = V.fit(points, V.R.axes([-.68,-1.,.8]), margin=1.15)
    pieces, shades, count = [], [], 0
    if obj is not None:
        pieces.append(obj.triangles); shades.append(np.tile(V.R.GREY,(len(obj.faces),1))); count+=len(obj.faces)
    pieces.append(body.triangles); shades.append(np.tile(ImageColor.getrgb(SUPPORT_COLOR),(len(body.faces),1))); count+=len(body.faces)
    first_head=count
    for mesh,color in zip(heads,colors):
        pieces.append(mesh.triangles);shades.append(np.tile(ImageColor.getrgb(color),(len(mesh.faces),1)));count+=len(mesh.faces)
    return V.render((np.vstack(pieces),np.vstack(shades),np.arange(first_head,count),[]),view,size)[0]


def legend(canvas, object_visible=True):
    ink = ImageDraw.Draw(canvas)
    entries = [(SUPPORT_COLOR, 'Support'), ('#dc9d47', 'Colored: heads')]
    if object_visible:
        entries.insert(1, (tuple(int(c) for c in V.R.GREY), 'Object'))
    for k, (color, label) in enumerate(entries):
        x = 20 + 230*k
        ink.rectangle((x, 48, x+16, 61), fill=color)
        ink.text((x+24, 43), label, font=V.R.font(18), fill=V.R.INK)


def publish(output):
    report=I.check_report(output/'data/report.json')
    assert report['schema']=='codesign_common_object_step4_v1' and report['constructed']
    # Refresh presentation only; preserve the original physical source hashes
    # and all saved mesh/reaction artifacts through the copied refresh helper.
    work = output/'data/codesign_body'
    refresh(work)
    page = (work/'index.html').read_text().replace('href="fixture.obj"', 'href="shape.obj"').replace('<a href="fixture_mm.stl" download>STL · mm</a>', '')
    (output/'shape.html').write_text(page)
    report['construction'] = json.loads((work/'report.json').read_text())
    report['artifacts']['../shape.html'] = I.sha256(output/'shape.html')
    for name in ('report.json', 'codesign_report.json'):
        I.save(output/'data'/name, report)
    case=read_case(output)
    bases,offsets,heads,_=registration(case)
    meshes=head_meshes(heads);colors=[PALETTE[h.pose%len(PALETTE)] for h in heads]
    body=trimesh.load(output/'shape.obj',force='mesh',process=False)
    assert body.is_watertight and body.is_winding_consistent
    with np.load(output/'data/codesign_body/geometry.npz') as z:
        np.testing.assert_allclose(body.vertices,z['vertices_m'],atol=1e-14,rtol=0)
        np.testing.assert_array_equal(body.faces,z['faces'])
        np.testing.assert_allclose(bases,z['rotations'],atol=1e-14,rtol=0)
        np.testing.assert_allclose(offsets,z['local_offsets_m'],atol=1e-14,rtol=0)
    images=[];artifacts={}
    for k,task in enumerate(case.tasks):
        actual=trimesh.Trimesh((body.vertices-offsets[k])@bases[k].T,body.faces,process=False)
        local_heads=[trimesh.Trimesh((m.vertices-offsets[k])@bases[k].T,m.faces,process=False) for m in meshes]
        picture=panel(actual,local_heads,colors,task.domain.mesh)
        canvas=Image.new('RGB',(850,970),'white');canvas.paste(picture,(0,65))
        ink=ImageDraw.Draw(canvas);ink.text((20,15),f'{case.pair.name} / {task.pose}',font=V.R.font(28),fill=V.R.INK)
        legend(canvas)
        state='ACCEPTED' if report['passed'] else 'CONNECTED CANDIDATE - NOT ACCEPTED'
        ink.text((20,935),state,font=V.R.font(20),fill='#287754' if report['passed'] else '#a64538')
        name=f'shape_{task.pose}.png';canvas.save(output/name);images.append(canvas)
        artifacts['../'+name]=I.sha256(output/name)
    picture=panel(body,meshes,colors,None)
    canvas=Image.new('RGB',(850,970),'white');canvas.paste(picture,(0,65))
    ImageDraw.Draw(canvas).text((20,15),'The same connected shape',font=V.R.font(28),fill=V.R.INK)
    legend(canvas, object_visible=False)
    images.append(canvas)
    columns=min(3,len(images));rows=(len(images)+columns-1)//columns
    page=Image.new('RGB',(850*columns,970*rows),'white')
    for k,im in enumerate(images):page.paste(im,(850*(k%columns),970*(k//columns)))
    page.save(output/'shape.png');artifacts['../shape.png']=I.sha256(output/'shape.png')
    artifacts['../shape.html'] = I.sha256(output/'shape.html')
    I.save(output/'data/codesign_visualization.json',dict(complete=True,
        source='Copied co-design Step5 visual_details raster and shared_geometry_viewer template',
        colors=dict(support=SUPPORT_COLOR, object='grey', heads='original pose colors'),
        original_geometry_unchanged=True, provenance=dict(inputs=I.hashes([output/'shape.obj',output/'data/report.json']),
            code=I.hashes([Path(__file__),Path(V.__file__),Path(V.R.__file__),
                Path(refresh.__code__.co_filename),Path(refresh.__code__.co_filename).with_name('shared_geometry_viewer.html')])),artifacts=artifacts))
    print(case.pair.name,'co-design visualization saved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('outputs',type=Path,nargs='+')
    for output in parser.parse_args().outputs:publish(output.resolve())
