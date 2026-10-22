"""One private motion bundle lets visual edits replay without replanning robots."""
import hashlib,json,zipfile
from pathlib import Path
from whole_task.step1.common import sha256

FILES=('plan.json','trajectory.npz','scene_meshes.npz','robot_meshes.npz','segments.json','motion_validation.json')
BUNDLE='video_motion.zip'


def save(work,native_sources):
    work=Path(work);data=work/'video'
    sources=[*native_sources,*(p for p in Path(__file__).parent.glob('*.py') if p.name not in ('raster.py','render.py'))]
    manifest=dict(station={name:sha256(work/name) for name in ('support.obj','layout.npz','execution.json')},
        sources={str(p):sha256(p) for p in sources},files={name:sha256(data/name) for name in FILES})
    with zipfile.ZipFile(work/BUNDLE,'w',compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('manifest.json',json.dumps(manifest))
        for name in FILES:archive.write(data/name,name)
    return BUNDLE


def restore(folder,work):
    folder=Path(folder);work=Path(work);bundle=folder/BUNDLE
    if not bundle.exists():return None
    with zipfile.ZipFile(bundle) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        if any(sha256(folder/name)!=digest for name,digest in manifest['station'].items()):return None
        if any(not Path(p).exists() or sha256(p)!=digest for p,digest in manifest['sources'].items()):return None
        contents={name:archive.read(name) for name in FILES}
        if any(hashlib.sha256(contents[name]).hexdigest()!=manifest['files'][name] for name in FILES):
            raise ValueError('Changed cached robot motion')
    data=work/'video';data.mkdir(exist_ok=True)
    for name,content in contents.items():(data/name).write_bytes(content)
    return json.loads(contents['motion_validation.json'])
