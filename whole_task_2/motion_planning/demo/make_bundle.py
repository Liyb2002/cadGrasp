"""Package this directory, including input geometry and the verified example, for another machine."""
import hashlib
import json
import shutil
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def build():
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and
                   not any(x in p.relative_to(HERE).parts for x in ('__pycache__', '.venv', '.pytest_cache')) and
                   p.suffix not in ('.zip', '.pyc', '.log') and p.name != 'bundle_manifest.json')
    manifest = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path = HERE / 'bundle_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    files.append(path)
    out = HERE / 'bunny_movement.zip'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in files:
            archive.write(p, 'bunny_movement/' + str(p.relative_to(HERE)))
    print(out, out.stat().st_size, 'bytes;', len(files), 'files')
    release = HERE / 'release.json'
    if release.exists():
        major = json.loads(release.read_text())['planner_version'].split('.')[0]
        versioned = HERE / f'bunny_movement_v{major}.zip'
        shutil.copy2(out, versioned)
        print(versioned)


if __name__ == '__main__':
    build()
