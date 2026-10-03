"""Run slide regression suites in isolated processes and print reproducible evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent.parent
SLIDES = HERE.parent
ROOT = SLIDES.parent


def main():
    directories = sorted({p.parent for p in SLIDES.rglob('test_*.py') if 'output' not in p.parts})
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    records = []
    for directory in directories:
        relative = str(directory.relative_to(ROOT))
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', relative, '-p', 'test_*.py']
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        output = result.stdout+result.stderr
        print('\n'+relative+'\n'+output, flush=True)
        count = sum(map(int, re.findall(r'Ran (\d+) tests?', output)))
        records.append(dict(directory=relative, returncode=result.returncode, test_count=count, command=command))
        print(relative, count, 'tests;', 'passed' if result.returncode == 0 else 'FAILED', flush=True)
    print(json.dumps(records, indent=2), flush=True)
    return int(any(r['returncode'] for r in records))


if __name__ == '__main__':
    sys.exit(main())
