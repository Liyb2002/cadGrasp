"""Copied baseline geometry/mechanics, with no completion bank in inference."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
# Resolve numbered scoring stages and geometry from this copy before train imports.
from step3_scheculer import stage_imports
from step2_local_support import geometry
TRAIN = Path(__file__).resolve().parents[2] / 'train'
sys.path.insert(0, str(TRAIN))
from bank_teacher import Context, object_directions, TerminalOracle, install_recorded_recovery
from train import DATA

class RuntimeContext(Context):
    """Reuse verified path witnesses; never load teacher completion banks."""
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cfg = json.loads((DATA/'pilot20_shared/config.json').read_text())
        I.check_hashes(self.cfg['source_inputs'])
        self.poses = sorted({p for _, ps in self.cfg['groups'] for p in ps}, key=lambda p:int(p.split('_')[1]))
        self.problems, self.pools, self.vectors, self.oracles = {}, {}, {}, {}
        install_recorded_recovery(self.output)
        for p in self.poses:
            copied = I.OUTPUTS/'B/independent_poses'/p
            original = I.ROOT/'slides/baseline_algo/output/B/independent_poses'/p
            for stage, filenames in [('step_1_needs', ['needs.json', 'samples.json']),
                                     ('step2_local_support', [f'candidates_{p}.json', f'candidates_{p}.npz'])]:
                for filename in filenames:
                    if I.sha256(copied/stage/filename) != I.sha256(original/stage/filename):
                        raise RuntimeError(f'Copied fixed-task input differs: {p}/{filename}')
            problem = read_task('B', p, folder=copied/'step_1_needs')
            cached = json.loads((DATA/'pilot20_shared/inputs'/f'paths_{p}.json').read_text())
            I.check_hashes(cached['signature'])
            contacts = {c['candidate_id']:c for c in I.read_contacts(copied/'step2_local_support'/f'candidates_{p}.npz')}
            pool = [dict(r, index=i, contact=contacts.get(r['id'])) for i,r in enumerate(cached['candidates'])]
            self.problems[p], self.pools[p] = problem, pool
            catalogue = np.asarray(cached['catalogue']['vectors'])
            self.vectors[p] = object_directions(problem, catalogue)
            cache = self.output/f'terminal_checks_{p}.json'
            if not cache.exists():
                # Reuse physical evidence only, never teacher action values.
                source = TRAIN/'policy_checks'/cache.name
                if not source.exists(): source = DATA/'pilot20_shared'/cache.name
                shutil.copyfile(source, cache)
            self.oracles[p] = TerminalOracle(problem, pool, cache)
        self.angle_cache = {}

    def save_label(self, *args, **kwargs):
        raise RuntimeError('Runtime Step3 cannot call the offline teacher')
