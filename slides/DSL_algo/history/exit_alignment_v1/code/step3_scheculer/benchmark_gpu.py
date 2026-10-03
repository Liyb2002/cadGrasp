"""Compare CPU and CUDA proposal kernels on finite differences of saved B heads."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I, contact_dsl as D
from step3_scheculer.run_dsl import saved_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', default='pose5+7')
    parser.add_argument('--pose', default='pose_5')
    args = parser.parse_args()
    root = I.OUTPUTS/'B'
    group = root/args.group
    schedule = I.check_report(group/'step3_scheculer/dsl/report.json')
    tasks = [saved_task(group, p) for p in schedule['poses']]
    task = next(t for t in tasks if t.pose == args.pose)
    artifact = group/'step3_scheculer/dsl'/task.pose/f'final_contacts_{task.pose}.npz'
    contacts = I.read_contacts(artifact)
    program = D.Program(tuple(D.Patch(i+1, tuple(c['center_m']), float(c['radius_m']))
                              for i, c in enumerate(contacts)))
    compiler = D.Compiler(task, tasks)
    x = D.parameters(program, compiler.scale)
    programs = []
    for j in range(len(x)-1):
        for sign in (-1, 1):
            y = x.copy(); y[j] += .004*sign
            programs.append(D.from_parameters(program, y, compiler.scale))
    programs += [program.delete(i) for i in range(len(program.patches))]
    fulls = [D.reduced_rays(task.supply(c)) for p in programs
             if (c := compiler.compile(p)) is not None]
    targets = task.targets[np.linspace(0, len(task.targets)-1, 24, dtype=int)]
    start = time.perf_counter()
    cpu = D.ConeBatch('cpu').solve(fulls, targets)
    cpu_seconds = time.perf_counter()-start
    backend = D.ConeBatch('cuda', 600)
    backend.solve(fulls, targets)  # Exclude initialization and warmup.
    start = time.perf_counter()
    gpu = backend.solve(fulls, targets)
    gpu_seconds = time.perf_counter()-start
    report = dict(complete=True, pose=task.pose, candidate_count=len(fulls), original_load_subset=24,
        cpu_seconds=cpu_seconds, gpu_seconds=gpu_seconds,
        kernel_speedup=cpu_seconds/gpu_seconds,
        max_absolute_residual_error=float(np.abs(cpu-gpu).max()),
        mean_absolute_residual_error=float(np.abs(cpu-gpu).mean()), backend=backend.info,
        maximum_cpu_residual=float(cpu.max()), nonzero_cpu_residual_count=int((cpu>1e-8).sum()),
        scope='Cone residual kernel only; excludes patch clipping, final LP, CUDA initialization',
        provenance=dict(inputs=I.hashes([artifact]+task.inputs),
                        code=I.hashes([Path(__file__)]+D.sources())))
    I.save(root/'pose2+9+13+15+17/step3_scheculer/dsl/gpu_benchmark.json', report)
    print(report)


if __name__ == '__main__':
    main()
