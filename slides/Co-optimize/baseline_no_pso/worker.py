"""Call the original Step4 worker with private output and a discrete controller."""
from pathlib import Path
import ast
import json
import sys

payload = json.loads(Path(sys.argv[1]).read_text())
tree = Path(payload['frozen'])
sys.path[:0] = [str(tree), str(tree / 'slides/Co-optimize/helper_func'),
               str(tree / 'slides/Co-optimize/step4.2')]
import _bootstrap
import run_oracle
import artifact_store as A

# The snapshot contains code; saved inputs remain in the real result store.
A.COOPT = run_oracle.ROOT / 'slides/Co-optimize'
A.RESULTS = A.COOPT / 'data/results'
A.INITIALIZATION_SOURCES = A.COOPT / 'data/initialization_sources'
A.VISUALS = A.COOPT / 'output'

if payload['mode'] != 'pso':
    from controller import PopulationNoPSOSearch
    import population_soft
    PopulationNoPSOSearch.local_adjustment = payload['mode'] == 'juxtapose_adjusted'
    population_soft.PopulationSoftSearch = PopulationNoPSOSearch

# Adapt only the output location in the unchanged production worker.
source = Path(run_oracle.__file__).read_text()
function = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == 'worker')
worker_source = ast.get_source_segment(source, function)
old = "out=base/'step4/step4.2'/options['output_name']"
new = "out=Path(" + repr(payload['directory']) + ")/'B'/label/'step4/step4.2'/options['output_name']"
assert worker_source.count(old) == 1
worker_source = worker_source.replace(old, new)
oracle_output = "oracle_out=A.VISUALS/'B'/label/'step4/step4.2'/options['output_name']/'oracle'"
assert worker_source.count(oracle_output) == 1
worker_source = worker_source.replace(oracle_output, "oracle_out=out/'oracle'")
exec(compile(worker_source, __file__, 'exec'), run_oracle.__dict__)
result = run_oracle.worker((payload['id'], payload['options']))
result.update(baseline_mode=payload['mode'], pso_enabled=payload['mode']=='pso')
modules = []
for name, module in sorted(sys.modules.items()):
    path = getattr(module, '__file__', None)
    if path and str(tree) in str(path):
        modules.append(dict(module=name, file=str(path)))
Path(payload['directory'], 'imported_sources.json').write_text(json.dumps(modules, indent=2)+'\n')
Path(payload['directory'], 'result.json').write_text(json.dumps(result, indent=2)+'\n')
