"""Adapt an older frozen Step4 caller that discards cleanup statistics.

AST checks require every original geometry operation, guard and tolerance to
remain identical. Only the unused second return value loses two report CSGs.
"""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap


def function_tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def install(directory):
    specification=importlib.util.find_spec('contact_regularization')
    coopt=Path(specification.origin).resolve().parents[1]
    for name in ['step5.2','step5.3']:
        path=str(coopt/name)
        if path not in sys.path:sys.path.insert(0,path)
    import contact_regularization as original
    from material_evaluator import MaterialEvaluator
    path=Path(directory)/'cleanup_without_unused_statistics.py'
    spec=importlib.util.spec_from_file_location('coopt_cleanup_without_unused_statistics',path)
    replacement=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=replacement;spec.loader.exec_module(replacement)
    before=function_tree(original.remove_object_interiors)
    after=function_tree(replacement.remove_object_interiors)
    assert ast.dump(before.args)==ast.dump(after.args),'Cleanup API changed'
    assert [ast.dump(x) for x in before.body[:-1]]==[ast.dump(x) for x in after.body[:-1]],'Numerical cleanup/guards changed'
    assert ast.dump(before.body[-1].value.elts[0])==ast.dump(after.body[-1].value.elts[0]),'Returned geometry changed'
    caller=function_tree(MaterialEvaluator.construct)
    uses=[node for node in ast.walk(caller) if isinstance(node,ast.Assign)
          and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
          and node.value.func.id=='remove_object_interiors']
    assert len(uses)==1 and ast.dump(uses[0].targets[0])==ast.dump(ast.parse('complete,_=None').body[0].targets[0])
    assert not any(isinstance(node,ast.Name) and node.id=='_' and isinstance(node.ctx,ast.Load)
                   for node in ast.walk(caller)),'Cleanup statistics consumed by search'
    original_source=Path(original.__file__)
    record=dict(original_source=str(original_source),
        original_source_sha256=hashlib.sha256(original_source.read_bytes()).hexdigest(),
        unchanged_numeric_statements=len(before.body)-1,
        omitted_unused_statistics=['removed_cm3','input_intrusion_cm3'],
        actual_geometry_and_all_guards_unchanged=True,
        reason='Step4 assigns the statistics to an unused local variable')
    original.remove_object_interiors=replacement.remove_object_interiors
    return record
