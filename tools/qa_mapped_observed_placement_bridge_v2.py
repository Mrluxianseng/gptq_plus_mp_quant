"""Versioned control-only observer substitution; formal admission stays disabled."""
import ast
import inspect
from tools import qa_mapped_observed_placement_bridge as ancestor
from tools.qa_control_head_observer import ControlHeadObserver


def head_ticket(head, **kwargs):
    if kwargs['enabled']:
        return ancestor.NaturalBatchMappedHeadStorage(head, **kwargs)
    return ControlHeadObserver(head, expected_shape=kwargs['expected_shape'],
        configured_batch_size=kwargs['configured_batch_size'], mark=kwargs['mark'],
        require_cuda=kwargs['require_cuda'])


def bridge_source():
    tree = ast.parse(inspect.getsource(ancestor.MappedObservedPlacementBridge))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'NaturalBatchMappedHeadStorage']
    if len(calls) != 1:
        raise RuntimeError('Frozen mapped bridge head-ticket contract changed')
    calls[0].func.id = '_qa_admitted_head_ticket'
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+'\n'


namespace = dict(ancestor.__dict__, _qa_admitted_head_ticket=head_ticket)
exec(compile(bridge_source(), 'qa_control_align_bridge_v2', 'exec'), namespace)
MappedObservedPlacementBridge = namespace['MappedObservedPlacementBridge']
