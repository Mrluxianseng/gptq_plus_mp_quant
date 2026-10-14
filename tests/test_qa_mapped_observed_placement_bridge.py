import ast
import inspect
import json
import pytest
import torch
from lm_eval.models.huggingface import HFLM
from tools.qa_mapped_observed_placement_bridge import MappedObservedPlacementBridge,control_batch_source
from tools.qa_complete_observer import validate_observer_pair
from tools.audit_memory_boundary_points import boundaries,compare
from test_qa_observed_placement_bridge import fixture


class Library:
    def qa_peak_bytes(self): return 0


def test_native_control_ast_only_adds_observer_callback():
    original=ast.parse(inspect.getsource(HFLM._loglikelihood_tokens))
    generated=ast.parse(control_batch_source(inspect.getsource(HFLM._loglikelihood_tokens)))
    class Remove(ast.NodeTransformer):
        def visit_Expr(self,node):
            return None if ast.unparse(node)=='_qa_release_batch()' else node
    assert ast.dump(original,include_attributes=False)==ast.dump(Remove().visit(generated),include_attributes=False)


def test_real_hflm_complete_task_and_original_footer_cpu_composition(tmp_path):
    original=(HFLM._model_call,HFLM._loglikelihood_tokens)
    for enabled,side in ((False,'control'),(True,'candidate')):
        model,tok,ev,dist,evaluation,memory,cuda,calls=fixture()
        original_simple=evaluation.simple_evaluate
        bridge=MappedObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=enabled,
            mapped_enabled=False,library=Library(),destination=tmp_path/side,
            expected_tasks=('fixture_a','fixture_b'),validation_only=True,device='cpu',exclusive_peak_owner=True)
        dist.distribute_model(model);assert ev.qa_eval(model,tok,16)==dict(fixture_a=0.5,fixture_b=0.5)
        record=json.loads((tmp_path/side/'qa_mapped_head_rank0.json').read_text())
        assert record['completed'] and record['formal_enabled'] is False
        assert record['head']['configured_batch_size']==16 and record['head']['actual_batch_sizes']==[16,16]
        assert memory.get_process_cuda_memory_peaks('cuda:3')==(5000,6000)
        bridge.restore();assert evaluation.simple_evaluate is original_simple
        assert (HFLM._model_call,HFLM._loglikelihood_tokens)==original
    assert validate_observer_pair(tmp_path/'control/qa_complete',tmp_path/'candidate/qa_complete',['fixture_a','fixture_b'])['valid']
    paths=[list((tmp_path/s/'qa_stage2/memory_profile').rglob('*.jsonl')) for s in ('control','candidate')]
    assert compare(*(boundaries(p) for p in paths))['valid']
    assert not torch.cuda.is_initialized()


def test_original_eval_error_restored_and_full_failure_evidence(tmp_path):
    model,tok,ev,dist,evaluation,memory,cuda,calls=fixture(fail=True)
    original=(HFLM._model_call,HFLM._loglikelihood_tokens)
    bridge=MappedObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=True,mapped_enabled=False,
        library=Library(),destination=tmp_path,expected_tasks=('fixture_a','fixture_b'),
        validation_only=True,device='cpu',exclusive_peak_owner=True)
    dist.distribute_model(model)
    with pytest.raises(ValueError,match='original evaluator'): ev.qa_eval(model,tok,16)
    record=json.loads((tmp_path/'qa_mapped_head_rank0.json').read_text())
    assert record['completed'] is False and record['error_type']=='ValueError'
    assert memory.get_process_cuda_memory_peaks('cuda:3')==(5000,6000)
    assert (HFLM._model_call,HFLM._loglikelihood_tokens)==original
    bridge.restore()


def test_mapping_cannot_be_enabled_for_cpu_control_or_formal(tmp_path):
    args=fixture();model,tok,ev,dist,evaluation,memory,cuda,calls=args
    with pytest.raises(RuntimeError,match='CUDA layerwise'):
        MappedObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=False,mapped_enabled=True,
            library=Library(),destination=tmp_path,expected_tasks=('fixture_a',),validation_only=True,device='cpu',exclusive_peak_owner=True)
    with pytest.raises(RuntimeError,match='validation-only'):
        MappedObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=True,mapped_enabled=False,
            library=Library(),destination=tmp_path,expected_tasks=('fixture_a',),validation_only=False,device='cpu',exclusive_peak_owner=True)
