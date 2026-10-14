import ast
from tools.run_qa_control_align_component import source, namespace


def test_actual_single_worker_entry_compiles_and_preserves_full_workload_and_math():
    text=source()
    scope=namespace()
    assert callable(scope['worker']) and callable(scope['audit'])
    assert text.count("requests.append((('natural-tail-unique','choice'),[11,12,13],[14]))") == 1
    assert text.count('qa_eval(model,tokenizer,16)') == 1
    assert "if torch.cuda.device_count()!=1:" in text
    assert "add_hook_to_module(model.lm_head,AlignDevicesHook(execution_device='cuda:0',io_same_device=True))" in text
    assert 'qa_control_head_observer.py' in text and 'inspect.getfile(_accelerate_hooks)' in text
    tree=ast.parse(text)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert not any(ast.unparse(node.func).endswith(('empty_cache','set_default_dtype')) for node in calls)
