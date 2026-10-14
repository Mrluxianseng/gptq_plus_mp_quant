import ast
import json
from types import SimpleNamespace
import torch
from tools.qa_quantized_recovery_checkpoint import install, state_digest
from tools.run_storage_qa_mapped_recovery import entry


def config():
    row = dict(rotate=False, rotation_seed=0, model='cpu_fixture',
               act_quant_aware_gptq=False, k_cache_quant_aware_gptq=False)
    for prefix in ('a', 'v', 'k'):
        row.update({prefix+'_bits':16, prefix+'_groupsize':-1,
                    prefix+'_asym':False, prefix+'_clip_ratio':1.0})
    return SimpleNamespace(**row)


def test_real_checkpoint_full_bytes_once_before_eval(tmp_path):
    model = torch.nn.Linear(3, 7).bfloat16().eval()
    analyzer = SimpleNamespace(model=model, tokenizer=None)
    calls = []
    module = SimpleNamespace(kl_ppl_eval=lambda *args, **kwargs: calls.append('eval'))
    destination = tmp_path/'quantized.pt'
    before = state_digest(model.state_dict())
    install(module, destination)
    module.kl_ppl_eval(config(), analyzer)
    receipt = json.loads(destination.with_suffix('.pt.receipt.json').read_text())
    assert receipt['completed'] and receipt['before'] == receipt['saved'] == receipt['after'] == before
    assert receipt['parameter_identity'] and calls == ['eval']
    saved_bytes = destination.read_bytes()
    module.kl_ppl_eval(config(), analyzer)
    assert destination.read_bytes() == saved_bytes and calls == ['eval', 'eval']
    try:
        install(module, destination)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing checkpoint would be overwritten')


def test_recovery_entry_retains_all_production_profile_anchors():
    from tools import run_storage_method_formal as formal
    from tools import run_storage_method_pipeline as pipeline
    from tools.run_storage_formal_multidevice_profile import footer
    from tools.run_storage_formal_qa_mapped_v8 import entry as old_entry
    original = formal.build_entry()
    generated = entry(original)
    old = old_entry(original)
    assert generated.count("logging.info('OVERNIGHT_METRICS %s',json.dumps(row))") == 1
    assert generated.count('"layer_output_fingerprints.json"') == 1
    assert 'qa_server_mapped_telemetry_v2' in generated
    assert 'qa_quantized_recovery_checkpoint' in generated
    anchor = "runpy.run_path('ptq.py', run_name='__main__')"
    # Removing only the declared new observation/I/O setup restores exactly
    # the older worker text; original math is not AST-reformatted or replaced.
    start = generated.index('if _qa_early_selected and _hinv_release:\n    from tools.qa_quantized_recovery_checkpoint')
    restored = generated[:start] + generated[generated.index(anchor, start):]
    restored = restored.replace('qa_server_mapped_telemetry_v2', 'qa_server_mapped_telemetry')
    assert restored == old
    previous = pipeline.PROCESS_PEAK_FOOTER
    try:
        pipeline.PROCESS_PEAK_FOOTER = footer(previous)
        final = pipeline.prepare_entry_source(generated)
    finally:
        pipeline.PROCESS_PEAK_FOOTER = previous
    compile(final, 'complete_recovery_production_entry', 'exec')
    calls = [n for n in ast.walk(ast.parse(final)) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == 'runpy.run_path']
    assert len(calls) == 1 and calls[0].args[0].value == 'ptq.py'
