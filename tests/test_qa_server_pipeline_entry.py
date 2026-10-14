def test_actual_formal_entry_passes_frozen_production_profile_installer():
    import ast
    from tools.run_storage_formal_qa_mapped_v8 import entry
    from tools import run_storage_method_formal as formal
    from tools.run_storage_method_pipeline import prepare_entry_source
    from tools.run_storage_formal_multidevice_profile import footer
    from tools import run_storage_method_pipeline as pipeline
    original=formal.build_entry();generated=entry(original)
    assert generated.count("logging.info('OVERNIGHT_METRICS %s',json.dumps(row))")==1
    assert generated.count('"layer_output_fingerprints.json"')==1
    assert generated.index('_qa_mapped_runtime_library')<generated.index('import torch')
    old=pipeline.PROCESS_PEAK_FOOTER
    try:
        pipeline.PROCESS_PEAK_FOOTER=footer(old)
        final=prepare_entry_source(generated)
    finally:pipeline.PROCESS_PEAK_FOOTER=old
    compile(final,'real_server_profiled_entry','exec')
    assert 'baseline-profile_metrics_rank' in final and '_collect_device_peaks' in final
    tree=ast.parse(final)
    runs=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='runpy.run_path']
    assert len(runs)==1 and runs[0].args[0].value=='ptq.py'
