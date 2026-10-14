import ast
from tools.run_qa_mapped_observed_bridge_component import source,namespace
from tools.audit_qa_mapped_observed_bridge_completed import expected_phases


def test_real_entry_single_composition_and_preserved_native_dispatch():
    text=source();compile(text,'real_composed_worker','exec')
    assert text.count('MappedObservedPlacementBridge as ObservedPlacementBridge')==1
    assert "dist_utils.distribute_model(model)" in text and "ev.qa_eval(model,tokenizer,16)" in text
    assert "range(32)" in text and "range(2)" in text
    assert text.count("requests.append((('natural-tail-unique'")==1
    assert text.index('prepare_runtime(os.environ')<text.index('torch.cuda.device_count()')
    assert namespace()['worker']
    assert len(expected_phases(4))==593
