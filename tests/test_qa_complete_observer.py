import json
from types import SimpleNamespace
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config
from tools.qa_complete_observer import CompleteQAObserver, validate_observer_pair


def model():
    torch.manual_seed(1)
    config = Qwen3Config(vocab_size=79, hidden_size=128, intermediate_size=256,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=8)
    return AutoModelForCausalLM.from_config(config).bfloat16().eval()


class FixtureLM:
    def _model_call(self, inputs):
        return self.model(inputs, use_cache=True).logits

    def _loglikelihood_tokens(self, requests):
        inputs = torch.tensor([r[1] for r in requests])
        logits = self._model_call(inputs)
        return [(float(value.float().sum()), bool(value.argmax().item() == 0)) for value in logits[:, -1]]


def test_full_real_28_layers_bytes_and_scores_preserved(tmp_path):
    m = model()
    lm = FixtureLM(); lm.model = m
    original = (FixtureLM._model_call, FixtureLM._loglikelihood_tokens)
    requests = [(('context', str(i)), [i%79, 1, 2], [3]) for i in range(16)]
    with torch.no_grad():
        expected = lm._loglikelihood_tokens(requests)
        for side in ('control', 'candidate'):
            with CompleteQAObserver(m, FixtureLM, tmp_path/side) as observer:
                with observer.task('cpu_fixture'):
                    assert lm._loglikelihood_tokens(requests) == expected
    assert (FixtureLM._model_call, FixtureLM._loglikelihood_tokens) == original
    report = validate_observer_pair(tmp_path/'control', tmp_path/'candidate', ['cpu_fixture'])
    assert report['valid']
    assert not validate_observer_pair(tmp_path/'control', tmp_path/'candidate', ['missing'])['valid']
    rows = [json.loads(line) for line in (tmp_path/'control/qa_complete_events.jsonl').read_text().splitlines()]
    assert len([r for r in rows if r['kind']=='layer']) == 28
    assert len([r for r in rows if r['kind']=='logits']) == 1
    assert not torch.cuda.is_initialized()


def test_original_failure_retained_hooks_methods_removed(tmp_path):
    m = model()
    original = (FixtureLM._model_call, FixtureLM._loglikelihood_tokens)
    with pytest.raises(ValueError, match='original evaluator'):
        with CompleteQAObserver(m, FixtureLM, tmp_path) as observer:
            with observer.task('cpu_fixture'):
                raise ValueError('original evaluator')
    assert (FixtureLM._model_call, FixtureLM._loglikelihood_tokens) == original
    assert not any(layer._forward_hooks for layer in m.model.layers)
    report = json.loads((tmp_path/'qa_complete_summary.json').read_text())
    assert not report['completed'] and report['error_type']=='ValueError'
    with pytest.raises(RuntimeError, match='Evidence exists'):
        CompleteQAObserver(m, FixtureLM, tmp_path)


def test_missing_task_no_forward_and_state_mutation_refused(tmp_path):
    m = model(); lm = FixtureLM(); lm.model = m
    with pytest.raises(RuntimeError, match='Missing task'):
        with CompleteQAObserver(m, FixtureLM, tmp_path/'missing'):
            lm._model_call(torch.ones((16,3), dtype=torch.long))
    with pytest.raises(RuntimeError, match='no complete'):
        with CompleteQAObserver(m, FixtureLM, tmp_path/'empty') as observer:
            with observer.task('empty'): pass
    requests = [(None, [1,2,3], [4])]*16
    with pytest.raises(RuntimeError, match='Incomplete QA'):
        with torch.no_grad(), CompleteQAObserver(m, FixtureLM, tmp_path/'mutated') as observer:
            with observer.task('fixture'):
                lm._loglikelihood_tokens(requests)
                m.model.layers[0].self_attn.q_proj.weight.add_(1)
    assert json.loads((tmp_path/'mutated/qa_complete_summary.json').read_text())['completed'] is False


def test_byte_evidence_tampering_refused(tmp_path):
    m = model(); lm = FixtureLM(); lm.model=m
    requests = [(None,[1,2,3],[4])]*16
    with torch.no_grad(), CompleteQAObserver(m, FixtureLM, tmp_path) as observer:
        with observer.task('fixture'): lm._loglikelihood_tokens(requests)
    path = tmp_path/'qa_complete_events.jsonl'
    path.write_text(path.read_text().replace('"count": 16', '"count": 15'))
    with pytest.raises(RuntimeError, match='SHA/count'):
        validate_observer_pair(tmp_path,tmp_path,['fixture'])


def test_actual_installed_hflm_and_ast_selected_methods(tmp_path):
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer
    from tools.qa_score_lifetime_prototype import build_method as rows
    from tools.qa_scoring_cache_offload_prototype import build_method as cache
    m = model()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/d/llamaModels/Qwen3-0.6B', local_files_only=True)
    hflm = HFLM(pretrained=m, tokenizer=tokenizer, batch_size=16, device='cpu')
    requests = [(('fixture', str(i)), [1,i+2,3], [4]) for i in range(16)]
    original_call, original_score = HFLM._model_call, HFLM._loglikelihood_tokens
    results = []
    try:
        for side in ('control', 'candidate'):
            HFLM._model_call = original_call if side=='control' else cache(original_call, offloading=False)
            HFLM._loglikelihood_tokens = original_score if side=='control' else rows(original_score)
            with torch.no_grad(), CompleteQAObserver(m, HFLM, tmp_path/side) as observer:
                with observer.task('cpu_fixture'):
                    results.append(hflm._loglikelihood_tokens(requests, disable_tqdm=True))
        assert results[0] == results[1]
        assert validate_observer_pair(tmp_path/'control',tmp_path/'candidate',['cpu_fixture'])['valid']
        record = json.loads((tmp_path/'candidate/qa_complete_summary.json').read_text())
        assert record['selected_call']['source_sha256'] is None
        assert record['selected_call']['selected_code_filename']=='qa_scoring_cache_storage_offload'
        assert len(record['selected_call']['code_sha256']) == 64
    finally:
        HFLM._model_call, HFLM._loglikelihood_tokens = original_call, original_score
    assert not torch.cuda.is_initialized()
