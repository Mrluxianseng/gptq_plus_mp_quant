from types import SimpleNamespace
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config
from lm_eval.models.huggingface import HFLM
from tools.qa_lm_head_storage_prototype import CpuLogitsHead, gpu_scored_rows_method
from tools.qa_score_lifetime_prototype import build_method


def test_complete_vocab_logits_and_parameter_identity():
    dtype = torch.bfloat16
    torch.manual_seed(1)
    m = AutoModelForCausalLM.from_config(Qwen3Config(vocab_size=79, hidden_size=128,
        intermediate_size=256, num_hidden_layers=2, num_attention_heads=16,
        num_key_value_heads=8, head_dim=8)).to(dtype).eval()
    inp = torch.arange(16 * 8).reshape(16, 8) % 79
    with torch.no_grad():
        reference = m(inp).logits
        parameter = m.lm_head.weight
        ticket = CpuLogitsHead(m.lm_head, enabled=True, chunk_rows=32)
        candidate = m(inp).logits
        ticket.close()
    assert torch.equal(reference, candidate)
    assert m.lm_head.weight is parameter
    assert ticket.records[0]['chunks'] == 4
    assert not torch.cuda.is_initialized()


def test_real_likelihood_selected_rows_scored_with_original_formula():
    selected = gpu_scored_rows_method(HFLM._loglikelihood_tokens)
    original = build_method(HFLM._loglikelihood_tokens)
    def fake():
        obj = SimpleNamespace(device='cpu', batch_size=16, max_length=32, AUTO_MODEL_CLASS=AutoModelForCausalLM,
            backend='causal', batch_size_per_gpu=16, _rank=0, prefix_token_id=0, logits_cache=True,
            cache_hook=SimpleNamespace(add_partial=lambda *args: None))
        obj._model_call = lambda tokens, **kw: torch.arange(tokens.numel() * 79, dtype=torch.float32).reshape(*tokens.shape, 79) / 79
        obj._select_cont_toks = lambda logits, contlen, inplen: logits[inplen-contlen:inplen]
        return obj
    requests = [(('context', 'choice'), [1, 2, 3], [4, 5])]
    assert selected(fake(), requests, disable_tqdm=True) == original(fake(), requests, disable_tqdm=True)


def test_non_linear_and_noncontiguous_hidden_are_rejected():
    with pytest.raises(RuntimeError, match='original Linear'):
        CpuLogitsHead(torch.nn.Sequential(torch.nn.Linear(8, 9)).eval(), enabled=True)
    with pytest.raises(RuntimeError, match='FP32 chunking differs'):
        CpuLogitsHead(torch.nn.Linear(8, 9).eval(), enabled=True)
    ticket = CpuLogitsHead(torch.nn.Linear(8, 9).bfloat16().eval(), enabled=True)
    with pytest.raises(RuntimeError, match='contiguous'):
        ticket.head(torch.zeros(2, 8, 8).transpose(1, 2))
    ticket.close()
