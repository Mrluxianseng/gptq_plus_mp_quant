import pytest
import torch
from test_qa_score_lifetime import test_real_library_scoring_cache_padding_and_truncation_are_exact as existing_case


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_actual_qwen_vocab_width_padding_cache_and_multitoken(dtype):
    existing_case(dtype,151936)
