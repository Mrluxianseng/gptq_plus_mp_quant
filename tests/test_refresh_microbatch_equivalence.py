import os

import torch
from torch import nn

# This is a text-only unit test. Some Windows dev environments install a
# torchvision binary built for a different PyTorch version; keep that optional
# vision extension from breaking imports of the text model utilities.
if os.name == "nt":
    try:
        import transformers.utils.import_utils as _transformers_import_utils

        _transformers_import_utils._torchvision_available = False
    except ImportError:
        pass

from gptq_utils.gptq_plus_utils import collect_true_weight_gradient


class _OneProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)

    def forward(self, hidden, **_kwargs):
        return (self.proj(hidden),)


class _Analyzer:
    def __init__(self):
        self.norm = nn.Identity()
        self.lm_head = nn.Linear(3, 7, bias=False)

    def get_layernorm_before_head(self):
        return self.norm

    def get_lm_head(self):
        return self.lm_head


def _collect(layer, analyzer, inps, fp_inps, refresh_mb):
    batch, seq_len, _ = inps.shape
    return collect_true_weight_gradient(
        layer=layer,
        analyzer=analyzer,
        module_name="proj",
        full={"proj": layer.proj},
        inps=inps,
        fp_inps=fp_inps,
        attention_mask=torch.ones(1, 1, seq_len, seq_len),
        position_ids=torch.zeros(1, seq_len, dtype=torch.long),
        position_embeddings=(
            torch.zeros(1, seq_len, 3),
            torch.zeros(1, seq_len, 3),
        ),
        bsz=batch,
        kl_topk=-1,
        dev=torch.device("cpu"),
        sample_indices=list(range(batch)),
        refresh_loss_type="kl",
        refresh_mb=refresh_mb,
    )


def test_kl_refresh_microbatch_preserves_full_batch_gradient_and_loss():
    torch.manual_seed(17)
    layer = _OneProjection()
    analyzer = _Analyzer()
    inps = torch.randn(4, 3, 3)
    fp_inps = torch.randn(4, 3, 3)

    full_grad, full_count, full_loss, _ = _collect(
        layer, analyzer, inps, fp_inps, refresh_mb=None
    )
    chunked_grad, chunked_count, chunked_loss, _ = _collect(
        layer, analyzer, inps, fp_inps, refresh_mb=2
    )

    assert full_count == chunked_count == 4
    torch.testing.assert_close(chunked_grad, full_grad, rtol=1e-5, atol=1e-7)
    assert abs(chunked_loss - full_loss) <= 1e-7 + 1e-5 * abs(full_loss)
