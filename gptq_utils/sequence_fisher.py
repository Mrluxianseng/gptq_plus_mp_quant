"""Opt-in, sample-aligned Fisher score sketches for REAL-Q refresh losses.

Scores are gradients of SUM-reduced sampled-label NLL w.r.t. the complete
block output, divided by the precompute loss scale before storage. Independent
label draws share the same teacher forward. Original saliency and aggregated
Fisher use only draw zero and are not changed by the additional probes.
"""
import hashlib
import logging

import torch


def score_losses(delta, scores):
    """Return (sequence, token-only) losses; delta B,T,H; scores B,K,T,H.

Both normalize by 2*B*K*T. The token control drops ONLY t != s terms.
Neither mixes different sequences. Work one probe at a time to bound FP32
temporaries; cached scores are constants, not part of the backward graph.
"""
    if delta.ndim != 3 or scores.ndim != 4:
        raise ValueError("Expected delta B,T,H and scores B,K,T,H")
    if scores.shape[0] != delta.shape[0] or scores.shape[2:] != delta.shape[1:]:
        raise ValueError("Score sample/token/channel axes must match delta")
    if scores.requires_grad:
        raise ValueError("Fisher scores must be detached constants")
    work_dtype = torch.float64 if delta.dtype == torch.float64 else torch.float32
    d = delta.to(work_dtype)
    seq_loss = d.new_zeros(())
    token_loss = d.new_zeros(())
    for k in range(scores.shape[1]):
        projection = (d * scores[:, k].to(work_dtype)).sum(dim=-1)
        seq_loss = seq_loss + projection.sum(dim=-1).square().sum()
        token_loss = token_loss + projection.square().sum()
    denominator = 2 * delta.shape[0] * scores.shape[1] * delta.shape[1]
    return seq_loss / denominator, token_loss / denominator


class SequenceFisherCache:
    """CPU score cache plus at most one resident layer on GPU (single rank)."""

    def __init__(self, samples, probes, layer_ids, mode, mix):
        self.samples, self.probes = int(samples), int(probes)
        self.layer_ids = set(layer_ids)
        self.mode, self.mix = mode, float(mix)
        self.cpu = {}
        self.written = {}
        self.batch_start = 0
        self.probe = 0
        self.active = False
        self.gpu_layer = None
        self.gpu = None
        self.logged = set()

    def capture(self, layer_idx, grad, loss_scale):
        if not self.active or layer_idx not in self.layer_ids:
            return
        if layer_idx not in self.cpu:
            self.cpu[layer_idx] = torch.empty(
                self.samples, self.probes, *grad.shape[1:],
                dtype=torch.bfloat16, device="cpu",
            )
            self.written[layer_idx] = torch.zeros(self.samples, self.probes, dtype=torch.bool)
        end = self.batch_start + grad.shape[0]
        if self.written[layer_idx][self.batch_start:end, self.probe].any():
            raise RuntimeError("Duplicate Fisher score capture")
        score = (grad.detach().float() / float(loss_scale)).to(torch.bfloat16).cpu()
        if not torch.isfinite(score).all():
            raise RuntimeError("Non-finite Fisher score")
        self.cpu[layer_idx][self.batch_start:end, self.probe].copy_(score)
        self.written[layer_idx][self.batch_start:end, self.probe] = True

    def validate(self):
        if set(self.cpu) != self.layer_ids:
            raise RuntimeError("Missing Fisher score layer")
        for idx in sorted(self.layer_ids):
            if not self.written[idx].all():
                raise RuntimeError(f"Incomplete Fisher score cache for layer {idx}")
            tensor = self.cpu[idx]
            digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
            logging.info("SEQUENCE_FISHER_CACHE layer=%d shape=%s bytes=%d sha256=%s",
                         idx, tuple(tensor.shape), tensor.numel() * tensor.element_size(), digest)

    def batch(self, layer_idx, indices, dev):
        if self.gpu_layer != layer_idx:
            self.gpu = None
            self.gpu = self.cpu[layer_idx].to(dev)
            self.gpu_layer = layer_idx
        return self.gpu[indices]

    def blend(self, original_loss, delta, layer_idx, indices, dev, module_name):
        if self.mix == 0:
            return original_loss  # exact no-op control, including its arithmetic
        seq_loss, token_loss = score_losses(delta, self.batch(layer_idx, indices, dev))
        if self.mode == "cross":
            # The score difference cancels all t == s terms and keeps only
            # cross-token interactions, while retaining REAL-Q's aggregate
            # Fisher loss at full strength.
            selected = original_loss + self.mix * (seq_loss - token_loss)
        else:
            selected = (seq_loss if self.mode == "sequence" else token_loss)
        key = (layer_idx, module_name)
        if key not in self.logged:
            logging.info(
                "SEQUENCE_FISHER_REFRESH layer=%d module=%s mode=%s mix=%g "
                "aggregate=%g sequence=%g token=%g",
                layer_idx, module_name, self.mode, self.mix,
                original_loss.detach().item(), seq_loss.detach().item(), token_loss.detach().item(),
            )
            self.logged.add(key)
        if self.mode == "cross":
            return selected
        return (1.0 - self.mix) * original_loss + self.mix * selected

    def release(self, layer_idx):
        if self.gpu_layer == layer_idx:
            self.gpu, self.gpu_layer = None, None
        self.cpu.pop(layer_idx, None)
        self.written.pop(layer_idx, None)
