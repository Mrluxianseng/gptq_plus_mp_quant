"""Isolate the Qwen3-0.6B full-sequence eval tensor lifetimes on CUDA.

This is a diagnostic, not a full REAL-Q benchmark: it uses Qwen3-0.6B's
actual LM-head dimensions/dtype with deterministic synthetic hidden states.
It measures the existing full-sequence LM-head + CE path and compares only
the token chunk size used for the KL intermediate calculation.
"""

import gc
import json
import time

import torch
import torch.nn.functional as F


def snapshot(stage):
    torch.cuda.synchronize()
    return {
        "stage": stage,
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


def run(chunk_tokens, head, student_hidden, teacher_hidden, input_ids):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    stages = [snapshot("start")]

    teacher_logits_cpu = F.linear(teacher_hidden, head).cpu()
    stages.append(snapshot("teacher_logits_to_cpu"))
    student_logits = F.linear(student_hidden, head)
    stages.append(snapshot("student_full_logits_bf16"))

    student_logits_fp32 = student_logits.float()
    nll = F.cross_entropy(
        student_logits_fp32[:, :-1, :].permute(0, 2, 1),
        input_ids[:, 1:],
        reduction="none",
    ).float().mean(dim=1)[0]
    del student_logits_fp32
    stages.append(snapshot("full_sequence_cross_entropy"))

    kl_parts = []
    for start in range(0, student_hidden.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, student_hidden.shape[1])
        s = student_logits[:, start:end, :].float()
        t = teacher_logits_cpu[:, start:end, :].to("cuda").float()
        tlp = F.log_softmax(t, dim=-1)
        slp = F.log_softmax(s, dim=-1)
        kl_parts.append((tlp.exp() * (tlp - slp)).sum(dim=-1).reshape(-1))
        del s, t, tlp, slp
    kl = torch.cat(kl_parts).mean()
    stages.append(snapshot("chunked_kl_complete"))
    result = (float(nll), float(kl))
    elapsed = time.perf_counter() - t0
    peak = snapshot("final_peak")
    del student_logits, teacher_logits_cpu, kl_parts, nll, kl
    return {"chunk_tokens": chunk_tokens, "metrics": result, "seconds": elapsed, "stages": stages, "peak": peak}


def probe_cross_entropy(student_logits, input_ids, chunk_tokens):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    labels = input_ids[:, 1:]
    logits = student_logits[:, :-1, :]
    per_token = F.cross_entropy(
        logits.float().permute(0, 2, 1), labels, reduction="none"
    ).float()
    full_nll = per_token.mean(dim=1)[0]
    full_peak = snapshot("full_ce")
    full_value = float(full_nll)
    del per_token, full_nll

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    chunk_sum = torch.zeros((), device="cuda", dtype=torch.float64)
    for start in range(0, logits.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, logits.shape[1])
        part = F.cross_entropy(
            logits[:, start:end, :].float().permute(0, 2, 1),
            labels[:, start:end],
            reduction="sum",
        )
        chunk_sum += part.double()
        del part
    chunked_value = float(chunk_sum / logits.shape[1])
    chunk_peak = snapshot("chunked_ce")
    return {
        "chunk_tokens": chunk_tokens,
        "full_nll": full_value,
        "chunked_nll": chunked_value,
        "abs_diff": abs(full_value - chunked_value),
        "full_peak_allocated": full_peak["peak_allocated"],
        "chunk_peak_allocated": chunk_peak["peak_allocated"],
        "full_peak_reserved": full_peak["peak_reserved"],
        "chunk_peak_reserved": chunk_peak["peak_reserved"],
        "seconds": time.perf_counter() - t0,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this memory diagnostic")
    torch.manual_seed(20261008)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, seq, hidden, vocab = 1, 1024, 1024, 151936
    head_weight = torch.empty(vocab, hidden, device=device, dtype=dtype).normal_(std=0.02)
    student_hidden = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    teacher_hidden = student_hidden + 0.002 * torch.randn_like(student_hidden)
    input_ids = torch.randint(vocab, (batch, seq), device=device)
    outputs = [run(c, head_weight, student_hidden, teacher_hidden, input_ids) for c in (16, 8, 4)]
    base = outputs[0]["metrics"]
    for item in outputs:
        item["metric_abs_diff_from_chunk16"] = [abs(a - b) for a, b in zip(item["metrics"], base)]
    # Reuse one full-shape BF16 head output to isolate whether the CE
    # intermediate, rather than the KL chunk, controls the peak.
    logits = F.linear(student_hidden, head_weight)
    ce_outputs = [probe_cross_entropy(logits, input_ids, c) for c in (256, 64, 16)]
    print(json.dumps({"device": torch.cuda.get_device_name(), "dtype": str(dtype), "shape": [batch, seq, hidden, vocab], "runs": outputs, "cross_entropy_chunk_probe": ce_outputs}, indent=2))


if __name__ == "__main__":
    main()
