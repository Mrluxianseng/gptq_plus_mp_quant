"""Scoring-only bounded GPU LM-head workspace with complete CPU logits.

Chunked GEMM must pass real GPU byte equality; theoretical equality is insufficient.
"""
import ast
import inspect
import textwrap
import torch
from tools.qa_score_lifetime_prototype import build_source


class CpuLogitsHead:
    def __init__(self, head, *, enabled, chunk_rows=256, mark=lambda name: None):
        if type(head) is not torch.nn.Linear:
            raise RuntimeError('Only original Linear LM head is supported')
        if head.training or chunk_rows <= 0:
            raise RuntimeError('Requires eval head and positive chunk size')
        if enabled and head.weight.dtype != torch.bfloat16:
            raise RuntimeError('Only BF16 head is admitted for exact GPU screening; FP32 chunking differs')
        self.head = head
        self.enabled = bool(enabled)
        self.chunk_rows = int(chunk_rows)
        self.mark = mark
        self.original = head.forward
        self.records = []
        self.weight_identity = id(head.weight)
        head.forward = self.forward

    def forward(self, hidden):
        if hidden.ndim != 3 or not hidden.is_contiguous():
            raise RuntimeError('Expected contiguous full batch hidden states')
        call = len(self.records) + 1
        self.mark(f'forward{call}_head_start')
        if not self.enabled:
            result = self.original(hidden)
            chunks = 1
        else:
            flat = hidden.view(-1, hidden.shape[-1])
            result = torch.empty((flat.shape[0], self.head.out_features), dtype=hidden.dtype, device='cpu')
            chunks = 0
            for start in range(0, flat.shape[0], self.chunk_rows):
                end = min(start + self.chunk_rows, flat.shape[0])
                block = self.original(flat[start:end])
                result[start:end].copy_(block, non_blocking=False)
                del block
                chunks += 1
            result = result.view(*hidden.shape[:-1], self.head.out_features)
        self.records.append(dict(forward=call, enabled=self.enabled, chunk_rows=self.chunk_rows,
            input_shape=list(hidden.shape), input_dtype=str(hidden.dtype),
            input_device=str(hidden.device), output_shape=list(result.shape),
            output_dtype=str(result.dtype), output_device=str(result.device), chunks=chunks))
        self.mark(f'forward{call}_head_end')
        return result

    def close(self):
        self.head.forward = self.original
        if id(self.head.weight) != self.weight_identity:
            raise RuntimeError('LM-head Parameter identity changed')


def gpu_scored_rows_method(original):
    """Copy only selected continuation rows back to the original scoring GPU."""
    tree = ast.parse(build_source(inspect.getsource(original)))
    targets = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and ast.unparse(n) == 'F.log_softmax(logits, dim=-1)']
    if len(targets) != 1:
        raise RuntimeError('Scored-row normalization contract changed')
    targets[0].args[0] = ast.parse('logits.to(self.device)', mode='eval').body
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, 'qa_gpu_scored_rows_from_cpu_logits', 'exec'), namespace)
    return namespace['_loglikelihood_tokens']
