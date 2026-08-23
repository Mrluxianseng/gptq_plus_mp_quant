"""Shared contracts for the Qwen3-4B true-tensor attention replay.

The experiment deliberately lives outside production REAL-Q code.  It captures
the tensors passed to the attention backend after Q/K norm and RoPE, freezes the
upstream gradient at the attention output, and replays local attention
forward/backward implementations on byte-identical inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT.parent / "experiment_data"
OUTPUT_ROOT = DATA_ROOT / "realq_fa4_q4_true_tensor_replay_20260822_v2"
PLAN_PATH = OUTPUT_ROOT / "plan.json"
LOCK_ROOT = DATA_ROOT / "_fair20_physical_gpu_locks_20260821"
HOST = "j-zogxxxduju-master-0"

MODEL_SLUG = "qwen3-4b"
MODEL_PATH = REPO_ROOT / "modelzoo/Qwen3/Qwen3-4B"
MODEL_CONFIG_SHA256 = (
    "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"
)
MODEL_WEIGHT_SHA256 = {
    "model-00001-of-00003.safetensors": (
        "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"
    ),
    "model-00002-of-00003.safetensors": (
        "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5"
    ),
    "model-00003-of-00003.safetensors": (
        "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1"
    ),
}

TOKEN_PATH = (
    DATA_ROOT
    / "realq_20group_20260808/shared_cache/qwen3-4b/tokens"
    / "Qwen3-4B_wikitext2_train_n256_sl2048_seed1.pt"
)
TOKEN_SERIALIZATION_SHA256 = (
    "21210e1929aa90ea23572e7904f8ebda7b3af75ebf8d9037e8196b3d2c52a399"
)
# Canonical list-order hash: for every tensor hash shape, dtype, and raw bytes.
TOKEN_SEMANTIC_SHA256 = (
    "5b7bd51b6f896a4b79e70f8ffa867974440dc5c63b6c8af7668c013d76998535"
)

SOURCE_PRODUCER_RECEIPT = (
    DATA_ROOT
    / "realq_fullmodel_retune_20260818_v6_run15/shared_cache/qwen3-4b"
    / "producer_success.json"
)
SOURCE_PRODUCER_RECEIPT_SHA256 = (
    "2405c6c6fa339e27e3b5d02f26564481c96c8f36e58d3231210ed533d047d2aa"
)

FIXED_LABEL_RESULT = (
    DATA_ROOT / "realq_backend_label_diagnostic_20260822_v1/sdpa/result.json"
)
FIXED_LABEL_RESULT_SHA256 = (
    "9634096ecf8bb50a13b07bed6a8460b05c87016d62bc7bf7912c86d20116d59b"
)
FIXED_LABEL_SUMMARY = (
    DATA_ROOT
    / "realq_backend_label_diagnostic_20260822_v1/sdpa/capture"
    / "capture_summary.json"
)
FIXED_LABEL_SUMMARY_SHA256 = (
    "8166f39657e693412fd8c61c2670b7dbe15de7be576f92c8a0053d95a3f98891"
)
FIXED_LABELS = (
    DATA_ROOT
    / "realq_backend_label_diagnostic_20260822_v1/sdpa/capture/labels.i64"
)
FIXED_LABELS_SHA256 = (
    "ae23e11d9096e7370ea7db3913a940f800510d8e06e2393a7841814358015741"
)

ALL_REQUESTED_LAYERS = (0, 5, 6, 12, 13, 35)
# The first immutable run is intentionally the smallest causal closure.  Once
# layer 6 establishes which local kernel is responsible, the remaining layers
# get a distinct follow-up output root/plan instead of mutating this run.
SELECTED_LAYERS = (6,)
BATCH_SIZE = 4
SEQUENCE_LENGTH = 2048
QUERY_HEADS = 32
KEY_VALUE_HEADS = 8
HEAD_DIM = 128
SCALING = HEAD_DIM**-0.5
QUERY_SHAPE = (BATCH_SIZE, QUERY_HEADS, SEQUENCE_LENGTH, HEAD_DIM)
KEY_VALUE_SHAPE = (BATCH_SIZE, KEY_VALUE_HEADS, SEQUENCE_LENGTH, HEAD_DIM)
OUTPUT_SHAPE = (BATCH_SIZE, SEQUENCE_LENGTH, QUERY_HEADS, HEAD_DIM)
TENSOR_DTYPE = torch.bfloat16
LOSS_GRAD_SCALE = 1000.0

CALIBRATION_SEED = 1
ROTATION_SEED = 0
REFRESH_SEED = 0
RANDOM_CONTROL_SEED = 2026082204
REPLAY_SEED = 2026082205

ARMS = ("math_sdpa", "fa4_default", "fa4_no2cta")
PAIRWISE_COMPARISONS = (
    ("fp32_reference", "math_sdpa"),
    ("fp32_reference", "fa4_default"),
    ("fp32_reference", "fa4_no2cta"),
    ("math_sdpa", "fa4_default"),
    ("math_sdpa", "fa4_no2cta"),
    ("fa4_default", "fa4_no2cta"),
)
OUTPUT_FIELDS = ("output", "dq", "dk", "dv")


class ReplayDiagnosticError(RuntimeError):
    """An immutable-input or numerical-replay contract was violated."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayDiagnosticError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _update_tensor_digest(digest: "hashlib._Hash", tensor: torch.Tensor) -> None:
    value = tensor.detach().to(device="cpu").contiguous()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"|")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"|")
    byte_view = value.view(torch.uint8).reshape(-1)
    # Avoid a second multi-GB temporary for future larger uses.
    chunk = 8 * 1024 * 1024
    for start in range(0, int(byte_view.numel()), chunk):
        digest.update(byte_view[start : start + chunk].numpy().tobytes(order="C"))


def tensors_semantic_sha256(tensors: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            raise ReplayDiagnosticError("semantic hash input is not a tensor")
        _update_tensor_digest(digest, tensor)
    return digest.hexdigest()


def named_tensors_semantic_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        _update_tensor_digest(digest, tensors[name])
        digest.update(b"\0")
    return digest.hexdigest()


def tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
    """Return layout metadata without normalising the tensor."""

    return {
        "shape": list(map(int, tensor.shape)),
        "dtype": str(tensor.dtype),
        "stride": list(map(int, tensor.stride())),
        "storage_offset": int(tensor.storage_offset()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def named_tensors_layout_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Bind tensor values and layout while retaining the content-only SHA."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_bytes(tensor_layout(tensor)))
        digest.update(b"\0")
        _update_tensor_digest(digest, tensor)
        digest.update(b"\0")
    return digest.hexdigest()


def tensor_contract(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu", memory_format=torch.preserve_format)
    if not torch.isfinite(value.float()).all().item():
        raise ReplayDiagnosticError("captured/replayed tensor contains non-finite values")
    return {
        **tensor_layout(value),
        "numel": int(value.numel()),
        "nbytes": int(value.numel() * value.element_size()),
        "semantic_sha256": tensors_semantic_sha256((value,)),
        "layout_semantic_sha256": named_tensors_layout_sha256({"tensor": value}),
        "rms": float(value.float().square().mean().sqrt().item()),
        "max_abs": float(value.float().abs().max().item()),
    }


def validate_qkvd(tensors: Mapping[str, torch.Tensor], *, batch: int = BATCH_SIZE) -> None:
    expected = {
        "q": (batch, QUERY_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
        "k": (batch, KEY_VALUE_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
        "v": (batch, KEY_VALUE_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
        "dout": (batch, SEQUENCE_LENGTH, QUERY_HEADS, HEAD_DIM),
    }
    if set(tensors) != set(expected):
        raise ReplayDiagnosticError(
            f"QKV+dO keys changed: {sorted(tensors)} != {sorted(expected)}"
        )
    for name, shape in expected.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape or tensor.dtype != TENSOR_DTYPE:
            raise ReplayDiagnosticError(
                f"{name} contract changed: shape={tuple(tensor.shape)} "
                f"dtype={tensor.dtype}, expected={shape}/{TENSOR_DTYPE}"
            )
        if not torch.isfinite(tensor.float()).all().item():
            raise ReplayDiagnosticError(f"{name} contains non-finite values")


def _contiguous_stride(shape: Sequence[int]) -> tuple[int, ...]:
    stride = []
    running = 1
    for dimension in reversed(tuple(map(int, shape))):
        stride.append(running)
        running *= dimension
    return tuple(reversed(stride))


def production_bhsd_stride(heads: int) -> tuple[int, int, int, int]:
    """Stride of a contiguous BSHD allocation viewed as BHSD."""

    return (
        SEQUENCE_LENGTH * int(heads) * HEAD_DIM,
        HEAD_DIM,
        int(heads) * HEAD_DIM,
        1,
    )


def validate_archive_qkvd(
    tensors: Mapping[str, torch.Tensor], *, batch: int = BATCH_SIZE
) -> None:
    """Validate the intentionally contiguous value-container representation."""

    validate_qkvd(tensors, batch=batch)
    for name, tensor in tensors.items():
        expected = _contiguous_stride(tensor.shape)
        if (
            tuple(tensor.stride()) != expected
            or tensor.storage_offset() != 0
            or not tensor.is_contiguous()
        ):
            raise ReplayDiagnosticError(
                f"archive {name} layout changed: {tensor_layout(tensor)}; "
                f"expected contiguous stride={expected}, storage_offset=0"
            )


def validate_kernel_qkv(
    tensors: Mapping[str, torch.Tensor], *, batch: int = BATCH_SIZE
) -> None:
    """Validate production BHSD and the adapter's contiguous BSHD views."""

    expected_shapes = {
        "q": (batch, QUERY_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
        "k": (batch, KEY_VALUE_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
        "v": (batch, KEY_VALUE_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
    }
    if set(tensors) != set(expected_shapes):
        raise ReplayDiagnosticError("kernel Q/K/V keys changed")
    expected_heads = {"q": QUERY_HEADS, "k": KEY_VALUE_HEADS, "v": KEY_VALUE_HEADS}
    for name, heads in expected_heads.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != expected_shapes[name] or tensor.dtype != TENSOR_DTYPE:
            raise ReplayDiagnosticError(
                f"kernel {name} value contract changed: {tensor_layout(tensor)}"
            )
        expected = production_bhsd_stride(heads)
        adapter_value = tensor.transpose(1, 2)
        adapter_expected = _contiguous_stride(adapter_value.shape)
        if (
            tuple(tensor.stride()) != expected
            or tensor.storage_offset() != 0
            or tensor.is_contiguous()
            or tuple(adapter_value.stride()) != adapter_expected
            or adapter_value.storage_offset() != 0
            or not adapter_value.is_contiguous()
        ):
            raise ReplayDiagnosticError(
                f"kernel {name} is not production BHSD->contiguous BSHD: "
                f"original={tensor_layout(tensor)}, adapter={tensor_layout(adapter_value)}"
            )


def validate_kernel_qkvd(
    tensors: Mapping[str, torch.Tensor], *, batch: int = BATCH_SIZE
) -> None:
    """Validate the exact layouts seen by production attention interfaces."""

    validate_qkvd(tensors, batch=batch)
    validate_kernel_qkv(
        {name: tensors[name] for name in ("q", "k", "v")}, batch=batch
    )
    dout = tensors["dout"]
    expected_dout = _contiguous_stride(dout.shape)
    if (
        tuple(dout.stride()) != expected_dout
        or dout.storage_offset() != 0
        or not dout.is_contiguous()
    ):
        raise ReplayDiagnosticError(
            f"kernel dO is not contiguous BSHD: {tensor_layout(dout)}"
        )


def reconstruct_kernel_qkvd(
    archive: Mapping[str, torch.Tensor], *, batch: int
) -> dict[str, torch.Tensor]:
    """Rebuild production non-contiguous BHSD views from value containers."""

    validate_archive_qkvd(archive, batch=batch)
    result = {
        name: (
            value.transpose(1, 2).contiguous().transpose(1, 2)
            if name in {"q", "k", "v"}
            else value
        )
        for name, value in archive.items()
    }
    validate_kernel_qkvd(result, batch=batch)
    return result


def validate_outcome(tensors: Mapping[str, torch.Tensor], *, batch: int) -> None:
    expected = {
        "output": (batch, SEQUENCE_LENGTH, QUERY_HEADS, HEAD_DIM),
        "dq": (batch, QUERY_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
        "dk": (batch, KEY_VALUE_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
        "dv": (batch, KEY_VALUE_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
    }
    if set(tensors) != set(expected):
        raise ReplayDiagnosticError("replay outcome keys changed")
    for name, shape in expected.items():
        value = tensors[name]
        if tuple(value.shape) != shape:
            raise ReplayDiagnosticError(
                f"{name} replay shape changed: {tuple(value.shape)} != {shape}"
            )
        if not torch.isfinite(value.float()).all().item():
            raise ReplayDiagnosticError(f"non-finite replay output: {name}")


def case_definitions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "real_b4",
            "source": "real",
            "source_indices": [0, 1, 2, 3],
            "batch": 4,
        },
        *(
            {
                "name": f"real_b1_s{sample}",
                "source": "real",
                "source_indices": [sample],
                "batch": 1,
            }
            for sample in range(BATCH_SIZE)
        ),
        {
            "name": "real_b4_perm_s3_to_s0",
            "source": "real",
            "source_indices": [3, 0, 1, 2],
            "batch": 4,
        },
        {
            "name": "real_b4_s3_repeated",
            "source": "real",
            "source_indices": [3, 3, 3, 3],
            "batch": 4,
        },
        {
            "name": "random_b4",
            "source": "random",
            "source_indices": [0, 1, 2, 3],
            "batch": 4,
        },
    )


def slice_case(
    tensors: Mapping[str, torch.Tensor], definition: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    validate_archive_qkvd(tensors)
    indices = tuple(map(int, definition.get("source_indices", ())))
    if len(indices) != int(definition["batch"]):
        raise ReplayDiagnosticError(f"case source indices changed: {definition}")
    if any(index < 0 or index >= BATCH_SIZE for index in indices):
        raise ReplayDiagnosticError(f"invalid case source indices: {indices}")
    if indices == tuple(range(BATCH_SIZE)):
        result = dict(tensors)
    else:
        index_tensor = torch.tensor(indices, dtype=torch.long, device="cpu")
        result = {
            name: value.index_select(0, index_tensor)
            for name, value in tensors.items()
        }
    validate_archive_qkvd(result, batch=int(definition["batch"]))
    return result


def b4_equivalent_case_count() -> float:
    return sum(int(item["batch"]) for item in case_definitions()) / BATCH_SIZE


def explicit_fp32_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scaling: float = SCALING,
) -> torch.Tensor:
    """Unfused causal GQA reference; Q/K/V must already include QK norm/RoPE."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ReplayDiagnosticError("explicit attention expects rank-4 Q/K/V")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ReplayDiagnosticError("explicit attention Q/K/V geometry changed")
    if q.shape[1] % k.shape[1]:
        raise ReplayDiagnosticError("Q heads are not divisible by KV heads")
    groups = int(q.shape[1] // k.shape[1])
    q32 = q.float()
    k32 = k.float().repeat_interleave(groups, dim=1)
    v32 = v.float().repeat_interleave(groups, dim=1)
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * float(scaling)
    sequence = int(q.shape[2])
    causal_mask = torch.ones(
        (sequence, sequence), dtype=torch.bool, device=q.device
    ).triu(diagonal=1)
    scores.masked_fill_(causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities, v32).transpose(1, 2).contiguous()


def full_tensor_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_elements: int = 1_048_576,
) -> dict[str, Any]:
    """Exact-all-element metrics with float64 accumulation in bounded chunks."""

    if left.shape != right.shape:
        raise ReplayDiagnosticError(
            f"metric shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    lhs = left.detach().reshape(-1)
    rhs = right.detach().reshape(-1)
    count = int(lhs.numel())
    left_square = right_square = difference_square = cross = absolute_sum = 0.0
    unequal = 0
    max_abs = 0.0
    for start in range(0, count, int(chunk_elements)):
        stop = min(count, start + int(chunk_elements))
        a = lhs[start:stop].to(dtype=torch.float64)
        b = rhs[start:stop].to(dtype=torch.float64)
        delta = b - a
        left_square += float(torch.dot(a, a).item())
        right_square += float(torch.dot(b, b).item())
        difference_square += float(torch.dot(delta, delta).item())
        cross += float(torch.dot(a, b).item())
        absolute_sum += float(delta.abs().sum().item())
        unequal += int(torch.count_nonzero(a != b).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        del a, b, delta
    left_rms = math.sqrt(left_square / count)
    right_rms = math.sqrt(right_square / count)
    difference_rms = math.sqrt(difference_square / count)
    denominator = math.sqrt(left_square * right_square)
    return {
        "all_values": True,
        "values": count,
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "unequal_values_after_fp64_cast": unequal,
        "unequal_fraction_after_fp64_cast": unequal / count,
        "left_rms": left_rms,
        "right_rms": right_rms,
        "right_to_left_rms_ratio": right_rms / left_rms if left_rms else None,
        "difference_rms": difference_rms,
        "relative_rms_to_left": difference_rms / left_rms if left_rms else None,
        "relative_rms_to_right": difference_rms / right_rms if right_rms else None,
        "cosine": cross / denominator if denominator else None,
        "mean_abs_difference": absolute_sum / count,
        "max_abs_difference": max_abs,
    }


def expected_capture_bytes_per_kind() -> int:
    elements = (
        math.prod(QUERY_SHAPE)
        + 2 * math.prod(KEY_VALUE_SHAPE)
        + math.prod(OUTPUT_SHAPE)
    )
    return elements * torch.tensor([], dtype=TENSOR_DTYPE).element_size()
