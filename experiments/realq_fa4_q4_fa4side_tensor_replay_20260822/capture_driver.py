#!/usr/bin/env python3
"""Capture Qwen3-4B's FA4-side post-RoPE Q/K/V and matching upstream dO.

This wrapper replaces only the Stage-0 loop inside its child process.  Model
loading, deterministic setup, weight rotation, and attention configuration are
still performed by ``realq.ptq``.  The replacement executes the exact first
global-loss batch with frozen SDPA labels, then returns a dummy ``StaticStats``
because the parent command exits immediately after precompute.
"""

from __future__ import annotations

from array import array
import json
import os
from pathlib import Path
import runpy
import traceback
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from experiments.realq_fa4_q4_fa4side_tensor_replay_20260822 import common
from realq import precompute
from realq.precompute import hooks as hooks_mod
from realq.precompute import static_e2e
from realq.utils import memory as mem_utils


class CaptureController:
    def __init__(self, root: Path, plan_fingerprint: str):
        self.root = root
        self.plan_fingerprint = plan_fingerprint
        self.attention_calls: dict[int, int] = {
            layer: 0 for layer in common.SELECTED_LAYERS
        }
        self.captured: dict[int, dict[str, torch.Tensor]] = {}
        self.production_layouts: dict[int, dict[str, Any]] = {}
        self.artifacts: dict[str, Any] = {}
        self.attention_mask_calls = 0
        self.attention_mask_contract: dict[str, Any] | None = None
        self.loss: float | None = None
        self.completed = False

    @staticmethod
    def _cpu_bf16_archive(value: torch.Tensor) -> torch.Tensor:
        """Store logical values in a declared contiguous archive container.

        Replay never passes this BHSD container directly to a kernel.  It
        reconstructs and validates the production transpose-stride first.
        """

        return value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()

    def _validate_fa4_contract(
        self,
        attention_mask: torch.Tensor | None,
        module: torch.nn.Module,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Bind the dense-causal contract used by the production FA4 path."""

        if attention_mask is not None:
            raise common.ReplayDiagnosticError(
                "FA4-side capture requires attention_mask=None"
            )
        raw_is_causal = kwargs.get("is_causal")
        module_is_causal = bool(getattr(module, "is_causal", True))
        effective_is_causal = (
            module_is_causal if raw_is_causal is None else bool(raw_is_causal)
        )
        if not effective_is_causal:
            raise common.ReplayDiagnosticError(
                "FA4-side capture stopped using causal attention"
            )
        contract = {
            "representation": "none_dense_unpadded",
            "attention_mask": None,
            "is_causal_argument": raw_is_causal,
            "module_is_causal": module_is_causal,
            "effective_is_causal": effective_is_causal,
        }
        if self.attention_mask_contract is None:
            self.attention_mask_contract = contract
        elif self.attention_mask_contract != contract:
            raise common.ReplayDiagnosticError(
                "decoder layers changed the frozen FA4 causal contract"
            )
        self.attention_mask_calls += 1

    def _attention_wrapper(self, original: Any):
        def wrapped(
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None,
            dropout: float = 0.0,
            scaling: float | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            layer = int(getattr(module, "layer_idx", -1))
            selected = layer in self.attention_calls
            self._validate_fa4_contract(attention_mask, module, kwargs)
            if float(dropout) != 0.0:
                raise common.ReplayDiagnosticError("attention dropout changed")
            actual_scale = common.SCALING if scaling is None else float(scaling)
            if actual_scale != common.SCALING:
                raise common.ReplayDiagnosticError(
                    f"attention scaling changed: {actual_scale}"
                )
            if kwargs.get("sliding_window") is not None:
                raise common.ReplayDiagnosticError("Qwen3-4B unexpectedly uses sliding window")
            if kwargs.get("softcap") is not None:
                raise common.ReplayDiagnosticError("Qwen3-4B unexpectedly uses softcap")

            if selected:
                if self.attention_calls[layer] != 0:
                    raise common.ReplayDiagnosticError(
                        f"selected layer {layer} attention ran more than once"
                    )
                expected = {
                    "q": common.QUERY_SHAPE,
                    "k": common.KEY_VALUE_SHAPE,
                    "v": common.KEY_VALUE_SHAPE,
                }
                for name, tensor in (("q", query), ("k", key), ("v", value)):
                    if tuple(tensor.shape) != expected[name] or tensor.dtype != common.TENSOR_DTYPE:
                        raise common.ReplayDiagnosticError(
                            f"layer {layer} {name} changed: {tuple(tensor.shape)}/{tensor.dtype}"
                        )
                common.validate_kernel_qkv({"q": query, "k": key, "v": value})
                self.attention_calls[layer] += 1
                self.captured[layer] = {
                    "q": self._cpu_bf16_archive(query),
                    "k": self._cpu_bf16_archive(key),
                    "v": self._cpu_bf16_archive(value),
                }
                self.production_layouts[layer] = {
                    name: {
                        "interface_bhsd": common.tensor_layout(tensor),
                        "adapter_bshd": common.tensor_layout(tensor.transpose(1, 2)),
                    }
                    for name, tensor in (("q", query), ("k", key), ("v", value))
                }

            output, weights = original(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                **kwargs,
            )
            if weights is not None:
                raise common.ReplayDiagnosticError("FA4 unexpectedly returned weights")

            if selected:
                if tuple(output.shape) != common.OUTPUT_SHAPE or output.dtype != common.TENSOR_DTYPE:
                    raise common.ReplayDiagnosticError(
                        f"layer {layer} attention output changed: "
                        f"{tuple(output.shape)}/{output.dtype}"
                    )
                if not output.requires_grad:
                    raise common.ReplayDiagnosticError("attention output has no gradient")

                def capture_dout(gradient: torch.Tensor, *, layer_index: int = layer):
                    record = self.captured[layer_index]
                    if "dout" in record:
                        raise common.ReplayDiagnosticError(
                            f"layer {layer_index} dO hook fired twice"
                        )
                    if tuple(gradient.shape) != common.OUTPUT_SHAPE:
                        raise common.ReplayDiagnosticError(
                            f"layer {layer_index} dO shape changed: {tuple(gradient.shape)}"
                        )
                    if gradient.dtype != common.TENSOR_DTYPE:
                        raise common.ReplayDiagnosticError(
                            f"layer {layer_index} dO dtype changed: {gradient.dtype}"
                        )
                    if not gradient.is_contiguous() or gradient.storage_offset() != 0:
                        raise common.ReplayDiagnosticError(
                            f"layer {layer_index} production dO layout changed: "
                            f"{common.tensor_layout(gradient)}"
                        )
                    self.production_layouts[layer_index]["dout"] = {
                        "interface_bshd": common.tensor_layout(gradient)
                    }
                    record["dout"] = self._cpu_bf16_archive(gradient)
                    return gradient

                output.register_hook(capture_dout)
            return output, weights

        return wrapped

    @staticmethod
    def _load_first_batch_tokens() -> tuple[list[torch.Tensor], torch.Tensor]:
        tokens = torch.load(common.TOKEN_PATH, map_location="cpu", weights_only=True)
        if not isinstance(tokens, list) or len(tokens) != 256:
            raise common.ReplayDiagnosticError("token artifact is not a 256-tensor list")
        if common.tensors_semantic_sha256(tokens) != common.TOKEN_SEMANTIC_SHA256:
            raise common.ReplayDiagnosticError("token semantic SHA256 changed")
        for index, tensor in enumerate(tokens):
            if tuple(tensor.shape) != (common.SEQUENCE_LENGTH,) or tensor.dtype != torch.int64:
                raise common.ReplayDiagnosticError(
                    f"token {index} shape/dtype changed: {tuple(tensor.shape)}/{tensor.dtype}"
                )
        first = torch.stack(tokens[: common.BATCH_SIZE], dim=0).contiguous()
        return tokens, first

    @staticmethod
    def _load_first_batch_labels() -> torch.Tensor:
        values = array("q")
        with common.FIXED_LABELS.open("rb") as handle:
            values.fromfile(handle, 256 * common.SEQUENCE_LENGTH)
        if len(values) != 256 * common.SEQUENCE_LENGTH:
            raise common.ReplayDiagnosticError("fixed-label element count changed")
        labels = torch.tensor(values, dtype=torch.long).reshape(
            256, common.SEQUENCE_LENGTH
        )
        return labels[: common.BATCH_SIZE].contiguous()

    @staticmethod
    def _matched_random(
        real: Mapping[str, torch.Tensor], layer: int
    ) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(common.RANDOM_CONTROL_SEED + int(layer))
        result: dict[str, torch.Tensor] = {}
        for name in ("q", "k", "v", "dout"):
            source = real[name]
            target_rms = source.float().square().mean().sqrt()
            candidate = torch.randn(
                source.shape,
                dtype=torch.float32,
                device="cpu",
                generator=generator,
            )
            candidate.mul_(target_rms / candidate.square().mean().sqrt())
            result[name] = candidate.to(dtype=torch.bfloat16).contiguous()
            del candidate
        common.validate_archive_qkvd(result)
        return result

    def _write_tensor_pair(self, layer: int) -> None:
        real = self.captured.pop(layer)
        common.validate_archive_qkvd(real)
        random = self._matched_random(real, layer)
        layer_records: dict[str, Any] = {}
        for kind, tensors in (("real", real), ("random", random)):
            path = self.root / "tensors" / f"layer{layer:02d}_{kind}.pt"
            common.atomic_torch_save(path, tensors)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                raise common.ReplayDiagnosticError("capture archive root changed")
            common.validate_archive_qkvd(loaded)
            semantic = common.named_tensors_semantic_sha256(loaded)
            layout_semantic = common.named_tensors_layout_sha256(loaded)
            if semantic != common.named_tensors_semantic_sha256(tensors):
                raise common.ReplayDiagnosticError("capture archive semantic roundtrip failed")
            if layout_semantic != common.named_tensors_layout_sha256(tensors):
                raise common.ReplayDiagnosticError("capture archive layout roundtrip failed")
            layer_records[kind] = {
                "path": str(path),
                "serialization_sha256": common.file_sha256(path),
                "semantic_sha256": semantic,
                "layout_semantic_sha256": layout_semantic,
                "representation": "contiguous_value_archive_not_kernel_input",
                "size_bytes": path.stat().st_size,
                "tensors": {
                    name: common.tensor_contract(value)
                    for name, value in sorted(loaded.items())
                },
            }
            del loaded
        self.artifacts[str(layer)] = layer_records

    def run(self, cfg: Any, analyzer: Any):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if (
            int(cfg.global_loss_bsz) != common.BATCH_SIZE
            or int(cfg.nsamples) != 256
            or int(cfg.seq_len) != common.SEQUENCE_LENGTH
            or str(cfg.attention_backend) != "flash_attention_4"
            or not bool(cfg.rotate)
            or int(cfg.seed) != common.CALIBRATION_SEED
            or int(cfg.rotation_seed) != common.ROTATION_SEED
            or int(cfg.refresh_seed) != common.REFRESH_SEED
            or int(cfg.grad_hessian_topk) != -1
        ):
            raise common.ReplayDiagnosticError("capture CLI scientific contract changed")
        if len(analyzer.get_layers()) != 36:
            raise common.ReplayDiagnosticError("Qwen3-4B layer count changed")

        _, input_ids_cpu = self._load_first_batch_tokens()
        labels_cpu = self._load_first_batch_labels()
        summary = common.read_json(common.FIXED_LABEL_SUMMARY)
        first_record = summary.get("records", [None])[0]
        if (
            summary.get("status") != "completed"
            or summary.get("backend") != "sdpa"
            or summary.get("labels_sha256") != common.FIXED_LABELS_SHA256
            or summary.get("invocations") != 64
            or not isinstance(first_record, dict)
            or first_record.get("global_sample_indices") != [0, 1, 2, 3]
            or first_record.get("shape") != [4, common.SEQUENCE_LENGTH]
            or first_record.get("base_seed") != 0
            or first_record.get("offset_labels") != 0
        ):
            raise common.ReplayDiagnosticError("fixed first-label record changed")

        model = analyzer.model
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        prior_use_cache = model.config.use_cache
        prior_training = bool(model.training)
        saved_requires_grad: list[tuple[torch.nn.Parameter, bool]] = []
        kick_off_handle = None
        if os.environ.get("FA_DISABLE_2CTA", "0") != "0":
            raise common.ReplayDiagnosticError(
                "FA4-side capture must use the production default backward kernel"
            )
        original_fa4 = ALL_ATTENTION_FUNCTIONS["flash_attention_4"]
        ALL_ATTENTION_FUNCTIONS["flash_attention_4"] = self._attention_wrapper(
            original_fa4
        )
        try:
            model.config.use_cache = False
            model.to(device)
            model.eval()
            saved_requires_grad = static_e2e._freeze_model_params(model)
            kick_off_handle = static_e2e._register_kick_off_hook(analyzer.get_layers()[0])
            input_ids = input_ids_cpu.to(device=device)
            labels = labels_cpu.to(device=device)
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if tuple(logits.shape[:2]) != (common.BATCH_SIZE, common.SEQUENCE_LENGTH):
                raise common.ReplayDiagnosticError("first-batch logits shape changed")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            self.loss = float(loss.detach().float().item())
            (loss * hooks_mod.LOSS_GRAD_SCALE).backward()
            torch.cuda.synchronize()
            if hooks_mod.LOSS_GRAD_SCALE != common.LOSS_GRAD_SCALE:
                raise common.ReplayDiagnosticError("production LOSS_GRAD_SCALE changed")
            for layer in common.SELECTED_LAYERS:
                if self.attention_calls[layer] != 1:
                    raise common.ReplayDiagnosticError(
                        f"layer {layer} capture calls={self.attention_calls[layer]}"
                    )
                if set(self.captured.get(layer, {})) != {"q", "k", "v", "dout"}:
                    raise common.ReplayDiagnosticError(
                        f"layer {layer} capture is incomplete"
                    )
            if self.attention_mask_calls != 36 or self.attention_mask_contract is None:
                raise common.ReplayDiagnosticError(
                    "FA4 dense-causal contract was not observed at every decoder layer"
                )
            del outputs, logits, labels, loss, input_ids
            model.zero_grad(set_to_none=True)
            for layer in common.SELECTED_LAYERS:
                self._write_tensor_pair(layer)
            self.completed = True
        finally:
            ALL_ATTENTION_FUNCTIONS["flash_attention_4"] = original_fa4
            if kick_off_handle is not None:
                kick_off_handle.remove()
            if saved_requires_grad:
                static_e2e._restore_requires_grad(saved_requires_grad)
            model.config.use_cache = prior_use_cache
            model.train(prior_training)
            model.cpu()
            mem_utils.cleanup_memory()

        # The enclosing production pipeline logs these fields, then obeys
        # exit_after_precompute=True.  No saliency/Fisher cache is produced.
        return precompute.StaticStats(
            saliency=[{} for _ in range(36)],
            fisher=[torch.empty(0) for _ in range(36)],
        )


def _validate_frozen_inputs() -> None:
    expected = {
        common.SOURCE_PRODUCER_RECEIPT: common.SOURCE_PRODUCER_RECEIPT_SHA256,
        common.TOKEN_PATH: common.TOKEN_SERIALIZATION_SHA256,
        common.FIXED_LABEL_RESULT: common.FIXED_LABEL_RESULT_SHA256,
        common.FIXED_LABEL_SUMMARY: common.FIXED_LABEL_SUMMARY_SHA256,
        common.FIXED_LABELS: common.FIXED_LABELS_SHA256,
    }
    for path, sha256 in expected.items():
        if not path.is_file() or common.file_sha256(path) != sha256:
            raise common.ReplayDiagnosticError(f"frozen capture input changed: {path}")


def main() -> int:
    root_value = os.environ.get("REALQ_TRUE_TENSOR_CAPTURE_ROOT")
    fingerprint = os.environ.get("REALQ_TRUE_TENSOR_PLAN_FINGERPRINT")
    if not root_value or not fingerprint:
        raise common.ReplayDiagnosticError("capture environment is incomplete")
    root = Path(root_value).resolve()
    if root.exists() or root.is_symlink():
        raise common.ReplayDiagnosticError(f"capture root is not fresh: {root}")
    _validate_frozen_inputs()
    root.mkdir(parents=True)
    controller = CaptureController(root, fingerprint)
    original_run = precompute.run
    exit_code = 1
    failure: dict[str, Any] | None = None
    try:
        precompute.run = controller.run
        runpy.run_module("realq.ptq", run_name="__main__", alter_sys=True)
        exit_code = 0
    except SystemExit as exc:
        exit_code = 0 if exc.code is None else int(exc.code)
        if exit_code:
            failure = {"kind": "SystemExit", "code": exit_code}
    except BaseException as exc:
        failure = {"kind": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        precompute.run = original_run

    completed = exit_code == 0 and controller.completed
    receipt = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "kind": "realq_qwen3_4b_fa4side_true_attention_tensor_capture",
        "plan_fingerprint": fingerprint,
        "selected_layers": list(common.SELECTED_LAYERS),
        "batch_size": common.BATCH_SIZE,
        "sequence_length": common.SEQUENCE_LENGTH,
        "qkv_boundary": "post_qk_norm_post_rope_pre_attention_backend",
        "dout_boundary": "attention_backend_output_pre_reshape_pre_o_proj",
        "attention_backend": "flash_attention_4",
        "fa_disable_2cta": os.environ.get("FA_DISABLE_2CTA", "0"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "attention_mask": {
            "calls": controller.attention_mask_calls,
            "contract": controller.attention_mask_contract,
            "passed_to_fa4_unchanged": True,
        },
        "fixed_label_source": str(common.FIXED_LABELS),
        "fixed_label_sha256": common.FIXED_LABELS_SHA256,
        "first_batch_indices": list(range(common.BATCH_SIZE)),
        "loss_reduction": "sum",
        "unscaled_loss": controller.loss,
        "loss_grad_scale": common.LOSS_GRAD_SCALE,
        "attention_calls": controller.attention_calls,
        "production_layouts": {
            str(layer): controller.production_layouts.get(layer)
            for layer in common.SELECTED_LAYERS
        },
        "production_layouts_sha256": common.canonical_sha256(
            {
                str(layer): controller.production_layouts.get(layer)
                for layer in common.SELECTED_LAYERS
            }
        ),
        "artifacts": controller.artifacts,
        "expected_bytes_per_real_or_random_layer": common.expected_capture_bytes_per_kind(),
        "exit_code": exit_code,
        "failure": failure,
    }
    common.atomic_json(root / "capture_receipt.json", receipt)
    if not completed:
        raise common.ReplayDiagnosticError("true-tensor capture did not complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
