"""High-level orchestration for RealQ.

    parse_cli  →  Config
                 │
                 ▼
              pipeline.run(cfg)
                 ├── load model + tokenizer
                 ├── (optional) reference logits for KL eval
                 ├── (optional) rotate
                 ├── stage 1: static precompute (saliency + Fisher)
                 ├── stage 2: per-layer quantisation (sub-task 3+)
                 └── (optional) PPL/KL eval
"""
from __future__ import annotations

import copy
import logging
import os
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig

from realq_moe import akv, fsdp as realq_fsdp, model_adapter, precompute, runner
from realq_moe.parallel import env as parallel_env
from realq_moe.utils import memory as mem_utils
from realq_moe.utils import nvtx
from utils.loss_utils import tokenwise_kl_from_logits
from utils import (
    checkpoint_utils,
    data_utils,
    dist_utils,
    eval_utils,
    model_utils,
    quant_utils,
    rotation_utils,
)

if TYPE_CHECKING:
    from realq_moe.config import Config


def _is_sparse_moe_analyzer(analyzer: model_utils.ModelAnalyzer) -> bool:
    return any(
        model_adapter.is_sparse_moe_layer(layer)
        for layer in analyzer.get_layers()
    )


def _model_source_declares_sparse_moe(model_source: object) -> bool:
    """Probe architecture without loading weights for cpu-master fail-closed."""

    if isinstance(model_source, str):
        config = AutoConfig.from_pretrained(
            model_source,
            trust_remote_code=True,
        )
    else:
        config = getattr(model_source, "config", None)
        if config is None:
            return False
    architectures = tuple(getattr(config, "architectures", ()) or ())
    return any(
        architecture in model_adapter.SUPPORTED_MOE_ARCHITECTURES
        for architecture in architectures
    )


def _assert_module_cuda(
    module: torch.nn.Module,
    dev: torch.device,
    *,
    label: str,
) -> None:
    expected_index = (
        dev.index if dev.index is not None else torch.cuda.current_device()
    )
    for tensor_name, tensor in (
        list(module.named_parameters(recurse=True))
        + list(module.named_buffers(recurse=True))
    ):
        if tensor.device.type != "cuda" or tensor.device.index != expected_index:
            raise RuntimeError(
                f"{label}.{tensor_name} must remain on cuda:{expected_index}; "
                f"got {tensor.device}."
            )


def _iter_tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_tensor_leaves(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_tensor_leaves(child)


def _assert_static_stats_cuda(
    static,
    dev: torch.device,
) -> None:
    expected_index = (
        dev.index if dev.index is not None else torch.cuda.current_device()
    )
    for family_name in ("saliency", "fisher", "routes"):
        family = getattr(static, family_name)
        tensor_count = 0
        for tensor in _iter_tensor_leaves(family):
            tensor_count += 1
            if (
                tensor.device.type != "cuda"
                or tensor.device.index != expected_index
            ):
                raise RuntimeError(
                    "Sparse REAL-Q MoE Stage-0/Stage-1 contract requires "
                    f"{family_name} tensors on cuda:{expected_index}; "
                    f"got {tensor.device}."
                )
        if tensor_count == 0:
            raise RuntimeError(
                f"Sparse REAL-Q MoE Stage-0 produced no {family_name} tensors."
            )


def _activate_moe_gpu_resident_runtime(
    cfg: "Config",
    analyzer: model_utils.ModelAnalyzer,
    *,
    upload_model: bool = True,
) -> bool:
    """Validate the sparse contract and optionally establish CUDA residency.

    The source checkpoint and copied rotation implementation are CPU-backed.
    With evaluation enabled we must upload once before rotation to capture the
    immutable FP target, then re-establish residency after rotation.  With
    evaluation skipped, ``upload_model=False`` avoids that unnecessary first
    whole-model transfer and the post-rotation call performs the sole upload.
    """

    sparse_moe = _is_sparse_moe_analyzer(analyzer)
    cfg.validate_moe_runtime_contract(sparse_moe=sparse_moe)
    if not sparse_moe:
        return False
    if parallel_env.get_world_size() != 1:
        raise ValueError(
            "The first joint/GPU-resident Qwen3-MoE runtime requires "
            "world_size=1."
        )
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    # ModelAnalyzer captures a state_dict at load time. Nothing in the runtime
    # consumes it; retaining those tensor aliases would keep the source
    # checkpoint storage alive when the module is later transferred.
    analyzer.state_dict = None
    if not upload_model:
        return True
    analyzer.model.to(dev)
    analyzer.model.eval()
    _assert_module_cuda(analyzer.model, dev, label="analyzer.model")
    return True


def _extract_last_hidden(outputs) -> torch.Tensor:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        hidden = outputs[0] if isinstance(outputs, tuple) else None
    if not torch.is_tensor(hidden):
        raise RuntimeError(
            "GPU-resident MoE eval expected base-model last_hidden_state."
        )
    return hidden


@torch.no_grad()
def _capture_gpu_hidden_states(
    analyzer: model_utils.ModelAnalyzer,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Capture post-final-norm hidden states without layer CPU streaming."""

    if input_ids.device.type != "cuda":
        raise RuntimeError("GPU-resident eval input_ids must be CUDA tensors.")
    base_model = getattr(analyzer.model, "model", None)
    if base_model is None:
        raise RuntimeError("Qwen3-MoE model is missing its base `model` module.")
    hidden_out = None
    for sample_idx in range(int(input_ids.shape[0])):
        outputs = base_model(
            input_ids=input_ids[sample_idx : sample_idx + 1],
            use_cache=False,
        )
        hidden = _extract_last_hidden(outputs)
        if hidden_out is None:
            hidden_out = torch.empty(
                (
                    int(input_ids.shape[0]),
                    int(hidden.shape[1]),
                    int(hidden.shape[2]),
                ),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        hidden_out[sample_idx : sample_idx + 1].copy_(hidden)
        del outputs, hidden
    if hidden_out is None:
        raise RuntimeError("GPU-resident eval received zero samples.")
    return hidden_out


def _setup_moe_gpu_eval(
    cfg: "Config",
    analyzer: model_utils.ModelAnalyzer,
):
    """Build immutable FP eval targets entirely on the resident GPU."""

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    test_tokens, ref_hidden = {}, {}
    for dataset in cfg.eval_datasets:
        loader = data_utils.get_loaders(
            dataset,
            split="test",
            tokenizer=analyzer.tokenizer,
            seq_len=cfg.eval_seq_len,
            num_samples=cfg.nsamples,
        )
        input_ids = loader.input_ids
        nsamples = int(input_ids.numel()) // int(cfg.eval_seq_len)
        tokens = (
            input_ids[:, : nsamples * cfg.eval_seq_len]
            .reshape(nsamples, cfg.eval_seq_len)
            .to(dev)
        )
        test_tokens[dataset] = tokens
        ref_hidden[dataset] = _capture_gpu_hidden_states(analyzer, tokens)
    orig_lm_head = copy.deepcopy(analyzer.get_lm_head()).to(dev)
    orig_lm_head.eval()
    _assert_module_cuda(orig_lm_head, dev, label="orig_lm_head")
    return test_tokens, ref_hidden, orig_lm_head


def _setup_eval(cfg: "Config", analyzer: model_utils.ModelAnalyzer):
    """Cache fp reference logits per dataset before any weight modification."""
    if (
        cfg.moe_gpu_resident
        and _is_sparse_moe_analyzer(analyzer)
    ):
        return _setup_moe_gpu_eval(cfg, analyzer)
    test_loaders, ref_logits = {}, {}
    orig_lm_head = None
    for ds in cfg.eval_datasets:
        loader = data_utils.get_loaders(
            ds, split="test", tokenizer=analyzer.tokenizer,
            seq_len=cfg.eval_seq_len, num_samples=cfg.nsamples,
        )
        rl, orig_lm_head = eval_utils.get_ref_logits(cfg, analyzer, ds, loader)
        test_loaders[ds] = loader
        ref_logits[ds] = rl
    return test_loaders, ref_logits, orig_lm_head


def _maybe_rotate(cfg: "Config", analyzer: model_utils.ModelAnalyzer) -> None:
    """QuaRot weight rotation + Hadamard activation wrappers."""
    if cfg.rotate:
        rotation_utils.fuse_layer_norms(analyzer)
        rotation_utils.rotate_model(cfg, analyzer)
        mem_utils.cleanup_memory()
        # Bit-exact agreement across DP ranks: cuSOLVER QR / cuBLAS GEMM can
        # pick slightly different kernels per physical GPU after rotate, so
        # broadcast every parameter from rank 0 before continuing.
        if parallel_env.get_world_size() > 1:
            cuda_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            for p in analyzer.model.parameters():
                data = p.data
                if not data.is_contiguous():
                    data = data.contiguous()
                if not data.is_cuda:
                    data = data.to(cuda_dev)
                dist.broadcast(data, src=0)
                if data.data_ptr() != p.data.data_ptr():
                    p.data = data
        rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
        model_adapter.repair_moe_down_rotation_wrappers(analyzer)
        # Mark the model so akv.install_actquant_wrappers stays a no-op.
        analyzer.model._realq_actquant_wrappers_installed = True
    else:
        # Even without rotate we still need ActQuantWrapper sites so that the
        # eval path has consistent module shapes.
        quant_utils.add_actquant(analyzer)
        analyzer.model._realq_actquant_wrappers_installed = True


def _prepare_loaded_runtime_wrappers(
    cfg: "Config",
    analyzer: model_utils.ModelAnalyzer,
) -> None:
    """Recreate wrapper topology without transforming weights.

    The checkpoint already contains fused/rotated/fake-quantized weights.
    Re-running rotation before overwriting them is both unnecessary and makes
    inference depend on the original optimized-rotation file.
    """
    if cfg.rotate:
        rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
        model_adapter.repair_moe_down_rotation_wrappers(analyzer)
        analyzer.model._realq_actquant_wrappers_installed = True
    else:
        akv.install_actquant_wrappers(analyzer)


@torch.no_grad()
def _moe_gpu_kl_ppl_eval(
    cfg: "Config",
    analyzer: model_utils.ModelAnalyzer,
    orig_lm_head: torch.nn.Module,
    test_tokens: dict[str, torch.Tensor],
    ref_hidden: dict[str, torch.Tensor],
) -> None:
    """Evaluate sparse MoE without invoking eval_utils' CPU layer streamer."""

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    _assert_module_cuda(analyzer.model, dev, label="analyzer.model")
    _assert_module_cuda(orig_lm_head, dev, label="orig_lm_head")
    base_model = getattr(analyzer.model, "model", None)
    if base_model is None:
        raise RuntimeError("Qwen3-MoE model is missing its base `model` module.")
    lm_head = analyzer.get_lm_head()
    head_chunk_tokens = max(1, int(cfg.blocksize))
    metric_vals: "OrderedDict[str, str]" = OrderedDict()

    for dataset in cfg.eval_datasets:
        input_ids = test_tokens[dataset]
        reference = ref_hidden[dataset]
        if input_ids.device.type != "cuda" or reference.device.type != "cuda":
            raise RuntimeError(
                f"GPU-resident eval tensors for {dataset} left CUDA."
            )
        nll_sum = torch.zeros((), dtype=torch.float32, device=dev)
        kl_sum = torch.zeros((), dtype=torch.float32, device=dev)
        nll_count = 0
        kl_count = 0
        seq_len = int(input_ids.shape[1])

        for sample_idx in range(int(input_ids.shape[0])):
            outputs = base_model(
                input_ids=input_ids[sample_idx : sample_idx + 1],
                use_cache=False,
            )
            student_hidden = _extract_last_hidden(outputs)
            teacher_hidden = reference[sample_idx : sample_idx + 1]
            for start in range(0, seq_len, head_chunk_tokens):
                end = min(start + head_chunk_tokens, seq_len)
                student_logits = lm_head(
                    student_hidden[:, start:end]
                ).float()
                teacher_logits = orig_lm_head(
                    teacher_hidden[:, start:end]
                ).float()

                # Causal NLL excludes the final position.  Compute before any
                # diagnostic top-k KL projection so PPL remains full-vocab.
                nll_end = min(end, seq_len - 1)
                nll_width = max(0, nll_end - start)
                if nll_width:
                    labels = input_ids[
                        sample_idx : sample_idx + 1,
                        start + 1 : nll_end + 1,
                    ]
                    nll_sum.add_(
                        F.cross_entropy(
                            student_logits[:, :nll_width].transpose(1, 2),
                            labels,
                            reduction="sum",
                        ).float()
                    )
                    nll_count += int(labels.numel())

                if cfg.kl_topk > 0:
                    teacher_for_kl, indices = teacher_logits.topk(
                        int(cfg.kl_topk),
                        dim=-1,
                        sorted=False,
                    )
                    student_for_kl = student_logits.gather(-1, indices)
                else:
                    teacher_for_kl = teacher_logits
                    student_for_kl = student_logits
                token_kl = tokenwise_kl_from_logits(
                    student_for_kl,
                    teacher_for_kl,
                )
                kl_sum.add_(token_kl.sum())
                kl_count += int(token_kl.numel())
                del (
                    student_logits,
                    teacher_logits,
                    teacher_for_kl,
                    student_for_kl,
                    token_kl,
                )
            del outputs, student_hidden

        if nll_count == 0 or kl_count == 0:
            raise RuntimeError(
                f"GPU-resident eval for {dataset} produced no tokens."
            )
        ppl = float(torch.exp(nll_sum / nll_count).item())
        kl_loss = float((kl_sum / kl_count).item())
        if kl_loss < -1e-7:
            raise RuntimeError(
                "Full-vocabulary fp32 KL became materially negative "
                f"({kl_loss:.3e}) for {dataset}."
            )
        kl_loss = max(0.0, kl_loss)
        metric_vals[f"KL-{dataset}"] = f"{kl_loss:.2e}"
        metric_vals[f"PPL-{dataset}"] = f"{ppl:.2f}"
        logging.info(
            "Exact GPU-resident KL&PPL on %s: %.17g, %.17g",
            dataset,
            kl_loss,
            ppl,
        )
    eval_utils.pretty_print_results(metric_vals)


def _run_lm_eval_if_requested(
    cfg: "Config",
    analyzer: model_utils.ModelAnalyzer,
) -> bool:
    """Run rank-0 lm_eval after releasing the torchrun process group.

    Returns ``True`` when lm_eval was requested and the caller must return
    immediately.  All ranks execute the pre-destroy barrier and destroy their
    process group. Dense legacy runs dispatch the full CPU model over visible
    GPUs. The accepted sparse MoE path is already single-GPU/CUDA-resident and
    passes that model directly to lm-eval without CPU dispatch.
    """
    if not (getattr(cfg, "lm_eval", False) and not cfg.skip_eval):
        return False

    run_lm_eval = parallel_env.is_main()
    if parallel_env.is_dist_available_and_initialized():
        parallel_env.barrier()
        dist.destroy_process_group()
    if run_lm_eval:
        gpu_resident_moe = (
            cfg.moe_gpu_resident
            and _is_sparse_moe_analyzer(analyzer)
        )
        if gpu_resident_moe:
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            _assert_module_cuda(
                analyzer.model,
                dev,
                label="analyzer.model",
            )
        else:
            dist_utils.distribute_model(analyzer.model)
        with nvtx.nvtx_range("ptq.eval_lm_eval"):
            eval_utils.qa_eval(
                analyzer.model,
                analyzer.tokenizer,
                cfg.lm_eval_batch_size,
            )
        if gpu_resident_moe:
            _assert_module_cuda(
                analyzer.model,
                dev,
                label="analyzer.model.after_lm_eval",
            )
    return True


def run(cfg: "Config") -> None:
    """End-to-end RealQ pipeline."""
    if cfg.cpu_master:
        cfg.validate_moe_runtime_contract(
            sparse_moe=_model_source_declares_sparse_moe(cfg.model)
        )
        return _run_cpu_master(cfg)

    loaded_checkpoint = None
    if cfg.load_qmodel_path:
        loaded_checkpoint = checkpoint_utils.load_quantized_checkpoint(
            cfg.load_qmodel_path,
            allow_unsafe_legacy=cfg.allow_unsafe_legacy_checkpoint,
        )
        checkpoint_utils.apply_runtime_manifest(cfg, loaded_checkpoint)
        checkpoint_utils.validate_artifact_identity(cfg, loaded_checkpoint)

    with nvtx.nvtx_range("ptq.load_model"):
        analyzer = model_utils.ModelAnalyzer(cfg.model, cfg.seq_len)
        model_adapter.validate_analyzer(analyzer)
    if loaded_checkpoint is not None:
        checkpoint_utils.validate_artifact_identity(
            cfg,
            loaded_checkpoint,
            model=analyzer.model,
            tokenizer=analyzer.tokenizer,
        )
    gpu_resident_moe = _activate_moe_gpu_resident_runtime(
        cfg,
        analyzer,
        upload_model=not cfg.skip_eval,
    )

    # 1. Reference logits (must be captured BEFORE rotate — KL eval compares
    # quantised lm_head against the unrotated lm_head).
    test_loaders = ref_logits = orig_lm_head = None
    if not cfg.skip_eval:
        with nvtx.nvtx_range("ptq.ref_logits"):
            test_loaders, ref_logits, orig_lm_head = _setup_eval(cfg, analyzer)

    # 2. Rotate (optional)
    with nvtx.nvtx_range("ptq.rotate"):
        if loaded_checkpoint is None:
            _maybe_rotate(cfg, analyzer)
        else:
            _prepare_loaded_runtime_wrappers(cfg, analyzer)
    if gpu_resident_moe:
        # The inherited rotation kernels deliberately write transformed
        # weights back to CPU.  Rotation is pre-Stage-0 setup; from this point
        # through capture, quantization and evaluation the entire sparse model
        # is required to remain on this GPU.
        _activate_moe_gpu_resident_runtime(cfg, analyzer)

    # A quantized artifact already contains the final fake-quantized weights.
    # Prepare the same wrapper topology, restore the weights, then reconstruct
    # both aware and unaware runtime A/V/K modes from its manifest.  Static
    # precompute and the weight quantization loop must not run again.
    if loaded_checkpoint is not None:
        # _maybe_rotate installs these in both branches today; keep this
        # idempotent call adjacent to state loading so a future rotate=False
        # refactor cannot accidentally load ``*.module.weight`` keys into an
        # unwrapped model.
        akv.install_actquant_wrappers(analyzer)
        checkpoint_utils.load_model_state(analyzer.model, loaded_checkpoint)
        with nvtx.nvtx_range("ptq.akv_restore"):
            akv.setup_aware_pre_quant(analyzer, cfg)
            akv.setup_unaware_post_quant(analyzer, cfg)
        if gpu_resident_moe:
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            _assert_module_cuda(analyzer.model, dev, label="analyzer.model")
        if cfg.save_qmodel_path and parallel_env.is_main():
            checkpoint_utils.save_quantized_checkpoint(
                cfg.save_qmodel_path, analyzer.model, cfg, analyzer.tokenizer
            )
        if not cfg.skip_eval:
            with nvtx.nvtx_range("ptq.eval_kl_ppl"):
                if gpu_resident_moe:
                    _moe_gpu_kl_ppl_eval(
                        cfg,
                        analyzer,
                        orig_lm_head,
                        test_loaders,
                        ref_logits,
                    )
                else:
                    analyzer.model.cpu()
                    eval_utils.kl_ppl_eval(
                        cfg, analyzer, orig_lm_head, test_loaders, ref_logits
                    )
        if _run_lm_eval_if_requested(cfg, analyzer):
            return
        if parallel_env.is_dist_available_and_initialized():
            parallel_env.barrier()
        return

    # 2b. FSDP single-stage wrap: shard the rotated model across all visible
    # GPUs so the precompute backward fits memory budgets that don't admit a
    # full replica per rank.  Do NOT configure aware A/V/K quantisation yet:
    # Stage 0 saliency + Fisher are FP teacher statistics in REAL-Q and must
    # be invariant to the later student-side aware quantisation settings.
    if cfg.fsdp:
        with nvtx.nvtx_range("ptq.fsdp_wrap"):
            realq_fsdp.fsdp_wrap_for_precompute(analyzer, cfg)

    # 3. Static precompute (saliency + Fisher).
    with nvtx.nvtx_range("ptq.precompute"):
        static = precompute.run(cfg, analyzer)
    if gpu_resident_moe:
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        _assert_module_cuda(analyzer.model, dev, label="analyzer.model")
        _assert_static_stats_cuda(static, dev)
    logging.info(
        "[realq] precompute done: %d layers, saliency[0] modules=%s, fisher[0].shape=%s",
        len(static.fisher),
        sorted(static.saliency[0].keys()) if static.saliency else "n/a",
        tuple(static.fisher[0].shape) if static.fisher else "n/a",
    )
    if cfg.exit_after_precompute:
        logging.info("[realq] exit_after_precompute=True — stopping before quantise.")
        return

    # 3b. Drop FSDP and rebuild a CPU master before quantisation. The save
    # path is collective (every rank calls save_pretrained); the reload is
    # rank-local and produces an unwrapped, CPU-resident model that the
    # per-layer streaming quant phase consumes one block at a time.
    if cfg.fsdp:
        with nvtx.nvtx_range("ptq.fsdp_unwrap"):
            ckpt_dir = realq_fsdp.save_post_precompute_checkpoint(analyzer, cfg)
            analyzer = realq_fsdp.reload_on_cpu(analyzer, ckpt_dir)
            model_adapter.validate_analyzer(analyzer)
            # Reinstall ActQuantWrapper sites on the freshly-loaded model — the
            # checkpoint stored only the inner Linear weights without wrappers.
            # Re-applying the rotation wrappers (which install had_K on down_proj
            # so the next stage's activation gets the inverse Hadamard) is
            # necessary when ``cfg.rotate=True``: the saved weights are already
            # rotated, but the runtime activation rotators are NOT in state and
            # must be re-attached. ``add_activation_quant_wrappers_for_rotation``
            # is an idempotent install (same model flag guard as before), so it
            # only sets the had_K buffers; weight values stay as-is.
            if cfg.rotate:
                rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
                model_adapter.repair_moe_down_rotation_wrappers(analyzer)
            else:
                akv.install_actquant_wrappers(analyzer)
            akv.setup_aware_pre_quant(analyzer, cfg)
    else:
        # Aware quantisation belongs exclusively to the student-side weight
        # GPTQ pass.  Configure it only after the FP Stage 0 statistics have
        # been computed (or loaded).  Unaware paths remain at FP16 here and
        # are configured after weight quantisation below.
        with nvtx.nvtx_range("ptq.akv_setup_pre_quant"):
            akv.setup_aware_pre_quant(analyzer, cfg)

    # 4. Quantise.
    # Reuse the same calibration tokens that drove precompute (same key, so
    # `data_utils.get_tokens` hits the on-disk cache).
    tokens_save_path = cfg.tokens_cache_file
    if tokens_save_path is None and cfg.tokens_cache_path:
        tokens_save_path = os.path.join(
            cfg.tokens_cache_path,
            f"{cfg.model_name}_{cfg.dataset}_train_n{cfg.nsamples}_sl{cfg.seq_len}_seed{cfg.seed}.pt",
        )
    with nvtx.nvtx_range("ptq.load_calibration_tokens"):
        trainloader = data_utils.get_tokens(
            cfg.dataset, "train", analyzer.tokenizer,
            cfg.seq_len, cfg.nsamples, tokens_save_path, cfg.seed,
        )
    with nvtx.nvtx_range("ptq.quant_loop"):
        runner.quantize_all_layers(cfg, analyzer, static, trainloader)

    # 5. Unaware AKV setup: configure the wrappers AFTER weight GPTQ when
    # ``act_quant_aware_gptq=False`` / ``k_cache_quant_aware_gptq=False``.
    # The wrappers were already installed by the rotate path or by
    # ``setup_aware_pre_quant``; this only pushes the quant params.
    with nvtx.nvtx_range("ptq.akv_setup_post_quant"):
        akv.setup_unaware_post_quant(analyzer, cfg)

    if cfg.save_qmodel_path and parallel_env.is_main():
        with nvtx.nvtx_range("ptq.save_qmodel"):
            checkpoint_utils.save_quantized_checkpoint(
                cfg.save_qmodel_path, analyzer.model, cfg, analyzer.tokenizer
            )
            logging.info(
                "[realq] reproducible quantized checkpoint saved → %s",
                cfg.save_qmodel_path,
            )

    # 6. Eval (PPL/KL)
    if not cfg.skip_eval:
        with nvtx.nvtx_range("ptq.eval_kl_ppl"):
            if gpu_resident_moe:
                _moe_gpu_kl_ppl_eval(
                    cfg,
                    analyzer,
                    orig_lm_head,
                    test_loaders,
                    ref_logits,
                )
            else:
                analyzer.model.cpu()
                eval_utils.kl_ppl_eval(
                    cfg,
                    analyzer,
                    orig_lm_head,
                    test_loaders,
                    ref_logits,
                )

    # 7. Optional lm_eval QA tasks.  Release the torchrun process group before
    # rank 0 dispatches the model over all visible GPUs.
    if _run_lm_eval_if_requested(cfg, analyzer):
        return

    if parallel_env.is_dist_available_and_initialized():
        parallel_env.barrier()


def _run_cpu_master(cfg: "Config") -> None:
    """End-to-end pipeline under cpu_master (rank0-CPU-master) mode.

    Phase A   rank0 builds + caches the rotated/untied checkpoint;
              rank>0 just barrier. Disk-cached so subsequent runs skip rotate.
    Phase A.5 rank0 captures ref_logits + orig_lm_head from the rotated
              checkpoint (NOT a duplicate model — Phase A's analyzer was freed
              already, this is a fresh load that's also freed before Phase B).
    Phase B   every rank: meta init from config → fully_shard the meta model
              → ``load_checkpoint_in_model(broadcast_from_rank0=True)`` fills
              shards from rank0's safetensors reads. Wrappers installed AFTER
              load (key matching).
    Phase C   precompute (unchanged; weights immutable).
    Phase D   drop FSDP analyzer; rank0 reloads full CPU model from the cached
              checkpoint, rank>0 builds a meta skeleton. Layer manager
              broadcasts each block from rank0 to all ranks during quant.
    Phase E   quantise via runner.quantize_all_layers (manager built inside).
    Phase F   eval rank0-only with barriers; lm_eval barriers + destroy
              process group, rank>0 returns. Mirrors ptq.py:55-206.
    """
    cfg.validate_moe_runtime_contract(
        sparse_moe=_model_source_declares_sparse_moe(cfg.model)
    )
    # Phase A — rotate cache (rank0 only, plus Phase B-needed checkpoint path).
    checkpoint_path, _is_rotated = realq_fsdp.prepare_rotated_checkpoint(cfg)

    # Phase A.5 — ref logits (rank 0 only; CPU peak still 1×M because Phase A
    # already freed its analyzer).
    test_loaders = ref_logits = orig_lm_head = None
    if not cfg.skip_eval:
        if parallel_env.is_main():
            analyzer_eval = model_utils.ModelAnalyzer(
                checkpoint_path, cfg.seq_len, tokenizer_source=checkpoint_path,
            )
            model_adapter.validate_analyzer(analyzer_eval)
            # Set the OLD attribute name so eval_utils.get_ref_logits computes
            # the same cache tag as the legacy path (utils/eval_utils.py:115).
            analyzer_eval.model._gptqplus_prepared_checkpoint_path = checkpoint_path
            if cfg.rotate:
                rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer_eval)
            else:
                quant_utils.add_actquant(analyzer_eval)
            test_loaders, ref_logits, orig_lm_head = _setup_eval(cfg, analyzer_eval)
            del analyzer_eval
            mem_utils.cleanup_memory()
        else:
            logging.info(
                "[realq.cpu_master] rank %d skipping ref_logits generation; "
                "rank0 owns CPU master eval.", parallel_env.get_rank(),
            )
        if parallel_env.is_dist_available_and_initialized():
            parallel_env.barrier()

    # Phase B — meta init + sharded broadcast load.
    analyzer = realq_fsdp.load_meta_for_precompute(cfg, checkpoint_path)
    model_adapter.validate_analyzer(analyzer)
    # Wrappers installed AFTER broadcast load (the loader matches keys against
    # vanilla state_dict; wrapping introduces ``.module.`` infix that would
    # turn into unmatched keys).
    if cfg.rotate:
        rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
        model_adapter.repair_moe_down_rotation_wrappers(analyzer)
    else:
        akv.install_actquant_wrappers(analyzer)
    # Sync the realq-side flag so akv.setup_aware_pre_quant's
    # install_actquant_wrappers() guard fires (it checks _realq_…, while
    # add_activation_quant_wrappers_for_rotation only sets _gptqplus_…).
    # Without this, setup_aware_pre_quant re-wraps and produces
    # Wrapper(Wrapper(Linear)). Mirrors _maybe_rotate at pipeline.py:69.
    analyzer.model._realq_actquant_wrappers_installed = True
    akv.setup_aware_pre_quant(analyzer, cfg)  # safe: aware flags both False under guard

    # Phase C — precompute. Same code as legacy; weights are not modified.
    static = precompute.run(cfg, analyzer)
    logging.info(
        "[realq.cpu_master] precompute done: %d layers, saliency[0] modules=%s, fisher[0].shape=%s",
        len(static.fisher),
        sorted(static.saliency[0].keys()) if static.saliency else "n/a",
        tuple(static.fisher[0].shape) if static.fisher else "n/a",
    )
    if cfg.exit_after_precompute:
        logging.info("[realq.cpu_master] exit_after_precompute=True — stopping before quantise.")
        return

    # Phase D — drop FSDP, asymmetric rebuild.
    del analyzer
    mem_utils.cleanup_memory()
    analyzer = realq_fsdp.rebuild_asymmetric_for_quant(cfg, checkpoint_path)
    model_adapter.validate_analyzer(analyzer)
    if cfg.rotate:
        rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
        model_adapter.repair_moe_down_rotation_wrappers(analyzer)
    else:
        akv.install_actquant_wrappers(analyzer)
    # Same flag-sync as Phase B (see comment there).
    analyzer.model._realq_actquant_wrappers_installed = True
    akv.setup_aware_pre_quant(analyzer, cfg)

    # Phase E — quant.
    tokens_save_path = cfg.tokens_cache_file
    if tokens_save_path is None and cfg.tokens_cache_path:
        tokens_save_path = os.path.join(
            cfg.tokens_cache_path,
            f"{cfg.model_name}_{cfg.dataset}_train_n{cfg.nsamples}_sl{cfg.seq_len}_seed{cfg.seed}.pt",
        )
    trainloader = data_utils.get_tokens(
        cfg.dataset, "train", analyzer.tokenizer,
        cfg.seq_len, cfg.nsamples, tokens_save_path, cfg.seed,
    )
    runner.quantize_all_layers(cfg, analyzer, static, trainloader)

    akv.setup_unaware_post_quant(analyzer, cfg)

    if cfg.save_qmodel_path and parallel_env.is_main():
        checkpoint_utils.save_quantized_checkpoint(
            cfg.save_qmodel_path, analyzer.model, cfg, analyzer.tokenizer
        )
        logging.info(
            "[realq.cpu_master] reproducible quantized checkpoint saved → %s",
            cfg.save_qmodel_path,
        )

    # Phase F — eval. NO analyzer.model.cpu() under cpu_master: rank 0's model
    # is already CPU-resident after Phase E (manager released every block back
    # to CPU), and rank>0's model is meta — calling .cpu() on a meta module
    # would raise NotImplementedError from _apply.
    if not cfg.skip_eval:
        if parallel_env.is_main():
            eval_utils.kl_ppl_eval(cfg, analyzer, orig_lm_head, test_loaders, ref_logits)
        else:
            logging.info(
                "[realq.cpu_master] rank %d waiting for rank0 PPL/KL eval.",
                parallel_env.get_rank(),
            )
        if parallel_env.is_dist_available_and_initialized():
            parallel_env.barrier()

    if _run_lm_eval_if_requested(cfg, analyzer):
        return

    if parallel_env.is_dist_available_and_initialized():
        parallel_env.barrier()
