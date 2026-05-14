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

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from realq import akv, fsdp as realq_fsdp, precompute, runner
from realq.parallel import env as parallel_env
from realq.utils import memory as mem_utils
from realq.utils import nvtx
from utils import data_utils, eval_utils, model_utils, quant_utils, rotation_utils

if TYPE_CHECKING:
    from realq.config import Config


def _setup_eval(cfg: "Config", analyzer: model_utils.ModelAnalyzer):
    """Cache fp reference logits per dataset before any weight modification."""
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
        # Mark the model so akv.install_actquant_wrappers stays a no-op.
        analyzer.model._realq_actquant_wrappers_installed = True
    else:
        # Even without rotate we still need ActQuantWrapper sites so that the
        # eval path has consistent module shapes.
        quant_utils.add_actquant(analyzer)
        analyzer.model._realq_actquant_wrappers_installed = True


def run(cfg: "Config") -> None:
    """End-to-end RealQ pipeline."""
    if cfg.cpu_master:
        return _run_cpu_master(cfg)
    with nvtx.nvtx_range("ptq.load_model"):
        analyzer = model_utils.ModelAnalyzer(cfg.model, cfg.seq_len)

    # 1. Reference logits (must be captured BEFORE rotate — KL eval compares
    # quantised lm_head against the unrotated lm_head).
    test_loaders = ref_logits = orig_lm_head = None
    if not cfg.skip_eval:
        with nvtx.nvtx_range("ptq.ref_logits"):
            test_loaders, ref_logits, orig_lm_head = _setup_eval(cfg, analyzer)

    # 2. Rotate (optional)
    with nvtx.nvtx_range("ptq.rotate"):
        _maybe_rotate(cfg, analyzer)

    # 2b. Aware AKV setup: install + configure the activation/K-cache wrappers
    # BEFORE weight GPTQ runs so the per-linear Hessian forward sees the
    # quantised input. Unaware paths defer this to step 6.
    with nvtx.nvtx_range("ptq.akv_setup_pre_quant"):
        akv.setup_aware_pre_quant(analyzer, cfg)

    # 2c. FSDP single-stage wrap: shard the (rotated, possibly aware-AKV-
    # configured) model across all visible GPUs so the precompute backward
    # fits memory budgets that don't admit a full replica per rank.
    if cfg.fsdp:
        with nvtx.nvtx_range("ptq.fsdp_wrap"):
            realq_fsdp.fsdp_wrap_for_precompute(analyzer, cfg)

    # 3. Static precompute (saliency + Fisher).
    with nvtx.nvtx_range("ptq.precompute"):
        static = precompute.run(cfg, analyzer)
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
            else:
                akv.install_actquant_wrappers(analyzer)
            akv.setup_aware_pre_quant(analyzer, cfg)

    # 4. Quantise.
    # Reuse the same calibration tokens that drove precompute (same key, so
    # `data_utils.get_tokens` hits the on-disk cache).
    tokens_save_path = None
    if cfg.tokens_cache_path:
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

    if cfg.save_qmodel_path and parallel_env.is_main():
        with nvtx.nvtx_range("ptq.save_qmodel"):
            os.makedirs(os.path.dirname(cfg.save_qmodel_path) or ".", exist_ok=True)
            torch.save({"model": analyzer.model.state_dict()}, cfg.save_qmodel_path)
            logging.info("[realq] quantised state_dict saved → %s", cfg.save_qmodel_path)

    # 5. Unaware AKV setup: configure the wrappers AFTER weight GPTQ when
    # ``act_quant_aware_gptq=False`` / ``k_cache_quant_aware_gptq=False``.
    # The wrappers were already installed by the rotate path or by
    # ``setup_aware_pre_quant``; this only pushes the quant params.
    with nvtx.nvtx_range("ptq.akv_setup_post_quant"):
        akv.setup_unaware_post_quant(analyzer, cfg)

    # 6. Eval (PPL/KL)
    if not cfg.skip_eval:
        with nvtx.nvtx_range("ptq.eval_kl_ppl"):
            analyzer.model.cpu()
            eval_utils.kl_ppl_eval(cfg, analyzer, orig_lm_head, test_loaders, ref_logits)

    # 7. Optional lm_eval QA tasks (Stage 2 — runs on rank 0 only).
    if getattr(cfg, "lm_eval", False) and parallel_env.is_main():
        # ``qa_eval`` dispatches the model across all visible GPUs via
        # accelerate; rank 0 alone is enough for the eval driver.
        with nvtx.nvtx_range("ptq.eval_lm_eval"):
            eval_utils.qa_eval(analyzer.model, analyzer.tokenizer, cfg.lm_eval_batch_size)

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
    # Wrappers installed AFTER broadcast load (the loader matches keys against
    # vanilla state_dict; wrapping introduces ``.module.`` infix that would
    # turn into unmatched keys).
    if cfg.rotate:
        rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
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
    if cfg.rotate:
        rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
    else:
        akv.install_actquant_wrappers(analyzer)
    # Same flag-sync as Phase B (see comment there).
    analyzer.model._realq_actquant_wrappers_installed = True
    akv.setup_aware_pre_quant(analyzer, cfg)

    # Phase E — quant.
    tokens_save_path = None
    if cfg.tokens_cache_path:
        tokens_save_path = os.path.join(
            cfg.tokens_cache_path,
            f"{cfg.model_name}_{cfg.dataset}_train_n{cfg.nsamples}_sl{cfg.seq_len}_seed{cfg.seed}.pt",
        )
    trainloader = data_utils.get_tokens(
        cfg.dataset, "train", analyzer.tokenizer,
        cfg.seq_len, cfg.nsamples, tokens_save_path, cfg.seed,
    )
    runner.quantize_all_layers(cfg, analyzer, static, trainloader)

    if cfg.save_qmodel_path and parallel_env.is_main():
        os.makedirs(os.path.dirname(cfg.save_qmodel_path) or ".", exist_ok=True)
        torch.save({"model": analyzer.model.state_dict()}, cfg.save_qmodel_path)
        logging.info("[realq.cpu_master] quantised state_dict saved → %s", cfg.save_qmodel_path)

    akv.setup_unaware_post_quant(analyzer, cfg)

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

    if getattr(cfg, "lm_eval", False):
        # Mirror ptq.py:191-203 verbatim: barrier all → destroy process group
        # → rank 0 runs qa_eval; rank>0 returns. The atexit hook in
        # realq.parallel.env._atexit_destroy is now try/except so the second
        # destroy is a no-op.
        run_lm_eval = parallel_env.is_main()
        if parallel_env.is_dist_available_and_initialized():
            parallel_env.barrier()
            dist.destroy_process_group()
        if run_lm_eval:
            eval_utils.qa_eval(analyzer.model, analyzer.tokenizer, cfg.lm_eval_batch_size)
        return

    if parallel_env.is_dist_available_and_initialized():
        parallel_env.barrier()
