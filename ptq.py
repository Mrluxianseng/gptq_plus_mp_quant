# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
# cuBLAS reads this setting when its CUDA context is created, so it must be
# present before importing torch (and before any transitive CUDA import).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import logging

import torch
import torch.distributed as dist

from process_args import parse_gen
from gptq_utils.main import quantize_weights
from gptq_utils.quant_aware_utils import (
    configure_activation_quantizers_for_gptq,
    configure_k_cache_quantizers_for_gptq,
)
from utils import checkpoint_utils, data_utils, dist_utils, eval_utils, model_utils, rotation_utils, \
                  memory_utils, quant_utils
from utils.reproducibility import configure_reproducibility

torch.backends.cuda.matmul.allow_tf32 = False


def main(args):
    # Global RNG is an algorithm-time fallback; calibration sampling uses a
    # local RNG keyed by args.seed and must be the only thing that changes in
    # a calibration-seed sweep.
    configure_reproducibility(args.refresh_seed, deterministic=True)

    # When launched via torchrun for DP, each rank must pin itself to its
    # assigned GPU BEFORE any CUDA / NCCL op; otherwise rank 1 defaults to
    # cuda:0 (same device as rank 0) and every downstream NCCL collective
    # deadlocks because the ranks aren't on distinct devices.
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist_utils.init_process_group()

    # Read the manifest before preparing the model: ``rotate`` determines
    # which runtime wrappers must be installed, while A/V/K attributes are
    # otherwise absent from state_dict.  Keep the normalized payload so the
    # weight-loading phase does not unpickle a legacy checkpoint twice.
    if args.load_qmodel_path:
        loaded_checkpoint = checkpoint_utils.load_quantized_checkpoint(
            args.load_qmodel_path,
            allow_unsafe_legacy=args.allow_unsafe_legacy_checkpoint,
        )
        checkpoint_utils.apply_runtime_manifest(args, loaded_checkpoint)
        checkpoint_utils.validate_artifact_identity(args, loaded_checkpoint)
        args._loaded_quantized_checkpoint = loaded_checkpoint

    if bool(getattr(args, "fsdp_meta_init", False)):
        analyzer = model_utils.load_model_fsdp_meta_for_precompute(args)
    elif bool(getattr(args, "stage2_cpu_master", False)):
        analyzer = model_utils.load_model_cpu_master_for_quantization(args)
    else:
        # A quantized artifact must be loaded over the original base model.
        # Reusing a prepared rotated checkpoint would make the FP reference
        # logits come from already-rotated weights and can also leave runtime
        # wrappers out of sync with the artifact manifest.
        analyzer = (
            None
            if args.load_qmodel_path
            else model_utils.load_model_from_prepared_checkpoint_for_quantization(args)
        )
        if analyzer is None:
            analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model
    tokenizer = analyzer.tokenizer
    if args.load_qmodel_path:
        checkpoint_utils.validate_artifact_identity(
            args,
            args._loaded_quantized_checkpoint,
            model=model,
            tokenizer=tokenizer,
        )

    model_pre_rotated = bool(getattr(model, "_gptqplus_checkpoint_is_rotated", False))

    def add_activation_quant_wrappers():
        rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)

    if not args.load_qmodel_path and args.rotate and model_pre_rotated:
        logging.info(
            "Model was loaded from a pre-rotated checkpoint; installing rotation wrappers "
            "before reference-logit generation."
        )
        add_activation_quant_wrappers()

    # Generate reference logits for KL eval
    test_loader_dict, ref_logits_dict = {}, {}
    orig_lm_head = None
    stage1_precompute_only = bool(getattr(args, "fsdp_precompute", False)) and bool(
        getattr(args, "exit_after_precompute", False)
    )
    if stage1_precompute_only and not args.skip_eval:
        logging.info(
            "Skipping eval/reference-logit generation for FSDP Stage 1 precompute-only run."
        )
        args.skip_eval = True
    rank_runs_eval = not bool(getattr(args, "stage2_cpu_master", False)) or dist_utils.is_main()
    if not args.skip_eval and rank_runs_eval:
        for eval_dataset in args.eval_datasets:
            test_loader = data_utils.get_loaders(eval_dataset, split="test", tokenizer=tokenizer,
                                                 seq_len=args.eval_seq_len, num_samples=args.nsamples)
            ref_logits, orig_lm_head = eval_utils.get_ref_logits(args, analyzer, eval_dataset, test_loader)
            test_loader_dict[eval_dataset] = test_loader
            ref_logits_dict[eval_dataset] = ref_logits
    if not args.skip_eval and not rank_runs_eval and dist.is_available() and dist.is_initialized():
        logging.info(
            "stage2_cpu_master: rank%d skipping PPL/KL reference-logit generation; rank0 owns CPU master eval.",
            dist_utils.get_rank(),
        )
    if (
        bool(getattr(args, "stage2_cpu_master", False))
        and not args.skip_eval
        and dist.is_available()
        and dist.is_initialized()
    ):
        dist.barrier()

    # Rotate the weights
    if args.load_qmodel_path:
        # The artifact already stores the transformed weights. Rebuild only
        # the activation wrapper topology after the FP reference pass and
        # before strict state_dict loading.
        if args.rotate:
            add_activation_quant_wrappers()
        else:
            quant_utils.add_actquant(analyzer)
    elif args.rotate and not model_pre_rotated:
        rotation_utils.fuse_layer_norms(analyzer)
        rotation_utils.rotate_model(args, analyzer)
        memory_utils.cleanup_memory()

        # DP hygiene: rotation's QR + per-layer W@R matmuls run on each rank
        # independently. Random matrices are generated via CPU RNG so they
        # agree, but cuSOLVER QR and cuBLAS GEMM can pick slightly different
        # kernels per physical GPU → rotated weights may drift by ~1e-7.
        # That drift would then be amplified through every downstream layer.
        # Force bit-exact agreement by broadcasting all parameters from rank 0.
        #
        # NCCL requires CUDA + contiguous tensors, and after rotate_model many
        # params are (a) still on CPU and (b) non-contiguous views produced by
        # in-place reshape/transpose inside the rotation routines. Normalise
        # both in-place before each broadcast. quantize_weights does a global
        # model.cpu() right after this block, so we don't restore the CPU
        # residency ourselves.
        if dist_utils.get_world_size() > 1:
            _cuda_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            for p in model.parameters():
                data = p.data
                if not data.is_contiguous():
                    data = data.contiguous()
                if not data.is_cuda:
                    data = data.to(_cuda_dev)
                dist.broadcast(data, src=0)
                # If contiguous()/to() returned a fresh tensor, update p.data.
                if data.data_ptr() != p.data.data_ptr():
                    p.data = data

        add_activation_quant_wrappers()
    elif args.rotate and model_pre_rotated:
        logging.info("Model was loaded from a pre-rotated checkpoint; skipping in-process rotation.")
    else:
        quant_utils.add_actquant(analyzer)

    # Quantize model weights
    model = quantize_weights(args, analyzer)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        configure_activation_quantizers_for_gptq(args, model)

    if args.k_bits < 16:
        configure_k_cache_quantizers_for_gptq(args, analyzer)

    # Save only after the deployed A/V/K runtime behavior has been configured.
    # The state_dict contains weights; the primitive manifest reconstructs
    # dynamic quantizers and the K-cache forward patch on reload.
    if args.save_qmodel_path and (
        not bool(getattr(model, "_gptqplus_stage2_cpu_master", False))
        or dist_utils.is_main()
    ):
        checkpoint_utils.save_quantized_checkpoint(
            args.save_qmodel_path, model, args, tokenizer
        )

    # Eval
    if not args.skip_eval and rank_runs_eval:
        eval_utils.kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict, ref_logits_dict)
        del orig_lm_head, ref_logits_dict
    if not args.skip_eval and not rank_runs_eval and dist.is_available() and dist.is_initialized():
        logging.info(
            "stage2_cpu_master: rank%d waiting for rank0 PPL/KL eval.",
            dist_utils.get_rank(),
        )
    if (
        bool(getattr(args, "stage2_cpu_master", False))
        and not args.skip_eval
        and dist.is_available()
        and dist.is_initialized()
    ):
        dist.barrier()

    if args.lm_eval and not args.skip_eval:
        # Run lm_eval only on the original rank 0 process. accelerate.dispatch_model
        # still does the model-parallel device placement inside that process; the
        # torchrun ranks must not sit in a later NCCL barrier while rank 0 spends
        # hours evaluating loglikelihood requests.
        run_lm_eval = dist_utils.is_main()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if run_lm_eval:
            dist_utils.distribute_model(model)
            eval_utils.qa_eval(model, tokenizer, args.lm_eval_batch_size)
        return

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_gen()
    main(args)
