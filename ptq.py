# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

import torch
import torch.distributed as dist
import transformers

from process_args import parse_gen
from gptq_utils.main import quantize_weights
from utils import data_utils, dist_utils, eval_utils, model_utils, rotation_utils, \
                  memory_utils, quant_utils, hadamard_utils

torch.backends.cuda.matmul.allow_tf32 = False


def main(args):
    # When launched via torchrun for DP, each rank must pin itself to its
    # assigned GPU BEFORE any CUDA / NCCL op; otherwise rank 1 defaults to
    # cuda:0 (same device as rank 0) and every downstream NCCL collective
    # deadlocks because the ranks aren't on distinct devices.
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist_utils.init_process_group()

    analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model
    tokenizer = analyzer.tokenizer

    # Generate reference logits for KL eval
    test_loader_dict, ref_logits_dict = {}, {}
    orig_lm_head = None
    if not args.skip_eval:
        for eval_dataset in args.eval_datasets:
            test_loader = data_utils.get_loaders(eval_dataset, split="test", tokenizer=tokenizer,
                                                 seq_len=args.eval_seq_len, num_samples=args.nsamples)
            ref_logits, orig_lm_head = eval_utils.get_ref_logits(args, analyzer, eval_dataset, test_loader)
            test_loader_dict[eval_dataset] = test_loader
            ref_logits_dict[eval_dataset] = ref_logits

    # Rotate the weights
    if args.rotate:
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

        quant_utils.add_actquant(analyzer)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = False
    else:
        quant_utils.add_actquant(analyzer)

    # Quantize model weights
    model = quantize_weights(args, analyzer)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=args.v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        rope_function_name = "apply_rotary_pos_emb"
        layers = analyzer.get_layers()
        k_quant_config = {
            "k_bits": args.k_bits,
            "k_groupsize": args.k_groupsize,
            "k_sym": not (args.k_asym),
            "k_clip_ratio": args.k_clip_ratio,
        }
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn,
                rope_function_name,
                head_dim=analyzer.head_dim,
                **k_quant_config,
            )

    # Eval
    if not args.skip_eval:
        eval_utils.kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict, ref_logits_dict)
        del orig_lm_head, ref_logits_dict

    if args.lm_eval and not args.skip_eval:
        dist_utils.distribute_model(model)
        eval_utils.qa_eval(model, tokenizer, args.lm_eval_batch_size)

    dist.barrier()


if __name__ == "__main__":
    args = parse_gen()
    main(args)
