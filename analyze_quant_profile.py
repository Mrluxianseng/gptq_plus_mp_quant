import argparse
import logging
import os

import torch

from gptq_utils.main import quantize_weights
from utils import memory_utils
from utils.log_utils import init_logging
from utils.model_utils import ModelAnalyzer


def build_parser():
    parser = argparse.ArgumentParser(description="Run GPTQ+ quantization with built-in NVTX ranges enabled")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF id")
    parser.add_argument("--exp", type=str, default="quant_profile", help="Experiment name")
    parser.add_argument(
        "--dataset",
        type=str,
        default="neuralmagic",
        choices=["wikitext2", "neuralmagic", "ultrachat_2k", "numinamath"],
    )
    parser.add_argument("--nsamples", type=int, default=512, help="Number of calibration samples")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_groups", type=int, default=4, help="Number of GPTQ+ row groups")
    parser.add_argument(
        "--fisher_num_groups",
        type=int,
        default=512,
        help="Optional number of groups used only for fisher_diag_mse layer-output Fisher weights. Defaults to --num_groups.",
    )
    parser.add_argument("--bsz", type=int, default=128, help="Batch size for gradient and saliency collection")
    parser.add_argument("--final_layer_stats_bsz", type=int, default=16, help="Optional statistics batch size override for the final transformer layer")
    parser.add_argument("--backward_samples", type=int, default=32, help="Refresh sample count (-1 means all samples)")
    parser.add_argument("--backward_bsz", type=int, default=32, help="Refresh batch size (-1 means reuse --bsz)")
    parser.add_argument("--final_layer_backward_bsz", type=int, default=16, help="Optional refresh batch size override for the final transformer layer")
    parser.add_argument("--w_bits", type=int, default=4, help="Weight bits")
    parser.add_argument("--w_method", type=str, default="gptq_plus", choices=["gptq_plus"], help="Weight quantization method")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Damping percent")
    parser.add_argument("--alpha", type=float, default=0.05, help="Gradient scaling coefficient")
    parser.add_argument("--grad_lr", type=float, default=1e-4, help="Learning rate for block-wise gradient descent")
    parser.add_argument("--final_layer_grad_lr", type=float, default=0.01, help="Optional final-layer lr override")
    parser.add_argument("--grad_optimizer", type=str, default="adam", choices=["sgd", "adam"], help="Optimizer for block_gd updates")
    parser.add_argument("--final_layer_grad_optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Optional final-layer optimizer override")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Elementwise clip threshold for SGD gradients")
    parser.add_argument("--grad_refresh_loss", type=str, default="fisher_diag_mse", choices=["kl", "hidden_mse", "fisher_diag_mse"], help="Refresh loss type")
    parser.add_argument(
        "--global_loss",
        dest="global_loss",
        action="store_true",
        help=(
            "Enable the frozen global-loss mode: run one end-to-end pre-quantization backward pass to cache "
            "saliency/Fisher coefficients and align GPTQ+ second-order terms with the configured refresh loss."
        ),
    )
    parser.add_argument(
        "--no_global_loss",
        dest="global_loss",
        action="store_false",
        help=(
            "Disable frozen global-loss caches and fall back to layerwise output-head saliency/Fisher collection, "
            "with GPTQ+ second-order terms using layerwise KL."
        ),
    )
    parser.set_defaults(global_loss=True)
    parser.add_argument(
        "--global_loss_bsz",
        type=int,
        default=16,
        help="Batch size used only for the frozen end-to-end global-loss backward pass.",
    )
    parser.add_argument("--pre_gd_steps", type=int, default=10, help="Number of pre-quantization GD steps after preclip/stat collection")
    parser.add_argument("--pre_grad_lr", type=float, default=3e-5, help="Pre-quantization GD learning rate for non-final layers")
    parser.add_argument("--pre_final_layer_grad_lr", type=float, default=0.3, help="Optional pre-quantization GD LR override for the final layer")
    parser.add_argument("--pre_grad_optimizer", type=str, default="adam", choices=["sgd", "adam"], help="Optimizer for the pre-quantization GD phase")
    parser.add_argument("--pre_final_layer_grad_optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Optional pre-quantization optimizer override for the final layer")
    parser.add_argument("--proj_lr_scale", type=float, default=1.0, help="LR multiplier for o_proj")
    parser.add_argument("--down_proj_lr_scale", type=float, default=1.0, help="LR multiplier for down_proj")
    parser.add_argument("--grad_reg_strategy", type=str, default="none", choices=["none", "l2", "hessian", "quant_error_gate", "quant_error_gate_optimized"], help="Regularization strategy")
    parser.add_argument("--grad_reg_lambda", type=float, default=0.01, help="Regularization strength")
    parser.add_argument("--grad_gate_floor", type=float, default=0.01, help="Minimum gate factor")
    parser.add_argument("--grad_gate_sharpness", type=float, default=5.0, help="Gate sharpness")
    parser.add_argument("--grad_gate_sine_amp", type=float, default=0.0005, help="Sine regularizer amplitude")
    parser.add_argument("--second_order_scale", type=float, default=1.0, help="Second-order update scale in block_gd")
    parser.add_argument("--kl_topk", type=int, default=20, help="Top-k logits for KL")
    parser.add_argument(
        "--grad_hessian_topk",
        type=int,
        default=20,
        help="When > 0, restrict grad/hessian label sampling, saliency NLL, and KL loss to fp top-k logits",
    )
    parser.add_argument("--blocksize", type=int, default=256, help="GPTQ block size")
    parser.add_argument("--block_atomic_quant", action="store_true", help="Quantize each block atomically")
    parser.add_argument("--w_groupsize", type=int, default=-1, help="Weight groupsize")
    parser.add_argument("--g_update_mode", type=str, default="block_gd", choices=["frozen", "surrogate_block", "surrogate_online", "block_backward", "block_gd"], help="First-order term update mode")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output root")
    parser.add_argument("--quant_profile_target_layers", type=str, default="all", help="Layer ids to profile in detail")
    parser.add_argument("--quant_profile_target_modules", type=str, default="all", help="Comma-separated module names to profile in detail")
    parser.add_argument(
        "--quant_stop_layer",
        type=str,
        default=None,
        help="Inclusive transformer layer index to stop after for profile runs; use none/all to run all layers",
    )
    parser.add_argument("--act_order", action="store_true", help="Enable act-order permutation")
    parser.add_argument("--w_asym", action="store_true", help="Use asymmetric weight quantization")
    parser.add_argument("--w_clip", action="store_true", help="Enable weight clipping")
    parser.add_argument(
        "--pre_clip",
        dest="pre_clip",
        action="store_true",
        help="Enable the manual pre-quantization clipping stage before stat collection and pre-GD.",
    )
    parser.add_argument(
        "--no_pre_clip",
        dest="pre_clip",
        action="store_false",
        help="Disable the manual pre-quantization clipping stage and skip pre-GD.",
    )
    parser.set_defaults(pre_clip=False)
    parser.add_argument("--offload_inps", action="store_true", help="Offload cached inputs to CPU")
    parser.add_argument("--final_layer_full_backward", action="store_true", help="Use all calibration samples for every final-layer refresh")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    args.model_name = model_name
    args.output_dir = os.path.join(args.output_dir, model_name, args.exp)
    args.log_dir = os.path.join(args.output_dir, "logs")
    args.tokens_cache_path = (
        f"{args.cache_dir}/tokens/{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}.pt"
    )
    args.saliency_cache_path = (
        f"{args.cache_dir}/saliency/{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_g{args.num_groups}"
    )
    args.gradients_cache_path = (
        f"{args.cache_dir}/gradients/{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_g{args.num_groups}.pt"
    )
    args.export_to_et = False
    args.enable_quant_profile = True
    args.load_qmodel_path = None
    args.save_qmodel_path = None
    args.rotate = True
    args.optimized_rotation_path = None
    args.a_bits = 16
    args.k_bits = 16
    args.v_bits = 16
    args.a_groupsize = -1
    args.k_groupsize = -1
    args.v_groupsize = -1
    args.a_asym = False
    args.k_asym = False
    args.v_asym = False
    args.a_clip_ratio = 1.0
    args.k_clip_ratio = 1.0
    args.v_clip_ratio = 1.0
    args.skip_eval = True
    args.lm_eval = False
    args.lm_eval_batch_size = 32
    args.eval_datasets = ["wikitext2", "ultrachat_2k", "numinamath"]
    args.enable_debug = False

    if args.fisher_num_groups is None:
        args.fisher_num_groups = args.num_groups

    if args.backward_samples == -1:
        args.backward_samples = args.nsamples
    if args.backward_samples <= 0:
        raise ValueError(f"`backward_samples` must be positive or -1. Got {args.backward_samples}.")
    if args.final_layer_stats_bsz is None:
        args.final_layer_stats_bsz = args.bsz
    if args.final_layer_stats_bsz <= 0:
        raise ValueError(
            f"`final_layer_stats_bsz` must be positive when provided. Got {args.final_layer_stats_bsz}."
        )
    if args.backward_bsz == -1:
        args.backward_bsz = args.bsz
    if args.backward_bsz <= 0:
        raise ValueError(f"`backward_bsz` must be positive or -1. Got {args.backward_bsz}.")
    if args.final_layer_backward_bsz is None:
        args.final_layer_backward_bsz = args.backward_bsz
    if args.final_layer_backward_bsz <= 0:
        raise ValueError(
            f"`final_layer_backward_bsz` must be positive when provided. Got {args.final_layer_backward_bsz}."
        )
    if args.global_loss_bsz is None:
        args.global_loss_bsz = args.bsz
    if args.global_loss_bsz <= 0:
        raise ValueError(f"`global_loss_bsz` must be positive when provided. Got {args.global_loss_bsz}.")
    if args.nsamples % args.backward_samples != 0:
        raise ValueError(
            f"`nsamples` ({args.nsamples}) must be divisible by `backward_samples` ({args.backward_samples})."
        )
    if args.grad_reg_lambda < 0:
        raise ValueError(f"`grad_reg_lambda` must be non-negative. Got {args.grad_reg_lambda}.")
    if args.grad_clip == 0:
        raise ValueError("`grad_clip` must be non-zero. Use a negative value to disable clipping.")
    if args.final_layer_grad_lr is not None and args.final_layer_grad_lr < 0:
        raise ValueError(f"`final_layer_grad_lr` must be non-negative when provided. Got {args.final_layer_grad_lr}.")
    if args.grad_hessian_topk == 0:
        raise ValueError("`grad_hessian_topk` must be positive or negative to disable. Use -1 to disable.")
    if args.pre_gd_steps < 0:
        raise ValueError(f"`pre_gd_steps` must be non-negative. Got {args.pre_gd_steps}.")
    if args.pre_grad_lr < 0:
        raise ValueError(f"`pre_grad_lr` must be non-negative. Got {args.pre_grad_lr}.")
    if args.pre_final_layer_grad_lr is not None and args.pre_final_layer_grad_lr < 0:
        raise ValueError(f"`pre_final_layer_grad_lr` must be non-negative when provided. Got {args.pre_final_layer_grad_lr}.")
    if args.pre_gd_steps > 0 and args.pre_clip and args.w_clip:
        effective_pre_final_lr = args.pre_final_layer_grad_lr if args.pre_final_layer_grad_lr is not None else args.pre_grad_lr
        if args.pre_grad_lr == 0 and effective_pre_final_lr == 0:
            raise ValueError("`pre_gd_steps > 0` requires `pre_grad_lr` or `pre_final_layer_grad_lr` to be positive.")
    if args.pre_gd_steps > 0 and not args.pre_clip:
        logging.info("`pre_clip` is disabled, so pre-GD will be skipped.")
    if args.pre_gd_steps > 0 and args.pre_clip and not args.w_clip:
        logging.info("`w_clip` is disabled, so pre-clip and pre-GD will be skipped.")
    if args.proj_lr_scale < 0:
        raise ValueError(f"`proj_lr_scale` must be non-negative. Got {args.proj_lr_scale}.")
    if args.down_proj_lr_scale < 0:
        raise ValueError(f"`down_proj_lr_scale` must be non-negative. Got {args.down_proj_lr_scale}.")
    if not (0.0 <= args.grad_gate_floor <= 1.0):
        raise ValueError(f"`grad_gate_floor` must be in [0, 1]. Got {args.grad_gate_floor}.")
    if args.grad_gate_sharpness < 0:
        raise ValueError(f"`grad_gate_sharpness` must be non-negative. Got {args.grad_gate_sharpness}.")
    if args.grad_gate_sine_amp < 0:
        raise ValueError(f"`grad_gate_sine_amp` must be non-negative. Got {args.grad_gate_sine_amp}.")
    if args.fisher_num_groups <= 0:
        raise ValueError(f"`fisher_num_groups` must be positive. Got {args.fisher_num_groups}.")

    init_logging(args.log_dir)
    logging.info(args)

    analyzer = ModelAnalyzer(args.model, args.seq_len)
    analyzer.model.cpu()
    quantize_weights(args, analyzer)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_utils.cleanup_memory(False)


if __name__ == "__main__":
    main()
