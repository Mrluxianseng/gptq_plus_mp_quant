import argparse
import os
import logging

import transformers

from utils import dist_utils
from utils.log_utils import init_logging


def parse_gen():
    parser = argparse.ArgumentParser(description="Quantize a model to any precision")
    parser.add_argument("--model", type=str, required=True, help="The model to quantize")
    parser.add_argument("--exp", type=str, required=True, help="Exp name")
    parser.add_argument("--seed", type=int, default=42,
                        help="The random state to use for reproducibility\n"
                             "[WARNING] May not be reproducible across different machines")
    # Paths
    parser.add_argument("--output_dir", type=str, default="./outputs", help="The directory to save results in")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="The directory to cache results in")
    # Datasets
    parser.add_argument("--dataset", type=str, default="neuralmagic",
                        choices=["wikitext2", "neuralmagic", "ultrachat_2k", "numinamath"], help="The dataset to use")
    parser.add_argument("--nsamples", type=int, default=1024, help="The number of examples to use")
    parser.add_argument("--seq_len", type=int, default=2048, help="The sequence length to use in calibration")
    parser.add_argument("--eval_seq_len", type=int, default=2048, help="The sequence length to use in PPL&KL evaluation")
    # Gradient profiling
    parser.add_argument("--mode", type=str, default="gradients", choices=["tokens", "gradients"],
                        help="The mode to run in")
    # Quantization configs
    parser.add_argument("--w_bits", type=int, default=16, help="Weight bits")
    parser.add_argument("--a_bits", type=int, default=16, help="Activation bits")
    parser.add_argument("--k_bits", type=int, default=16, help="K cache bits")
    parser.add_argument("--v_bits", type=int, default=16, help="V cache bits")
    parser.add_argument("--w_groupsize", type=int, default=-1, help="Weight group size")
    parser.add_argument("--a_groupsize", type=int, default=-1, help="Activation group size")
    parser.add_argument("--k_groupsize", type=int, default=-1, help="K cache group size")
    parser.add_argument("--v_groupsize", type=int, default=-1, help="V cache group size")
    parser.add_argument("--w_asym", action="store_true", help="Weight asymmetric quantization")
    parser.add_argument("--a_asym", action="store_true", help="Activation asymmetric quantization")
    parser.add_argument("--k_asym", action="store_true", help="K cache asymmetric quantization")
    parser.add_argument("--v_asym", action="store_true", help="V cache asymmetric quantization")
    parser.add_argument("--w_clip", action="store_true", help="Enable weight clipping")
    parser.add_argument(
        "--pre_clip",
        dest="pre_clip",
        action="store_true",
        help="Enable the manual pre-quantization weight clipping stage before stat collection and pre-GD.",
    )
    parser.add_argument(
        "--no_pre_clip",
        dest="pre_clip",
        action="store_false",
        help="Disable the manual pre-quantization weight clipping stage and skip pre-GD.",
    )
    parser.set_defaults(pre_clip=True)
    parser.add_argument("--a_clip_ratio", type=float, default=1.0, help="Activation clipping ratio")
    parser.add_argument("--k_clip_ratio", type=float, default=1.0, help="K cache clipping ratio")
    parser.add_argument("--v_clip_ratio", type=float, default=1.0, help="V cache clipping ratio")
    parser.add_argument("--export_to_et", action="store_true", help="Export quantized model (TODO)")
    # Rotate
    parser.add_argument("--optimized_rotation_path", type=str, default=None, help="The path to rotation ckpt")
    parser.add_argument("--rotate", action="store_true", help="Rotate model (SpinQuant)")
    # GPTQ
    parser.add_argument("--w_method", type=str, default="gptq",
                        choices=["rtn", "gptq", "gptaq", "gptq_guided", "gptq_plus"], help="The path to rotation ckpt")
    parser.add_argument("--act_order", action="store_true", help="Activation reorder (with static groups)")
    parser.add_argument("--num_groups", type=int, default=4,
                        help="Number of groups $g$ to use for block-diagonal Hessian")
    parser.add_argument(
        "--fisher_num_groups",
        type=int,
        default=None,
        help="Optional number of groups used only for fisher_diag_mse layer-output Fisher weights. Defaults to --num_groups.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--alpha", type=float, default=1, help="Dynamic down-scaling factor for the gradient update term")
    parser.add_argument(
        "--grad_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the independent block-wise gradient descent update.",
    )
    parser.add_argument(
        "--final_layer_grad_lr",
        type=float,
        default=None,
        help="Optional override for the block-wise gradient descent learning rate used only in the final transformer layer.",
    )
    parser.add_argument(
        "--grad_optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer used for the block-wise first-order update after each refresh.",
    )
    parser.add_argument(
        "--final_layer_grad_optimizer",
        type=str,
        default=None,
        choices=["sgd", "adam"],
        help="Optional override for the block-wise first-order optimizer used only in the final transformer layer.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Elementwise clip threshold applied to the true refresh gradient before optimizer updates. Set negative to disable clipping.",
    )
    parser.add_argument(
        "--grad_refresh_loss",
        type=str,
        default="kl",
        choices=["kl", "hidden_mse", "fisher_diag_mse"],
        help="Loss used to compute the true refresh gradient in block_backward/block_gd.",
    )
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
    parser.set_defaults(global_loss=False)
    parser.add_argument(
        "--global_loss_bsz",
        type=int,
        default=None,
        help=(
            "Batch size used only for the frozen end-to-end global-loss backward pass that collects static "
            "saliency/Fisher caches. Defaults to --bsz when not provided."
        ),
    )
    parser.add_argument(
        "--pre_gd_steps",
        type=int,
        default=0,
        help="Number of gradient descent steps to run after preclip/stat collection and before GPTQ quantization.",
    )
    parser.add_argument(
        "--pre_grad_lr",
        type=float,
        default=0.0,
        help="Learning rate for the pre-quantization gradient descent phase on non-final transformer layers.",
    )
    parser.add_argument(
        "--pre_final_layer_grad_lr",
        type=float,
        default=None,
        help="Optional override for pre-quantization gradient descent learning rate in the final transformer layer.",
    )
    parser.add_argument(
        "--pre_grad_optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer used for the pre-quantization gradient descent phase.",
    )
    parser.add_argument(
        "--pre_final_layer_grad_optimizer",
        type=str,
        default=None,
        choices=["sgd", "adam"],
        help="Optional override for the pre-quantization optimizer used only in the final transformer layer.",
    )
    parser.add_argument(
        "--proj_lr_scale",
        type=float,
        default=0.1,
        help="Multiplier applied to block_gd lr for attention output projections (o_proj).",
    )
    parser.add_argument(
        "--down_proj_lr_scale",
        type=float,
        default=0.1,
        help="Multiplier applied to block_gd lr for MLP down projections.",
    )
    parser.add_argument(
        "--grad_reg_strategy",
        type=str,
        default="none",
        choices=["none", "l2", "hessian", "quant_error_gate", "quant_error_gate_optimized"],
        help="Regularization strategy applied to the block-wise first-order update.",
    )
    parser.add_argument(
        "--grad_reg_lambda",
        type=float,
        default=0.0,
        help="Regularization strength for l2/hessian-weighted first-order regularization.",
    )
    parser.add_argument(
        "--grad_gate_floor",
        type=float,
        default=0.1,
        help="Minimum multiplicative lr factor used by quantization-error gating.",
    )
    parser.add_argument(
        "--grad_gate_sharpness",
        type=float,
        default=1.0,
        help="Sharpness of the quantization-error lr gate; larger means faster transition to full lr.",
    )
    parser.add_argument(
        "--grad_gate_sine_amp",
        type=float,
        default=0.0,
        help="Amplitude A of the periodic sine regularization used by quant_error_gate_optimized.",
    )
    parser.add_argument(
        "--second_order_scale",
        type=float,
        default=1.0,
        help="Scale applied to the pure GPTQ second-order update in block_gd mode.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="GPTQ block size used by the quantization inner loop.",
    )
    parser.add_argument(
        "--block_atomic_quant",
        action="store_true",
        help="Quantize each block atomically without any block-internal GPTQ/GPTQ+ updates; only apply cross-block updates after the block is quantized.",
    )
    parser.add_argument(
        "--g_update_mode",
        type=str,
        default="frozen",
        choices=["frozen", "surrogate_block", "surrogate_online", "block_backward", "block_gd"],
        help="How to update the first-order term during GPTQ+ quantization.",
    )
    parser.add_argument("--kl_topk", type=int, default=-1, help="Top-k KL loss")
    parser.add_argument(
        "--grad_hessian_topk",
        type=int,
        default=-1,
        help=(
            "When > 0, restrict the grad/hessian label sampling, saliency NLL, and KL loss "
            "to the full-precision top-k logits support. Disabled when <= 0."
        ),
    )
    parser.add_argument("--bsz", type=int, default=1, help="Batch size for computing hessians and gradients")
    parser.add_argument(
        "--final_layer_stats_bsz",
        type=int,
        default=None,
        help="Optional override for the statistics-collection batch size used only in the final transformer layer.",
    )
    parser.add_argument(
        "--backward_samples",
        type=int,
        default=-1,
        help="Number of calibration samples used in each block-backward refresh (-1 means all samples).",
    )
    parser.add_argument(
        "--backward_bsz",
        type=int,
        default=-1,
        help="Batch size used inside each block-backward refresh (-1 means reuse --bsz).",
    )
    parser.add_argument(
        "--final_layer_backward_bsz",
        type=int,
        default=None,
        help="Optional override for the refresh backward batch size used only in the final transformer layer.",
    )
    parser.add_argument(
        "--final_layer_full_backward",
        action="store_true",
        help="Use all calibration samples for every block refresh in the final transformer layer while keeping earlier layers on --backward_samples.",
    )
    parser.add_argument("--load_qmodel_path", type=str, default=None, help="The path to load quantized model ckpt")
    parser.add_argument("--save_qmodel_path", type=str, default=None, help="The path to save quantized model ckpt")
    parser.add_argument("--offload_inps", action="store_true", help="Offload inputs to CPU")
    parser.add_argument("--enable_quant_profile", action="store_true", help="Emit NVTX ranges for Nsight profiling during GPTQ+ quantization.")
    parser.add_argument("--quant_profile_target_layers", type=str, default="all", help="Layer ids to profile in detail when quant profiling is enabled.")
    parser.add_argument("--quant_profile_target_modules", type=str, default="all", help="Comma-separated module names to profile in detail when quant profiling is enabled.")
    parser.add_argument("--quant_stop_layer", type=str, default=None, help="Inclusive transformer layer index to stop after; intended for short profiling/debug runs.")
    # Eval
    parser.add_argument("--skip_eval", action="store_true", help="Skip KL/PPL and QA evaluation")
    parser.add_argument("--lm_eval", action="store_true", help="Enable QA eval")
    parser.add_argument("--lm_eval_batch_size", type=int, default=32, help="Batch size for QA tasks")
    parser.add_argument("--eval_datasets", type=list[str], default=["wikitext2", "ultrachat_2k", "numinamath"],
                        help="Datasets for PPL & KL eval")
    # Exp
    parser.add_argument("--enable_debug", action="store_true", help="Enable debugging")
    # Diagnostic dump: saves raw matrices at every fasterquant checkpoint for
    # selected (layer_idx, module_substring) pairs. Zero overhead when off.
    parser.add_argument(
        "--diagnose_targets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of `<layer_idx>:<module_substring>` tuples; dumps "
            "W/Q/Err/H/updates/refresh_grad/etc per block for each matching module. "
            "Example: `2:mlp.down_proj,1:mlp.down_proj,2:self_attn.q_proj`."
        ),
    )
    parser.add_argument(
        "--diagnose_dir",
        type=str,
        default=None,
        help=(
            "Where to write diagnostic dumps. Defaults to `<output_dir>/diagnostics/` "
            "when --diagnose_targets is set."
        ),
    )
    parser.add_argument(
        "--diagnose_spike_ratio",
        type=float,
        default=3.0,
        help="Flag a block as `is_spike` in meta.json when loss[i]/loss[i-1] exceeds this.",
    )

    args = parser.parse_args()

    # set paths & others
    args.model_name = args.model.split("/")[-1]
    args.output_dir = os.path.join(args.output_dir, args.model_name, args.exp)
    args.log_dir = os.path.join(args.output_dir, "logs")
    args.tokens_cache_path = (f"{args.cache_dir}/tokens/"
                              f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}.pt")
    if args.num_groups is not None:
        args.saliency_cache_path = (f"{args.cache_dir}/saliency/"
                                    f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_g{args.num_groups}")
        args.gradients_cache_path = (f"{args.cache_dir}/gradients/"
                                    f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_g{args.num_groups}.pt")
    else:
        args.saliency_cache_path = None
        args.gradients_cache_path = None

    transformers.set_seed(args.seed)

    init_logging(args.log_dir)

    if args.fisher_num_groups is None:
        args.fisher_num_groups = args.num_groups

    if args.backward_samples == -1:
        args.backward_samples = args.nsamples
    if args.backward_samples <= 0:
        raise ValueError(f"`backward_samples` must be positive or -1. Got {args.backward_samples}.")

    # DP divisibility constraints. When running under torchrun with N>1 ranks,
    # sample counts and batch sizes must split evenly across ranks. We read
    # WORLD_SIZE from the env so these checks fire at parse time — failing late
    # inside gptq_fwrd after loading the model would waste a lot of startup.
    _dp_world = int(os.environ.get("WORLD_SIZE", 1))
    if _dp_world > 1:
        if args.nsamples % _dp_world != 0:
            raise ValueError(
                f"DP requires nsamples ({args.nsamples}) divisible by WORLD_SIZE ({_dp_world})."
            )
        if args.backward_samples % _dp_world != 0:
            raise ValueError(
                f"DP requires backward_samples ({args.backward_samples}) divisible by WORLD_SIZE ({_dp_world})."
            )
        if args.bsz % _dp_world != 0:
            raise ValueError(
                f"DP requires bsz ({args.bsz}) divisible by WORLD_SIZE ({_dp_world})."
            )
        if args.global_loss_bsz is not None and args.global_loss_bsz > 0 and args.global_loss_bsz % _dp_world != 0:
            raise ValueError(
                f"DP requires global_loss_bsz ({args.global_loss_bsz}) divisible by WORLD_SIZE ({_dp_world})."
            )
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
    if args.grad_hessian_topk == 0:
        raise ValueError("`grad_hessian_topk` must be positive or negative to disable. Use -1 to disable.")
    if args.fisher_num_groups <= 0:
        raise ValueError(f"`fisher_num_groups` must be positive. Got {args.fisher_num_groups}.")
    logging.info(args)

    # Disable parallelism in tokenizers to prevent warnings when forking in the seed generation step
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.enable_debug:
        import debugpy
        debugpy.listen(5678 + dist_utils.get_rank())
        logging.info("Waiting for debugger attach")
        debugpy.wait_for_client()
        # debugpy.breakpoint()

    return args
