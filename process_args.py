import argparse
import os
import logging
import hashlib

import transformers

from utils import dist_utils
from utils.log_utils import init_logging


def parse_gen():
    parser = argparse.ArgumentParser(description="Quantize a model to any precision")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision")
    parser.add_argument("--mp_target_avg_bits", type=float, default=3.5, help="Target average bits")
    parser.add_argument("--mp_high_bits", type=int, default=4, help="the higher weight bits")
    parser.add_argument("--mp_low_bits", type=int, default=3, help="the lower weight bits")
    parser.add_argument(
        "--mp_saliency_metric",
        type=str,
        default="fisher_mean",
        choices=["fisher_mean", "fisher_max", "fisher_topk", "random"],
        help=(
            "Metric used to score each module's sensitivity for mixed-precision bit allocation. "
            "'fisher_mean': mean of per-token squared gradient (default). "
            "'fisher_max': max of per-token squared gradient (peak sensitivity). "
            "'fisher_topk': mean of top-k%% most sensitive tokens (see --mp_topk_ratio). "
            "'random': random scores; ablation baseline to test whether saliency ordering matters."
        ),
    )
    parser.add_argument(
        "--mp_granularity",
        type=str,
        default="module",
        choices=["module", "layer", "type"],
        help=(
            "Granularity of mixed-precision bit allocation. "
            "'module': each linear module gets an independent bit width (finest, default). "
            "'layer': all modules within a transformer layer share the same bits. "
            "'type': all modules of the same role (e.g. q_proj, mlp.down_proj) share bits across layers."
        ),
    )
    parser.add_argument(
        "--mp_topk_ratio",
        type=float,
        default=0.1,
        help="Fraction of tokens kept by the fisher_topk saliency metric (default 0.1 = top 10%%).",
    )
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
    parser.add_argument(
        "--a_loss_ratio",
        type=float,
        default=1.0,
        help=(
            "Fisher-MSE delta-y clipping quantile. 1.0 leaves delta_y unchanged; "
            "values <1 scale elements with |delta_y| above that quantile down to "
            "the quantile magnitude before computing the Fisher/refined MSE loss."
        ),
    )
    parser.add_argument("--k_clip_ratio", type=float, default=1.0, help="K cache clipping ratio")
    parser.add_argument("--v_clip_ratio", type=float, default=1.0, help="V cache clipping ratio")
    parser.add_argument(
        "--act_quant_aware_gptq",
        action="store_true",
        help=(
            "GPTQ+ only: enable activation quantization during the GPTQ+ student "
            "path so Hessian accumulation, refresh gradients, and layer replay see "
            "the same A-quantized activations used at evaluation. FP teacher paths "
            "still disable activation quantization."
        ),
    )
    parser.add_argument(
        "--k_cache_quant_aware_gptq",
        action="store_true",
        help=(
            "GPTQ+ only: install K-cache quantization during the GPTQ+ student "
            "path so Hessian accumulation, refresh gradients, and layer replay see "
            "online K fake quantization. FP teacher paths keep K quantization disabled."
        ),
    )
    parser.add_argument(
        "--rtn_act_error_debug",
        action="store_true",
        help=(
            "RTN only: run a temporary FP-weight activation-quantization diagnostic. "
            "Use with --w_bits 16 and --a_bits/--v_bits <16 to print layer-output "
            "relative error after enabling each linear layer's activation fake quant."
        ),
    )
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
        "--final_layer_grad_clip",
        type=float,
        default=None,
        help=(
            "Optional override for --grad_clip used only in the final transformer "
            "layer. The final layer's backward path goes through lm_head + final "
            "norm and often produces much larger gradients than earlier blocks, "
            "so a looser (or tighter) clip can help. Leave unset to reuse "
            "--grad_clip. Set negative to disable clipping on the final layer."
        ),
    )
    parser.add_argument(
        "--grad_refresh_loss",
        type=str,
        default="kl",
        choices=["kl", "hidden_mse", "fisher_diag_mse", "residual_kl", "refined_residual_kl", "refined_mse", "refined_mix"],
        help=(
            "Loss used to compute the true refresh gradient in block_backward/block_gd. "
            "'residual_kl' assumes the current-layer output delta flows through the "
            "remaining residual stream unchanged and only measures its effect after the "
            "final norm + lm_head (cheap approximation of end-to-end KL). "
            "'refined_residual_kl' replaces that zero-order approximation with a first-order "
            "Jacobian f(x+Δx) ≈ f(x) + A·Δx; the shared H×H matrix A per layer is fit via "
            "least squares on (dy, dx-dy) pairs during the static end-to-end backward. "
            "'refined_mse' adds a first-order term g·Δy (where g = ∂KL/∂layer_output, "
            "collected end-to-end per layer just before its quant loop opens) on top of "
            "fisher_diag_mse's second-order term — the full Taylor expansion of end-to-end "
            "KL in the layer output. "
            "'refined_mix' splits the layer stack: [0, split) use fisher_diag_mse, "
            "[split, N-1) use refined_residual_kl, and the final layer keeps its forced kl. "
            "Front-half layers skip the per-layer end-to-end grad pool collection entirely "
            "(fisher_diag_mse only reads the precomputed per-layer fisher); precompute only "
            "collects fisher for the front half and A for the back half."
        ),
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
        "--loss_slide_window",
        action="store_true",
        help=(
            "In block_gd, linearly blend the current-layer fisher_diag_mse loss with the "
            "next-layer fisher_diag_mse (weights α=1→0 across per-block refreshes). "
            "Requires --grad_refresh_loss=fisher_diag_mse and --global_loss. Skipped for the "
            "last layer and the second-to-last layer."
        ),
    )
    parser.add_argument(
        "--ignore_attention_sink",
        action="store_true",
        help=(
            "Skip the first --attention_sink_size tokens of every calibration sequence when "
            "computing any loss (NLL / KL / hidden_mse / fisher_diag_mse / refined_mse / "
            "residual_kl / refined_residual_kl / refined_diag_residual_kl). Sink tokens still "
            "participate in forward/KV (so downstream tokens see them as context), but their "
            "loss contribution and the gradients flowing back through them are zero. "
            "Affects static saliency/Fisher precompute and per-layer stats; the on-disk static "
            "cache key is augmented with `_sink{N}` so caches don't cross-contaminate. "
            "Default off (no slicing)."
        ),
    )
    parser.add_argument(
        "--attention_sink_size",
        type=int,
        default=256,
        help=(
            "Number of leading tokens to drop from every loss calculation when "
            "--ignore_attention_sink is set. Must be < --seq_len. Default 256."
        ),
    )
    parser.add_argument(
        "--dp_global_shuffle",
        action="store_true",
        help=(
            "Use a single globally-shared shuffle of all nsamples across ranks so every "
            "rank selects the same global sample ids each refresh. Each rank filters to "
            "its own shard and contributes a partial (sum, count). Makes DP results "
            "numerically match a 1-GPU run with the same seed. Default (off) keeps the "
            "per-rank stratified scheduler."
        ),
    )
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
        "--fsdp_precompute",
        action="store_true",
        help=(
            "Wrap the model with FSDP2 during `collect_static_end_to_end_saliency_and_fisher` so the "
            "end-to-end backward fits on multi-GPU setups where the full model + grads don't fit on a "
            "single card (Llama-2-70B, Llama-3-70B, etc). Requires torchrun launch. Combine with "
            "`--fsdp_cpu_offload` to spill param shards to CPU between layers."
        ),
    )
    parser.add_argument(
        "--fsdp_cpu_offload",
        action="store_true",
        help="When --fsdp_precompute is set, offload param shards to CPU (pinned) between layer forwards.",
    )
    parser.add_argument(
        "--fsdp_meta_init",
        action="store_true",
        help=(
            "Precompute-only memory saver: initialize the model on meta, apply FSDP2 before "
            "checkpoint load, then load directly into sharded FSDP params. This avoids each "
            "torchrun rank materializing a full CPU copy before FSDP. Requires "
            "--fsdp_precompute --exit_after_precompute."
        ),
    )
    parser.add_argument(
        "--fsdp_prepared_max_shard_size",
        type=str,
        default="5GB",
        help=(
            "Max shard size used when --fsdp_meta_init has to write a rank0-prepared "
            "checkpoint (rotated or untied) before FSDP sharded loading."
        ),
    )
    parser.add_argument(
        "--static_cache_path",
        type=str,
        default=None,
        help=(
            "Optional directory to persist the static end-to-end saliency/fisher caches. "
            "If set and the cache exists (keyed by model/dataset/nsamples/seq_len/num_groups/"
            "full_fisher/grad_hessian_topk/global_loss_bsz/seed/rotate), the precompute is skipped and the saved "
            "tensors are loaded per rank. First run writes, subsequent runs read."
        ),
    )
    parser.add_argument(
        "--exit_after_precompute",
        action="store_true",
        help=(
            "Exit right after `collect_static_end_to_end_saliency_and_fisher` finishes and the results "
            "are saved to `--static_cache_path`. Useful for the FSDP two-stage workflow: run precompute "
            "under a torchrun that wraps the model with FSDP, then run the quantization pass separately "
            "(which just reads the cache)."
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
        choices=["sgd", "adam", "h_adam"],
        help="Optimizer used for the pre-quantization gradient descent phase. "
             "`h_adam` preconditions with the KFAC curvature diagonal "
             "(output-Fisher ⊗ input-Hessian) instead of Adam's exp_avg_sq.",
    )
    parser.add_argument(
        "--pre_final_layer_grad_optimizer",
        type=str,
        default=None,
        choices=["sgd", "adam", "h_adam"],
        help="Optional override for the pre-quantization optimizer used only in the final transformer layer.",
    )
    parser.add_argument(
        "--h_adam_curvature_damping",
        type=float,
        default=0.1,
        help="Relative damping added to the (mean-normalised) H-Adam KFAC "
             "curvature diagonal before the sqrt-preconditioner. Larger values "
             "make H-Adam behave closer to plain SGD/Adam on near-zero-curvature "
             "channels. Only used when a pre-grad optimizer is `h_adam`.",
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
        "--grad_lr_layer_schedule",
        type=str,
        default="none",
        choices=["none", "cosine", "linear", "sqrt"],
        help=(
            "Per-layer ramp for --grad_lr and --pre_grad_lr. "
            "'none' keeps grad_lr constant across layers (default). "
            "The ramp goes 0 → 1 from layer 0 to the last layer: "
            "'cosine' = 0.5*(1-cos(π·x)), 'linear' = x, 'sqrt' = √x, "
            "where x = layer_idx / (num_layers - 1). "
            "Does NOT scale --final_layer_grad_lr / --pre_final_layer_grad_lr "
            "(the final layer keeps its dedicated LR)."
        ),
    )
    parser.add_argument(
        "--grad_lr_layer_base_ratio",
        type=float,
        default=0.01,
        help=(
            "Base LR as a fraction of --grad_lr / --pre_grad_lr for per-layer "
            "schedules. Scheduled non-final layers use "
            "base_lr + (target_lr - base_lr) * schedule_scale, where "
            "base_lr = target_lr * ratio. Default 0.01 makes layer 0 start at "
            "1% of the target LR. Set 0 to recover the old zero-base ramp."
        ),
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
    parser.add_argument(
        "--saliency_clip_percentile",
        type=float,
        default=0.99,
        help=(
            "In `collect_static_end_to_end_saliency_and_fisher`, clip per-token saliency "
            "(grad² of end-to-end NLL wrt module output) to this percentile before caching. "
            "Prevents a handful of extreme-gradient tokens in deep layers from collapsing "
            "the downstream weighted Hessian `inp.T @ diag(s) @ inp` to near rank-1, which "
            "makes Cholesky fail even with large damp. Set to 1.0 to disable clipping."
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
        "--hessian_accum_bsz",
        type=int,
        default=None,
        help=(
            "Batch size for the Hessian accumulation forward loop (add_batch). "
            "Defaults to --bsz when unset. Independent of stats collection bsz "
            "so you can lower this if add_batch's `weighted` tensor spikes memory."
        ),
    )
    parser.add_argument(
        "--enable_gptq_plus",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "When 0, bypass all GPTQ+ first-order extensions and run pure GPTQ: "
            "no pre-quant GD, no block refresh / gradient descent, no GHinv / Z "
            "term in the per-column inner loop, no fisher precompute, no gradient "
            "reference-loss backward during stats collection."
        ),
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
    parser.add_argument("--eval_datasets", nargs='+', default=["wikitext2", "ultrachat_2k", "numinamath"],
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
    # analyze_grad_cosine.py only
    parser.add_argument(
        "--target_layers",
        type=str,
        default="1,5,10,15",
        help=(
            "Comma-separated transformer block indices at which to measure gradient cosine "
            "(true KL vs fisher_diag_mse vs residual_kl). Use 'all' for every layer. Index 0 "
            "is dropped with a warning because inps==fp_inps there makes the surrogate grads zero."
        ),
    )
    parser.add_argument(
        "--measure_samples",
        type=int,
        default=64,
        help="Number of calibration samples used to compute gradient cosine at each target layer.",
    )
    parser.add_argument(
        "--measure_batch_size",
        type=int,
        default=4,
        help="Batch size for each backward pass in the cosine measurement loop.",
    )
    parser.add_argument(
        "--refined_rkl_damp",
        type=float,
        default=0.01,
        help=(
            "Damping coefficient for the refined_residual_kl least-squares fit of A. "
            "H = Σ dyᵀdy gets a diagonal bump of damp · trace(H)/H · I before inversion. "
            "Default 0.01 matches GPTQ-style percent damping."
        ),
    )
    parser.add_argument(
        "--refined_rkl_num_A",
        type=int,
        default=1,
        help=(
            "Number of A matrices per layer for refined_residual_kl. Samples are split "
            "into num_A contiguous groups of size nsamples/num_A, each group fits its "
            "own A via LS. At inference time, batch picks A based on its sample id. "
            "Requires nsamples % num_A == 0 and (nsamples/num_A) divisible by batch sizes "
            "used during fit and measurement. Default 1 = original single-A behaviour."
        ),
    )
    parser.add_argument(
        "--refresh_full_metrics",
        action="store_true",
        help=(
            "Record the full per-refresh diagnostic set (trailing_grad / second_order / "
            "first_order / regularizer abs_mean / row_l2 / abs_max / q99, plus slide_alpha "
            "split losses). Each metric requires a torch op + .item() on GPU → "
            "cudaStreamSynchronize, and in DP mode adds several scalar allreduces per "
            "refresh; on a multi-layer sweep this dominates true_gradient_refresh wall "
            "time. With this flag OFF (default) only `mean_refresh_loss` is kept, and the "
            "per-refresh aggregation allreduce is packed into a single collective. Turn "
            "ON for diagnosing loss spikes or validating new regularizer ideas."
        ),
    )
    parser.add_argument(
        "--num_samples_for_refined_mse",
        type=int,
        default=32,
        help=(
            "refined_mse only: per-rank size of the end-to-end grad pool collected "
            "fresh before each transformer block's quant loop opens. For each layer, "
            "a random subset of the rank-local calibration samples (seeded with "
            "seed + layer_idx) is forwarded end-to-end in FP and backward'd to capture "
            "g = ∂KL/∂(layer output) per (sample, token). Refresh mini-batches pull "
            "exact g for samples in the pool and fall back to the pool mean (over "
            "sample, seq) for everyone else. Must be >0, ≤ nsamples // world, and "
            "divisible by (global_loss_bsz // world)."
        ),
    )
    parser.add_argument(
        "--refined_mix_split_layer",
        type=int,
        default=None,
        help=(
            "refined_mix only: layers [0, split) use fisher_diag_mse, [split, N-1) use "
            "refined_residual_kl, final layer always kl. Default (None) resolves to "
            "N // 2 at runtime where N is the number of transformer blocks."
        ),
    )
    parser.add_argument(
        "--refined_mix_rkl_lr_ratio",
        type=float,
        default=1.0,
        help=(
            "refined_mix only: multiplier applied to --grad_lr and --pre_grad_lr for "
            "the refined_residual_kl half of layers. 1.0 = same LR for both halves. "
            "The final layer keeps --final_layer_grad_lr independently."
        ),
    )
    parser.add_argument(
        "--measure_losses",
        type=str,
        default="fisher_diag_mse,residual_kl,refined_residual_kl,refined_diag_residual_kl",
        help=(
            "Comma-separated subset of surrogate losses to measure against the true "
            "end-to-end KL gradient in analyze_grad_cosine. Choices: fisher_diag_mse, "
            "residual_kl, refined_residual_kl, refined_diag_residual_kl, refined_mse. "
            "Only the fits / backward passes needed for the selected set are run "
            "(saves memory and time — e.g. skipping refined_residual_kl avoids the "
            "H×H A fit)."
        ),
    )
    parser.add_argument(
        "--enable_dynamic_saliency",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Dynamic saliency update. When 1, refresh per-module saliency at each of the "
            "4 module-group boundaries (qkv / o_proj / up+gate / down) using a low-rank "
            "decomposition of the per-module end-to-end gradient matrix G = U·Σ·V^T "
            "(captured once per module in precompute). At each boundary, ΔY = current_Y − "
            "FP_Y drives the Fisher-linearized gradient perturbation Δg ≈ (1/N)·G^T G·ΔY, "
            "which in turn updates the per-group saliency S = E[g²]. Requires --global_loss. "
            "Adds ~1 extra end-to-end backward in precompute (for U) plus 1 FP forward + "
            "1 current-state forward per layer-group boundary. See saliency_dynamic_update_design.md."
        ),
    )
    parser.add_argument(
        "--dyn_sal_rank",
        type=int,
        default=16,
        help=(
            "--enable_dynamic_saliency only: low-rank dimension R kept after EVD of per-module "
            "G^T G. Higher R = finer Δg approximation but larger per-module U_Sigma cache "
            "(N_local · T · R) on CPU and more matmul per boundary. Default 16. Typical sweep: "
            "8 / 16 / 32 / 64."
        ),
    )
    parser.add_argument(
        "--dyn_sal_evd_thresh",
        type=float,
        default=1e-6,
        help=(
            "--enable_dynamic_saliency only: after EVD of G^T G, drop eigenvalues below "
            "max(Σ²) · thresh to avoid Σ⁻¹ blowup on near-zero singular values. Default 1e-6."
        ),
    )
    parser.add_argument(
        "--dyn_sal_refresh_mode",
        type=str,
        default="per_boundary",
        choices=["per_boundary", "per_layer"],
        help=(
            "--enable_dynamic_saliency only: when to refresh saliency inside each "
            "transformer layer's quant loop.\n"
            "  per_boundary (default): refresh 4× per layer, once BEFORE each of the "
            "    four module-group boundaries (qkv / o / up+gate / down). Captures both "
            "    upstream drift AND the accumulated effect of already-quantized modules "
            "    in this layer.\n"
            "  per_layer: refresh ONCE per layer, at layer entry with all weights still "
            "    FP, and accumulate Hessian ONCE for the whole layer before the four "
            "    module-group quantization boundaries. Captures upstream drift only, "
            "    then reuses the per-module H through qkv / o / up+gate / down."
        ),
    )

    args = parser.parse_args()

    # set paths & others
    args.model_name = args.model.split("/")[-1]
    args.output_dir = os.path.join(args.output_dir, args.model_name, args.exp)
    args.log_dir = os.path.join(args.output_dir, "logs")
    args.tokens_cache_path = (f"{args.cache_dir}/tokens/"
                              f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}.pt")
    if args.num_groups is not None:
        rotation_cache_tag = f"rot{int(bool(args.rotate))}_seed{args.seed}"
        if args.optimized_rotation_path is not None:
            opt_hash = hashlib.sha1(str(args.optimized_rotation_path).encode()).hexdigest()[:8]
            rotation_cache_tag += f"_opt{opt_hash}"
        args.saliency_cache_path = (f"{args.cache_dir}/saliency/"
                                    f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_"
                                    f"{rotation_cache_tag}_g{args.num_groups}")
        args.gradients_cache_path = (f"{args.cache_dir}/gradients/"
                                    f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_"
                                    f"{rotation_cache_tag}_g{args.num_groups}.pt")
    else:
        args.saliency_cache_path = None
        args.gradients_cache_path = None

    transformers.set_seed(args.seed)

    init_logging(args.log_dir)

    if args.w_bits < 16:
        if args.w_asym:
            raise ValueError(
                "--w_asym is temporarily unsupported for weight quantization. "
                "The GPTQ/GPTQ+ fake-quant path currently does not preserve asymmetric zero-points."
            )
        if args.w_groupsize != -1 and args.w_groupsize != args.blocksize:
            raise ValueError(
                "--w_groupsize must be -1 or equal to --blocksize for now. "
                f"Got w_groupsize={args.w_groupsize}, blocksize={args.blocksize}."
            )

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
    if not (0.0 < args.a_loss_ratio <= 1.0):
        raise ValueError(f"`a_loss_ratio` must be in (0, 1]. Got {args.a_loss_ratio}.")
    if args.h_adam_curvature_damping < 0:
        raise ValueError(
            f"`h_adam_curvature_damping` must be non-negative. Got {args.h_adam_curvature_damping}."
        )
    if getattr(args, "act_quant_aware_gptq", False):
        if args.w_method != "gptq_plus":
            raise ValueError("--act_quant_aware_gptq is currently implemented only for --w_method=gptq_plus.")
        if args.grad_refresh_loss == "refined_mse":
            raise ValueError(
                "--act_quant_aware_gptq does not support --grad_refresh_loss=refined_mse yet. "
                "The refined_mse grad-pool path and layer-0 shortcut still assume an FP student path."
            )
    if getattr(args, "k_cache_quant_aware_gptq", False):
        if args.w_method != "gptq_plus":
            raise ValueError("--k_cache_quant_aware_gptq is currently implemented only for --w_method=gptq_plus.")
        if args.k_bits >= 16:
            raise ValueError("--k_cache_quant_aware_gptq requires --k_bits < 16.")
        if args.grad_refresh_loss == "refined_mse":
            raise ValueError(
                "--k_cache_quant_aware_gptq does not support --grad_refresh_loss=refined_mse yet. "
                "The refined_mse grad-pool path still assumes an FP student path."
            )
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
    if getattr(args, "loss_slide_window", False):
        if args.g_update_mode != "block_gd":
            raise ValueError("--loss_slide_window requires --g_update_mode=block_gd.")
        if args.grad_refresh_loss not in ("fisher_diag_mse", "residual_kl", "refined_residual_kl", "refined_mse", "refined_mix"):
            raise ValueError(
                "--loss_slide_window requires --grad_refresh_loss in "
                "{fisher_diag_mse, residual_kl, refined_residual_kl, refined_mse, refined_mix}."
            )
        if args.grad_refresh_loss == "fisher_diag_mse" and not args.global_loss:
            raise ValueError(
                "--loss_slide_window + fisher_diag_mse requires --global_loss "
                "(so next-layer fisher is cached)."
            )
        if args.grad_refresh_loss == "refined_mse" and not args.global_loss:
            raise ValueError(
                "--loss_slide_window + refined_mse requires --global_loss "
                "(next-layer fisher + fp_inps_final are both cached under the "
                "global-loss precompute path)."
            )
        # refined_mix uses both halves' stats; --global_loss is required for
        # either half's slide-window dependency (fisher[i+1] / A[i+1] /
        # fp_inps_final). The per-type --global_loss check below also catches
        # this, but surface it next to the other slide_window guards for
        # locality.
        if args.grad_refresh_loss == "refined_mix" and not args.global_loss:
            raise ValueError(
                "--loss_slide_window + refined_mix requires --global_loss."
            )
    # residual_kl + slide_window is supported: the next-layer loss computes
    # δ_next = next_layer(out_hidden) - fp_inps_next, then runs the same
    # residual-stream shortcut using fp_inps_final. No incompatibility.
    # refined_residual_kl + slide_window is also supported: the next-layer
    # loss uses A_{i+1} in place of A_i. --global_loss is required for both
    # fp_inps_final and A to be cached, which is enforced below.
    if args.grad_refresh_loss == "refined_residual_kl":
        # refined_residual_kl needs the per-layer A matrix, which is fit inside
        # `collect_static_end_to_end_saliency_and_fisher`. That precompute only
        # runs when global_loss is enabled, so refuse early rather than silently
        # fall back to residual_kl-style behavior with no A.
        if not args.global_loss:
            raise ValueError(
                "--grad_refresh_loss=refined_residual_kl requires --global_loss "
                "(the per-layer A matrix is fit during the global-loss precompute)."
            )
    if getattr(args, "refined_rkl_num_A", 1) < 1:
        raise ValueError(
            f"`refined_rkl_num_A` must be >= 1. Got {args.refined_rkl_num_A}."
        )
    if args.refined_rkl_num_A > 1 and args.nsamples % args.refined_rkl_num_A != 0:
        raise ValueError(
            f"refined_rkl_num_A ({args.refined_rkl_num_A}) must divide nsamples "
            f"({args.nsamples}) evenly."
        )
    # refined_mse validation: needs end-to-end forward machinery (→ global_loss
    # enables fp_inps_final precompute), per-rank pool size bounded by local
    # shard + divisible by per-rank backward batch so the collection loop
    # doesn't trail a short batch.
    if args.grad_refresh_loss == "refined_mse":
        if not args.global_loss:
            raise ValueError(
                "--grad_refresh_loss=refined_mse requires --global_loss (fp_inps_final "
                "is only cached under the global-loss precompute path)."
            )
        if int(getattr(args, "enable_gptq_plus", 1)) != 0:
            raise ValueError(
                "--grad_refresh_loss=refined_mse requires --enable_gptq_plus 0 in v1. "
                "With GPTQ+ on (alpha≠0) the reference-loss backward inside "
                "collect_layer_grad_hessian_stats would call compute_refresh_loss("
                "'refined_mse', ...) without the per-layer grad pool plumbed through; "
                "pushing pool args down that code path is deferred."
            )
        if args.num_samples_for_refined_mse <= 0:
            raise ValueError(
                f"--num_samples_for_refined_mse must be positive. "
                f"Got {args.num_samples_for_refined_mse}."
            )
        # Per-rank divisibility checks: dp_world inferred from WORLD_SIZE env
        # (falls back to 1 in single-GPU runs). nsamples/global_loss_bsz div
        # by world is already validated further up, so //world is integer.
        _dp_world_r = int(os.environ.get("WORLD_SIZE", "1"))
        _n_local = args.nsamples // _dp_world_r
        if args.num_samples_for_refined_mse > _n_local:
            raise ValueError(
                f"--num_samples_for_refined_mse ({args.num_samples_for_refined_mse}) "
                f"must be <= nsamples // world ({_n_local})."
            )
        _bwd_bsz_local = args.global_loss_bsz // _dp_world_r
        if _bwd_bsz_local <= 0 or args.num_samples_for_refined_mse % _bwd_bsz_local != 0:
            raise ValueError(
                f"--num_samples_for_refined_mse ({args.num_samples_for_refined_mse}) "
                f"must be divisible by per-rank backward bsz ({_bwd_bsz_local} = "
                f"global_loss_bsz // world)."
            )
    # refined_mix validation: front half uses fisher_diag_mse (no per-layer
    # end-to-end backward, no pool), back half uses refined_residual_kl (A
    # precomputed once, fp_inps_final as teacher), final layer forced to kl.
    # Both halves need the static precompute pass, so --global_loss is
    # required. No refined_mse-style enable_gptq_plus=0 limit here because
    # neither half routes the reference-loss backward through refined_mse's
    # pool-dependent code path.
    if args.grad_refresh_loss == "refined_mix":
        if not args.global_loss:
            raise ValueError(
                "--grad_refresh_loss=refined_mix requires --global_loss (the precompute "
                "pass fits fisher for the fisher_diag_mse front half, A for the "
                "refined_residual_kl back half, and captures fp_inps_final for the back half)."
            )
        if args.refined_mix_split_layer is not None and args.refined_mix_split_layer < 1:
            raise ValueError(
                f"--refined_mix_split_layer ({args.refined_mix_split_layer}) must be >= 1 "
                "(upper bound is checked against len(layers) at runtime)."
            )
        if args.refined_mix_rkl_lr_ratio <= 0:
            raise ValueError(
                f"--refined_mix_rkl_lr_ratio must be positive. "
                f"Got {args.refined_mix_rkl_lr_ratio}."
            )
    if int(getattr(args, "enable_dynamic_saliency", 0)) == 1:
        if not args.global_loss:
            raise ValueError(
                "--enable_dynamic_saliency=1 requires --global_loss (per-module G^T G is "
                "collected during the same end-to-end backward as saliency/fisher)."
            )
        if args.dyn_sal_rank <= 0:
            raise ValueError(
                f"--dyn_sal_rank must be positive when --enable_dynamic_saliency=1. "
                f"Got {args.dyn_sal_rank}."
            )
        if not (0.0 < args.dyn_sal_evd_thresh < 1.0):
            raise ValueError(
                f"--dyn_sal_evd_thresh must be in (0, 1). Got {args.dyn_sal_evd_thresh}."
            )
    if getattr(args, "fsdp_meta_init", False):
        if not args.fsdp_precompute:
            raise ValueError("--fsdp_meta_init requires --fsdp_precompute.")
        if not args.exit_after_precompute:
            raise ValueError(
                "--fsdp_meta_init requires --exit_after_precompute. The meta/FSDP-loaded "
                "model remains DTensor-wrapped and is intended only for the Stage 1 "
                "static precompute process."
            )
        if args.static_cache_path is None:
            raise ValueError(
                "--fsdp_meta_init requires --static_cache_path so the Stage 1 "
                "precompute result can be saved and reused by Stage 2."
            )
    if args.fsdp_precompute and args.exit_after_precompute and not args.skip_eval:
        logging.info(
            "--fsdp_precompute --exit_after_precompute is a Stage 1 cache build; "
            "forcing --skip_eval because reference-logit eval is unused and "
            "incompatible with FSDP/DTensor-managed params."
        )
        args.skip_eval = True
    if args.nsamples % args.backward_samples != 0:
        raise ValueError(
            f"`nsamples` ({args.nsamples}) must be divisible by `backward_samples` ({args.backward_samples})."
        )
    if args.grad_reg_lambda < 0:
        raise ValueError(f"`grad_reg_lambda` must be non-negative. Got {args.grad_reg_lambda}.")
    if getattr(args, "ignore_attention_sink", False):
        if args.attention_sink_size <= 0:
            raise ValueError(
                f"`--attention_sink_size` must be > 0 when `--ignore_attention_sink` is set. "
                f"Got {args.attention_sink_size}."
            )
        if args.attention_sink_size >= args.seq_len:
            raise ValueError(
                f"`--attention_sink_size` ({args.attention_sink_size}) must be strictly less than "
                f"`--seq_len` ({args.seq_len}); otherwise every loss has 0 valid tokens."
            )
    if args.grad_clip == 0:
        raise ValueError("`grad_clip` must be non-zero. Use a negative value to disable clipping.")
    if args.final_layer_grad_clip is not None and args.final_layer_grad_clip == 0:
        raise ValueError(
            "`final_layer_grad_clip` must be non-zero when provided. "
            "Use a negative value to disable clipping on the final layer."
        )
    if args.final_layer_grad_lr is not None and args.final_layer_grad_lr < 0:
        raise ValueError(f"`final_layer_grad_lr` must be non-negative when provided. Got {args.final_layer_grad_lr}.")
    if args.pre_gd_steps < 0:
        raise ValueError(f"`pre_gd_steps` must be non-negative. Got {args.pre_gd_steps}.")
    if args.pre_grad_lr < 0:
        raise ValueError(f"`pre_grad_lr` must be non-negative. Got {args.pre_grad_lr}.")
    if args.pre_final_layer_grad_lr is not None and args.pre_final_layer_grad_lr < 0:
        raise ValueError(f"`pre_final_layer_grad_lr` must be non-negative when provided. Got {args.pre_final_layer_grad_lr}.")
    if args.grad_lr_layer_base_ratio < 0:
        raise ValueError(f"`grad_lr_layer_base_ratio` must be non-negative. Got {args.grad_lr_layer_base_ratio}.")
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
