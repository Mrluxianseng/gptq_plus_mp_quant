"""Configuration for RealQ.

A single dataclass replaces ``process_args.py``'s large argparse surface.
It retains the numerical and runtime controls used by the refactored pipeline,
including FSDP precompute, A/V/K fake quantization, the analytical GPTQ+ term,
Block-GD, and regularization compatibility settings. Deprecated pre-GD knobs
are accepted only where legacy CLI/checkpoint compatibility requires them.

The dataclass exposes attribute names that match what `utils.eval_utils` /
`utils.rotation_utils` / `utils.data_utils` already read from `args.X`, so we
can pass `Config` instances straight into those helpers without an adapter.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class Config:
    # ----- model / data ----------------------------------------------------
    model: str = ""
    dataset: str = "wikitext2"
    eval_datasets: list[str] = field(default_factory=lambda: ["wikitext2"])
    # ``seed`` deliberately means calibration sampling only.  The paper's
    # seed sweep changes the calibration set while holding every other
    # stochastic artifact fixed.
    seed: int = 42
    rotation_seed: int = 0
    refresh_seed: int = 0
    nsamples: int = 128
    seq_len: int = 2048
    eval_seq_len: int = 2048

    # ----- weight quantisation --------------------------------------------
    w_bits: int = 4
    w_groupsize: int = -1     # -1 = per-row
    w_asym: bool = False
    w_clip: bool = True       # MSE-based clip search in find_params

    # ----- RealQ algorithm -------------------------------------------------
    num_groups: int = 4       # Hessian groups per linear (output-row sharing)
    percdamp: float = 0.01
    blocksize: int = 128
    act_order: bool = True
    # When ``rank``, output-row sharding inside ``RealQLayer.quantize``: each
    # DP rank owns ``out_features / world_size`` consecutive rows for
    # find_params + per-row inner block update. Rank shards are gathered at
    # the END of quantize so module.weight ends up replicated on every rank.
    # ``none`` keeps the legacy redundant-replica path (each rank runs the
    # full quantize). Old code's tensor-mode is not ported (RealQ already
    # vectorises across NUM_GROUPS in the per-group fallback).
    group_parallel_quant: str = "rank"  # one of: none, rank

    # ----- static end-to-end precompute -----------------------------------
    global_loss_bsz: int = 16
    saliency_clip_percentile: float = 0.99
    grad_hessian_topk: int = -1   # <= 0 = full vocab (paper protocol)
    static_cache_path: Optional[str] = None
    exit_after_precompute: bool = False
    # Optional old/new numerical-alignment trace. Only distributed rank zero
    # writes JSONL; all ranks still aggregate the traced loss statistics.
    alignment_trace_path: Optional[str] = None
    alignment_run_id: str = "default"

    # ----- block_gd refresh (Adam + reverse-cosine + grad_clip) -----------
    grad_lr: float = 0.0003
    grad_clip: float = 1.0
    # Optional final-transformer-block override. ``None`` reuses
    # ``grad_clip`` exactly like the legacy implementation.
    final_layer_grad_clip: Optional[float] = None
    # Per-layer lr schedule. "cosine" uses the paper's literal all-L
    # sin(π·x/2) indexing; the final block has a separate true-KL override, so
    # the deepest scheduled non-final block does not reach `grad_lr`. "none"
    # disables the ramp entirely so every layer uses `grad_lr` (base_ratio is
    # ignored). Mirrors legacy `--grad_lr_layer_schedule` (process_args.py:393).
    grad_lr_layer_schedule: str = "cosine"
    grad_lr_layer_base_ratio: float = 0.01
    backward_samples: int = 32
    # These are GLOBAL gradient-accumulation chunk sizes, matching legacy
    # GPTQ+; they do not change `backward_samples`, hence do not change the
    # number of samples in one Adam step.
    # Each rank uses value // world_size after validating divisibility.
    backward_bsz: int = 32
    # ``None`` means inherit ``backward_bsz``. An explicit final-layer
    # override changes only chunking, not the 32-sample optimizer step.
    final_layer_backward_bsz: Optional[int] = None
    # Per-element |delta| clip applied to the refresh-loss delta
    # (= q_out - fp_out) before the fisher quadratic. ``a_loss_ratio`` is
    # the kept-fraction quantile: ratio < 1 caps the top (1 - ratio)
    # fraction of |delta| at its quantile threshold to stop a few outlier
    # tokens from dominating the gradient. Old GPTQ+ ``--a_loss_ratio``
    # (process_args.py:60); see ``_scale_delta_by_abs_quantile`` and
    # the ``_activation_clip_threshold`` torch.quantile/topk fallback.
    # Default 1.0 = disabled (delta passes through).
    a_loss_ratio: float = 1.0
    # The paper specifies P95 clipping but not the percentile population.
    # ``global_refresh`` computes one exact threshold over every selected
    # sample/token/channel in a refresh, across ranks and accumulation chunks.
    # ``local_backward_chunk`` preserves the historical paper-code behavior:
    # every rank/backward chunk computes its own threshold in-line, with no
    # prepass. It is the default because the paper tables predate the newer
    # partition-invariant implementation.
    a_loss_clip_scope: str = "local_backward_chunk"

    # ----- batch / memory -------------------------------------------------
    bsz: int = 64
    hessian_accum_bsz: int = 64

    # ----- FSDP single-stage (Stage 3) ------------------------------------
    # When True, wrap the model with FSDP2 for the precompute backward pass,
    # save the post-rotate weights to a temp checkpoint, then drop FSDP and
    # reload from the checkpoint on CPU master so per-layer quant streams
    # one block at a time to GPU. Default cpu_master ⇒ master copy on CPU
    # for the quant phase. Set to False for single-GPU / small-model runs.
    fsdp: bool = False
    fsdp_cpu_offload: bool = False         # forwarded to CPUOffloadPolicy
    fsdp_max_shard_size: str = "5GB"       # for save_pretrained sharding
    fsdp_prepared_dir: Optional[str] = None  # cached checkpoint between phases
    # Opt-in switch for the rank0-CPU-master path. When True the pipeline
    # goes through Phase A/B/D (rank0-only rotate cache + meta init +
    # sharded broadcast load + asymmetric rebuild) and the per-node CPU
    # peak is ~1×M instead of N×M — necessary for large models on
    # CPU-constrained hosts. When False the model is replicated on every
    # rank (legacy fsdp=True path), which has faster setup, no broadcast
    # overhead per quant block, and supports aware AKV. The two paths are
    # bit-exact equivalent (verified on Qwen3-0.6B, 2-rank, with and
    # without block_gd); the choice is purely a memory-vs-overhead
    # trade-off and BOTH are long-term supported. See
    # realq/TODO_CPU_MASTER.md for the cpu_master path's current
    # functional restrictions (e.g. no aware AKV).
    cpu_master: bool = False

    # ----- AKV quantisation (Stage 2) -------------------------------------
    # A applies to the linear's INPUT (pre-matmul activation).
    # V applies to v_proj's OUTPUT (the value cached for attention).
    # K applies to the K projection AFTER RoPE rotation.
    # All default to fp16 (no quant). Set bits<16 to enable.
    a_bits: int = 16
    a_groupsize: int = -1
    a_asym: bool = False
    # ``None`` selects the paper preset conditionally: 0.9 only when the
    # corresponding tensor is actually quantised, otherwise 1.0.  This keeps
    # W4A16 configuration/log/cache identities free of meaningless A/K/V clip
    # changes while making W*x*A4KV4 use the documented clipping by default.
    a_clip_ratio: Optional[float] = None
    v_bits: int = 16
    v_groupsize: int = -1
    v_asym: bool = False
    v_clip_ratio: Optional[float] = None
    k_bits: int = 16
    k_groupsize: int = -1
    k_asym: bool = False
    k_clip_ratio: Optional[float] = None
    # The first flag makes A/V quantisers fire during the weight-Hessian
    # student forward; K-cache awareness is controlled independently by the
    # second flag. When false, the corresponding fake quantiser is installed
    # only after weight quantization for runtime use.
    act_quant_aware_gptq: bool = False
    k_cache_quant_aware_gptq: bool = False

    # ----- loss_slide_window (Stage 2) ------------------------------------
    loss_slide_window: bool = True  # raised by layer_loop; see REFACTOR_NOTES

    # ----- final layer override (Stage 2) ---------------------------------
    final_layer_grad_lr: Optional[float] = 0.00001
    kl_topk: int = -1  # <= 0 = full vocab (paper protocol)

    # ----- rotate (QuaRot) -------------------------------------------------
    rotate: bool = True
    optimized_rotation_path: Optional[str] = None

    # ----- eval -----------------------------------------------------------
    skip_eval: bool = False
    lm_eval: bool = False                # run QA tasks via lm-eval-harness
    lm_eval_batch_size: int = 8

    # ----- caching --------------------------------------------------------
    tokens_cache_path: str = "./cache/tokens"
    cache_dir: str = "./cache"

    # ----- debug ----------------------------------------------------------
    quant_stop_layer: Optional[int] = None
    # Optional synchronized wall/memory probe for one transformer block.
    # ``None`` is a strict no-op: the ordinary quantize_one_layer call remains
    # direct and performs no timing, CUDA-stat, barrier, or file operations.
    perf_measure_layer: Optional[int] = None
    # When True, emit NVTX ranges for every major pipeline phase so that
    # `nsys profile -t cuda,nvtx ...` traces can be inspected. Off by
    # default — when off, the wrapper is a nullcontext (no behavior change,
    # no synchronization, no allocation).
    nsys_profile: bool = False
    # ----- output ---------------------------------------------------------
    load_qmodel_path: Optional[str] = None
    allow_unsafe_legacy_checkpoint: bool = False
    save_qmodel_path: Optional[str] = None
    output_dir: str = "./output"
    exp: str = "realq"

    # ----- derived (auto-filled by __post_init__) -------------------------
    model_name: str = ""
    # Appended after every pre-existing field to preserve Config's positional
    # constructor ABI. Optional human-readable diagnostic: rank zero logs the
    # globally averaged objective actually sent to backward after every
    # refreshed column block, plus its learning-rate and loss-slide metadata.
    # The final block has no trailing weights and hence no backward objective.
    log_column_block_loss: bool = False

    # Performance knobs are appended after the pre-existing dataclass fields
    # so positional Config construction keeps its ABI. P01--P06/P10 use their
    # optimized implementations by default; every legacy implementation
    # remains selectable for rollback.
    # Prevalidate WeightQuantizer state and grouped natural-column coordinates
    # once per GPTQ block, then use the private exact-arithmetic primitive.
    quantizer_inner_fastpath: bool = True
    # Reduce finite symmetric weight-clip candidates from the historical
    # Cartesian scan to the exact endpoint union.
    w_clip_search_impl: str = "symmetric_union_exact"
    # Stage current/slide Fisher matrices as FP32 once per layer instead of
    # repeatedly expanding the persisted BF16 matrices in refresh losses.
    fisher_fp32_cache: bool = True
    # In rank-parallel act-order refreshes, rebuild the complete permuted
    # weight from the already-gathered quantized prefix and working suffix
    # instead of issuing a redundant full-weight all-gather.  Keep the
    # historical collective as the default until distributed CUDA exactness
    # and timing gates pass.
    act_order_stitch_impl: str = "prefix_q_trailing_w_exact"
    # Replace the per-candidate CUDA-tensor Python guard in the exact
    # Cartesian weight-clip search with fixed-shape ``torch.where(..., out=)``
    # updates.  The optimized implementation is restricted to the production
    # FP32/no-grad observer domain; every other input follows the historical
    # guarded implementation exactly.
    w_clip_update_impl: str = "where_out"
    # Store grouped weight scale/zero either once per natural column
    # (historical) or once per natural group. Compact is an opt-in exact
    # candidate until the CUDA E0--E4 gates are complete.
    w_group_param_layout: str = "compact"
    # Campaign consumers set this after a dedicated cache producer succeeds.
    # Appended to preserve Config's positional constructor ABI. A miss then
    # fails instead of silently recomputing Stage 0 concurrently.
    require_static_cache_hit: bool = False
    # The dedicated producer also materialises FP WikiText-2 reference
    # hidden states. Sweep consumers refuse to regenerate them concurrently.
    require_reference_cache_hit: bool = False
    # Optional exact calibration artifact.  Controlled comparisons use this
    # to make base and instruct checkpoints consume the identical saved token
    # tensor even though their derived ``model_name`` values differ.
    tokens_cache_file: Optional[str] = None
    # P10: evaluate the invariant tensor lower clamp bound once per prepared
    # P01 block rather than once per quantized column.
    prepared_clamp_bound_cache: bool = True
    # Fuse the sequential quantize + in-block GPTQ compensation loop into one
    # autotuned Triton program. The cross-block Err@Hinv compensation remains
    # a separate one-shot GEMM/BMM.
    triton_column_block: bool = True
    # Normal single-rank Block-GD fast path: consume the single backward's
    # low-precision gradient directly and fuse cast, clipping, Adam moment
    # updates, active-column indexing, and master/update writeback into one
    # streaming Triton kernel. Multi-rank/multi-chunk runs retain the exact
    # FP32 accumulation path before entering the same fused Adam update.
    fused_block_adam: bool = True

    # ----- generation-based inference evaluation -------------------------
    reasoning_eval: bool = False
    reasoning_tasks: list[str] = field(
        default_factory=lambda: [
            "gsm8k",
            "math_500",
            "humaneval_plus",
            "livecodebench_lite",
        ]
    )
    reasoning_data_dir: str = "./datasets/reasoning_eval"
    reasoning_output_dir: Optional[str] = None
    reasoning_batch_size: int = 1
    reasoning_limit: int = -1
    reasoning_max_new_tokens: int = 4096
    reasoning_num_samples: int = 1
    reasoning_apply_chat_template: bool = True
    reasoning_enable_thinking: bool = True
    reasoning_do_sample: bool = True
    reasoning_temperature: float = 0.6
    reasoning_top_p: float = 0.95
    reasoning_top_k: int = 20
    reasoning_seed: int = 1234
    reasoning_resume: bool = True
    reasoning_protocol: str = "realq_zero_shot_v1"
    reasoning_system_prompt: str = (
        "You are a careful reasoning assistant. Follow the requested output "
        "format exactly."
    )
    reasoning_lcb_release: str = "release_v6"
    reasoning_lcb_source_dir: str = (
        "./datasets/reasoning_eval/vendor/LiveCodeBench"
    )
    # Allows generation evaluation without first materializing WikiText KL/PPL
    # references. ``skip_eval`` still disables every evaluator.
    skip_kl_ppl_eval: bool = False

    # ----- Qwen3-MoE policy / lifecycle ----------------------------------
    moe_route_pack_impl: str = "stable_csr_reference"
    moe_expert_chunk_assignments: int = 4096
    moe_min_expert_assignments: int = 1
    moe_min_expert_unique_tokens: int = 1
    moe_min_expert_unique_samples: int = 1
    moe_fail_on_teacher_cold: bool = True
    moe_fail_on_student_cold: bool = True
    moe_zero_route_fallback: str = "rtn"
    moe_gpu_resident: bool = True
    moe_joint_column_block: bool = True
    moe_expert_loss_slide_window: bool = True
    # Optional CUDA-profiler capture window, using zero-based transformer
    # layer indices.  Appended to preserve Config's positional constructor
    # ABI.  When both endpoints are set, the runner calls cudaProfilerStart
    # immediately before the first layer and cudaProfilerStop immediately
    # after the last layer.  The nsys launcher must use
    # ``--capture-range=cudaProfilerApi``.  Layers outside the inclusive
    # window still execute normally but do not enter the trace.
    nsys_capture_start_layer: Optional[int] = None
    nsys_capture_end_layer: Optional[int] = None
    # ``flash_attention_4`` uses the pinned CuTe DSL FA4 package and is
    # intentionally limited to RealQ's dense, unpadded, causal calibration /
    # refresh path. ``sdpa`` preserves the historical Transformers behavior.
    attention_backend: str = "sdpa"
    # Use TensorFloat-32 multiply with FP32 accumulation/output for the
    # Hessian X^T diag(s) X matrix multiplication.  The persistent Hessian
    # buffer and every cross-batch ``H.add_`` accumulation remain FP32.
    # Appended to preserve Config's positional constructor ABI.
    hessian_tf32: bool = True
    # REAL-Q Plus extends every refresh from the current linear's unquantized
    # suffix to all not-yet-quantized linears in the current Transformer
    # block (and, for an active sliding arm, the next block).  Appended to
    # preserve Config's positional constructor ABI.  The default is the
    # paper/original single-linear refresh scope; REAL-Q Plus campaigns must
    # opt into the expanded Transformer-block update explicitly.
    full_block_refresh: bool = False

    def __post_init__(self) -> None:
        if not self.model_name:
            # Mirror the old process_args convention: model_name = basename
            # of the model path, used as a cache-key fragment by eval_utils.
            self.model_name = os.path.basename(self.model.rstrip("/")) or "model"
        if self.attention_backend not in ("sdpa", "flash_attention_4"):
            raise ValueError(
                "`attention_backend` must be 'sdpa' or "
                f"'flash_attention_4'. Got {self.attention_backend!r}."
            )
        if type(self.hessian_tf32) is not bool:
            raise ValueError(
                "`hessian_tf32` must be bool. Got "
                f"{self.hessian_tf32!r}."
            )
        if type(self.full_block_refresh) is not bool:
            raise ValueError(
                "`full_block_refresh` must be bool. Got "
                f"{self.full_block_refresh!r}."
            )
        if type(self.require_static_cache_hit) is not bool:
            raise ValueError(
                "`require_static_cache_hit` must be bool. Got "
                f"{self.require_static_cache_hit!r}."
            )
        if self.require_static_cache_hit and not self.static_cache_path:
            raise ValueError(
                "`require_static_cache_hit=True` requires "
                "`static_cache_path`."
            )
        if type(self.require_reference_cache_hit) is not bool:
            raise ValueError(
                "`require_reference_cache_hit` must be bool. Got "
                f"{self.require_reference_cache_hit!r}."
            )
        capture_start = self.nsys_capture_start_layer
        capture_end = self.nsys_capture_end_layer
        if (capture_start is None) != (capture_end is None):
            raise ValueError(
                "`nsys_capture_start_layer` and "
                "`nsys_capture_end_layer` must be set together."
            )
        if capture_start is not None:
            if capture_start < 0 or capture_end < 0:
                raise ValueError("nsys capture layer indices must be non-negative.")
            if capture_start > capture_end:
                raise ValueError(
                    "`nsys_capture_start_layer` must be <= "
                    "`nsys_capture_end_layer`."
                )
            if not self.nsys_profile:
                raise ValueError(
                    "layer-scoped nsys capture requires `nsys_profile=True`."
                )
            if (
                self.quant_stop_layer is not None
                and capture_end > self.quant_stop_layer
            ):
                raise ValueError(
                    "`nsys_capture_end_layer` must be <= `quant_stop_layer` "
                    "when an early stop is configured."
                )
        if not self.reasoning_tasks:
            raise ValueError("`reasoning_tasks` must not be empty.")
        if self.reasoning_batch_size <= 0:
            raise ValueError("`reasoning_batch_size` must be positive.")
        if self.reasoning_limit == 0 or self.reasoning_limit < -1:
            raise ValueError(
                "`reasoning_limit` must be -1 or a positive integer."
            )
        if self.reasoning_max_new_tokens <= 0:
            raise ValueError("`reasoning_max_new_tokens` must be positive.")
        if self.reasoning_num_samples <= 0:
            raise ValueError("`reasoning_num_samples` must be positive.")
        if self.reasoning_protocol != "realq_zero_shot_v1":
            raise ValueError(
                "`reasoning_protocol` must be 'realq_zero_shot_v1'."
            )
        if self.reasoning_do_sample and self.reasoning_temperature <= 0:
            raise ValueError(
                "`reasoning_temperature` must be positive when sampling."
            )
        if not 0.0 < self.reasoning_top_p <= 1.0:
            raise ValueError("`reasoning_top_p` must be in (0, 1].")
        if self.reasoning_top_k < 0:
            raise ValueError("`reasoning_top_k` must be non-negative.")
        if self.reasoning_lcb_release == "release_latest":
            raise ValueError(
                "`reasoning_lcb_release=release_latest` is not reproducible; "
                "pin an explicit LiveCodeBench release."
            )
        if type(self.skip_kl_ppl_eval) is not bool:
            raise ValueError("`skip_kl_ppl_eval` must be bool.")
        if not (0.0 < self.a_loss_ratio <= 1.0):
            raise ValueError(
                f"`a_loss_ratio` must be in (0, 1]. Got {self.a_loss_ratio}."
            )
        if self.a_loss_clip_scope not in (
            "global_refresh",
            "local_backward_chunk",
        ):
            raise ValueError(
                "`a_loss_clip_scope` must be 'global_refresh' or "
                "'local_backward_chunk'. Got "
                f"{self.a_loss_clip_scope!r}."
            )
        if (
            not isinstance(self.w_bits, int)
            or isinstance(self.w_bits, bool)
            or not 2 <= self.w_bits <= 16
        ):
            raise ValueError(
                "`w_bits` must be an integer in [2, 16], where 16 disables "
                f"weight quantization. Got {self.w_bits!r}."
            )
        if (
            not isinstance(self.w_groupsize, int)
            or isinstance(self.w_groupsize, bool)
            or (self.w_groupsize != -1 and self.w_groupsize <= 0)
        ):
            raise ValueError(
                "`w_groupsize` must be -1 (per-row) or a positive integer. "
                f"Got {self.w_groupsize!r}."
            )
        if self.w_bits < 16:
            if self.w_asym:
                raise ValueError(
                    "`w_asym=True` is unsupported: REAL-Q's weight fake-quant "
                    "path does not preserve asymmetric zero-points."
                )
            if self.w_groupsize != -1:
                if self.blocksize % self.w_groupsize != 0:
                    raise ValueError(
                        "`blocksize` must be an integer multiple of "
                        "`w_groupsize` for grouped weight quantization. "
                        f"Got w_groupsize={self.w_groupsize}, "
                        f"blocksize={self.blocksize}."
                    )
                if (
                    not self.act_order
                    and self.w_groupsize != self.blocksize
                ):
                    raise ValueError(
                        "`w_groupsize != blocksize` is supported only with "
                        "`act_order=True`, where natural-column group "
                        "parameters are observed statically before the "
                        "permutation. Dynamic non-act-order groups still "
                        "require equality to preserve legacy observer timing. "
                        f"Got w_groupsize={self.w_groupsize}, "
                        f"blocksize={self.blocksize}."
                    )
        if self.w_clip_search_impl not in (
            "cartesian_legacy",
            "symmetric_union_exact",
        ):
            raise ValueError(
                "`w_clip_search_impl` must be 'cartesian_legacy' or "
                "'symmetric_union_exact'. Got "
                f"{self.w_clip_search_impl!r}."
            )
        if self.act_order_stitch_impl not in (
            "full_weight_legacy",
            "prefix_q_trailing_w_exact",
        ):
            raise ValueError(
                "`act_order_stitch_impl` must be 'full_weight_legacy' or "
                "'prefix_q_trailing_w_exact'. Got "
                f"{self.act_order_stitch_impl!r}."
            )
        if self.w_clip_update_impl not in ("guarded", "where_out"):
            raise ValueError(
                "`w_clip_update_impl` must be 'guarded' or 'where_out'. Got "
                f"{self.w_clip_update_impl!r}."
            )
        if self.w_group_param_layout not in ("expanded", "compact"):
            raise ValueError(
                "`w_group_param_layout` must be 'expanded' or 'compact'. Got "
                f"{self.w_group_param_layout!r}."
            )
        if self.num_groups <= 0:
            raise ValueError(
                f"`num_groups` must be positive. Got {self.num_groups}."
            )
        if self.blocksize <= 0:
            raise ValueError(
                f"`blocksize` must be positive. Got {self.blocksize}."
            )
        if self.percdamp <= 0:
            raise ValueError(
                f"`percdamp` must be positive. Got {self.percdamp}."
            )
        if self.group_parallel_quant not in ("none", "rank"):
            raise ValueError(
                "`group_parallel_quant` must be 'none' or 'rank'. Got "
                f"{self.group_parallel_quant!r}."
            )
        if type(self.log_column_block_loss) is not bool:
            raise ValueError(
                "`log_column_block_loss` must be bool. Got "
                f"{self.log_column_block_loss!r}."
            )
        if type(self.quantizer_inner_fastpath) is not bool:
            raise ValueError(
                "`quantizer_inner_fastpath` must be bool. Got "
                f"{self.quantizer_inner_fastpath!r}."
            )
        if type(self.prepared_clamp_bound_cache) is not bool:
            raise ValueError(
                "`prepared_clamp_bound_cache` must be bool. Got "
                f"{self.prepared_clamp_bound_cache!r}."
            )
        if (
            self.prepared_clamp_bound_cache
            and not self.quantizer_inner_fastpath
        ):
            raise ValueError(
                "`prepared_clamp_bound_cache=True` requires "
                "`quantizer_inner_fastpath=True`."
            )
        if type(self.triton_column_block) is not bool:
            raise ValueError(
                "`triton_column_block` must be bool. Got "
                f"{self.triton_column_block!r}."
            )
        if type(self.fisher_fp32_cache) is not bool:
            raise ValueError(
                "`fisher_fp32_cache` must be bool. Got "
                f"{self.fisher_fp32_cache!r}."
            )
        if self.moe_route_pack_impl != "stable_csr_reference":
            raise ValueError(
                "`moe_route_pack_impl` currently supports only "
                f"'stable_csr_reference'; got {self.moe_route_pack_impl!r}."
            )
        if (
            not isinstance(self.moe_expert_chunk_assignments, int)
            or isinstance(self.moe_expert_chunk_assignments, bool)
            or self.moe_expert_chunk_assignments <= 0
        ):
            raise ValueError(
                "`moe_expert_chunk_assignments` must be a positive integer; "
                f"got {self.moe_expert_chunk_assignments!r}."
            )
        for threshold_name in (
            "moe_min_expert_assignments",
            "moe_min_expert_unique_tokens",
            "moe_min_expert_unique_samples",
        ):
            threshold = getattr(self, threshold_name)
            if (
                not isinstance(threshold, int)
                or isinstance(threshold, bool)
                or threshold <= 0
            ):
                raise ValueError(
                    f"`{threshold_name}` must be a positive integer; "
                    f"got {threshold!r}."
                )
        for policy_name in (
            "moe_fail_on_teacher_cold",
            "moe_fail_on_student_cold",
            "moe_gpu_resident",
            "moe_joint_column_block",
            "moe_expert_loss_slide_window",
        ):
            if type(getattr(self, policy_name)) is not bool:
                raise ValueError(
                    f"`{policy_name}` must be bool; "
                    f"got {getattr(self, policy_name)!r}."
                )
        if not self.moe_fail_on_teacher_cold:
            raise ValueError(
                "The accepted Qwen3-MoE policy requires "
                "`moe_fail_on_teacher_cold=True`."
            )
        if not self.moe_fail_on_student_cold:
            raise ValueError(
                "The accepted Qwen3-MoE policy requires "
                "`moe_fail_on_student_cold=True`."
            )
        if self.moe_zero_route_fallback != "rtn":
            raise ValueError(
                "The accepted Qwen3-MoE zero-route policy requires "
                "`moe_zero_route_fallback='rtn'`; got "
                f"{self.moe_zero_route_fallback!r}."
            )
        if not self.moe_gpu_resident:
            raise ValueError(
                "The accepted Qwen3-MoE policy requires "
                "`moe_gpu_resident=True`."
            )
        if not self.moe_joint_column_block:
            raise ValueError(
                "The accepted Qwen3-MoE policy requires "
                "`moe_joint_column_block=True`."
            )
        if not self.moe_expert_loss_slide_window:
            raise ValueError(
                "The accepted Qwen3-MoE joint policy requires "
                "`moe_expert_loss_slide_window=True`."
            )
        if not (0.0 < self.saliency_clip_percentile <= 1.0):
            raise ValueError(
                "`saliency_clip_percentile` must be in (0, 1]. Got "
                f"{self.saliency_clip_percentile}."
            )
        for ratio_name, bits in (
            ("a_clip_ratio", self.a_bits),
            ("k_clip_ratio", self.k_bits),
            ("v_clip_ratio", self.v_bits),
        ):
            ratio = getattr(self, ratio_name)
            if ratio is None:
                ratio = 0.9 if bits < 16 else 1.0
                setattr(self, ratio_name, ratio)
            if not (0.0 < float(ratio) <= 1.0):
                raise ValueError(
                    f"`{ratio_name}` must be in (0, 1]. Got {ratio}."
                )
        for tensor_name in ("a", "k", "v"):
            bits = getattr(self, f"{tensor_name}_bits")
            groupsize = getattr(self, f"{tensor_name}_groupsize")
            if not isinstance(bits, int) or isinstance(bits, bool) or not (
                2 <= bits <= 16
            ):
                raise ValueError(
                    f"`{tensor_name}_bits` must be an integer in [2, 16], "
                    f"where 16 disables fake quantization. Got {bits!r}."
                )
            if (
                not isinstance(groupsize, int)
                or isinstance(groupsize, bool)
                or (groupsize != -1 and groupsize <= 0)
            ):
                raise ValueError(
                    f"`{tensor_name}_groupsize` must be -1 (per-token) or a "
                    f"positive integer. Got {groupsize!r}."
                )
        if self.k_cache_quant_aware_gptq and self.k_bits >= 16:
            raise ValueError(
                "`k_cache_quant_aware_gptq=True` requires `k_bits < 16`; "
                "an aware K-cache path without K fake quantization is invalid."
            )
        if self.grad_clip == 0:
            raise ValueError(
                "`grad_clip` must be non-zero; use a negative value to disable clipping."
            )
        if self.final_layer_grad_clip == 0:
            raise ValueError(
                "`final_layer_grad_clip` must be non-zero when provided; "
                "use a negative value to disable clipping."
            )
        if self.grad_lr_layer_schedule not in ("none", "cosine"):
            raise ValueError(
                "`grad_lr_layer_schedule` must be 'none' or 'cosine'. "
                f"Got {self.grad_lr_layer_schedule!r}."
            )
        if self.backward_samples <= 0:
            raise ValueError(
                f"`backward_samples` must be positive. Got {self.backward_samples}."
            )
        if self.backward_bsz <= 0:
            raise ValueError(
                f"`backward_bsz` must be positive. Got {self.backward_bsz}."
            )
        if self.final_layer_backward_bsz is None:
            self.final_layer_backward_bsz = self.backward_bsz
        if self.final_layer_backward_bsz <= 0:
            raise ValueError(
                "`final_layer_backward_bsz` must be positive. Got "
                f"{self.final_layer_backward_bsz}."
            )
        if type(self.fused_block_adam) is not bool:
            raise ValueError(
                "`fused_block_adam` must be bool. Got "
                f"{self.fused_block_adam!r}."
            )
        if self.cpu_master and not self.fsdp:
            raise ValueError("`cpu_master=True` requires `fsdp=True`.")
        if self.cpu_master and self.load_qmodel_path:
            raise ValueError(
                "`cpu_master=True` does not support `load_qmodel_path`; load "
                "the artifact with the ordinary replicated runtime path."
            )
        if self.cpu_master and (
            self.act_quant_aware_gptq or self.k_cache_quant_aware_gptq
        ):
            raise ValueError(
                "`cpu_master=True` does not support aware AKV "
                f"(act_quant_aware_gptq={self.act_quant_aware_gptq}, "
                f"k_cache_quant_aware_gptq={self.k_cache_quant_aware_gptq}). "
                "See realq/TODO_CPU_MASTER.md."
            )
        if (
            self.perf_measure_layer is not None
            and (
                not isinstance(self.perf_measure_layer, int)
                or isinstance(self.perf_measure_layer, bool)
                or self.perf_measure_layer < 0
            )
        ):
            raise ValueError(
                "`perf_measure_layer` must be None or a non-negative integer. "
                f"Got {self.perf_measure_layer!r}."
            )

    def validate_moe_runtime_contract(self, *, sparse_moe: bool) -> None:
        """Fail closed on unsupported offload/distributed sparse runtimes."""

        if not sparse_moe:
            return
        if not self.loss_slide_window:
            raise ValueError(
                "Sparse REAL-Q MoE full-slide refresh requires "
                "`loss_slide_window=True`."
            )
        incompatible = []
        if self.cpu_master:
            incompatible.append("cpu_master=True")
        if self.fsdp_cpu_offload:
            incompatible.append("fsdp_cpu_offload=True")
        if self.fsdp:
            incompatible.append("fsdp=True (post-Stage-0 reload_on_cpu)")
        if incompatible:
            raise ValueError(
                "Sparse REAL-Q MoE formal runs are fully GPU-resident and "
                "reject " + ", ".join(incompatible) + "."
            )

    @property
    def activation_aware_quantization_enabled(self) -> bool:
        """Whether an aware flag activates a real low-bit A/V/K path.

        Merely passing an aware flag alongside A16/V16/K16 must not alter the
        learning-rate schedule.
        """
        return (
            self.act_quant_aware_gptq
            and (self.a_bits < 16 or self.v_bits < 16)
        ) or (
            self.k_cache_quant_aware_gptq and self.k_bits < 16
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_BOOL_TRUE = {"1", "true", "yes", "y", "t"}
_BOOL_FALSE = {"0", "false", "no", "n", "f"}


def _str2bool(v: str) -> bool:
    s = v.lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {v!r}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RealQ post-training quantisation")
    defaults = Config()
    for f in fields(Config):
        if f.name == "model_name":
            continue  # derived
        default = getattr(defaults, f.name)
        flag = f"--{f.name}"
        if f.name in {"a_clip_ratio", "k_clip_ratio", "v_clip_ratio"}:
            # Preserve the omitted-vs-explicit distinction so Config can apply
            # its conditional paper preset after all bit-widths are known.
            p.add_argument(flag, type=float, default=None)
        elif f.name == "final_layer_grad_clip":
            p.add_argument(flag, type=float, default=None)
        elif f.name == "allow_unsafe_legacy_checkpoint":
            p.add_argument(flag, action="store_true", default=False)
        elif f.type is bool or isinstance(default, bool):
            p.add_argument(flag, type=_str2bool, default=default)
        elif f.name in {"eval_datasets", "reasoning_tasks"}:
            p.add_argument(flag, type=str, nargs="+", default=default)
        elif default is None:
            # Optional[str] / Optional[int]: leave default None, accept str
            # — the dataclass field type carries the runtime intent.
            p.add_argument(flag, type=str, default=None)
        elif isinstance(default, int) and not isinstance(default, bool):
            p.add_argument(flag, type=int, default=default)
        elif isinstance(default, float):
            p.add_argument(flag, type=float, default=default)
        else:
            p.add_argument(flag, type=str, default=default)
    return p


def parse_cli(argv: list[str] | None = None) -> Config:
    """Parse argv into a Config. Optional fields stay None when omitted."""
    ns = _build_parser().parse_args(argv)
    raw = vars(ns)

    # Re-cast Optional[int] CLI strings back to int.
    for f in fields(Config):
        if f.name == "model_name":
            continue
        v = raw.get(f.name)
        if v is None:
            continue
        # Optional[int] fields are accepted as strings by the generic parser.
        if f.name in {
            "quant_stop_layer",
            "final_layer_backward_bsz",
            "perf_measure_layer",
            "nsys_capture_start_layer",
            "nsys_capture_end_layer",
        } and isinstance(v, str):
            raw[f.name] = int(v)
    return Config(**raw)
