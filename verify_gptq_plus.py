"""
Verification harness for GPTQ+ optimization work.

Runs the same quantization flow as scripts/quant_profile_quick.sh (via
analyze_quant_profile.build_parser) and either (a) saves the final quantized
weights as a golden baseline, or (b) reloads a saved baseline and compares it
to the current run.

DP support: launch via `torchrun --nproc_per_node=N` to run multi-GPU DP.
All ranks execute quantize_weights; only rank 0 handles the save/check I/O
and the final pass/fail report.

Usage:
    # save baseline before optimizing (single GPU)
    python verify_gptq_plus.py --verify_mode save \
        --verify_output ./outputs/verify/baseline.pt [<analyze args...>]

    # DP run (2 GPUs)
    torchrun --nproc_per_node=2 verify_gptq_plus.py --verify_mode save \
        --verify_output ./outputs/verify/baseline_2gpu.pt [<analyze args...>]
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import logging
import random
import sys

import numpy as np
import torch

from analyze_quant_profile import build_parser
from gptq_utils.main import quantize_weights
from utils import memory_utils, dist_utils
from utils.log_utils import init_logging
from utils.model_utils import ModelAnalyzer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Determinism: kernel selection + cuBLAS workspace.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    # Disable SDPA fast paths that may pick non-deterministic kernels.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except AttributeError:
        pass


def collect_quantized_tensors(model) -> dict:
    out = {}
    for name, module in model.named_modules():
        if not name.startswith("model.layers."):
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            out[f"{name}.weight"] = weight.data.detach().cpu().clone()
        for attr in ("scale", "int_weight"):
            maybe = getattr(module, attr, None)
            if isinstance(maybe, torch.Tensor):
                out[f"{name}.{attr}"] = maybe.detach().cpu().clone()
    return out


def compare_tensors(golden: dict, current: dict, atol: float, rtol: float) -> bool:
    ok = True
    missing = sorted(set(golden) - set(current))
    extra = sorted(set(current) - set(golden))
    if missing:
        ok = False
        print(f"[verify] keys only in golden ({len(missing)}): {missing[:5]}...")
    if extra:
        ok = False
        print(f"[verify] keys only in current ({len(extra)}): {extra[:5]}...")

    print(f"[verify] comparing {len(set(golden) & set(current))} tensors atol={atol} rtol={rtol}")
    worst = []
    mismatches = 0
    for key in sorted(set(golden) & set(current)):
        a = golden[key].float()
        b = current[key].float()
        if a.shape != b.shape:
            print(f"[verify] SHAPE  {key}: golden={a.shape} current={b.shape}")
            ok = False
            continue
        diff = (a - b).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        ref = a.abs().max().item()
        denom = a.abs().clamp(min=1e-12)
        max_rel = (diff / denom).max().item()
        matches = torch.allclose(a, b, atol=atol, rtol=rtol)
        if not matches:
            ok = False
            mismatches += 1
        worst.append((max_abs, max_rel, mean_abs, ref, matches, key))

    worst.sort(reverse=True)
    print(f"[verify] mismatches: {mismatches}; top 10 by max_abs:")
    for max_abs, max_rel, mean_abs, ref, matches, key in worst[:10]:
        tag = "OK  " if matches else "FAIL"
        print(
            f"  [{tag}] {key}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
            f"max_rel={max_rel:.3e} ref={ref:.3e}"
        )
    return ok


def finalize_args(args) -> None:
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
    args.enable_quant_profile = False
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
    args.enable_debug = getattr(args, "enable_debug", False)

    if args.fisher_num_groups is None:
        args.fisher_num_groups = args.num_groups
    if args.backward_samples == -1:
        args.backward_samples = args.nsamples
    if args.final_layer_stats_bsz is None:
        args.final_layer_stats_bsz = args.bsz
    if args.backward_bsz == -1:
        args.backward_bsz = args.bsz
    if args.final_layer_backward_bsz is None:
        args.final_layer_backward_bsz = args.backward_bsz
    if args.global_loss_bsz is None:
        args.global_loss_bsz = args.bsz


def _is_torchrun_launched() -> bool:
    """torchrun sets RANK/WORLD_SIZE/LOCAL_RANK env vars."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def main() -> None:
    parser = build_parser()
    parser.add_argument("--verify_mode", required=True, choices=["save", "check"])
    parser.add_argument("--verify_output", required=True, type=str)
    parser.add_argument("--verify_tol_abs", type=float, default=0.0)
    parser.add_argument("--verify_tol_rel", type=float, default=0.0)
    parser.add_argument("--enable_debug", action="store_true",
                        help="Enable cross-rank bit-exact assertions after each layer's fasterquant.")
    parser.add_argument(
        "--diagnose_targets",
        type=str,
        default=None,
        help="Comma-separated `<layer_idx>:<module_substring>` list (e.g. `2:mlp.down_proj,1:mlp.down_proj`).",
    )
    parser.add_argument("--diagnose_dir", type=str, default=None)
    parser.add_argument("--diagnose_spike_ratio", type=float, default=3.0)
    args = parser.parse_args()
    finalize_args(args)

    # DP init. When launched single-GPU without torchrun we skip init entirely
    # so the code paths stay as before (world_size=1 everywhere).
    if _is_torchrun_launched():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist_utils.init_process_group()

    if dist_utils.is_main():
        init_logging(args.log_dir)
        logging.info("[verify] mode=%s output=%s world_size=%d",
                     args.verify_mode, args.verify_output, dist_utils.get_world_size())
        logging.info(args)

    seed_everything(args.seed)

    analyzer = ModelAnalyzer(args.model, args.seq_len)
    analyzer.model.cpu()
    quantize_weights(args, analyzer)

    # After quantization all ranks' model.layer weights are identical (verified
    # via assert_bit_exact when --enable_debug is set). Only rank 0 needs to
    # do the save/check I/O.
    state = collect_quantized_tensors(analyzer.model)
    if not state:
        if dist_utils.is_main():
            print("[verify] no tensors collected — is quant_stop_layer set correctly?")
        dist_utils.barrier()
        sys.exit(1)

    if dist_utils.is_main():
        if args.verify_mode == "save":
            os.makedirs(os.path.dirname(args.verify_output) or ".", exist_ok=True)
            torch.save(state, args.verify_output)
            print(f"[verify] saved baseline: {args.verify_output} ({len(state)} tensors)")
            exit_code = 0
        else:
            if not os.path.exists(args.verify_output):
                print(f"[verify] golden file not found: {args.verify_output}")
                exit_code = 1
            else:
                golden = torch.load(args.verify_output, map_location="cpu")
                if compare_tensors(golden, state, args.verify_tol_abs, args.verify_tol_rel):
                    print("[verify] PASS — all tensors match within tolerance")
                    exit_code = 0
                else:
                    print("[verify] FAIL — tensor mismatch detected")
                    exit_code = 1
    else:
        exit_code = 0

    dist_utils.barrier()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_utils.cleanup_memory(False)

    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
