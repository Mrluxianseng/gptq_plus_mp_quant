"""Write portable, explicitly scoped REAL-Q memory A/B manifests.

Paper preset aligns disclosed W4A16 resource/optimizer/rotation settings, but
does not claim an exact reproduction of unpublished clipping/draw details.
No experiment is started by this preparation command.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("pilot", "paper"), default="pilot")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--group-parallel", choices=("none", "rank"), default="none")
    parser.add_argument("--activation-deduplicate", action="store_true", help="Candidate-only sharing of identical saved CPU activation copies")
    parser.add_argument("--candidate-hinv-downproj-offload", action="store_true", help="Candidate-only per-block CPU staging of all MLP down_proj inverse factors; requires group-parallel=none")
    parser.add_argument("--nsamples", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--eval-samples", type=int, help="0 evaluates the full WikiText-2 test split; default: pilot=2, paper=0")
    parser.add_argument("--eval-seq-len", type=int, help="default: pilot=1024, paper=2048")
    args = parser.parse_args()
    model = args.model_path.resolve()
    config = json.loads((model / "config.json").read_text())
    layers = int(config["num_hidden_layers"])
    if args.world_size <= 0:
        parser.error("world-size must be positive")
    if args.candidate_hinv_downproj_offload and args.group_parallel != "none":
        parser.error("candidate Hinv staging currently requires group-parallel=none")
    paper = args.preset == "paper"
    if args.eval_samples is None:
        args.eval_samples = 0 if paper else 2
    if args.eval_seq_len is None:
        args.eval_seq_len = 2048 if paper else 1024
    samples = args.nsamples if args.nsamples is not None else (2048 if paper else 2)
    length = args.seq_len if args.seq_len is not None else (2048 if paper else 1024)
    if min(samples, length, args.eval_seq_len) <= 0 or args.eval_samples < 0 or samples % args.world_size:
        parser.error("use positive sizes, nonnegative eval-samples, and nsamples divisible by world-size")
    if paper and samples < 32:
        parser.error("paper preset requires at least 32 global gradient/calibration samples; use pilot for smaller runs")
    if paper:
        hidden = int(config["hidden_size"])
        rates = {1024: "3e-4", 2048: "5e-4"}
        if hidden not in rates or layers != 28 or config.get("model_type") != "qwen3":
            parser.error("paper preset currently specifies Table-6 LRs only for Qwen3-0.6B/1.7B")
        lr, final_lr, seed = rates[hidden], "1e-5", "1"
    else:
        lr, final_lr, seed = "5e-5", "1e-6", "44"
    gradient_samples = 32 if paper else min(16, samples)
    if gradient_samples % args.world_size:
        parser.error("global gradient samples must divide evenly across world-size")
    # This is one process launch; rendezvous uses a free port rather than a
    # hard-coded value left over from a prior experiment.
    common = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
              f"--nproc_per_node={args.world_size}", str(ROOT / "tools/realq_memory_profile_entry.py"),
              "baseline-profile", "--exp", "memory-manifest-template", "--model", str(model),
              "--dataset", "wikitext2", "--nsamples", str(samples), "--seq_len", str(length),
              "--w_method", "gptq_plus", "--w_bits", "4", "--a_bits", "16", "--k_bits", "16", "--v_bits", "16", "--w_groupsize", "-1", "--w_clip", "--percdamp", "0.01",
              "--num_groups", "4", "--blocksize", "128", "--act_order", "--kl_topk", "-1",
              "--bsz", str(args.world_size), "--final_layer_stats_bsz", str(args.world_size), "--alpha", "0", "--enable_gptq_plus", "0",
              "--backward_samples", str(gradient_samples), "--backward_bsz", str(args.world_size), "--final_layer_backward_bsz", str(args.world_size),
              "--g_update_mode", "block_gd", "--grad_lr", lr, "--final_layer_grad_lr", final_lr,
              "--grad_optimizer", "adam", "--grad_refresh_loss", "fisher_diag_mse", "--global_loss",
              "--global_loss_bsz", str(args.world_size), "--grad_clip", "5e-5", "--final_layer_grad_clip", "5e-4",
              "--group_parallel_quant", args.group_parallel, "--eval_seq_len", str(args.eval_seq_len),
              "--eval_datasets", "wikitext2", "--eval_samples", str(args.eval_samples), "--eval_start_sample", "0",
              "--seed", seed, "--gptq_inner_kernel", "torch", "--kl_logit_chunk_tokens", "16",
              "--static_fisher_label_chunk_tokens", "16", "--enable_quant_profile",
              "--quant_profile_target_layers", ",".join(map(str, range(layers))), "--quant_stop_layer", str(layers-1),
              "--pre_clip_search_impl", "cartesian_legacy", "--quantizer_search_impl", "cartesian_legacy",
              "--eval_logit_chunk_tokens", "16"]
    if paper:
        common += ["--rotate", "--a_loss_ratio", "0.95"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    description = {
        "preset": args.preset,
        "paper_resource_scale": paper and samples == 2048 and length == 2048,
        "exact_paper_table_reproduction_claimed": False,
        "explicit_shared_conditions": "FSDP CPU-offload/meta-init; static categorical label chunk16; KL chunk16; microbatch1; shared grad clips; per-row W4A16; full-vocabulary final KL",
        "unpublished_conditions_and_baseline_scope": "See docs/REALQ_PAPER_PROTOCOL.md; drawing/chunking and clip details are explicit shared A/B conditions, not an exact table reconstruction.",
        "eval_samples": args.eval_samples, "eval_seq_len": args.eval_seq_len,
        "full_test_split_evaluation": args.eval_samples == 0,
        "multi_gpu_local_validation_pending": args.world_size > 1,
    }
    for side in ("control", "candidate"):
        stage0 = common + ["--fsdp_precompute", "--fsdp_cpu_offload", "--fsdp_meta_init", "--exit_after_precompute", "--skip_eval"]
        if side == "candidate":
            stage0 += ["--static_saved_tensor_offload_mib", "4"]
            if args.activation_deduplicate:
                stage0 += ["--static_saved_tensor_offload_deduplicate"]
        for stage, command in (("stage0", stage0), ("stage2", common.copy())):
            if stage == "stage2" and side == "candidate" and args.candidate_hinv_downproj_offload:
                command += ["--stage2_hinv_cpu_offload_modules", ",".join(f"{layer}.mlp.down_proj" for layer in range(layers))]
            (args.output_dir / f"{side}-{stage}.json").write_text(json.dumps({**description, "command": command},indent=2))
    (args.output_dir / "protocol.json").write_text(json.dumps({**description, "nsamples": samples, "seq_len": length, "gradient_samples": gradient_samples, "world_size": args.world_size, "group_parallel": args.group_parallel, "grad_lr": lr, "final_layer_grad_lr": final_lr, "calibration_seed": int(seed), "rotation": paper, "candidate_activation_deduplicate": args.activation_deduplicate, "candidate_hinv_downproj_offload": args.candidate_hinv_downproj_offload},indent=2))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
