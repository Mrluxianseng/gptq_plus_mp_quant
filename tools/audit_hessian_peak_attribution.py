"""Disambiguate solver intervals from work after an after-inverse marker."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit(run, layer):
    status = json.loads((run / "status.json").read_text())
    before = f"layer_{layer}_mlp_down_proj_before_hessian_inverse"
    after = f"layer_{layer}_mlp_down_proj_after_hessian_inverse"
    rows = []
    for path in status["memory_profile_files"]:
        if "stage2" not in path or "cuda_memory_quantization" not in path:
            continue
        profile = ROOT / path
        for line in profile.read_text().splitlines():
            row = json.loads(line)
            if row.get("phase") in (before, after):
                rows.append({key: row[key] for key in (
                    "phase", "next_phase", "peak_measurement_mode", "elapsed_seconds",
                    "start_allocated_bytes", "sampled_peak_allocated_bytes", "end_allocated_bytes",
                    "sampled_peak_reserved_bytes", "metadata")})
    solver = [r for r in rows if r["phase"] == before and r["next_phase"] == after]
    following = [r for r in rows if r["phase"] == after]
    if len(solver) != 4 or len(following) != 4:
        raise RuntimeError("expected actual four-group last-layer inverse intervals")
    if any(r["peak_measurement_mode"] != "exact" for r in rows):
        raise RuntimeError("attribution requires exact allocator intervals")
    return dict(run=str(run), layer=layer, intervals=rows,
        solver_peak_allocated_bytes=max(r["sampled_peak_allocated_bytes"] for r in solver),
        post_solver_peak_allocated_bytes=max(r["sampled_peak_allocated_bytes"] for r in following),
        full_process_peak_allocated_bytes=status["allocator_peak_allocated_bytes_any_stage"],
        interpretation="before_hessian_inverse -> after_hessian_inverse covers the solver. "
                       "after_hessian_inverse -> activation_replay_and_cleanup includes remaining "
                       "module quantization and refresh work, not solely inverse computation.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--layer", type=int, default=27)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = dict(control=audit(args.control, args.layer), candidate=audit(args.candidate, args.layer),
        scope="Read-only attribution audit; no new optimization or benchmark.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    for side in ("control", "candidate"):
        row = result[side]
        print(side, "solver_MiB=", row["solver_peak_allocated_bytes"] / 1024**2,
              "post_solver_MiB=", row["post_solver_peak_allocated_bytes"] / 1024**2)


if __name__ == "__main__":
    main()
