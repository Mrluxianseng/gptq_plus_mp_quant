"""
Diagnostic recorder for GPTQ+ block-by-block analysis.

Dumps full raw matrices (not just statistics) at every checkpoint inside
GPTQPlus.fasterquant, so we can post-hoc investigate why a particular block
loss spikes / why block_gd behaves pathologically on rotated down_proj, etc.

Enabled per (layer_idx, module_name_substring) via command-line flag, so one
run can simultaneously capture the suspect block and a baseline block for
side-by-side comparison (same seed, same data, different position).

Disk layout (per target):

    <root>/layer_<L>_<module_sanitized>/
        meta.json                               # summary + detected loss spikes
        module_level/
            original_weight_rotated.pt          # FP weight after rotate, before any quant
            quantizer_scale.pt
            quantizer_maxq.pt
            H.pt                                 # (G, d, d) Hessian after finalize
            act_square.pt
            gradients.pt                         # (d_out, d_in)
            saliency_per_module_sample.pt        # optional, can be big
            reference_loss.pt                    # scalar, mean refresh loss pre-quant
        subgroup_<G>/
            Hinv_init.pt                         # inverse Hessian
            Hinv_upper_cholesky.pt
            beta.pt                              # GPTQ+ safe-beta per row
            Z.pt                                 # beta * gradients @ Hinv.T
            GHinv_init.pt                        # Z @ Hinv
            W_sub_initial.pt                     # slice of rotated W for this subgroup
            block_<B>/
                W1_start.pt                     # W_sub[:, i1:i2] before quant
                W_trailing_start.pt             # W_sub[:, i2:] before outer/adam
                Hinv1.pt                        # block-local inverse
                Q1.pt                           # quantized block
                W_int1.pt                       # integer codes
                Scale1.pt                       # per-column scale
                Err1.pt                         # quantization residual / diag_hinv
                W1_after_inner.pt               # W1 after inner-loop err propagation
                second_order_update.pt          # Err1 @ Hinv[i1:i2, i2:]
                G_Update.pt                     # count*GHinv_rest - einsum(...)
                total_outer_update.pt           # second_order + G_Update
                W_trailing_after_outer.pt       # W_sub[:, i2:] after outer update
                refreshed_grad_subgroup.pt      # per-subgroup slice of refreshed_grad
                mean_refresh_loss.pt            # scalar
                optimizer_update.pt             # Adam step
                W_trailing_after_adam.pt        # W_sub[:, i2:] after adam apply
                applied_second_order_update.pt  # second_order_update * second_order_scale

Reading: each .pt is a CPU fp32 torch.save; load with torch.load("path/to/*.pt").
"""
import json
import os
from typing import List, Optional, Tuple

import torch


def _sanitize(module_name: str) -> str:
    """Make a module name filesystem-friendly."""
    return module_name.replace(".", "_").replace("/", "_")


def parse_diagnose_targets(spec: str) -> List[Tuple[int, str]]:
    """Parse a CLI --diagnose_targets spec.

    Format: "layer_idx:module_substring,layer_idx:module_substring,..."
    Example: "2:mlp.down_proj,1:mlp.down_proj,2:self_attn.q_proj"
    Returns a list of (layer_idx, module_substring) tuples; empty list if spec is empty/None.
    """
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"diagnose_targets entry `{chunk}` must be of the form <layer_idx>:<module_substring>"
            )
        layer_str, module_sub = chunk.split(":", 1)
        out.append((int(layer_str.strip()), module_sub.strip()))
    return out


def target_matches(targets: List[Tuple[int, str]], layer_idx: int, module_name: str) -> bool:
    """Return True if the (layer_idx, module_name) pair matches any target spec."""
    for t_layer, t_module_sub in targets:
        if t_layer == layer_idx and t_module_sub in module_name:
            return True
    return False


class DiagnosticRecorder:
    """Per-module dumper; one instance per (layer, module) target.

    Save methods are no-ops when the recorder is None, so call sites can use
        rec and rec.save_block(sub_idx, block_idx, tag, tensor)
    to conditionally dump.
    """

    def __init__(self, root_dir: str, layer_idx: int, module_name: str, spike_ratio: float = 3.0):
        module_safe = _sanitize(module_name)
        self.root = os.path.join(root_dir, f"layer_{layer_idx:02d}_{module_safe}")
        os.makedirs(self.root, exist_ok=True)
        self.layer_idx = layer_idx
        self.module_name = module_name
        self._block_losses: dict = {}  # (sub_idx, block_idx) -> loss
        self._block_meta: list = []    # list of dicts with extra per-block info
        self.spike_ratio = spike_ratio

    # -- internal --------------------------------------------------------
    def _save(self, path: str, tensor):
        """Save a tensor (or Python scalar) as a .pt file, always on CPU and fp32."""
        if tensor is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(tensor, torch.Tensor):
            t = tensor.detach().cpu()
            if t.dtype.is_floating_point and t.dtype != torch.float32:
                t = t.float()
            torch.save(t, path)
        else:
            torch.save(torch.tensor(tensor, dtype=torch.float64), path)

    # -- save helpers ----------------------------------------------------
    def save_module_level(self, tag: str, tensor):
        self._save(os.path.join(self.root, "module_level", f"{tag}.pt"), tensor)

    def save_subgroup(self, sub_idx: int, tag: str, tensor):
        self._save(
            os.path.join(self.root, f"subgroup_{sub_idx:02d}", f"{tag}.pt"),
            tensor,
        )

    def save_block(self, sub_idx: int, block_idx: int, tag: str, tensor):
        self._save(
            os.path.join(
                self.root,
                f"subgroup_{sub_idx:02d}",
                f"block_{block_idx:02d}",
                f"{tag}.pt",
            ),
            tensor,
        )

    def record_loss(self, sub_idx: int, block_idx: int, loss: float):
        self._block_losses[(sub_idx, block_idx)] = float(loss) if loss is not None else None

    def record_block_meta(self, entry: dict):
        """Arbitrary small key/value info, lands in meta.json."""
        self._block_meta.append(entry)

    # -- cleanup ---------------------------------------------------------
    def finalize(self):
        """Write a meta.json summarising what was captured, including detected spikes."""
        per_sub = {}
        for (sub, blk), loss in sorted(self._block_losses.items()):
            prev = self._block_losses.get((sub, blk - 1))
            is_spike = bool(
                loss is not None and prev is not None and prev > 0 and loss / prev > self.spike_ratio
            )
            per_sub.setdefault(str(sub), []).append(
                {
                    "block": int(blk),
                    "loss": None if loss is None else float(loss),
                    "prev_loss": None if prev is None else float(prev),
                    "is_spike": is_spike,
                }
            )
        meta = {
            "layer_idx": self.layer_idx,
            "module_name": self.module_name,
            "spike_ratio_threshold": self.spike_ratio,
            "per_subgroup_blocks": per_sub,
            "extra_block_meta": self._block_meta,
        }
        with open(os.path.join(self.root, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


class DiagnosticRegistry:
    """Routes (layer_idx, module_name) to the right recorder, or None."""

    def __init__(self, root_dir: Optional[str], targets: List[Tuple[int, str]], spike_ratio: float = 3.0):
        self.root_dir = root_dir
        self.targets = targets
        self.spike_ratio = spike_ratio
        self._recorders: dict = {}

    def get_or_create(self, layer_idx: int, module_name: str) -> Optional[DiagnosticRecorder]:
        if not self.root_dir or not self.targets:
            return None
        if not target_matches(self.targets, layer_idx, module_name):
            return None
        key = (layer_idx, module_name)
        if key not in self._recorders:
            self._recorders[key] = DiagnosticRecorder(
                self.root_dir, layer_idx, module_name, spike_ratio=self.spike_ratio
            )
        return self._recorders[key]

    def finalize_all(self):
        for rec in self._recorders.values():
            rec.finalize()
