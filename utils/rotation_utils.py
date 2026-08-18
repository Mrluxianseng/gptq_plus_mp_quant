import functools
import typing
import torch
import math
import types
import weakref
from tqdm import tqdm

from utils import (
    hadamard_utils,
    memory_utils,
    model_utils,
    monkeypatch,
    quant_utils,
    triton_qwen3_fusions,
)


@torch.inference_mode()
def fuse_ln_linears(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
    layernorm.weight.fill_(1.)


@torch.inference_mode()
def fuse_layer_norms(analyzer: model_utils.ModelAnalyzer) -> None:
    if analyzer.tie_word_embeddings:
        return
    layers = analyzer.get_layers()
    for layer in tqdm(layers, desc="Fusing LN"):
        for ln, linears in zip(
            analyzer.get_layernorms(layer),
            analyzer.get_perlayer_input_modules(layer)
        ):
            fuse_ln_linears(ln, linears)
    fuse_ln_linears(analyzer.get_layernorm_before_head(), [analyzer.get_lm_head()])

    memory_utils.cleanup_memory()


def get_orthogonal_matrix(size, mode, device="cuda", generator=None):
    if mode == "random":
        return hadamard_utils.random_orthogonal_matrix(size, device, generator=generator)
    elif mode == "hadamard":
        return hadamard_utils.random_hadamard_matrix(size, device, generator=generator)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(analyzer: model_utils.ModelAnalyzer, R1: torch.Tensor) -> None:
    if analyzer.tie_word_embeddings:
        return
    # Rotate the embeddings.
    for W in [analyzer.get_embed_layer()]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_head(analyzer: model_utils.ModelAnalyzer, R1: torch.Tensor) -> None:
    if analyzer.tie_word_embeddings:
        return
    # Rotate the head.
    W = analyzer.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_mlp_inputs(analyzer: model_utils.ModelAnalyzer, layer, R1) -> None:
    if analyzer.tie_word_embeddings:
        return
    for layers in analyzer.get_perlayer_input_modules(layer):
        for W in layers:
            dtype = W.weight.dtype
            W_ = W.weight.to(device="cuda", dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_mlp_output(analyzer: model_utils.ModelAnalyzer, layer, R1) -> None:
    if analyzer.tie_word_embeddings:
        return
    for layers in analyzer.get_perlayer_output_modules(layer):
        for W in layers:
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
            W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device="cuda", dtype=torch.float64)
                W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_down_proj(analyzer: model_utils.ModelAnalyzer, layer):
    # apply exact (inverse) hadamard on the weights of mlp output
    for W in analyzer.get_perlayer_down_proj(layer):
        hadamard_utils.apply_exact_had_to_linear(
            W, had_dim=-1, output=False
        )


def rotate_ov_proj(analyzer: model_utils.ModelAnalyzer, layer, R2=None):
    v_proj = analyzer.get_perlayer_v_proj(layer)
    o_proj = analyzer.get_perlayer_o_proj(layer)

    hadamard_utils.apply_exact_had_to_linear(v_proj, had_dim=analyzer.head_dim, output=True, R2=R2)
    hadamard_utils.apply_exact_had_to_linear(o_proj, had_dim=analyzer.head_dim, output=False, R2=R2)


@torch.inference_mode()
def rotate_model(args, analyzer: model_utils.ModelAnalyzer):
    if args.optimized_rotation_path is not None:
        R_cpk = args.optimized_rotation_path
        R1 = torch.load(R_cpk)["R1"].to(
            device="cuda", dtype=torch.float64
        )
        rotation_gen = None
    else:
        # Rotation is an algorithmic artifact, not part of calibration
        # sampling. Keep it stable when ``--seed`` is swept in the paper's
        # calibration-seed experiment.
        seed = int(getattr(args, "rotation_seed", 0))
        rotation_gen = torch.Generator(device="cpu").manual_seed(seed)
        R1 = get_orthogonal_matrix(
            analyzer.hidden_size,
            "hadamard",
            generator=rotation_gen,
        )

    rotate_embeddings(analyzer, R1)
    rotate_head(analyzer, R1)
    memory_utils.cleanup_memory()
    layers = analyzer.get_layers()
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating")):
        if args.optimized_rotation_path is not None:
            key = f"model.layers.{idx}.self_attn.R2"
            R2 = torch.load(R_cpk)[key].to(
                device="cuda", dtype=torch.float64
            )
        else:
            R2 = get_orthogonal_matrix(
                analyzer.head_dim,
                "hadamard",
                generator=rotation_gen,
            )
        rotate_attention_mlp_inputs(analyzer, layer, R1)
        rotate_attention_mlp_output(analyzer, layer, R1)
        rotate_down_proj(analyzer, layer)
        rotate_ov_proj(analyzer, layer, R2=R2)


def add_activation_quant_wrappers_for_rotation(analyzer: model_utils.ModelAnalyzer) -> None:
    model = analyzer.model
    if bool(getattr(model, "_gptqplus_rotation_wrappers_installed", False)):
        return
    quant_utils.add_actquant(analyzer)
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = False
    model._gptqplus_rotation_wrappers_installed = True


def prepare_model_for_rotated_quantization(args, analyzer: model_utils.ModelAnalyzer) -> None:
    model = analyzer.model
    model_pre_rotated = bool(getattr(model, "_gptqplus_checkpoint_is_rotated", False))
    if getattr(args, "rotate", False) and not model_pre_rotated:
        fuse_layer_norms(analyzer)
        rotate_model(args, analyzer)
        memory_utils.cleanup_memory()
        add_activation_quant_wrappers_for_rotation(analyzer)
    elif getattr(args, "rotate", False) and model_pre_rotated:
        add_activation_quant_wrappers_for_rotation(analyzer)
    else:
        quant_utils.add_actquant(analyzer)


def _deferred_qk_rmsnorm_forward(_module, hidden_states):
    """Leave Q/K unnormalised until the fused post-projection RoPE site."""

    return hidden_states


class QKRotationWrapper(torch.nn.Module):
    def __init__(
        self,
        func,
        head_dim,
        *args,
        q_norm=None,
        k_norm=None,
        **kwargs,
    ):
        super().__init__()
        if not hadamard_utils.is_pow2(head_dim):
            raise ValueError(
                "K-cache Q/K Hadamard rotation requires a power-of-two "
                f"head_dim; got {head_dim}."
            )
        self.func = func
        self.head_dim = head_dim
        self.k_quantizer = quant_utils.ActQuantizer()
        # K-cache quant params are re-estimated online every forward; keeping
        # them out of checkpoints avoids extra state_dict keys when GPTQ+ installs
        # this wrapper during weight quantization.
        self.k_quantizer._non_persistent_buffers_set.update({"maxq", "scale", "zero"})
        self.k_bits = 16
        self.k_groupsize = -1
        self.k_sym = False
        self.k_clip_ratio = 1.0
        self.k_quant_enabled = False
        # Weak references avoid registering q_norm/k_norm a second time below
        # this wrapper, which would duplicate state_dict paths.
        object.__setattr__(self, "_q_norm_ref", None)
        object.__setattr__(self, "_k_norm_ref", None)
        if kwargs:
            self.configure_k_quant(**kwargs)
        if q_norm is not None or k_norm is not None:
            self.configure_qk_norm_fusion(q_norm=q_norm, k_norm=k_norm)

    def configure_qk_norm_fusion(self, *, q_norm, k_norm) -> None:
        if q_norm is None or k_norm is None:
            raise ValueError("Q/K RMSNorm fusion requires both norm modules.")
        for label, norm in (("q_norm", q_norm), ("k_norm", k_norm)):
            weight = getattr(norm, "weight", None)
            if weight is None or tuple(weight.shape) != (self.head_dim,):
                raise ValueError(
                    f"{label}.weight must have shape ({self.head_dim},)."
                )
            if not hasattr(norm, "variance_epsilon"):
                raise ValueError(f"{label} does not expose variance_epsilon.")
            if not hasattr(norm, "_realq_unfused_forward"):
                # REAL-Q never optimises RMSNorm parameters during Block-GD;
                # freezing makes that existing contract explicit and lets the
                # custom backward omit an otherwise expensive dweight reduce.
                norm.weight.requires_grad_(False)
                norm._realq_unfused_forward = norm.forward
                norm.forward = types.MethodType(
                    _deferred_qk_rmsnorm_forward, norm
                )
        object.__setattr__(self, "_q_norm_ref", weakref.ref(q_norm))
        object.__setattr__(self, "_k_norm_ref", weakref.ref(k_norm))

    def _deferred_norms(self):
        q_ref = object.__getattribute__(self, "_q_norm_ref")
        k_ref = object.__getattribute__(self, "_k_norm_ref")
        if q_ref is None or k_ref is None:
            return None, None
        q_norm = q_ref()
        k_norm = k_ref()
        if q_norm is None or k_norm is None:
            raise RuntimeError("Deferred Q/K RMSNorm module was released.")
        return q_norm, k_norm

    def configure_k_quant(
        self,
        head_dim=None,
        k_bits=None,
        k_groupsize=None,
        k_sym=None,
        k_clip_ratio=None,
        k_quant_enabled=True,
    ):
        if head_dim is not None and int(head_dim) != self.head_dim:
            raise ValueError(
                f"Existing QKRotationWrapper head_dim={self.head_dim} cannot be "
                f"reconfigured with head_dim={head_dim}."
            )
        if k_bits is not None:
            self.k_bits = int(k_bits)
        if k_groupsize is not None:
            self.k_groupsize = int(k_groupsize)
        if k_sym is not None:
            self.k_sym = bool(k_sym)
        if k_clip_ratio is not None:
            self.k_clip_ratio = float(k_clip_ratio)
        if self.k_groupsize not in (-1, self.head_dim):
            raise ValueError(
                "K-cache groupsize must be -1 (one per-token scale across all "
                f"KV heads) or head_dim={self.head_dim} (one scale per head); "
                f"got {self.k_groupsize}."
            )
        self.k_quantizer.configure(
            bits=self.k_bits,
            groupsize=-1,  # token-wise; head-wise is handled explicitly below.
            sym=self.k_sym,
            clip_ratio=self.k_clip_ratio,
        )
        self.k_quant_enabled = bool(k_quant_enabled)

    def forward(self, *args, **kwargs):
        q_norm, k_norm = self._deferred_norms()
        if q_norm is None:
            q, k = self.func(*args, **kwargs)
        else:
            if len(args) < 2:
                raise ValueError("RoPE wrapper requires positional Q and K inputs.")
            q, k = args[:2]
            cos = args[2] if len(args) > 2 else kwargs.get("cos")
            sin = args[3] if len(args) > 3 else kwargs.get("sin")
            unsqueeze_dim = (
                args[5]
                if len(args) > 5
                else kwargs.get("unsqueeze_dim", 1)
            )
            if (
                cos is not None
                and sin is not None
                and triton_qwen3_fusions.can_fuse_qk_rmsnorm_rope(
                    q,
                    k,
                    q_norm.weight,
                    k_norm.weight,
                    cos,
                    sin,
                    unsqueeze_dim=unsqueeze_dim,
                )
            ):
                q, k = triton_qwen3_fusions.fused_qk_rmsnorm_rope(
                    q,
                    k,
                    q_norm.weight,
                    k_norm.weight,
                    cos,
                    sin,
                    q_eps=float(q_norm.variance_epsilon),
                    k_eps=float(k_norm.variance_epsilon),
                )
            else:
                # CPU, unsupported layouts and non-default RoPE broadcasting
                # retain the original eager equation exactly.
                q = q_norm._realq_unfused_forward(q)
                k = k_norm._realq_unfused_forward(k)
                forwarded_args = (q, k, *args[2:])
                q, k = self.func(*forwarded_args, **kwargs)
        dtype = q.dtype
        # FP32 input/output casts are precision-critical here: the Q/K online
        # rotation was defined in FP32.  FHT fuses the normalization multiply
        # into its CUDA store; only the final deployed-dtype cast remains.
        q = hadamard_utils.scaled_hadamard_transform(
            q.float(), scale=1.0 / math.sqrt(q.shape[-1])
        ).to(dtype)
        k = hadamard_utils.scaled_hadamard_transform(
            k.float(), scale=1.0 / math.sqrt(k.shape[-1])
        ).to(dtype)
        if not self.k_quant_enabled or self.k_bits >= 16:
            return q, k

        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    q_norm=None,
    k_norm=None,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    if hasattr(module, attr_name):
        wrapper = getattr(module, attr_name)
        if not isinstance(wrapper, QKRotationWrapper):
            raise TypeError(
                f"{type(module).__name__}.{attr_name} already exists but is not "
                "a QKRotationWrapper; refusing to stack an ambiguous patch."
            )
        if args or kwargs:
            wrapper.configure_k_quant(*args, **kwargs)
        if q_norm is not None or k_norm is not None:
            wrapper.configure_qk_norm_fusion(
                q_norm=q_norm, k_norm=k_norm
            )
        return wrapper

    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        lambda original_func: QKRotationWrapper(
            original_func,
            *args,
            q_norm=q_norm,
            k_norm=k_norm,
            **kwargs,
        ),
    )
    setattr(module, attr_name, wrapper)
    return wrapper


def disable_k_cache_quant(module):
    state = []
    for _, m in module.named_modules():
        if isinstance(m, QKRotationWrapper):
            state.append((m, m.k_quant_enabled))
            m.k_quant_enabled = False
    return state


def enable_k_cache_quant(module, state):
    for wrapper, enabled in state:
        wrapper.k_quant_enabled = enabled
