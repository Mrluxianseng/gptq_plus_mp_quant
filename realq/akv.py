"""Activation / KV cache quantisation wrappers.

Two phases:

* **Unaware** (default): wrappers are configured AFTER the weight GPTQ pass
  has finished. Weight Hessian was accumulated against fp activations; the
  wrappers only kick in for the eval/runtime forward path.
* **Aware** (``act_quant_aware_gptq`` / ``k_cache_quant_aware_gptq``):
  wrappers are configured BEFORE quantisation starts so the per-linear
  Hessian forward sees the quantised activation, matching what the deployed
  model will see at inference time.

Both modes leverage the existing ``utils.quant_utils.ActQuantWrapper`` and
``utils.rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward``
infrastructure — RealQ only owns the orchestration wrapper.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from utils import quant_utils, rotation_utils

if TYPE_CHECKING:
    from realq.config import Config
    from utils.model_utils import ModelAnalyzer


def install_actquant_wrappers(analyzer: "ModelAnalyzer") -> None:
    """Wrap every quantisable linear in ``ActQuantWrapper``. Idempotent.

    The model is tagged with ``_realq_actquant_wrappers_installed`` after
    the first call so subsequent invocations are a no-op. Without this guard
    a second ``add_actquant`` pass would double-wrap (the inner ``Linear`` is
    still discoverable via ``named_modules()`` after the first wrap, so
    ``add_actquant`` happily wraps it again, leaving us with
    ``Wrapper(Wrapper(Linear))`` and breaking every downstream isinstance check.
    """
    model = analyzer.model
    if bool(getattr(model, "_realq_actquant_wrappers_installed", False)):
        return
    quant_utils.add_actquant(analyzer)
    model._realq_actquant_wrappers_installed = True


def configure_a_v_quantizers(analyzer: "ModelAnalyzer", cfg: "Config") -> None:
    """Set the A and V quant params on the already-wrapped linears.

    A applies to the wrapper's input (pre-linear activation).
    V applies to v_proj's wrapper OUTPUT.
    Skipped for ``lm_head`` (always fp16).
    """
    if cfg.a_bits >= 16 and cfg.v_bits >= 16:
        return
    qlayers = quant_utils.find_qlayers(analyzer.model, layers=[quant_utils.ActQuantWrapper])
    for name, w in qlayers.items():
        # V output quantiser only on v_proj.
        if cfg.v_bits < 16 and "v_proj" in name:
            w.out_quantizer.configure(
                bits=cfg.v_bits,
                groupsize=cfg.v_groupsize,
                sym=not cfg.v_asym,
                clip_ratio=cfg.v_clip_ratio,
            )
        # A input quantiser on every wrapper EXCEPT lm_head.
        if cfg.a_bits < 16:
            bits = 16 if "lm_head" in name else cfg.a_bits
            w.quantizer.configure(
                bits=bits,
                groupsize=cfg.a_groupsize,
                sym=not cfg.a_asym,
                clip_ratio=cfg.a_clip_ratio,
            )


def install_k_cache_wrappers(analyzer: "ModelAnalyzer", cfg: "Config") -> None:
    """Splice ``QKRotationWrapper`` after RoPE so K is quantised before cache.

    Idempotent at the wrapper level — old code's helper sees the existing
    wrapper and skips; we don't have to track that here.
    """
    if cfg.k_bits >= 16:
        return
    rope_function_name = "apply_rotary_pos_emb"
    layers = analyzer.get_layers()
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn,
            rope_function_name,
            head_dim=analyzer.head_dim,
            k_bits=cfg.k_bits,
            k_groupsize=cfg.k_groupsize,
            k_sym=not cfg.k_asym,
            k_clip_ratio=cfg.k_clip_ratio,
        )


def setup_aware_pre_quant(analyzer: "ModelAnalyzer", cfg: "Config") -> None:
    """Install + configure A/K wrappers BEFORE weight GPTQ runs.

    With ``act_quant_aware_gptq=True`` the per-linear Hessian forward sees
    the quantised input; with ``k_cache_quant_aware_gptq=True`` the K cache
    is rotated + quantised inside the attention computation, so o_proj's
    Hessian forward also sees the quantised K path.
    """
    install_actquant_wrappers(analyzer)
    if cfg.act_quant_aware_gptq:
        configure_a_v_quantizers(analyzer, cfg)
    if cfg.k_cache_quant_aware_gptq:
        install_k_cache_wrappers(analyzer, cfg)


def setup_unaware_post_quant(analyzer: "ModelAnalyzer", cfg: "Config") -> None:
    """Configure A/V/K wrappers AFTER weight GPTQ finishes.

    Old ``ptq.py`` lines 130-172 do exactly this. ``add_actquant`` was
    already called by ``install_actquant_wrappers`` (or by the rotate path);
    we only need to push the quant params into the existing wrappers and
    splice the K cache rotator.
    """
    install_actquant_wrappers(analyzer)
    if not cfg.act_quant_aware_gptq:
        configure_a_v_quantizers(analyzer, cfg)
    if not cfg.k_cache_quant_aware_gptq:
        install_k_cache_wrappers(analyzer, cfg)
