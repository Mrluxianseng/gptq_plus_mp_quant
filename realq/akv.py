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

from gptq_utils.quant_aware_utils import (
    configure_activation_quantizers_for_gptq,
    configure_k_cache_quantizers_for_gptq,
)
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
    if (
        bool(getattr(model, "_realq_actquant_wrappers_installed", False))
        or bool(getattr(model, "_gptqplus_rotation_wrappers_installed", False))
    ):
        # QuaRot owns the shared wrapper installation but historically set
        # only its legacy marker.  Synchronise the RealQ marker rather than
        # rediscovering and wrapping each inner Linear a second time.
        model._realq_actquant_wrappers_installed = True
        return
    quant_utils.add_actquant(analyzer)
    model._realq_actquant_wrappers_installed = True


def configure_a_v_quantizers(analyzer: "ModelAnalyzer", cfg: "Config") -> None:
    """Set the A and V quant params on the already-wrapped linears.

    A applies to the wrapper's input (pre-linear activation).
    V applies to v_proj's wrapper OUTPUT.
    Skipped for ``lm_head`` (always fp16).
    """
    configure_activation_quantizers_for_gptq(cfg, analyzer.model)


def install_k_cache_wrappers(analyzer: "ModelAnalyzer", cfg: "Config") -> None:
    """Splice ``QKRotationWrapper`` after RoPE so K is quantised before cache.

    Idempotent at the wrapper level — old code's helper sees the existing
    wrapper and skips; we don't have to track that here.
    """
    if cfg.k_bits >= 16:
        return
    configure_k_cache_quantizers_for_gptq(cfg, analyzer)


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
