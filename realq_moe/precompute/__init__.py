"""Static end-to-end saliency + Fisher precompute.

The public entry is ``run(cfg, analyzer)`` from ``static_e2e``. Other
submodules are implementation detail.
"""
from realq_moe.precompute.static_e2e import StaticStats, run

__all__ = ["StaticStats", "run"]
