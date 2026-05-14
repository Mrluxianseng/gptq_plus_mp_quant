"""Cross-rank collective wrappers — re-exports of the helpers realq actually uses."""
from utils.dist_utils import allreduce_sum_, allreduce_sum_scalar

__all__ = [
    "allreduce_sum_",
    "allreduce_sum_scalar",
]
