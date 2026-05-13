"""Cross-rank collective wrappers.

Sub-task 1 only re-exports what utils.dist_utils already provides. Custom
collectives (e.g. all_gather of per-rank scale/zero in find_params) land in
sub-task 4.
"""
from utils.dist_utils import (
    allreduce_mean_scalar,
    allreduce_sum_,
    allreduce_sum_scalar,
    assert_bit_exact,
    broadcast_tensor_,
    broadcast_parameters,
    gather_into_tensor,
)

__all__ = [
    "allreduce_sum_",
    "allreduce_sum_scalar",
    "allreduce_mean_scalar",
    "broadcast_tensor_",
    "broadcast_parameters",
    "gather_into_tensor",
    "assert_bit_exact",
]
