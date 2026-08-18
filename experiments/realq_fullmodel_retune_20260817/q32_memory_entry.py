"""Dedicated PTQ entry for Qwen3-32B deterministic-memory retries."""

from __future__ import annotations

from experiments.realq_fullmodel_retune_20260817.deterministic_sdpa_memory import (
    install,
)


def main() -> None:
    install()
    print(
        "[realq.q32_memory] wrappers installed: checkpoint=true "
        "math_sdpa_batch_tile=4 logical_backward_bsz_unchanged=true",
        flush=True,
    )
    from realq.ptq import main as ptq_main

    ptq_main()


if __name__ == "__main__":
    main()
