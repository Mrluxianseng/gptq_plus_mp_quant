#!/usr/bin/env python3
"""Bind the dynamic formal controller to deterministic-SDPA V2."""

from __future__ import annotations

import sys
from typing import Sequence

from experiments.realq_sdpa_frozen_lr20_20260822 import controller as core
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v2 as formal


def _activate() -> None:
    campaign._bootstrap()
    formal._activate()
    core.campaign = campaign
    core.formal = formal
    core.CONTROLLER_CLAIM = ".sdpa_v2_controller_claim"
    core.TERMINAL_FAILURE = "controller_v2_terminal_failure.json"


def main(argv: Sequence[str] | None = None) -> int:
    _activate()
    return int(core.main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
