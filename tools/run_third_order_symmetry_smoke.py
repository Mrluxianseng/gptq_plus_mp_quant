#!/usr/bin/env python3
"""Replay a saved REAL-Q Namespace for matched third-order experiments.

The source log preserves the exact matched-pair configuration. This helper
changes only the output name, optional inclusive layer stop, evaluation toggle,
and explicitly selected refresh estimator. Launch it through torchrun because
ptq.main initializes the distributed group.
"""

import argparse
import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptq import main
from utils.log_utils import init_logging


def namespace_from_log(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    start = first_line.find("Namespace(")
    end = first_line.rfind(")")
    if start < 0 or end < start:
        raise ValueError(f"No Namespace(...) configuration found in {path}")
    expression = ast.parse(first_line[start : end + 1], mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("Saved configuration is not a Namespace call")
    return {
        item.arg: ast.literal_eval(item.value)
        for item in expression.keywords
        if item.arg is not None
    }


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--stop-after-layer", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--estimator",
        choices=("off", "central_cubic", "exact_downstream"),
        default="central_cubic",
    )
    args_cli = parser.parse_args()

    config = namespace_from_log(args_cli.source_log)
    config["exp"] = args_cli.exp
    config["quant_stop_layer"] = args_cli.stop_after_layer
    config["skip_eval"] = args_cli.skip_eval
    if args_cli.estimator == "off":
        config["third_order_refresh_layer"] = -1
        # The CLI validates the default estimator as inert when the selected
        # layer is -1; preserve that exact baseline-compatible state.
        config["third_order_refresh_estimator"] = "central_cubic"
    else:
        config["third_order_refresh_estimator"] = args_cli.estimator
    config["output_dir"] = os.path.join(
        "outputs", args_cli.exp, config["model_name"], args_cli.exp
    )
    config["log_dir"] = os.path.join(config["output_dir"], "logs")
    init_logging(config["log_dir"])

    from argparse import Namespace

    main(Namespace(**config))


if __name__ == "__main__":
    main_cli()
