"""Run the pinned EvalPlus evaluator for an existing samples file."""

import importlib
import importlib.machinery
import os
import sys
import types


if len(sys.argv) != 3:
    raise SystemExit(
        "usage: lowbit_activation_evalplus_entrypoint.py SAMPLES PARALLEL"
    )

samples_path = sys.argv[1]
parallel = sys.argv[2]

# EvalPlus imports the optional generation frontend even for samples-only
# scoring. The frontend is unused here and its pinned tree-sitter wheel targets
# a different Python ABI, so expose only the unused symbol.
codegen = types.ModuleType("evalplus.codegen")
codegen.__spec__ = importlib.machinery.ModuleSpec("evalplus.codegen", loader=None)


def _unused_codegen(*args, **kwargs):
    raise RuntimeError("run_codegen is unavailable in samples-only scoring")


codegen.run_codegen = _unused_codegen
sys.modules["evalplus.codegen"] = codegen
sys.argv = [
    "evalplus.evaluate",
    "--dataset",
    "humaneval",
    "--samples",
    samples_path,
    "--parallel",
    parallel,
]
os.environ.setdefault("REALQ_ALLOW_UNTRUSTED_CODE", "1")
importlib.import_module("evalplus.evaluate").main()
