"""Opt-in algorithm-core timing hooks for formal low-bit experiments.

Python imports ``sitecustomize`` before the selected entrypoint.  The external
formal timed executor prepends this directory to ``PYTHONPATH`` only for a
timed child and supplies a complete, hashed timing spec in the environment.
All wrappers call the original numerical function exactly once and preserve
its arguments and return value.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


_PREFIX = "LOWBIT_FORMAL_TIMING_"
_TIMING_DIR = os.environ.get(f"{_PREFIX}DIR")
_EXPECTED_WORLD = int(
    os.environ.get(f"{_PREFIX}EXPECT_WORLD_SIZE", "0") or "0"
)
_MODE = os.environ.get(f"{_PREFIX}MODE")
_DISTRIBUTED_WORKER = "LOCAL_RANK" in os.environ or "RANK" in os.environ
_ACTIVE = bool(
    _TIMING_DIR
    and _EXPECTED_WORLD > 0
    and (
        (_MODE == "guided_precompute" and _EXPECTED_WORLD == 1)
        or _DISTRIBUTED_WORKER
    )
)


def _fatal_bootstrap(message: str) -> None:
    """Abort instead of letting Python ignore a sitecustomize exception."""

    try:
        sys.stderr.write(f"FATAL formal timing bootstrap: {message}\n")
        sys.stderr.flush()
    finally:
        os._exit(86)


if _ACTIVE:
    _EXPECTED_SITE_SHA = os.environ.get(
        f"{_PREFIX}SITECUSTOMIZE_SHA256"
    )
    try:
        _SELF_BYTES = Path(__file__).resolve().read_bytes()
    except OSError as exc:
        _fatal_bootstrap(f"cannot read instrumentation source: {exc}")
    _ACTUAL_SITE_SHA = hashlib.sha256(_SELF_BYTES).hexdigest()
    if (
        not _EXPECTED_SITE_SHA
        or _ACTUAL_SITE_SHA != _EXPECTED_SITE_SHA
    ):
        _fatal_bootstrap(
            "instrumentation SHA256 mismatch "
            f"(expected={_EXPECTED_SITE_SHA!r}, "
            f"actual={_ACTUAL_SITE_SHA!r})"
        )

    import torch
    import torch.distributed as dist

    _VALID_MODES = {
        "realq_shared_precompute",
        "realq_quantization",
        "legacy_quantization",
        "guided_precompute",
    }
    if _MODE not in _VALID_MODES:
        _fatal_bootstrap(f"unsupported timing mode {_MODE!r}")

    _state: dict[str, Any] = {
        "started_monotonic_ns": None,
        "ended_monotonic_ns": None,
        "rotation_ended_monotonic_ns": None,
        "complete": False,
        "written": False,
        "error": None,
        "cache_lookups": [],
        "cache_writes": [],
        "artifacts": [],
    }

    def _rank() -> int:
        return int(
            os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        )

    def _local_rank() -> int:
        return int(os.environ.get("LOCAL_RANK", "0"))

    def _world_size() -> int:
        return int(
            os.environ.get("WORLD_SIZE", str(_EXPECTED_WORLD or 1))
        )

    def _visible_devices() -> list[int]:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        try:
            values = [int(value.strip()) for value in raw.split(",")]
        except ValueError as exc:
            _fatal_bootstrap(
                f"CUDA_VISIBLE_DEVICES is not an integer list: {exc}"
            )
        if (
            not values
            or len(set(values)) != len(values)
            or any(value < 0 for value in values)
        ):
            _fatal_bootstrap(
                f"invalid CUDA_VISIBLE_DEVICES value {raw!r}"
            )
        return values

    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _start_interval() -> None:
        if _state["started_monotonic_ns"] is not None:
            raise RuntimeError("formal timing interval started twice")
        _sync_cuda()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        _sync_cuda()
        _state["started_monotonic_ns"] = time.perf_counter_ns()

    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    value,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            directory_fd = os.open(path.parent, flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
            raise

    def _nullable_env(name: str) -> str | None:
        value = os.environ.get(f"{_PREFIX}{name}", "")
        return value or None

    def _evidence() -> dict[str, Any]:
        start = _state["started_monotonic_ns"]
        end = _state["ended_monotonic_ns"]
        elapsed = (
            (end - start) / 1_000_000_000
            if isinstance(start, int)
            and isinstance(end, int)
            and end >= start
            else None
        )
        visible = _visible_devices()
        local_rank = _local_rank()
        if local_rank < 0 or local_rank >= len(visible):
            _fatal_bootstrap(
                f"LOCAL_RANK={local_rank} is outside visible devices"
            )
        complete = bool(_state["complete"])
        timing_status = (
            "complete"
            if complete
            else ("partial" if isinstance(start, int) else "not_started")
        )
        return {
            "schema_version": 1,
            "scope_id": "algorithm_core_v1",
            "source_set_sha256": os.environ.get(
                f"{_PREFIX}SOURCE_SET_SHA256"
            ),
            "sitecustomize_sha256": _ACTUAL_SITE_SHA,
            "spec_sha256": os.environ.get(f"{_PREFIX}SPEC_SHA256"),
            "run_id": os.environ.get(f"{_PREFIX}RUN_ID"),
            "mode": _MODE,
            "method": os.environ.get(f"{_PREFIX}METHOD"),
            "stage": os.environ.get(f"{_PREFIX}STAGE"),
            "model": os.environ.get(f"{_PREFIX}MODEL"),
            "setting": _nullable_env("SETTING"),
            "phase": _nullable_env("PHASE"),
            "target_phase": _nullable_env("TARGET_PHASE"),
            "rank": _rank(),
            "local_rank": local_rank,
            "world_size": _world_size(),
            "physical_gpu_index": visible[local_rank],
            "clock": "time.perf_counter_ns",
            "started_monotonic_ns": start,
            "rotation_ended_monotonic_ns": _state[
                "rotation_ended_monotonic_ns"
            ],
            "ended_monotonic_ns": end,
            "elapsed_seconds": elapsed,
            "synchronized_start": isinstance(start, int),
            "synchronized_end": complete,
            "timing_status": timing_status,
            "complete": complete,
            "error": _state["error"],
            "cache_lookups": list(_state["cache_lookups"]),
            "cache_writes": list(_state["cache_writes"]),
            "artifacts": list(_state["artifacts"]),
        }

    def _write_evidence() -> None:
        if (
            _state["written"]
            or not _TIMING_DIR
        ):
            return
        path = Path(_TIMING_DIR) / (
            f"phase_timing_rank{_rank()}.json"
        )
        _atomic_json(path, _evidence())
        _state["written"] = True

    def _finish_interval(
        *,
        complete: bool,
        error: BaseException | None = None,
        synchronize: bool = True,
    ) -> None:
        if _state["written"]:
            return
        if _state["ended_monotonic_ns"] is not None:
            _write_evidence()
            return
        if _state["started_monotonic_ns"] is None:
            _state["error"] = (
                f"{type(error).__name__}: {error}"
                if error is not None
                else "process exited before the timing boundary started"
            )
            _write_evidence()
            return
        if complete:
            try:
                if synchronize:
                    _sync_cuda()
                ended = time.perf_counter_ns()
                if (
                    synchronize
                    and dist.is_available()
                    and dist.is_initialized()
                ):
                    dist.barrier()
                if synchronize:
                    _sync_cuda()
            except BaseException as exc:
                _state["ended_monotonic_ns"] = time.perf_counter_ns()
                _state["error"] = f"{type(exc).__name__}: {exc}"
                _write_evidence()
                raise
            _state["ended_monotonic_ns"] = ended
            _state["complete"] = True
        else:
            if synchronize:
                with contextlib.suppress(BaseException):
                    _sync_cuda()
            _state["ended_monotonic_ns"] = time.perf_counter_ns()
            _state["error"] = (
                f"{type(error).__name__}: {error}"
                if error is not None
                else "process exited before the timing boundary completed"
            )
        _write_evidence()

    _previous_signal_handlers: dict[int, Any] = {}

    def _signal_evidence(signum: int, frame: Any) -> None:
        name = signal.Signals(signum).name
        _finish_interval(
            complete=False,
            error=RuntimeError(f"received {name} ({signum})"),
            synchronize=False,
        )
        previous = _previous_signal_handlers.get(signum, signal.SIG_DFL)
        if previous == signal.SIG_IGN:
            return
        if callable(previous):
            previous(signum, frame)
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for _signal_name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if not hasattr(signal, _signal_name):
            continue
        _signum = int(getattr(signal, _signal_name))
        _previous_signal_handlers[_signum] = signal.getsignal(_signum)
        signal.signal(_signum, _signal_evidence)

    def _atexit_evidence() -> None:
        if _state["written"]:
            return
        _finish_interval(complete=False)

    atexit.register(_atexit_evidence)

    if _MODE in {
        "realq_shared_precompute",
        "realq_quantization",
    }:
        from realq_benchmark import pipeline
        from realq_benchmark.precompute import cache as cache_mod

        _original_rotate = pipeline._maybe_rotate

        def _timed_rotate(*args: Any, **kwargs: Any) -> Any:
            _start_interval()
            try:
                result = _original_rotate(*args, **kwargs)
            except BaseException as exc:
                _finish_interval(complete=False, error=exc)
                raise
            _sync_cuda()
            _state["rotation_ended_monotonic_ns"] = (
                time.perf_counter_ns()
            )
            return result

        pipeline._maybe_rotate = _timed_rotate

        _original_try_load = cache_mod.try_load

        def _timed_try_load(
            cache_dir: str,
            key: str,
            world_size: int,
            rank: int,
        ) -> Any:
            result = _original_try_load(
                cache_dir, key, world_size, rank
            )
            _state["cache_lookups"].append(
                {
                    "key": key,
                    "path": cache_mod.cache_path(
                        cache_dir, key, world_size, rank
                    ),
                    "hit": result is not None,
                }
            )
            return result

        cache_mod.try_load = _timed_try_load

        _original_save = cache_mod.save

        def _timed_save(
            cache_dir: str,
            key: str,
            world_size: int,
            rank: int,
            payload: dict[str, Any],
        ) -> str:
            path = _original_save(
                cache_dir, key, world_size, rank, payload
            )
            _state["cache_writes"].append(
                {"key": key, "path": path}
            )
            return path

        cache_mod.save = _timed_save

        if _MODE == "realq_shared_precompute":
            _original_precompute = pipeline.precompute.run

            def _timed_precompute(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = _original_precompute(*args, **kwargs)
                except BaseException as exc:
                    _finish_interval(complete=False, error=exc)
                    raise
                _finish_interval(complete=True)
                return result

            pipeline.precompute.run = _timed_precompute
        else:
            _original_post_quant = (
                pipeline.akv.setup_unaware_post_quant
            )

            def _timed_post_quant(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = _original_post_quant(*args, **kwargs)
                except BaseException as exc:
                    _finish_interval(complete=False, error=exc)
                    raise
                _finish_interval(complete=True)
                return result

            pipeline.akv.setup_unaware_post_quant = _timed_post_quant

    elif _MODE == "legacy_quantization":
        from utils import checkpoint_utils, eval_utils, rotation_utils

        _original_fuse = rotation_utils.fuse_layer_norms

        def _timed_fuse(*args: Any, **kwargs: Any) -> Any:
            _start_interval()
            try:
                result = _original_fuse(*args, **kwargs)
            except BaseException as exc:
                _finish_interval(complete=False, error=exc)
                raise
            _sync_cuda()
            _state["rotation_ended_monotonic_ns"] = (
                time.perf_counter_ns()
            )
            return result

        rotation_utils.fuse_layer_norms = _timed_fuse

        _original_checkpoint_save = (
            checkpoint_utils.save_quantized_checkpoint
        )

        def _timed_checkpoint_save(*args: Any, **kwargs: Any) -> Any:
            # The requested safeguard checkpoint is intentionally outside the
            # formal quantization GPU-hour boundary.
            _finish_interval(complete=True)
            return _original_checkpoint_save(*args, **kwargs)

        checkpoint_utils.save_quantized_checkpoint = (
            _timed_checkpoint_save
        )

        _original_kl_ppl = eval_utils.kl_ppl_eval

        def _timed_kl_ppl(*args: Any, **kwargs: Any) -> Any:
            # Defensive fallback for a non-formal caller. Formal commands are
            # required to save first, so their interval already ended before
            # checkpoint I/O.
            if not _state["written"]:
                _finish_interval(complete=True)
            return _original_kl_ppl(*args, **kwargs)

        eval_utils.kl_ppl_eval = _timed_kl_ppl

    elif _MODE == "guided_precompute":
        from utils import data_utils, gradients as gradients_mod

        _original_get_tokens = data_utils.get_tokens

        def _timed_get_tokens(*args: Any, **kwargs: Any) -> Any:
            _start_interval()
            try:
                return _original_get_tokens(*args, **kwargs)
            except BaseException as exc:
                _finish_interval(complete=False, error=exc)
                raise

        data_utils.get_tokens = _timed_get_tokens

        _original_get_gradients = gradients_mod.get_gradients

        def _timed_get_gradients(*args: Any, **kwargs: Any) -> Any:
            try:
                result = _original_get_gradients(*args, **kwargs)
            except BaseException as exc:
                _finish_interval(complete=False, error=exc)
                raise
            gradients_path = kwargs.get(
                "gradients_path",
                args[2] if len(args) > 2 else None,
            )
            saliency_path = kwargs.get(
                "saliency_path",
                args[3] if len(args) > 3 else None,
            )
            num_groups = kwargs.get(
                "num_groups",
                args[4] if len(args) > 4 else None,
            )
            _state["artifacts"].append(
                {
                    "gradients_path": gradients_path,
                    "saliency_path": saliency_path,
                    "num_groups": num_groups,
                }
            )
            _finish_interval(complete=True)
            return result

        gradients_mod.get_gradients = _timed_get_gradients
