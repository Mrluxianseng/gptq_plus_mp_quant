import atexit
import inspect
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Dict, Union, List, Optional

import torch
import torch.nn as nn


_process_profile_peaks = {}


def get_process_cuda_memory_peaks(device=None):
    """Include completed exact-phase intervals plus later allocations (e.g. eval)."""
    device = torch.device(device if device is not None else f"cuda:{torch.cuda.current_device()}")
    index = torch.cuda.current_device() if device.index is None else device.index
    previous = _process_profile_peaks.get(index, (0, 0))
    return (
        max(previous[0], int(torch.cuda.max_memory_allocated(device))),
        max(previous[1], int(torch.cuda.max_memory_reserved(device))),
    )


class CudaPhaseMemoryProfiler:
    """Opt-in CUDA memory observations between explicit phase markers.

    Default phase peaks are sampled; CUDA cumulative counters are not reset.
    REALQ_PHASE_PEAK_MODE=exact opts into exact allocator interval counters.
    That diagnostic mode resets CUDA peak counters between markers and carries
    their maxima here; consumers must use get_process_cuda_memory_peaks for
    cumulative reporting. Each marker synchronizes to separate phases.
    """

    def __init__(self, output_dir: str, device, *, tag: str = "quantization"):
        self.device = torch.device(device)
        self.peak_mode = os.environ.get("REALQ_PHASE_PEAK_MODE", "sampled").strip().lower()
        if self.peak_mode not in {"sampled", "exact"}:
            raise ValueError("REALQ_PHASE_PEAK_MODE must be sampled or exact")
        self.enabled = self.device.type == "cuda" and torch.cuda.is_available()
        self.rows = []
        self.phase = None
        self.started_at = None
        self.start_allocated = 0
        self.start_reserved = 0
        self.start_device_used = None
        self.start_device_free = None
        self.total_device = None
        self.sampled_peak_allocated = 0
        self.sampled_peak_reserved = 0
        self._sample_lock = threading.Lock()
        self._stop_sampler = threading.Event()
        self._sampler_thread = None
        self._run_peak_allocated = 0
        self._run_peak_reserved = 0
        self.path = None
        if not self.enabled:
            return

        rank = int(os.environ.get("RANK", "0") or 0)
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(
            output_dir,
            f"cuda_memory_{tag}_rank{rank}_pid{os.getpid()}.jsonl",
        )
        self._file = open(self.path, "w", encoding="utf-8")
        self._run_peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
        self._run_peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        self._publish_peaks()
        self._begin("gptq_fwrd_setup")
        if self.peak_mode == "exact":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sampler_thread = threading.Thread(
            target=self._sample_loop,
            name="gptq-plus-cuda-memory-sampler",
            daemon=True,
        )
        if self.peak_mode == "sampled":
            self._sampler_thread.start()
        atexit.register(self.close)

    def _publish_peaks(self):
        index = torch.cuda.current_device() if self.device.index is None else self.device.index
        previous = _process_profile_peaks.get(index, (0, 0))
        _process_profile_peaks[index] = (
            max(previous[0], self._run_peak_allocated),
            max(previous[1], self._run_peak_reserved),
        )

    def _device_snapshot(self):
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return int(total_bytes - free_bytes), int(free_bytes), int(total_bytes)

    def _allocator_boundary_stats(self):
        # These are synchronized boundary values, not interval maxima.
        # They distinguish live workspaces from inactive split/cache blocks.
        stats = torch.cuda.memory_stats(self.device)
        return {key: int(stats.get(key, 0)) for key in (
            "active_bytes.all.current", "inactive_split_bytes.all.current",
            "segment.all.current", "num_alloc_retries", "num_ooms",
        )}

    def _begin(self, phase):
        torch.cuda.synchronize(self.device)
        self.phase = str(phase)
        self.started_at = time.perf_counter()
        self.start_allocated = int(torch.cuda.memory_allocated(self.device))
        self.start_reserved = int(torch.cuda.memory_reserved(self.device))
        self.start_allocator_stats = self._allocator_boundary_stats()
        self.start_device_used, self.start_device_free, self.total_device = (
            self._device_snapshot()
        )
        self.sampled_peak_allocated = self.start_allocated
        self.sampled_peak_reserved = self.start_reserved

    def _sample_once(self):
        with self._sample_lock:
            if self.phase is None:
                return
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
            self.sampled_peak_allocated = max(self.sampled_peak_allocated, allocated)
            self.sampled_peak_reserved = max(self.sampled_peak_reserved, reserved)

    def _sample_loop(self):
        while not self._stop_sampler.wait(0.005):
            try:
                self._sample_once()
            except Exception:
                # Keep the profiler from affecting a run if a transient CUDA
                # error prevents one sample; explicit markers still report it.
                continue

    def mark(self, next_phase, **metadata):
        """Close the current interval and start `next_phase`."""
        if not self.enabled or self.phase is None:
            return
        torch.cuda.synchronize(self.device)
        self._sample_once()
        end_allocated = int(torch.cuda.memory_allocated(self.device))
        end_reserved = int(torch.cuda.memory_reserved(self.device))
        end_allocator_stats = self._allocator_boundary_stats()
        end_device_used, end_device_free, _ = self._device_snapshot()
        interval_peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
        interval_peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        cumulative_peak_allocated = max(self._run_peak_allocated, interval_peak_allocated)
        cumulative_peak_reserved = max(self._run_peak_reserved, interval_peak_reserved)
        with self._sample_lock:
            sampled_peak_allocated = self.sampled_peak_allocated
            sampled_peak_reserved = self.sampled_peak_reserved
        if self.peak_mode == "exact":
            sampled_peak_allocated = interval_peak_allocated
            sampled_peak_reserved = interval_peak_reserved
        row = {
            "peak_measurement_mode": self.peak_mode,
            "phase": self.phase,
            "next_phase": str(next_phase),
            "elapsed_seconds": time.perf_counter() - self.started_at,
            "start_allocated_bytes": self.start_allocated,
            "sampled_peak_allocated_bytes": sampled_peak_allocated,
            "end_allocated_bytes": end_allocated,
            "start_reserved_bytes": self.start_reserved,
            "sampled_peak_reserved_bytes": sampled_peak_reserved,
            "end_reserved_bytes": end_reserved,
            "start_allocator_stats": self.start_allocator_stats,
            "end_allocator_stats": end_allocator_stats,
            "cumulative_peak_allocated_bytes": cumulative_peak_allocated,
            "cumulative_peak_reserved_bytes": cumulative_peak_reserved,
            "new_run_high_watermark": bool(
                cumulative_peak_allocated > self._run_peak_allocated
                or cumulative_peak_reserved > self._run_peak_reserved
            ),
            "device_total_bytes": self.total_device,
            "start_device_used_bytes": self.start_device_used,
            "start_device_free_bytes": self.start_device_free,
            "end_device_used_bytes": end_device_used,
            "end_device_free_bytes": end_device_free,
            "metadata": metadata,
        }
        self._write(row)
        self._run_peak_allocated = max(self._run_peak_allocated, cumulative_peak_allocated)
        self._run_peak_reserved = max(self._run_peak_reserved, cumulative_peak_reserved)
        self._publish_peaks()
        with self._sample_lock:
            self.phase = str(next_phase)
            self.started_at = time.perf_counter()
            self.start_allocated = end_allocated
            self.start_reserved = end_reserved
            self.start_allocator_stats = end_allocator_stats
            self.start_device_used = end_device_used
            self.start_device_free = end_device_free
            self.sampled_peak_allocated = end_allocated
            self.sampled_peak_reserved = end_reserved
        if self.peak_mode == "exact":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _write(self, row):
        self.rows.append(row)
        self._file.write(json.dumps(row, sort_keys=True) + "\n")
        self._file.flush()

    def finish(self, status="completed"):
        """Mark a normal return from the instrumented quantization path."""
        self._close(status=status, terminal_phase="complete")

    def close(self):
        """Finalize at process exit without claiming the run completed.

        This is registered with ``atexit`` as a best-effort fallback. A normal
        call path should use :meth:`finish`; process exit can also follow an
        exception or interruption, so it must not be labeled ``complete``.
        """
        self._close(
            status="process_exit_unconfirmed",
            terminal_phase="process_exit_unconfirmed",
        )

    def _close(self, *, status, terminal_phase):
        if not self.enabled or self.phase is None:
            return
        try:
            self.mark(terminal_phase)
            self._write({
                "type": "summary",
                "termination_status": str(status),
                "phase_count": len(self.rows),
                "max_sampled_phase_peak_allocated_bytes": max(
                    (row["sampled_peak_allocated_bytes"] for row in self.rows), default=0
                ),
                "max_sampled_phase_peak_reserved_bytes": max(
                    (row["sampled_peak_reserved_bytes"] for row in self.rows), default=0
                ),
                "run_peak_allocated_bytes": self._run_peak_allocated,
                "run_peak_reserved_bytes": self._run_peak_reserved,
            })
        except Exception as exc:
            try:
                self._write({"type": "summary_error", "error": repr(exc)})
            except Exception:
                pass
        finally:
            self._stop_sampler.set()
            if self._sampler_thread is not None and self._sampler_thread.is_alive():
                self._sampler_thread.join(timeout=1.0)
            self._file.close()
            self.phase = None


def cleanup_memory(verbos=False, trim_cpu=False) -> None:
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    gc.collect()

    if trim_cpu:
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    memory_after = total_reserved_mem()
    if verbos:
        logger.info(
            f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        )


def get_param_size_in_bytes(dtype: torch.dtype) -> int:
    """
    Returns the size of a single element (in bytes) for a given PyTorch dtype.
    """
    # PyTorch Tensors have an element_size() method, but defining a map 
    # here ensures we can report on the calculation clearly.
    # Note: torch.float is usually float32 (4 bytes).
    dtype_map = {
        torch.float32: 4,
        torch.float: 4,
        torch.float64: 8,
        torch.double: 8,
        torch.float16: 2,
        torch.half: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.int: 4,
        torch.int64: 8,
        torch.long: 8,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
        torch.complex64: 8, # 2 * float32
        torch.complex128: 16 # 2 * float64
    }
    return dtype_map.get(dtype, 0) # Return 0 for unsupported/unknown types


def compute_trainable_param_memory(
    model: nn.Module, 
    unit: str = 'GB'
) -> Dict[str, Union[float, int]]:
    """
    Computes the total memory space occupied by the trainable parameters 
    in a PyTorch model and returns the result in the specified unit (MB, GB, or Bytes).

    Args:
        model (nn.Module): The PyTorch model instance.
        unit (str): The desired output unit ('B', 'MB', or 'GB'). Case-insensitive.

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing total bytes, 
                                      total trainable parameters count, and 
                                      the calculated size in the desired unit.
    """
    total_bytes = 0
    total_trainable_params = 0

    # Iterate over all named parameters of the model
    for name, param in model.named_parameters():
        # Check if the parameter requires a gradient (i.e., is trainable)
        if param.requires_grad:
            # Get the number of elements in the tensor
            num_elements = param.numel()
            
            # Get the size of a single element in bytes based on its dtype
            element_bytes = get_param_size_in_bytes(param.dtype)
            
            # Calculate the total bytes for this parameter
            param_bytes = num_elements * element_bytes
            
            # Accumulate totals
            total_bytes += param_bytes
            total_trainable_params += num_elements

    # Conversion factors
    UNIT_FACTORS = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3
    }
    
    # Calculate the size in the requested unit
    unit_upper = unit.upper()
    factor = UNIT_FACTORS.get(unit_upper, UNIT_FACTORS['GB']) # Default to GB
    
    size_in_unit = total_bytes / factor

    return {
        "unit": unit_upper,
        "size": round(size_in_unit, 4),
        "total_bytes": total_bytes,
        "trainable_params": total_trainable_params,
    }


def enable_memory_visualize(
    trace_alloc_max_entries: int = 200_000,
    stack_depth: int = 32,
    context: str = "all",
    stacks: str = "all",
    devices=None,
    record_context: bool = True,
):
    """
    Enables memory history recording for CUDA allocations. This function
    should be called before any large-scale CUDA allocations. For DDP or
    multi-process setups, it must be called on each rank.

    Args:
        trace_alloc_max_entries (int): Maximum number of allocation entries
            to record.
        stack_depth (int): The depth of the call stack to capture for each
            allocation. (Supported by some PyTorch versions).
        context (str): The type of memory events to record.
            'alloc': records only allocation events.
            'state': records memory state changes.
            'all': records both.
        stacks (str): The type of call stacks to record.
            'python': records Python stacks.
            'cpp': records C++ stacks (available in some versions).
            'all': records both.
        devices (Union[int, list[int], None]): The device for which to enable
            memory history. `None` enables it for the current default device.
        record_context (bool): Whether to record context information for
            allocations. Required by older PyTorch versions.
    """
    f = torch.cuda.memory._record_memory_history
    params = set(inspect.signature(f).parameters.keys())

    def _one_call(dev_kw=None):
        kwargs = {}
        if "context" in params:
            kwargs["context"] = context
        if "stacks" in params:
            kwargs["stacks"] = stacks
        if "max_entries" in params:
            kwargs["max_entries"] = trace_alloc_max_entries
        elif "trace_alloc_max_entries" in params:
            kwargs["trace_alloc_max_entries"] = trace_alloc_max_entries
        if "stack_depth" in params:
            kwargs["stack_depth"] = stack_depth
        if dev_kw is not None:
            if "device" in params:
                kwargs["device"] = dev_kw
            elif "devices" in params:
                kwargs["devices"] = dev_kw if isinstance(dev_kw, list) else [dev_kw]
        if "record_context" in params:
            kwargs["record_context"] = record_context

        try:
            f(**kwargs)
            return "native", kwargs
        except TypeError:
            try:
                if "trace_alloc_max_entries" in params and "record_context" in params:
                    f(enabled=True, trace_alloc_max_entries=trace_alloc_max_entries, record_context=True)
                    return "legacy", {
                        "enabled": True,
                        "trace_alloc_max_entries": trace_alloc_max_entries,
                        "record_context": True,
                    }
                else:
                    f(enabled=True)
                    return "legacy-min", {"enabled": True}
            except Exception:
                raise

    if devices is None or isinstance(devices, str | int | torch.device):
        mode, used = _one_call(devices if devices is not None else None)
    else:
        mode, used = "multi-device", {}
        for d in list(devices):
            _mode, _used = _one_call(d)
            used[f"dev{d}"] = _used

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    rank = int(os.environ.get("RANK", "0") or 0)
    logger.info(f"[memory_visualize][rank {rank}] recording enabled ({mode}); args={used}")


class MemorySnapshotSampler:
    """
    A utility class that dumps GPU memory snapshots.
    This is useful for monitoring memory usage over a long-running process.

    The dumped files can be visualized with https://docs.pytorch.org/memory_viz

    Args:
        out_dir (str): The directory where the snapshots will be saved.
        tag (str): A tag for the snapshot filenames.
    """

    def __init__(self, out_dir: str = "./mem_snapshots", tag: str = "periodic"):
        self.out_dir = out_dir
        self.tag = tag

    def dump_memory_snapshot(self, out_dir: str = "./outputs/mem_snapshots", tag: str = "snapshot", sub_dir: str = None):
        """
        Generates a memory snapshot and saves it as a pickle file in a specified directory.
        The files are organized by timestamp in subdirectories, with all ranks' files
        placed in the same timestamp subdirectory.

        Args:
            out_dir (str): The directory where the snapshot file will be saved.
                The directory is created if it does not exist.
            tag (str): A string tag to prepend to the filename for easier identification.
            sub_dir (str): A subdirectory to place the snapshot file in.
        """
        if sub_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            out_path = Path(out_dir) / timestamp
        else:
            out_path = Path(out_dir) / sub_dir
        out_path.mkdir(parents=True, exist_ok=True)

        # get the GPU rank on the current process
        rank = os.environ.get("RANK", "0")
        pid = os.getpid()
        # todo(chenyang): check wether we need to sync all ranks before dump
        fname = f"{tag}_rank{rank}_pid{pid}.pickle"
        path = out_path / fname

        try:
            torch.cuda.synchronize()
            # Memory snapshot is CUDA-specific functionality
            torch.cuda.memory._dump_snapshot(str(path))
            logger.info(f"[memory_visualize] dumped: {path}")
        except Exception as e:
            logger.info(f"[memory_visualize][warn] dump failed: {e}")
