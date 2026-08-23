#!/usr/bin/env python3
"""Pure, restart-stable ETA estimation for the low-bit campaign.

The public entry point, :func:`estimate_campaign_eta`, reads only the supplied
campaign state, plan, and wall-clock timestamp.  It performs no filesystem,
process, GPU, or network I/O, so a controller can call it on every heartbeat
and persist the returned JSON object in ``state.json``.

The estimate is deliberately a range rather than a single promise:

* completed-task timestamps provide progressively broader duration history;
* otherwise explicit low-confidence fallback ranges are used;
* running work subtracts its timestamp-derived elapsed time;
* tuning work not created yet is represented as min/likely/max candidate
  counts, with every scenario bounded by the per-model/setting hard cap;
* remaining work is scheduled on eight GPU units, with four-GPU jobs assigned
  first to the two contiguous four-GPU blocks.

The ETA is compute-only.  Human approval, LR-plan commit, preflight
regeneration, and an externally occupied/failed GPU have no finite duration
that can be inferred safely; such blockers are reported in ``assumptions`` and
``blocked`` instead of silently inventing a wait time.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ETA_SCHEMA_VERSION = 1
GPU_CAPACITY = 8
ACTIVE_STATES = frozenset(
    {"LAUNCHING", "RUNNING", "STALLED", "VALIDATING"}
)
TERMINAL_STATES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "OOM",
        "INVALID_RESULT",
        "ORPHANED",
        "SUPERSEDED",
        "CANCELLED",
        "LAUNCH_FAILED",
    }
)
SUCCESS_STATES = frozenset({"SUCCEEDED"})

BOUNDS = ("optimistic", "median", "conservative")
BOUND_INDEX = {"optimistic": 0, "median": 1, "conservative": 2}
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}

# These are intentionally broad wall-clock ranges.  They are only used until
# this campaign has timestamped observations.  Model-size scaling is applied
# separately, and every fallback use is made visible in the returned sources.
_FALLBACK_SECONDS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("tuning", "realq_static"): (15 * 60, 30 * 60, 75 * 60),
    ("tuning", "realq"): (20 * 60, 50 * 60, 2 * 60 * 60),
    ("formal", "realq_static"): (30 * 60, 90 * 60, 4 * 60 * 60),
    ("formal", "realq"): (60 * 60, 3 * 60 * 60, 8 * 60 * 60),
    ("formal", "guided_saliency"): (20 * 60, 60 * 60, 3 * 60 * 60),
    ("formal", "baseline"): (20 * 60, 75 * 60, 4 * 60 * 60),
}
_GENERIC_FALLBACK_SECONDS = (30 * 60, 90 * 60, 5 * 60 * 60)


@dataclass(frozen=True)
class _Observation:
    model: str
    stage: str
    kind: str
    method: str
    profile: int
    generation: int
    gpu_count: int
    seconds: float


@dataclass(frozen=True)
class _DurationEstimate:
    seconds: tuple[float, float, float]
    source: str
    confidence: str
    samples: int


@dataclass(frozen=True)
class _Job:
    job_id: str
    stage: str
    kind: str
    method: str
    setting: str | None
    gpu_count: int
    priority: int
    seconds: tuple[float, float, float]
    source: str
    confidence: str
    running_gpus: tuple[int, ...] = ()


def _parse_utc(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str) and value:
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(candidate)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat()


def _finite_nonnegative(value: float) -> float:
    return value if math.isfinite(value) and value >= 0 else 0.0


def _stage_for_spec(spec: Mapping[str, Any]) -> str:
    target = spec.get("target_phase")
    if target == "tune":
        return "tuning"
    if target == "final":
        return "formal"
    phase = spec.get("phase")
    if phase == "tune":
        return "tuning"
    if phase == "final":
        return "formal"
    purpose = str(spec.get("purpose", ""))
    if purpose.startswith("tune_"):
        return "tuning"
    return "formal"


def _kind_for_spec(spec: Mapping[str, Any]) -> str:
    kind = spec.get("kind")
    if isinstance(kind, str) and kind:
        return kind
    method = spec.get("method")
    return str(method) if method else "unknown"


def _method_for_spec(spec: Mapping[str, Any]) -> str:
    method = spec.get("method")
    if isinstance(method, str) and method:
        return method
    return _kind_for_spec(spec)


def _integer(value: Any, default: int = 0) -> int:
    if type(value) is int:
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _task_duration(task: Mapping[str, Any]) -> float | None:
    launched = _parse_utc(task.get("launched_utc"))
    finished = _parse_utc(task.get("finished_utc"))
    if launched is None or finished is None or finished < launched:
        return None
    seconds = (finished - launched).total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return seconds


def _task_elapsed(task: Mapping[str, Any], now: dt.datetime) -> float:
    launched = _parse_utc(task.get("launched_utc"))
    if launched is None or launched > now:
        return 0.0
    return _finite_nonnegative((now - launched).total_seconds())


def _model_multiplier(model: str) -> float:
    """Scale cold-start fallbacks sublinearly by a model's ``*B`` suffix."""
    matches = re.findall(r"(?i)(\d+(?:\.\d+)?)b", model)
    if not matches:
        return 1.0
    size_b = max(float(value) for value in matches)
    return min(4.0, max(1.0, math.sqrt(size_b / 4.0)))


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _range_from_samples(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) == 1:
        value = values[0]
        return value * 0.75, value, value * 1.5
    optimistic = _quantile(values, 0.20)
    median = _quantile(values, 0.50)
    conservative = _quantile(values, 0.90) * 1.20
    return (
        max(1.0, optimistic),
        max(optimistic, median),
        max(median, conservative),
    )


def _observations(tasks: Iterable[Mapping[str, Any]]) -> list[_Observation]:
    result: list[_Observation] = []
    for task in tasks:
        # Failed, OOM, interrupted, invalid, and superseded attempts often end
        # after only a few seconds.  They describe failure latency, not the
        # duration of a successful experiment, and must never train the ETA.
        if str(task.get("status")) not in SUCCESS_STATES:
            continue
        duration = _task_duration(task)
        if duration is None:
            continue
        spec_value = task.get("spec")
        spec = spec_value if isinstance(spec_value, Mapping) else {}
        result.append(
            _Observation(
                model=str(spec.get("model", "unknown")),
                stage=_stage_for_spec(spec),
                kind=_kind_for_spec(spec),
                method=_method_for_spec(spec),
                profile=_integer(spec.get("profile_level")),
                generation=_integer(spec.get("generation")),
                gpu_count=max(1, _integer(task.get("gpu_count"), 1)),
                seconds=duration,
            )
        )
    return result


def _history_levels(
    *,
    model: str,
    stage: str,
    kind: str,
    method: str,
    profile: int,
    generation: int,
) -> tuple[tuple[str, Any], ...]:
    return (
        (
            "observed:model_phase_kind_profile_generation",
            lambda item: (
                item.model == model
                and item.stage == stage
                and item.kind == kind
                and item.method == method
                and item.profile == profile
                and item.generation == generation
            ),
        ),
        (
            "observed:model_phase_kind_profile",
            lambda item: (
                item.model == model
                and item.stage == stage
                and item.kind == kind
                and item.method == method
                and item.profile == profile
            ),
        ),
        (
            "observed:model_phase_kind",
            lambda item: (
                item.model == model
                and item.stage == stage
                and item.kind == kind
                and item.method == method
            ),
        ),
    )


def _duration_estimate(
    spec: Mapping[str, Any],
    *,
    gpu_count: int,
    observations: Sequence[_Observation],
) -> _DurationEstimate:
    model = str(spec.get("model", "unknown"))
    stage = _stage_for_spec(spec)
    kind = _kind_for_spec(spec)
    method = _method_for_spec(spec)
    profile = _integer(spec.get("profile_level"))
    generation = _integer(spec.get("generation"))
    for level_index, (source, predicate) in enumerate(
        _history_levels(
            model=model,
            stage=stage,
            kind=kind,
            method=method,
            profile=profile,
            generation=generation,
        )
    ):
        matching = [item for item in observations if predicate(item)]
        values = [item.seconds for item in matching]
        if not values:
            continue
        source += ":successful"
        if level_index <= 1 and len(values) >= 3:
            confidence = "high"
        elif level_index <= 3 and len(values) >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        return _DurationEstimate(
            seconds=_range_from_samples(values),
            source=source,
            confidence=confidence,
            samples=len(values),
        )

    fallback = _FALLBACK_SECONDS.get(
        (stage, kind),
        _GENERIC_FALLBACK_SECONDS,
    )
    multiplier = _model_multiplier(model)
    return _DurationEstimate(
        seconds=tuple(value * multiplier for value in fallback),
        source=f"fallback:{stage}:{kind}:model_size_scaled",
        confidence="low",
        samples=0,
    )


def _known_job(
    task: Mapping[str, Any],
    *,
    now: dt.datetime,
    observations: Sequence[_Observation],
) -> _Job | None:
    status = str(task.get("status", "PENDING"))
    if status in TERMINAL_STATES:
        return None
    spec_value = task.get("spec")
    spec = spec_value if isinstance(spec_value, Mapping) else {}
    gpu_count = max(1, min(GPU_CAPACITY, _integer(task.get("gpu_count"), 1)))
    estimate = _duration_estimate(
        spec,
        gpu_count=gpu_count,
        observations=observations,
    )
    seconds = estimate.seconds
    running_gpus: tuple[int, ...] = ()
    if status in ACTIVE_STATES:
        elapsed = _task_elapsed(task, now)
        overdue_tails = (
            max(60.0, elapsed * 0.05),
            max(5 * 60.0, elapsed * 0.25),
            max(15 * 60.0, elapsed * 0.75),
        )
        overdue = tuple(value <= elapsed for value in seconds)
        seconds = tuple(
            (
                max(0.0, value - elapsed)
                if not overdue[index]
                else overdue_tails[index]
            )
            for index, value in enumerate(seconds)
        )
        if any(overdue):
            estimate = _DurationEstimate(
                seconds=estimate.seconds,
                source=estimate.source + ":overdue_runtime",
                confidence="low",
                samples=estimate.samples,
            )
        raw_gpus = task.get("gpus")
        if isinstance(raw_gpus, Sequence) and not isinstance(
            raw_gpus, (str, bytes)
        ):
            parsed = tuple(
                value
                for value in (_integer(item, -1) for item in raw_gpus)
                if 0 <= value < GPU_CAPACITY
            )
            if len(parsed) == gpu_count and len(set(parsed)) == gpu_count:
                running_gpus = parsed
    return _Job(
        job_id=str(task.get("task_id", "unknown")),
        stage=_stage_for_spec(spec),
        kind=_kind_for_spec(spec),
        method=str(spec.get("method", "")),
        setting=(
            str(spec.get("setting"))
            if spec.get("setting") is not None
            else None
        ),
        gpu_count=gpu_count,
        priority=_integer(task.get("priority")),
        seconds=seconds,
        source=estimate.source,
        confidence=estimate.confidence,
        running_gpus=running_gpus,
    )


def _synthetic_job(
    *,
    job_id: str,
    spec: Mapping[str, Any],
    gpu_count: int,
    priority: int,
    observations: Sequence[_Observation],
    source_suffix: str = "",
) -> _Job:
    estimate = _duration_estimate(
        spec,
        gpu_count=gpu_count,
        observations=observations,
    )
    source = estimate.source + source_suffix
    return _Job(
        job_id=job_id,
        stage=_stage_for_spec(spec),
        kind=_kind_for_spec(spec),
        method=str(spec.get("method", "")),
        setting=(
            str(spec.get("setting"))
            if spec.get("setting") is not None
            else None
        ),
        gpu_count=gpu_count,
        priority=priority,
        seconds=estimate.seconds,
        source=source,
        confidence=estimate.confidence,
    )


def _allocate_contiguous(
    availability: Sequence[float],
    count: int,
) -> tuple[int, ...]:
    capacity = len(availability)
    if count == 4 and capacity == 8:
        candidates = ((0, 1, 2, 3), (4, 5, 6, 7))
    else:
        candidates = tuple(
            tuple(range(start, start + count))
            for start in range(0, capacity - count + 1)
        )
    return min(
        candidates,
        key=lambda group: (
            max(availability[index] for index in group),
            sum(availability[index] for index in group),
            group,
        ),
    )


def _simulate_jobs(
    jobs: Sequence[_Job],
    *,
    bound: str,
    capacity: int = GPU_CAPACITY,
) -> float:
    """Greedy list-schedule jobs on GPU units and return the makespan.

    Active jobs are restored first using their persisted GPU assignments.
    Pending four-GPU jobs are then assigned before one-GPU backfill.  With the
    campaign's fixed capacity of eight this models two contiguous four-GPU
    workers or eight independent one-GPU workers.
    """
    index = BOUND_INDEX[bound]
    availability = [0.0] * capacity
    running = [job for job in jobs if job.running_gpus]
    pending = [job for job in jobs if not job.running_gpus]
    for job in sorted(running, key=lambda item: item.job_id):
        duration = _finite_nonnegative(job.seconds[index])
        assigned = tuple(
            gpu for gpu in job.running_gpus if 0 <= gpu < capacity
        )
        if (
            len(assigned) != job.gpu_count
            or len(set(assigned)) != job.gpu_count
        ):
            assigned = _allocate_contiguous(availability, job.gpu_count)
        finish = max(availability[gpu] for gpu in assigned) + duration
        for gpu in assigned:
            availability[gpu] = finish

    pending.sort(
        key=lambda item: (
            0 if item.gpu_count == 4 else 1,
            -item.priority,
            item.job_id,
        )
    )
    for job in pending:
        duration = _finite_nonnegative(job.seconds[index])
        if job.gpu_count == 1:
            gpu = min(range(capacity), key=lambda value: availability[value])
            availability[gpu] += duration
            continue
        assigned = _allocate_contiguous(availability, job.gpu_count)
        start = max(availability[gpu] for gpu in assigned)
        finish = start + duration
        for gpu in assigned:
            availability[gpu] = finish
    return max(availability, default=0.0)


def _simulate_stage_jobs(
    jobs: Sequence[_Job],
    *,
    stage: str,
    bound: str,
) -> float:
    """Schedule dependency waves without pretending all DAG nodes are ready."""
    if stage == "tuning":
        static = [job for job in jobs if job.kind == "realq_static"]
        static_ids = {job.job_id for job in static}
        arms = [job for job in jobs if job.job_id not in static_ids]
        return _simulate_jobs(static, bound=bound) + _simulate_jobs(
            arms,
            bound=bound,
        )

    # Formal dependencies:
    #   static -> REAL-Q canary -> remaining REAL-Q settings
    #   saliency -> GuidedGPTQ
    # Independent BF16/GPTAQ work is conservatively placed in the first wave.
    precompute_or_independent: list[_Job] = []
    canary_or_guided: list[_Job] = []
    remaining_realq: list[_Job] = []
    for job in jobs:
        is_realq = job.kind == "realq"
        is_guided = job.method == "guided_gptq"
        if is_realq and job.setting != "2W4A":
            remaining_realq.append(job)
        elif is_realq or is_guided:
            canary_or_guided.append(job)
        else:
            precompute_or_independent.append(job)
    return (
        _simulate_jobs(precompute_or_independent, bound=bound)
        + _simulate_jobs(canary_or_guided, bound=bound)
        + _simulate_jobs(remaining_realq, bound=bound)
    )


def _tasks_mapping(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = state.get("tasks")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(key): value
        for key, value in raw.items()
        if isinstance(value, Mapping)
    }


def _known_counts(
    tasks: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    result = {
        "known": 0,
        "completed": 0,
        "running": 0,
        "pending": 0,
        "succeeded": 0,
        "failed_or_oom": 0,
        "skipped": 0,
    }
    for task in tasks:
        result["known"] += 1
        status = str(task.get("status", "PENDING"))
        if status in ACTIVE_STATES:
            result["running"] += 1
        elif status in TERMINAL_STATES:
            result["completed"] += 1
        else:
            result["pending"] += 1
        if status == "SUCCEEDED":
            result["succeeded"] += 1
        elif status in {"FAILED", "OOM", "INVALID_RESULT", "ORPHANED", "LAUNCH_FAILED"}:
            result["failed_or_oom"] += 1
        elif status in {"SUPERSEDED", "CANCELLED"}:
            result["skipped"] += 1
    return result


def _group_pending_count(
    tasks: Iterable[Mapping[str, Any]],
    *,
    model: str,
    setting: str,
    profile: int,
    generation: int,
) -> int:
    count = 0
    for task in tasks:
        spec_value = task.get("spec")
        spec = spec_value if isinstance(spec_value, Mapping) else {}
        if (
            task.get("status") == "PENDING"
            and spec.get("kind") == "realq"
            and spec.get("phase") == "tune"
            and str(spec.get("model")) == model
            and str(spec.get("setting")) == setting
            and _integer(spec.get("profile_level")) == profile
            and _integer(spec.get("generation")) == generation
        ):
            count += 1
    return count


def _tuning_forecast(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tuning_plan_value = plan.get("tuning")
    tuning_plan = (
        tuning_plan_value if isinstance(tuning_plan_value, Mapping) else {}
    )
    max_attempts = max(
        0,
        _integer(
            tuning_plan.get("max_attempts_per_model_setting"),
            20,
        ),
    )
    coarse_value = tuning_plan.get("lr_candidates")
    coarse_count = (
        len(coarse_value)
        if isinstance(coarse_value, Sequence)
        and not isinstance(coarse_value, (str, bytes))
        else 9
    )
    policy_value = tuning_plan.get("search_policy")
    policy = policy_value if isinstance(policy_value, Mapping) else {}
    reserve = max(
        0,
        _integer(policy.get("reserved_refinement_attempts"), 3),
    )
    normal_likely_total = min(max_attempts, coarse_count + reserve)

    models_value = plan.get("models")
    models = (
        [str(value) for value in models_value]
        if isinstance(models_value, Mapping)
        else []
    )
    settings_value = plan.get("settings")
    settings = (
        [str(value) for value in settings_value]
        if isinstance(settings_value, Mapping)
        else []
    )
    groups_value = state.get("tuning")
    groups = groups_value if isinstance(groups_value, Mapping) else {}
    if not models or not settings:
        discovered = [
            str(key).split("/", 1)
            for key in groups
            if isinstance(key, str) and "/" in key
        ]
        if not models:
            models = list(dict.fromkeys(parts[0] for parts in discovered))
        if not settings:
            settings = list(dict.fromkeys(parts[1] for parts in discovered))

    details: dict[str, Any] = {}
    totals = {"optimistic": 0, "median": 0, "conservative": 0}
    for model in models:
        for setting in settings:
            key = f"{model}/{setting}"
            value = groups.get(key)
            group = value if isinstance(value, Mapping) else {}
            status = str(group.get("status", "WAIT_STATIC"))
            profile = max(0, _integer(group.get("profile_level")))
            generation = max(0, _integer(group.get("generation")))
            attempted = max(0, _integer(group.get("attempt_count")))
            pending = _group_pending_count(
                tasks,
                model=model,
                setting=setting,
                profile=profile,
                generation=generation,
            )
            remaining_slots = max(0, max_attempts - attempted - pending)
            selected_or_blocked = status in {
                "SELECTED",
                "NEEDS_USER_ACTION",
            }
            if selected_or_blocked:
                future = (0, 0, 0)
            else:
                oom_restart = (
                    "OOM" in status.upper()
                    or any(
                        str(task.get("status")) == "OOM"
                        and isinstance(task.get("spec"), Mapping)
                        and task["spec"].get("model") == model
                        and task["spec"].get("setting") == setting
                        and task["spec"].get("phase") == "tune"
                        and _integer(task["spec"].get("generation"))
                        == generation
                        for task in tasks
                    )
                )
                if oom_restart:
                    likely_needed = coarse_count + reserve
                else:
                    likely_needed = max(
                        0,
                        normal_likely_total - attempted - pending,
                    )
                required_minimum = min(remaining_slots, likely_needed)
                future = (
                    required_minimum,
                    required_minimum,
                    remaining_slots,
                )
            details[key] = {
                "status": status,
                "profile_level": profile,
                "generation": generation,
                "attempted": attempted,
                "known_pending": pending,
                "hard_cap": max_attempts,
                "remaining_cap_slots": remaining_slots,
                "future_candidates": {
                    "optimistic": future[0],
                    "median": future[1],
                    "conservative": future[2],
                },
                "conservative_includes_oom_replacements": True,
            }
            for index, bound in enumerate(BOUNDS):
                totals[bound] += future[index]
    return {
        "hard_cap_per_model_setting": max_attempts,
        "coarse_candidates": coarse_count,
        "reserved_refinement_attempts": reserve,
        "future_candidates": totals,
        "groups": details,
    }


def _tuning_synthetic_jobs(
    forecast: Mapping[str, Any],
    *,
    bound: str,
    observations: Sequence[_Observation],
) -> list[_Job]:
    result: list[_Job] = []
    groups_value = forecast.get("groups")
    groups = groups_value if isinstance(groups_value, Mapping) else {}
    for key, value in groups.items():
        if not isinstance(value, Mapping) or "/" not in str(key):
            continue
        model, setting = str(key).split("/", 1)
        candidates_value = value.get("future_candidates")
        candidates = (
            candidates_value
            if isinstance(candidates_value, Mapping)
            else {}
        )
        count = max(0, _integer(candidates.get(bound)))
        for index in range(count):
            spec = {
                "kind": "realq",
                "method": "realq",
                "phase": "tune",
                "model": model,
                "setting": setting,
                "profile_level": _integer(value.get("profile_level")),
                "generation": _integer(value.get("generation")),
                "purpose": "eta_future_tune_candidate",
            }
            result.append(
                _synthetic_job(
                    job_id=f"eta:tune:{bound}:{key}:{index}",
                    spec=spec,
                    gpu_count=1,
                    priority=80,
                    observations=observations,
                    source_suffix=":forecast_candidate",
                )
            )
    return result


def _tuning_static_future_jobs(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    tasks: Sequence[Mapping[str, Any]],
    observations: Sequence[_Observation],
) -> list[_Job]:
    models_value = plan.get("models")
    models = (
        [str(value) for value in models_value]
        if isinstance(models_value, Mapping)
        else []
    )
    represented = {
        _logical_identity(spec)
        for task in tasks
        if str(task.get("status")) in (ACTIVE_STATES | SUCCESS_STATES | {"PENDING"})
        for spec in [
            task.get("spec")
            if isinstance(task.get("spec"), Mapping)
            else {}
        ]
        if _stage_for_spec(spec) == "tuning"
    }
    caches_value = state.get("static_caches")
    caches = caches_value if isinstance(caches_value, Mapping) else {}
    jobs: list[_Job] = []
    for model in models:
        cache_value = caches.get(f"tune/{model}")
        cache = cache_value if isinstance(cache_value, Mapping) else {}
        if cache.get("status") == "READY":
            continue
        spec = {
            "kind": "realq_static",
            "method": "realq_static",
            "phase": "precompute",
            "target_phase": "tune",
            "model": model,
            "setting": None,
            "profile_level": max(0, _integer(cache.get("profile_level"))),
            "generation": 0,
            "purpose": "eta_future_tune_static",
        }
        if _logical_identity(spec) in represented:
            continue
        jobs.append(
            _synthetic_job(
                job_id=f"eta:tune:static:{model}",
                spec=spec,
                gpu_count=1,
                priority=100,
                observations=observations,
                source_suffix=":forecast_tune_static",
            )
        )
    return jobs


def _formal_templates(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any], int, int]]:
    models_value = plan.get("models")
    settings_value = plan.get("settings")
    models = (
        [str(value) for value in models_value]
        if isinstance(models_value, Mapping)
        else []
    )
    settings = (
        [str(value) for value in settings_value]
        if isinstance(settings_value, Mapping)
        else []
    )
    formal_value = state.get("formal")
    formal = formal_value if isinstance(formal_value, Mapping) else {}
    model_states_value = formal.get("models")
    model_states = (
        model_states_value
        if isinstance(model_states_value, Mapping)
        else {}
    )
    templates: list[tuple[str, dict[str, Any], int, int]] = []
    for model in models:
        model_state_value = model_states.get(model)
        model_state = (
            model_state_value
            if isinstance(model_state_value, Mapping)
            else {}
        )
        profile = max(0, _integer(model_state.get("realq_profile_level")))
        generation = max(0, _integer(model_state.get("realq_generation")))
        templates.extend(
            [
                (
                    f"formal:static:{model}",
                    {
                        "kind": "realq_static",
                        "method": "realq_static",
                        "phase": "precompute",
                        "target_phase": "final",
                        "model": model,
                        "setting": None,
                        "profile_level": max(
                            0,
                            _integer(model_state.get("static_profile_level")),
                        ),
                        "generation": 0,
                    },
                    4,
                    100,
                ),
                (
                    f"formal:guided-saliency:{model}",
                    {
                        "kind": "guided_saliency",
                        "method": "guided_saliency",
                        "phase": "precompute",
                        "model": model,
                        "setting": None,
                        "profile_level": 0,
                        "generation": 0,
                    },
                    1,
                    60,
                ),
                (
                    f"formal:bf16:{model}",
                    {
                        "kind": "baseline",
                        "method": "bf16",
                        "phase": "final",
                        "model": model,
                        "setting": None,
                        "profile_level": 0,
                        "generation": 0,
                    },
                    1,
                    45,
                ),
            ]
        )
        for setting in settings:
            for method in ("gptaq", "guided_gptq"):
                templates.append(
                    (
                        f"formal:{method}:{model}:{setting}",
                        {
                            "kind": "baseline",
                            "method": method,
                            "phase": "final",
                            "model": model,
                            "setting": setting,
                            "profile_level": 0,
                            "generation": 0,
                        },
                        1,
                        45,
                    )
                )
            templates.append(
                (
                    f"formal:realq:{model}:{setting}",
                    {
                        "kind": "realq",
                        "method": "realq",
                        "phase": "final",
                        "model": model,
                        "setting": setting,
                        "profile_level": profile,
                        "generation": generation,
                    },
                    4,
                    90,
                )
            )
    return templates


def _logical_identity(spec: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _stage_for_spec(spec),
        _kind_for_spec(spec),
        spec.get("method"),
        spec.get("model"),
        spec.get("setting"),
    )


def _formal_future_jobs(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    tasks: Sequence[Mapping[str, Any]],
    observations: Sequence[_Observation],
    conservative_retry: bool,
) -> tuple[list[_Job], int]:
    relevant_statuses = ACTIVE_STATES | SUCCESS_STATES | {"PENDING"}
    represented: set[tuple[Any, ...]] = set()
    for task in tasks:
        if str(task.get("status")) not in relevant_statuses:
            continue
        spec_value = task.get("spec")
        spec = spec_value if isinstance(spec_value, Mapping) else {}
        if _stage_for_spec(spec) == "formal":
            represented.add(_logical_identity(spec))

    formal_value = state.get("formal")
    formal = formal_value if isinstance(formal_value, Mapping) else {}
    model_states_value = formal.get("models")
    model_states = (
        model_states_value
        if isinstance(model_states_value, Mapping)
        else {}
    )
    static_value = state.get("static_caches")
    static_caches = static_value if isinstance(static_value, Mapping) else {}
    jobs: list[_Job] = []
    retry_candidates: list[tuple[str, dict[str, Any], int, int]] = []
    for job_id, spec, gpu_count, priority in _formal_templates(state, plan):
        model = str(spec.get("model"))
        kind = _kind_for_spec(spec)
        model_state_value = model_states.get(model)
        model_state = (
            model_state_value
            if isinstance(model_state_value, Mapping)
            else {}
        )
        if (
            kind == "realq_static"
            and isinstance(static_caches.get(f"final/{model}"), Mapping)
            and static_caches[f"final/{model}"].get("status") == "READY"
        ):
            continue
        if (
            kind == "guided_saliency"
            and model_state.get("guided_status") == "READY"
        ):
            continue
        if _logical_identity(spec) in represented:
            continue
        jobs.append(
            _synthetic_job(
                job_id=f"eta:{job_id}",
                spec=spec,
                gpu_count=gpu_count,
                priority=priority,
                observations=observations,
                source_suffix=":forecast_formal",
            )
        )
        if gpu_count == 4:
            retry_candidates.append((job_id, spec, gpu_count, priority))

    retry_count = 0
    if conservative_retry:
        # A single reviewed replacement wave is included.  Later heartbeats
        # replace this risk allowance with concrete generation/profile tasks.
        for job_id, spec, gpu_count, priority in retry_candidates:
            replacement = dict(spec)
            replacement["profile_level"] = (
                _integer(spec.get("profile_level")) + 1
            )
            replacement["generation"] = _integer(spec.get("generation")) + 1
            jobs.append(
                _synthetic_job(
                    job_id=f"eta:{job_id}:oom-replacement",
                    spec=replacement,
                    gpu_count=gpu_count,
                    priority=priority,
                    observations=observations,
                    source_suffix=":forecast_one_oom_replacement",
                )
            )
            retry_count += 1
    return jobs, retry_count


def _known_formal_oom_replacements(jobs: Sequence[_Job]) -> list[_Job]:
    result: list[_Job] = []
    for job in jobs:
        if job.stage != "formal" or job.gpu_count != 4:
            continue
        result.append(
            _Job(
                job_id=f"{job.job_id}:eta-oom-replacement",
                stage=job.stage,
                kind=job.kind,
                method=job.method,
                setting=job.setting,
                gpu_count=job.gpu_count,
                priority=job.priority,
                seconds=job.seconds,
                source=job.source + ":forecast_one_oom_replacement",
                confidence="low",
                running_gpus=(),
            )
        )
    return result


def _source_summary(jobs: Sequence[_Job]) -> tuple[dict[str, Any], str]:
    sources: dict[str, dict[str, Any]] = {}
    confidence = "high"
    for job in jobs:
        entry = sources.setdefault(
            job.source,
            {"jobs": 0, "confidence": job.confidence},
        )
        entry["jobs"] += 1
        if CONFIDENCE_ORDER[job.confidence] < CONFIDENCE_ORDER[
            entry["confidence"]
        ]:
            entry["confidence"] = job.confidence
        if CONFIDENCE_ORDER[job.confidence] < CONFIDENCE_ORDER[confidence]:
            confidence = job.confidence
    return sources, confidence


def _eta_payload(seconds: float, now: dt.datetime) -> dict[str, Any]:
    normalized = round(_finite_nonnegative(seconds), 3)
    return {
        "remaining_seconds": normalized,
        "remaining_hours": round(normalized / 3600.0, 3),
        "finish_utc": _iso_utc(now + dt.timedelta(seconds=normalized)),
    }


def _campaign_elapsed(
    state: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    now: dt.datetime,
) -> tuple[float, str]:
    state_created = _parse_utc(state.get("created_utc"))
    task_timestamps = [
        parsed
        for task in tasks
        for parsed in (
            _parse_utc(task.get("created_utc")),
            _parse_utc(task.get("launched_utc")),
        )
        if parsed is not None
    ]
    candidates = [
        value for value in [state_created, *task_timestamps] if value is not None
    ]
    created = min(candidates) if candidates else now
    if state_created is not None and created == state_created:
        source = "minimum_of_state_and_task_timestamps:state.created_utc"
    elif task_timestamps:
        source = "minimum_of_state_and_task_timestamps:earliest_task"
    else:
        source = "no_timestamp_available"
    return (
        round(_finite_nonnegative((now - created).total_seconds()), 3),
        source,
    )


def estimate_campaign_eta(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    now_utc: str | dt.datetime,
) -> dict[str, Any]:
    """Return a JSON-serializable compute ETA for one campaign heartbeat.

    ``now_utc`` is mandatory so callers and tests control the clock.  Repeating
    the call after a controller restart with the same persisted state and
    timestamp produces the same result.
    """
    now = _parse_utc(now_utc)
    if now is None:
        raise ValueError("now_utc must be an ISO-8601 timestamp or datetime")
    tasks_by_id = _tasks_mapping(state)
    tasks = list(tasks_by_id.values())
    observations = _observations(tasks)
    known_jobs = [
        job
        for task in tasks
        for job in [_known_job(task, now=now, observations=observations)]
        if job is not None
    ]
    stage_tasks = {
        stage: [
            task
            for task in tasks
            if _stage_for_spec(
                task.get("spec")
                if isinstance(task.get("spec"), Mapping)
                else {}
            )
            == stage
        ]
        for stage in ("tuning", "formal")
    }
    tuning_forecast = _tuning_forecast(state, plan, tasks=tasks)
    tuning_static_jobs = _tuning_static_future_jobs(
        state,
        plan,
        tasks=tasks,
        observations=observations,
    )

    stage_jobs: dict[str, dict[str, list[_Job]]] = {
        stage: {bound: [] for bound in BOUNDS}
        for stage in ("tuning", "formal")
    }
    for job in known_jobs:
        for bound in BOUNDS:
            stage_jobs[job.stage][bound].append(job)
    for bound in BOUNDS:
        stage_jobs["tuning"][bound].extend(tuning_static_jobs)
        stage_jobs["tuning"][bound].extend(
            _tuning_synthetic_jobs(
                tuning_forecast,
                bound=bound,
                observations=observations,
            )
        )
        formal_jobs, _ = _formal_future_jobs(
            state,
            plan,
            tasks=tasks,
            observations=observations,
            conservative_retry=(bound == "conservative"),
        )
        stage_jobs["formal"][bound].extend(formal_jobs)
        if bound == "conservative":
            stage_jobs["formal"][bound].extend(
                _known_formal_oom_replacements(known_jobs)
            )

    formal_base, _ = _formal_future_jobs(
        state,
        plan,
        tasks=tasks,
        observations=observations,
        conservative_retry=False,
    )
    formal_conservative, formal_retry_count = _formal_future_jobs(
        state,
        plan,
        tasks=tasks,
        observations=observations,
        conservative_retry=True,
    )
    stage_payloads: dict[str, Any] = {}
    stage_seconds: dict[str, dict[str, float]] = {}
    for stage in ("tuning", "formal"):
        estimates: dict[str, Any] = {}
        seconds_by_bound: dict[str, float] = {}
        for bound in BOUNDS:
            seconds = _simulate_stage_jobs(
                stage_jobs[stage][bound],
                stage=stage,
                bound=bound,
            )
            seconds_by_bound[bound] = seconds
            estimates[bound] = _eta_payload(seconds, now)
            estimates[bound]["work_items"] = len(stage_jobs[stage][bound])
        likely_jobs = stage_jobs[stage]["median"]
        sources, confidence = _source_summary(likely_jobs)
        counts = _known_counts(stage_tasks[stage])
        if stage == "tuning":
            future_counts = {
                bound: (
                    _integer(
                        tuning_forecast["future_candidates"].get(bound)
                    )
                    + len(tuning_static_jobs)
                )
                for bound in BOUNDS
            }
        else:
            future_counts = {
                "optimistic": len(formal_base),
                "median": len(formal_base),
                "conservative": len(formal_conservative),
            }
        stage_payloads[stage] = {
            "counts": counts,
            "forecast_future_tasks": future_counts,
            "eta": estimates,
            "confidence": confidence,
            "duration_sources": sources,
        }
        stage_seconds[stage] = seconds_by_bound

    whole_estimates: dict[str, Any] = {}
    for bound in BOUNDS:
        # Tuning must finish and selected LRs must be committed before formal
        # compute is valid, so stage makespans are added rather than overlapped.
        seconds = sum(stage_seconds[stage][bound] for stage in stage_seconds)
        whole_estimates[bound] = _eta_payload(seconds, now)
        whole_estimates[bound]["work_items"] = sum(
            len(stage_jobs[stage][bound]) for stage in stage_jobs
        )
    whole_sources, whole_confidence = _source_summary(
        stage_jobs["tuning"]["median"] + stage_jobs["formal"]["median"]
    )
    elapsed, elapsed_source = _campaign_elapsed(state, tasks, now)
    blockers_value = state.get("blockers")
    blockers = (
        [str(value) for value in blockers_value]
        if isinstance(blockers_value, Sequence)
        and not isinstance(blockers_value, (str, bytes))
        else []
    )
    status = str(state.get("status", "UNKNOWN"))
    blocked = bool(blockers) or status.startswith("WAIT_") or status == (
        "NEEDS_USER_ACTION"
    )
    fallback_jobs = sum(
        int(item["jobs"])
        for source, item in whole_sources.items()
        if source.startswith("fallback:")
    )
    assumptions = [
        (
            "compute-only ETA; excludes human LR commit, preflight "
            "regeneration, queueing, and external GPU blockage"
        ),
        (
            "capacity is eight GPUs: at most eight one-GPU jobs or two "
            "contiguous four-GPU jobs; ready four-GPU work is prioritized"
        ),
        (
            "running-task residual is estimated total duration minus elapsed "
            "time from launched_utc"
        ),
        (
            "duration history uses only SUCCEEDED tasks and falls back from "
            "model×phase×kind/method×profile×generation to broader cohorts "
            "within the same model, phase, and kind/method; otherwise it "
            "retains the explicit model-scaled cold-start range"
        ),
        (
            "tuning conservative candidates consume all remaining hard-cap "
            "slots, including possible OOM profile replacements"
        ),
        (
            "formal conservative ETA includes one additional four-GPU OOM "
            "replacement wave; later heartbeats replace it with concrete work"
        ),
    ]
    if blocked:
        assumptions.append(
            "campaign is currently blocked; finish timestamps assume the "
            "blocker clears immediately"
        )
    if fallback_jobs:
        assumptions.append(
            f"{fallback_jobs} likely-scenario jobs use low-confidence cold-start "
            "duration ranges"
        )
    overdue_running_tasks = [
        job.job_id for job in known_jobs if job.source.endswith(":overdue_runtime")
    ]
    return {
        "schema_version": ETA_SCHEMA_VERSION,
        "generated_at_utc": _iso_utc(now),
        "elapsed_seconds": elapsed,
        "elapsed_hours": round(elapsed / 3600.0, 3),
        "elapsed_source": elapsed_source,
        "counts": _known_counts(tasks),
        "overdue_running_tasks": overdue_running_tasks,
        "capacity": {
            "gpu_count": GPU_CAPACITY,
            "one_gpu_parallelism": 8,
            "four_gpu_parallelism": 2,
            "four_gpu_priority": True,
        },
        "blocked": blocked,
        "campaign_status": status,
        "blockers": blockers,
        "stages": stage_payloads,
        "whole": {
            "eta": whole_estimates,
            "confidence": whole_confidence,
            "duration_sources": whole_sources,
        },
        "tuning_candidate_forecast": tuning_forecast,
        "tuning_static_forecast": {
            "base_missing_tasks": len(tuning_static_jobs),
        },
        "formal_forecast": {
            "base_missing_tasks": len(formal_base),
            "conservative_one_wave_oom_replacements": (
                formal_retry_count
                + len(_known_formal_oom_replacements(known_jobs))
            ),
        },
        "assumptions": assumptions,
    }


__all__ = ["ETA_SCHEMA_VERSION", "GPU_CAPACITY", "estimate_campaign_eta"]
