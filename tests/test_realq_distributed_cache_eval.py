from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from realq import pipeline
from realq.precompute import cache as cache_mod
from realq.precompute import static_e2e


@pytest.mark.parametrize(
    ("case", "local_hit", "global_min", "expected"),
    [
        ("all-hit", True, 1, True),
        ("partial-hit-local-hit", True, 0, False),
        ("partial-hit-local-miss", False, 0, False),
        ("all-miss", False, 0, False),
    ],
)
def test_static_cache_hit_requires_collective_all_hit(
    monkeypatch,
    case,
    local_hit,
    global_min,
    expected,
):
    del case
    calls = []
    monkeypatch.setattr(
        static_e2e.parallel_env,
        "is_dist_available_and_initialized",
        lambda: True,
    )
    monkeypatch.setattr(static_e2e.dist, "get_backend", lambda: "gloo")

    def fake_all_reduce(tensor, op):
        calls.append((tensor.item(), tensor.device.type, op))
        tensor.fill_(global_min)

    monkeypatch.setattr(static_e2e.dist, "all_reduce", fake_all_reduce)

    assert static_e2e._all_ranks_have_cache(local_hit, world=2) is expected
    assert calls == [
        (int(local_hit), "cpu", static_e2e.dist.ReduceOp.MIN),
    ]


def test_static_cache_consensus_rejects_uninitialized_multi_rank_group(monkeypatch):
    monkeypatch.setattr(
        static_e2e.parallel_env,
        "is_dist_available_and_initialized",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="initialized process group"):
        static_e2e._all_ranks_have_cache(True, world=2)


def test_static_cache_single_rank_needs_no_collective(monkeypatch):
    monkeypatch.setattr(
        static_e2e.dist,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("unexpected all_reduce"),
    )
    assert static_e2e._all_ranks_have_cache(True, world=1)
    assert not static_e2e._all_ranks_have_cache(False, world=1)


@pytest.mark.parametrize(
    (
        "case",
        "cached",
        "all_hit",
        "fsdp",
        "returns_cached",
        "expected_teardown",
    ),
    [
        (
            "all-hit-non-fsdp",
            {"saliency": ["rank-local"], "fisher": ["global"]},
            True,
            False,
            True,
            ["model.cpu", "cleanup"],
        ),
        (
            "all-hit-fsdp",
            {"saliency": ["rank-local"], "fisher": ["global"]},
            True,
            True,
            True,
            [],
        ),
        (
            "partial-hit",
            {"saliency": ["rank-local"], "fisher": ["global"]},
            False,
            False,
            False,
            [],
        ),
        ("all-miss", None, False, False, False, []),
    ],
)
def test_static_e2e_only_returns_cache_on_collective_all_hit(
    monkeypatch,
    case,
    cached,
    all_hit,
    fsdp,
    returns_cached,
    expected_teardown,
):
    del case

    class RecomputeStarted(Exception):
        pass

    teardown = []

    class Model:
        def cpu(self):
            teardown.append("model.cpu")
            return self

    cfg = SimpleNamespace(
        static_cache_path="/unused-cache",
        tokens_cache_path=None,
        dataset="unused",
        seq_len=4,
        nsamples=2,
        seed=0,
        fsdp=fsdp,
    )
    analyzer = SimpleNamespace(tokenizer=object(), model=Model())
    monkeypatch.setattr(static_e2e.parallel_env, "get_rank", lambda: 0)
    monkeypatch.setattr(static_e2e.parallel_env, "get_world_size", lambda: 2)
    monkeypatch.setattr(cache_mod, "build_cache_key", lambda *_args: "key")
    monkeypatch.setattr(cache_mod, "try_load", lambda *_args: cached)
    monkeypatch.setattr(
        static_e2e.mem_utils,
        "cleanup_memory",
        lambda: teardown.append("cleanup"),
    )

    def consensus(local_hit, world):
        assert local_hit is (cached is not None)
        assert world == 2
        return all_hit

    monkeypatch.setattr(static_e2e, "_all_ranks_have_cache", consensus)

    def raise_recompute(*_args, **_kwargs):
        raise RecomputeStarted

    monkeypatch.setattr(static_e2e.data_utils, "get_tokens", raise_recompute)

    if returns_cached:
        result = static_e2e.run(cfg, analyzer)
        assert result.saliency == cached["saliency"]
        assert result.fisher == cached["fisher"]
    else:
        with pytest.raises(RecomputeStarted):
            static_e2e.run(cfg, analyzer)
    assert teardown == expected_teardown


def test_static_e2e_required_cache_hit_refuses_recompute(monkeypatch):
    cfg = SimpleNamespace(
        static_cache_path="/required-cache",
        require_static_cache_hit=True,
        fsdp=False,
    )
    analyzer = SimpleNamespace(tokenizer=object(), model=object())
    monkeypatch.setattr(static_e2e.parallel_env, "get_rank", lambda: 0)
    monkeypatch.setattr(static_e2e.parallel_env, "get_world_size", lambda: 1)
    monkeypatch.setattr(cache_mod, "build_cache_key", lambda *_args: "key")
    monkeypatch.setattr(cache_mod, "try_load", lambda *_args: None)

    with pytest.raises(RuntimeError, match="required static cache hit"):
        static_e2e.run(cfg, analyzer)


def test_static_cache_save_atomically_replaces_in_same_directory(
    tmp_path,
    monkeypatch,
):
    key = "cache-key"
    target = Path(cache_mod.cache_path(str(tmp_path), key, world_size=2, rank=1))
    target.write_bytes(b"old-complete-cache")
    temporary_paths = []

    def fake_torch_save(payload, path):
        assert payload == {"saliency": [], "fisher": []}
        tmp = Path(path)
        temporary_paths.append(tmp)
        assert tmp.parent == target.parent
        assert tmp != target
        # Until serialization completes and os.replace runs, a concurrent
        # reader still sees the previous complete cache.
        assert target.read_bytes() == b"old-complete-cache"
        tmp.write_bytes(b"new-complete-cache")

    monkeypatch.setattr(cache_mod.torch, "save", fake_torch_save)

    written = cache_mod.save(
        str(tmp_path),
        key,
        world_size=2,
        rank=1,
        payload={"saliency": [], "fisher": []},
    )

    assert written == str(target)
    assert target.read_bytes() == b"new-complete-cache"
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_static_cache_save_cleans_temp_and_preserves_old_file_on_failure(
    tmp_path,
    monkeypatch,
):
    key = "cache-key"
    target = Path(cache_mod.cache_path(str(tmp_path), key, world_size=1, rank=0))
    target.write_bytes(b"old-complete-cache")
    temporary_paths = []

    def failing_torch_save(_payload, path):
        tmp = Path(path)
        temporary_paths.append(tmp)
        tmp.write_bytes(b"partial")
        raise OSError("serialization failed")

    monkeypatch.setattr(cache_mod.torch, "save", failing_torch_save)

    with pytest.raises(OSError, match="serialization failed"):
        cache_mod.save(
            str(tmp_path),
            key,
            world_size=1,
            rank=0,
            payload={"saliency": [], "fisher": []},
        )

    assert target.read_bytes() == b"old-complete-cache"
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_static_cache_unreadable_or_invalid_payload_is_a_miss(tmp_path, monkeypatch):
    key = "cache-key"
    path = Path(cache_mod.cache_path(str(tmp_path), key, world_size=1, rank=0))
    path.write_bytes(b"broken")

    def failing_load(*_args, **_kwargs):
        raise RuntimeError("bad archive")

    monkeypatch.setattr(cache_mod.torch, "load", failing_load)
    assert cache_mod.try_load(str(tmp_path), key, world_size=1, rank=0) is None

    monkeypatch.setattr(cache_mod.torch, "load", lambda *_args, **_kwargs: {})
    assert cache_mod.try_load(str(tmp_path), key, world_size=1, rank=0) is None


@pytest.mark.parametrize(
    ("is_main", "is_distributed", "expected_events"),
    [
        (
            True,
            True,
            ["barrier", "destroy", "distribute", "qa_eval"],
        ),
        (
            False,
            True,
            ["barrier", "destroy"],
        ),
        (
            True,
            False,
            ["distribute", "qa_eval"],
        ),
    ],
)
def test_lm_eval_lifecycle_matches_legacy(
    monkeypatch,
    is_main,
    is_distributed,
    expected_events,
):
    events = []
    model = object()
    tokenizer = object()
    analyzer = SimpleNamespace(model=model, tokenizer=tokenizer)
    cfg = SimpleNamespace(
        lm_eval=True,
        skip_eval=False,
        lm_eval_batch_size=7,
    )

    monkeypatch.setattr(pipeline.parallel_env, "is_main", lambda: is_main)
    monkeypatch.setattr(
        pipeline.parallel_env,
        "is_dist_available_and_initialized",
        lambda: is_distributed,
    )
    monkeypatch.setattr(
        pipeline.parallel_env,
        "barrier",
        lambda: events.append("barrier"),
    )
    monkeypatch.setattr(
        pipeline.dist,
        "destroy_process_group",
        lambda: events.append("destroy"),
    )

    def distribute(current):
        assert current is model
        events.append("distribute")

    def qa_eval(current_model, current_tokenizer, batch_size):
        assert current_model is model
        assert current_tokenizer is tokenizer
        assert batch_size == 7
        events.append("qa_eval")

    monkeypatch.setattr(pipeline.dist_utils, "distribute_model", distribute)
    monkeypatch.setattr(pipeline.eval_utils, "qa_eval", qa_eval)

    assert pipeline._run_lm_eval_if_requested(cfg, analyzer)
    assert events == expected_events


@pytest.mark.parametrize(
    ("lm_eval", "skip_eval"),
    [
        (False, False),
        (False, True),
        (True, True),
    ],
)
def test_lm_eval_respects_skip_eval(monkeypatch, lm_eval, skip_eval):
    analyzer = SimpleNamespace(model=object(), tokenizer=object())
    cfg = SimpleNamespace(
        lm_eval=lm_eval,
        skip_eval=skip_eval,
        lm_eval_batch_size=1,
    )
    monkeypatch.setattr(
        pipeline.parallel_env,
        "is_main",
        lambda: pytest.fail("lm_eval lifecycle should not start"),
    )

    assert not pipeline._run_lm_eval_if_requested(cfg, analyzer)
